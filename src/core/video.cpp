#include "src/core/video.hpp"

#include <fcntl.h>
#include <signal.h>
#include <spawn.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#ifndef GUFO_FFMPEG_EXECUTABLE
#define GUFO_FFMPEG_EXECUTABLE "ffmpeg"
#endif

#ifndef GUFO_FFPROBE_EXECUTABLE
#define GUFO_FFPROBE_EXECUTABLE "ffprobe"
#endif

// POSIX requires the process environment through a mutable-pointer ABI even
// though posix_spawnp does not mutate it.
extern char**
    environ;  // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)

namespace gufo::core {
namespace {

// The child sees the memory file here; /proc reopens it seekably.
constexpr int kChildInput = 3;
constexpr std::string_view kChildPath = "/proc/self/fd/3";
constexpr std::size_t kProbeOutputLimit = 1U << 20U;

const char* Program(const char* variable, const char* fallback) {
  const char* override = std::getenv(variable);
  return override != nullptr && *override != '\0' ? override : fallback;
}

void CloseFd(int& descriptor) {
  if (descriptor >= 0) {
    while (close(descriptor) != 0 && errno == EINTR) {
    }
    descriptor = -1;
  }
}

/// A child whose stdout is a pipe and whose stdin/stderr are /dev/null.
class Child {
public:
  Child(const char* program, std::vector<std::string> arguments, int input) {
    std::vector<char*> argv;
    argv.reserve(arguments.size() + 1);
    for (auto& argument : arguments)
      argv.push_back(argument.data());
    argv.push_back(nullptr);
    int stream[2] = {-1, -1};
    if (pipe2(stream, O_CLOEXEC) != 0)
      throw std::runtime_error("cannot create video decoder pipe");
    posix_spawn_file_actions_t actions;
    int code = posix_spawn_file_actions_init(&actions);
    if (code == 0)
      code = posix_spawn_file_actions_addopen(&actions, STDIN_FILENO,
                                              "/dev/null", O_RDONLY, 0);
    if (code == 0)
      code = posix_spawn_file_actions_adddup2(&actions, stream[1],
                                              STDOUT_FILENO);
    if (code == 0)
      code = posix_spawn_file_actions_addopen(&actions, STDERR_FILENO,
                                              "/dev/null", O_WRONLY, 0);
    if (code == 0)
      code = posix_spawn_file_actions_adddup2(&actions, input, kChildInput);
    if (code == 0)
      code = posix_spawnp(&pid_, program, &actions, nullptr, argv.data(),
                          environ);
    (void)posix_spawn_file_actions_destroy(&actions);
    CloseFd(stream[1]);
    if (code != 0) {
      CloseFd(stream[0]);
      throw std::runtime_error("cannot start " + std::string(program) + ": " +
                               std::strerror(code));
    }
    output_ = stream[0];
    program_ = program;
  }
  ~Child() {
    CloseFd(output_);
    if (pid_ > 0) {
      (void)kill(pid_, SIGKILL);
      (void)Wait();
    }
  }
  Child(const Child&) = delete;
  Child& operator=(const Child&) = delete;

  /// Fills `buffer` unless the stream ends first; returns the bytes read.
  std::size_t Read(std::span<std::uint8_t> buffer) {
    std::size_t filled = 0;
    while (filled < buffer.size()) {
      const ssize_t amount =
          read(output_, buffer.data() + filled, buffer.size() - filled);
      if (amount < 0 && errno == EINTR)
        continue;
      if (amount <= 0)
        break;
      filled += static_cast<std::size_t>(amount);
    }
    return filled;
  }

  /// Closes the pipe and requires a clean exit.
  void Finish() {
    CloseFd(output_);
    const int status = Wait();
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
      throw std::invalid_argument(
          "cannot decode video: " + program_ + " exited with status " +
          std::to_string(WIFEXITED(status) ? WEXITSTATUS(status) : -1));
    }
  }

private:
  int Wait() {
    int status = 0;
    while (waitpid(pid_, &status, 0) < 0 && errno == EINTR) {
    }
    pid_ = -1;
    return status;
  }

  pid_t pid_{-1};
  int output_{-1};
  std::string program_;
};

bool ParseRate(std::string_view text, double& output) {
  const auto slash = text.find('/');
  const auto parse = [](std::string_view value, double& result) {
    const auto [end, error] =
        std::from_chars(value.data(), value.data() + value.size(), result);
    return error == std::errc{} && end == value.data() + value.size();
  };
  if (slash == std::string_view::npos)
    return parse(text, output) && std::isfinite(output) && output > 0;
  double numerator = 0;
  double denominator = 0;
  if (!parse(text.substr(0, slash), numerator) ||
      !parse(text.substr(slash + 1), denominator) || denominator <= 0)
    return false;
  output = numerator / denominator;
  return std::isfinite(output) && output > 0;
}

template <typename T>
bool ParseInteger(std::string_view text, T& output) {
  const auto [end, error] =
      std::from_chars(text.data(), text.data() + text.size(), output);
  return error == std::errc{} && end == text.data() + text.size();
}
}  // namespace

EncodedVideo::EncodedVideo(std::span<const std::uint8_t> bytes) {
  if (bytes.empty() || bytes.size() > kMaxEncodedVideoBytes)
    throw std::invalid_argument("video is empty or exceeds encoded byte limit");
  descriptor_ = memfd_create("gufo-video", MFD_CLOEXEC);
  if (descriptor_ < 0)
    throw std::runtime_error("cannot allocate video memory file");
  for (std::size_t written = 0; written < bytes.size();) {
    const ssize_t amount =
        write(descriptor_, bytes.data() + written, bytes.size() - written);
    if (amount < 0 && errno == EINTR)
      continue;
    if (amount <= 0) {
      CloseFd(descriptor_);
      throw std::runtime_error("cannot stage video bytes");
    }
    written += static_cast<std::size_t>(amount);
  }

  // Packet counting demuxes without decoding; container frame counts are
  // optional and sometimes wrong. V excludes attached cover pictures.
  Child probe(Program("GUFO_FFPROBE", GUFO_FFPROBE_EXECUTABLE),
              {"ffprobe", "-v", "error", "-select_streams", "V:0",
               "-count_packets", "-show_entries",
               "stream=width,height,avg_frame_rate,r_frame_rate,nb_read_"
               "packets:stream_side_data=rotation",
               "-of", "default=noprint_wrappers=1", std::string(kChildPath)},
              descriptor_);
  std::string text(kProbeOutputLimit, '\0');
  text.resize(probe.Read(
      {reinterpret_cast<std::uint8_t*>(text.data()), text.size()}));
  probe.Finish();
  double average_rate = 0;
  double base_rate = 0;
  long rotation = 0;
  std::string_view rest = text;
  while (!rest.empty()) {
    const auto newline = rest.find('\n');
    auto line = rest.substr(0, newline);
    rest = newline == std::string_view::npos ? std::string_view{}
                                             : rest.substr(newline + 1);
    if (!line.empty() && line.back() == '\r')
      line.remove_suffix(1);
    const auto equals = line.find('=');
    if (equals == std::string_view::npos)
      continue;
    const auto key = line.substr(0, equals);
    const auto value = line.substr(equals + 1);
    if (key == "width")
      (void)ParseInteger(value, info_.width);
    else if (key == "height")
      (void)ParseInteger(value, info_.height);
    else if (key == "nb_read_packets")
      (void)ParseInteger(value, info_.frame_count);
    else if (key == "avg_frame_rate")
      (void)ParseRate(value, average_rate);
    else if (key == "r_frame_rate")
      (void)ParseRate(value, base_rate);
    else if (key == "rotation")
      (void)ParseInteger(value, rotation);
  }
  info_.fps = average_rate > 0 ? average_rate : base_rate;
  // FFmpeg autorotates while decoding; report display dimensions.
  if (std::abs(rotation) % 180 == 90)
    std::swap(info_.width, info_.height);
  if (info_.width == 0 || info_.height == 0 || info_.frame_count == 0 ||
      info_.fps <= 0 ||
      std::uint64_t{info_.width} * info_.height > kMaxImagePixels) {
    CloseFd(descriptor_);
    throw std::invalid_argument(
        "video has no decodable stream or unsupported dimensions");
  }
}

EncodedVideo::~EncodedVideo() { CloseFd(descriptor_); }

std::size_t EncodedVideo::Decode(
    std::span<const std::uint64_t> indices,
    const std::function<void(Image)>& frame) const {
  if (indices.empty())
    return 0;
  std::string select = "select='";
  for (std::size_t i = 0; i < indices.size(); ++i) {
    if (i != 0 && indices[i] <= indices[i - 1])
      throw std::invalid_argument("video frame indices must ascend");
    if (i != 0)
      select.push_back('+');
    select += "eq(n," + std::to_string(indices[i]) + ")";
  }
  select.push_back('\'');
  Child decoder(Program("GUFO_FFMPEG", GUFO_FFMPEG_EXECUTABLE),
                {"ffmpeg", "-nostdin", "-v", "error", "-i",
                 std::string(kChildPath), "-map", "0:V:0", "-an", "-sn", "-dn",
                 "-vf", select, "-fps_mode", "passthrough", "-f", "rawvideo",
                 "-pix_fmt", "rgb24", "pipe:1"},
                descriptor_);
  const std::size_t frame_bytes = std::size_t{info_.width} * info_.height * 3;
  std::size_t delivered = 0;
  while (delivered < indices.size()) {
    Image image{info_.width, info_.height,
                std::vector<std::uint8_t>(frame_bytes)};
    const auto amount = decoder.Read(image.pixels);
    if (amount == 0)
      break;
    if (amount != frame_bytes)
      throw std::invalid_argument("video decoder returned a partial frame");
    frame(std::move(image));
    ++delivered;
  }
  // Trailing output means the probe and decoder disagree on frame geometry.
  std::array<std::uint8_t, 1> extra{};
  if (decoder.Read(extra) != 0)
    throw std::invalid_argument("video decoder returned unexpected frames");
  decoder.Finish();
  if (delivered == 0)
    throw std::invalid_argument("video produced no decodable frames");
  return delivered;
}

}  // namespace gufo::core
