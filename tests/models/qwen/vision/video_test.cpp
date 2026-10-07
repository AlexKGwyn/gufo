#include "src/core/video.hpp"

#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/core/image.hpp"
#include "src/models/qwen/chat_template.hpp"
#include "src/models/qwen/control_tokens.hpp"
#include "src/models/qwen/vision/prompt.hpp"

namespace {
using namespace gufo;
namespace vision = models::qwen::vision;

void Expect(bool condition, std::string_view message) {
  if (!condition)
    throw std::runtime_error(std::string(message));
}

bool Near(double a, double b) {
  return std::abs(a - b) < 1e-9;
}

// Expected values follow the Qwen3-VL video processor by hand:
// int(2220 / 29.97 * 2) = 148 frames, linspace(0, 2219, 148).round(), and
// smart_resize(148, 1080, 1920, max_pixels=25165824) = 288 x 544.
void TestPlanLongVideo() {
  const auto plan = vision::PlanVideo(2220, 30000.0 / 1001, 1920, 1080);
  Expect(plan.frames.size() == 148, "2 fps sampling of a 74 s clip");
  Expect(plan.frames.front() == 0 && plan.frames.back() == 2219,
         "sampling spans the first and last frame");
  Expect(plan.frames[1] == 15, "linspace step 2219/147 rounds to 15");
  Expect(plan.seconds.size() == 74, "one timestamp per frame pair");
  Expect(Near(plan.seconds[0], 7.5 / (30000.0 / 1001)),
         "pair timestamp is the mean of its frames");
  Expect(plan.width == 544 && plan.height == 288,
         "one pixel budget is shared by every frame");
}

// Fewer source frames than the minimum: keep every frame, then repeat the
// last one so the count fills whole temporal patches.
void TestPlanShortVideo() {
  const auto plan = vision::PlanVideo(3, 30, 64, 64);
  Expect((plan.frames == std::vector<std::uint64_t>{0, 1, 2, 2}),
         "short video repeats its last frame to an even count");
  Expect(plan.seconds.size() == 2 && Near(plan.seconds[0], 0.5 / 30) &&
             Near(plan.seconds[1], 2.0 / 30),
         "short video timestamps");
  Expect(plan.width == 64 && plan.height == 64,
         "small frames keep their rounded size above the minimum budget");
}

void TestPlanRejectsInvalid() {
  bool threw = false;
  try {
    (void)vision::PlanVideo(0, 30, 64, 64);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  Expect(threw, "empty video is rejected");
}

void TestTemplateVideoMarker() {
  using tokenization::ChatMessage;
  using tokenization::ChatRole;
  auto bytes = std::make_shared<const std::vector<std::uint8_t>>(1, 0);
  std::vector<ChatMessage> messages{{ChatRole::kUser, "Describe."}};
  messages[0].images.push_back({0, bytes, true});
  messages[0].images.push_back({0, bytes});
  tokenization::ChatTemplateOptions options;
  options.add_vision_id = true;
  std::vector<std::size_t> offsets;
  const auto rendered = tokenization::QwenChatTemplate::Render(
      messages, {}, options, nullptr, &offsets);
  Expect(rendered.has_value() && offsets.size() == 2, "two vision parts");
  const std::string video_marker = std::string("Video 1: ") +
                                   std::string(tokenization::kVisionStart) +
                                   std::string(tokenization::kVideoPad) +
                                   std::string(tokenization::kVisionEnd);
  Expect(rendered->find(video_marker) != std::string::npos,
         "video renders a numbered video placeholder");
  Expect(rendered->find("Picture 1: ") != std::string::npos,
         "images and videos are numbered separately");
  Expect(rendered->substr(offsets[0], tokenization::kVideoPad.size()) ==
             tokenization::kVideoPad,
         "video offset addresses its pad");
}

// Decoding uses the FFmpeg executables, as generated-media output does.
void TestDecodeSyntheticVideo() {
  const auto directory = std::filesystem::temp_directory_path();
  const auto path =
      directory / ("gufo-video-test-" + std::to_string(getpid()) + ".mp4");
  const std::string command =
      "ffmpeg -v error -y -f lavfi -i testsrc=size=96x64:rate=10 -t 1 "
      "-pix_fmt yuv420p '" +
      path.string() + "'";
  if (std::system(command.c_str()) != 0) {
    std::cout << "SKIP decode: ffmpeg cannot synthesize a test video\n";
    return;
  }
  const auto bytes = [&] {
    std::vector<std::uint8_t> data(std::filesystem::file_size(path));
    FILE* file = std::fopen(path.c_str(), "rb");
    Expect(file != nullptr &&
               std::fread(data.data(), 1, data.size(), file) == data.size(),
           "read synthetic video");
    std::fclose(file);
    std::filesystem::remove(path);
    return data;
  }();
  const core::EncodedVideo video(bytes);
  Expect(video.info().width == 96 && video.info().height == 64,
         "probe reports dimensions");
  Expect(video.info().frame_count == 10 && Near(video.info().fps, 10),
         "probe reports frame count and rate");
  std::vector<core::Image> frames;
  const std::vector<std::uint64_t> wanted{0, 5, 9, 40};
  const auto count = video.Decode(
      wanted, [&](core::Image frame) { frames.push_back(std::move(frame)); });
  Expect(count == 3 && frames.size() == 3,
         "indices past the end are not delivered");
  Expect(frames[0].pixels != frames[1].pixels,
         "selected frames are distinct pictures");
  // The maximum sampled frame count must still form a valid FFmpeg select.
  std::vector<std::uint64_t> many(vision::kVideoMaxFrames);
  for (std::size_t i = 0; i < many.size(); ++i)
    many[i] = i % 10 == 0 ? i / 10 : 1000 + i;
  std::ranges::sort(many);
  many.erase(std::ranges::unique(many).begin(), many.end());
  std::size_t delivered = 0;
  Expect(video.Decode(many, [&](core::Image) { ++delivered; }) == 10 &&
             delivered == 10,
         "hundreds of selected indices decode the frames that exist");
  Expect(frames[2].width == 96 && frames[2].height == 64 &&
             frames[2].pixels.size() == 96 * 64 * 3,
         "frames are packed RGB8");
}
}  // namespace

int main() {
  try {
    TestPlanLongVideo();
    TestPlanShortVideo();
    TestPlanRejectsInvalid();
    TestTemplateVideoMarker();
    TestDecodeSyntheticVideo();
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
  std::cout << "PASS\n";
  return 0;
}
