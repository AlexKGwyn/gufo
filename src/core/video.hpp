#ifndef GUFO_CORE_VIDEO_HPP_
#define GUFO_CORE_VIDEO_HPP_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>

#include "src/core/image.hpp"

namespace gufo::core {

/// Display-oriented stream facts; width and height already follow rotation.
struct VideoInfo {
  std::uint32_t width{0};
  std::uint32_t height{0};
  std::uint64_t frame_count{0};
  double fps{0};
};

/// In-memory container bytes decoded by the bundled FFmpeg. The bytes stay
/// in an anonymous memory file, so demuxers may seek (MP4 moov at the end).
class EncodedVideo {
public:
  explicit EncodedVideo(std::span<const std::uint8_t> bytes);
  ~EncodedVideo();
  EncodedVideo(const EncodedVideo&) = delete;
  EncodedVideo& operator=(const EncodedVideo&) = delete;

  [[nodiscard]] const VideoInfo& info() const noexcept { return info_; }
  /// Decodes the given ascending, unique frame indices as display-oriented
  /// RGB8 at native resolution, one callback per decoded frame in order.
  /// Indices past the stream's actual end are not delivered; the return
  /// value counts delivered frames.
  std::size_t Decode(std::span<const std::uint64_t> indices,
                     const std::function<void(Image)>& frame) const;

private:
  int descriptor_{-1};
  VideoInfo info_;
};

}  // namespace gufo::core
#endif
