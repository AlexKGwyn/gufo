#ifndef GUFO_MODELS_QWEN_VISION_PROMPT_HPP_
#define GUFO_MODELS_QWEN_VISION_PROMPT_HPP_

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "src/core/image.hpp"
#include "src/models/qwen/chat_template.hpp"

namespace gufo::models::qwen::vision {

inline constexpr std::uint32_t kImageToken = 248056;
inline constexpr std::uint32_t kPatchSize = 16;
inline constexpr std::uint32_t kMergeSize = 2;

struct ImageGrid {
  std::uint32_t offset{0};  ///< physical first image-pad token
  std::uint32_t height{0};  ///< merged language-token grid, not ViT patches
  std::uint32_t width{0};
  bool operator==(const ImageGrid&) const = default;
};

/// Compact request-owned position state, sufficient to restore RoPE without
/// retaining pixels or encoder scratch in a continuation snapshot.
struct RopeLayout {
  std::vector<ImageGrid> images;
  [[nodiscard]] std::array<std::int32_t, 3> Position(
      std::uint32_t physical) const;
  [[nodiscard]] std::int32_t Delta() const;
  [[nodiscard]] std::uint32_t PrefixLength() const;
  /// Images contributing at least one embedding to the computed prefix.
  [[nodiscard]] RopeLayout Prefix(std::uint32_t token_count) const;
  void Validate(std::uint32_t max_context) const;
  bool operator==(const RopeLayout&) const = default;
};

struct PreparedImage {
  core::Image pixels;  ///< resized RGB8; temporal repetition happens on GPU
  /// Second temporal frame of a video pair, same size as `pixels`. Empty
  /// for still images, whose single frame fills both temporal slots.
  core::Image next;
  ImageGrid grid;
  std::array<std::uint8_t, 32> prefix_identity{};
};

struct Prompt {
  std::vector<tokenization::TokenId> tokens;
  /// Frontier unchanged when an interrupted assistant is followed by a user.
  std::size_t stable_prefix_tokens{0};
  RopeLayout rope;
  std::vector<PreparedImage> images;
  /// SHA-256 covers decoded pixels, grid placement, preprocessing version,
  /// and the model-specific encoder identity. Tokens remain a separate key.
  std::vector<std::uint8_t> cache_identity;
  [[nodiscard]] std::span<const std::uint8_t> IdentityForPrefix(
      std::size_t token_count) const;
};

[[nodiscard]] core::Image ResizeImage(const core::Image& image);
/// Antialiased bicubic resize to an exact, already-chosen size.
[[nodiscard]] core::Image ResizeImageTo(const core::Image& image,
                                        std::uint32_t width,
                                        std::uint32_t height);

/// Reference video sampling: frames at kVideoFps, at least kVideoMinFrames,
/// one pixel budget shared by every sampled frame.
inline constexpr double kVideoFps = 2.0;
inline constexpr std::uint64_t kVideoMinFrames = 4;
inline constexpr std::uint64_t kVideoMaxFrames = 768;
inline constexpr std::uint64_t kVideoMinPixels = 4096;
inline constexpr std::uint64_t kVideoMaxPixels = 25165824;
inline constexpr std::uint32_t kTemporalPatchSize = 2;

struct VideoPlan {
  /// Sampled source frame indices, padded with the last to an even count.
  std::vector<std::uint64_t> frames;
  /// One timestamp per frame pair: the pair's mean source time.
  std::vector<double> seconds;
  std::uint32_t width{0};
  std::uint32_t height{0};
};
[[nodiscard]] VideoPlan PlanVideo(std::uint64_t frame_count, double fps,
                                  std::uint32_t width, std::uint32_t height);
[[nodiscard]] Prompt Prepare(
    const tokenization::QwenTokenizer& tokenizer,
    std::span<const tokenization::ChatMessage> messages,
    std::span<const tokenization::ChatTool> tools,
    const tokenization::ChatTemplateOptions& options,
    std::string_view encoder_identity, std::uint32_t max_context);

}  // namespace gufo::models::qwen::vision
#endif
