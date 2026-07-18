// Copyright the Hastur authors.
// SPDX-License-Identifier: LicenseRef-SAM-License
//
// MeshAssets.h -- loads the versioned flat binary `mhr_assets.bin` (produced by
// tools/extract_mhr_assets.py) that holds every STATIC MHR buffer the C++ LBS
// needs: template/rest vertices, identity blend basis, sparse skin weights, the
// skeleton (parameter transform, prerotations, FK prefix-multiply indices,
// inverse bind pose), the hand-PCA + scale bases, and the keypoint_mapping.
//
// The file format is documented in tools/mhr_binfmt.py and docs/MHR_ASSETS.md.
// This header is Eigen-free so it can be included widely; MhrModel does the math.

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "MeshTypes.h"

namespace hastur {

class MeshAssets {
 public:
  enum class DType : uint32_t { F32 = 0, I32 = 1, I64 = 2 };

  struct Block {
    DType dtype{};
    uint32_t ndim{};
    int64_t shape[4]{1, 1, 1, 1};
    const void* data{nullptr};  // into `raw_`
    uint64_t nbytes{};
    int64_t numel() const { return shape[0] * shape[1] * shape[2] * shape[3]; }
  };

  // Loads and validates the binary. Throws std::runtime_error on failure.
  static std::shared_ptr<MeshAssets> Load(const std::string& path);

  // Typed accessors (throw if missing / wrong dtype).
  const float* f32(const std::string& name) const;
  const int32_t* i32(const std::string& name) const;
  const int64_t* i64(const std::string& name) const;
  const Block& block(const std::string& name) const;
  bool has(const std::string& name) const { return blocks_.count(name) != 0; }

  // Shared static topology (kNumFaces*3), matching Mesh::faces.
  std::shared_ptr<const std::vector<int32_t>> faces() const { return faces_; }

  uint32_t version() const { return version_; }

 private:
  std::vector<uint8_t> raw_;
  std::unordered_map<std::string, Block> blocks_;
  std::shared_ptr<const std::vector<int32_t>> faces_;
  uint32_t version_{};
};

}  // namespace hastur
