// Copyright the Hastur authors.
// SPDX-License-Identifier: LicenseRef-SAM-License
//
// MattePartition.cpp — see MattePartition.h for the contract. Pure C++17: reuses
// VoronoiLabels for the nearest-id assignment and BuildCryptomatte for the packing.

#include "MattePartition.h"

#include <cmath>

#include "Cryptomatte.h"
#include "VoronoiLabel.h"

namespace hastur {

namespace {

// A pixel's rank-0 coverage must exceed this to count as owned (a true seed).
// Feathered Cryptomatte edges below this are treated as unseeded and are instead
// captured by the Voronoi fill from the solid interior.
constexpr float kSeedCoverageThresh = 0.5f;

// Float ids are exact bit patterns (CryptoIdFloat), but compare with a small
// relative epsilon so a value that survived a float32 layer round-trip still
// matches its manifest id.
inline bool IdEquals(float a, float b) {
  return std::fabs(a - b) <= 1e-6f * (1.0f + std::fabs(b));
}

}  // namespace

CryptoResult PartitionCryptomatte(const std::vector<std::vector<float>>& in_layers,
                                  const std::vector<CryptoLabel>& ids,
                                  const std::vector<float>& pool, int W, int H,
                                  float max_dist, int levels,
                                  const std::string& type_name) {
  CryptoResult empty;
  empty.width = W;
  empty.height = H;
  empty.type_name = type_name;
  empty.manifest = "{}";
  if (W <= 0 || H <= 0 || ids.empty()) return empty;

  const size_t npix = static_cast<size_t>(W) * static_cast<size_t>(H);

  // (1) Seed labels from rank 0 of the front-most input layer. Entry = person
  // index whose id owns this pixel (cov0 > thresh), else -1 (unseeded).
  std::vector<int> seed_labels(npix, -1);
  const bool have_input = !in_layers.empty() && in_layers[0].size() >= npix * 4;
  if (have_input) {
    const std::vector<float>& l0 = in_layers[0];
    for (size_t px = 0; px < npix; ++px) {
      const float id0 = l0[px * 4 + 0];
      const float cov0 = l0[px * 4 + 1];
      if (cov0 <= kSeedCoverageThresh) continue;
      for (size_t k = 0; k < ids.size(); ++k) {
        if (IdEquals(id0, ids[k].id)) {
          seed_labels[px] = static_cast<int>(k);
          break;
        }
      }
    }
  }

  // (2) Nearest-owned-id Voronoi partition, clamped at max_dist.
  std::vector<int> labels;
  VoronoiLabels(seed_labels, W, H, max_dist, labels);

  // (3) Partition the pooled soft alpha by the Voronoi labelling: each person gets
  // pool[px] on the pixels nearest to it (and only where pool > 0). Persons that
  // end up all-zero are skipped so they drop out of both layers and the manifest.
  const bool have_pool = pool.size() >= npix;
  std::vector<CryptoPerson> persons;
  persons.reserve(ids.size());
  for (size_t k = 0; k < ids.size(); ++k) {
    std::vector<float> coverage(npix, 0.0f);
    bool any = false;
    for (size_t px = 0; px < npix; ++px) {
      if (labels[px] != static_cast<int>(k)) continue;
      const float a = have_pool ? pool[px] : 0.0f;
      if (a > 0.0f) {
        coverage[px] = a;
        any = true;
      }
    }
    if (any) persons.push_back(CryptoPerson{ids[k].name, std::move(coverage)});
  }

  if (persons.empty()) return empty;

  // (4) Repack into a fresh Cryptomatte. Persons are already in input (front-to-
  // back) order; Voronoi labels are pixel-exclusive so occlusion is a no-op here.
  return BuildCryptomatte(W, H, type_name, persons, levels);
}

}  // namespace hastur
