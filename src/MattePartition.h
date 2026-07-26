// Copyright the Hastur authors.
// SPDX-License-Identifier: LicenseRef-SAM-License
//
// MattePartition — the Matte Partition CORE (Phase 6b). Pure C++17, no OFX / GPU /
// OpenCV / ONNX, so it is unit-testable in isolation.
//
// It fuses two inputs into a fresh Cryptomatte:
//   1. an input Cryptomatte from Hastur — per-id, depth-ordered *ownership* of the
//      frame (who owns each pixel, hard/anti-aliased regions), and
//   2. a pooled *soft* alpha — the MatAnyone2 / Kthanid soft foreground matte, a
//      single W*H channel in [0,1] with feathered external edges.
//
// For each pixel it reads the front-most (rank-0) Cryptomatte id, matches it to a
// person, and uses those owned pixels as Voronoi seeds. VoronoiLabels then assigns
// every pixel to the *nearest* owned id region (clamped at `max_dist`). The pooled
// soft alpha is then partitioned by that Voronoi labelling: each person gets the
// soft alpha of the pixels nearest to it. The result:
//   * external edges keep MatAnyone2's softness (the soft alpha is carried through
//     verbatim on the pixels it covers),
//   * internal seams between people follow the Cryptomatte ownership (Voronoi
//     bisectors between the seed regions), and
//   * soft-alpha pixels with no id within `max_dist` are dropped — this kills the
//     "ex-player phantom" halo left over when a tracked person has exited.
//
// This mirrors the validated skylab `partition.py` prototype, but with
// VoronoiLabels standing in for cv2.distanceTransformWithLabels and
// BuildCryptomatte for the packing.

#pragma once

#include <string>
#include <vector>

#include "MeshTypes.h"

namespace hastur {

// One person's Cryptomatte identity: the manifest name plus its Cryptomatte id
// float (== CryptoIdFloat(name)). Carried explicitly so the caller can match the
// ids that were packed into `in_layers` without re-hashing here.
struct CryptoLabel {
  std::string name;  // e.g. "person_00"
  float id;          // CryptoIdFloat(name)
};

// Partition a pooled soft alpha into a fresh Cryptomatte along Voronoi regions of
// the input Cryptomatte's front-most ids.
//
//   in_layers  the packed input Cryptomatte layers (each W*H*4 = ranked
//              [id0,cov0,id1,cov1] pairs, two ranks per layer; rank 0 = front-
//              most). Only rank 0 of layer 0 is read for seeding. May be empty.
//   ids        the manifest: one CryptoLabel per person (name + id float).
//   pool       W*H pooled soft alpha in [0,1], top-down HWC (same layout as
//              CryptoPerson::coverage).
//   W, H       frame dimensions (> 0).
//   max_dist   Voronoi clamp radius in pixels; soft-alpha pixels farther than this
//              from any owned seed are dropped. <= 0 disables the clamp.
//   levels     number of Cryptomatte layers to pack in the result.
//   type_name  Cryptomatte typename for the result (default "person").
//
// Returns a fresh CryptoResult carrying one soft matte per person that received
// any coverage; persons whose partitioned coverage is all-zero (e.g. fully
// occluded, or entirely dropped by the clamp) are omitted from both the layers and
// the manifest. When every input id is present in the frame the manifest/names
// therefore round-trip the input. If W/H are invalid or there are no ids, an empty
// CryptoResult is returned.
CryptoResult PartitionCryptomatte(const std::vector<std::vector<float>>& in_layers,
                                  const std::vector<CryptoLabel>& ids,
                                  const std::vector<float>& pool, int W, int H,
                                  float max_dist, int levels,
                                  const std::string& type_name = "person");

}  // namespace hastur
