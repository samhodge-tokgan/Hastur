// Copyright the Hastur authors.
// SPDX-License-Identifier: LicenseRef-SAM-License
//
// VoronoiLabel — multi-label Voronoi / nearest-seed-label assignment on a pixel
// grid, with a maximum-distance clamp. This is the dependency-free C++ core that
// the Matte Partition node (Phase 6) is built on: it replaces the
// `cv2.distanceTransformWithLabels` used in the skylab `partition.py` prototype
// (the OFX plugin cannot depend on OpenCV).
//
// Given a per-pixel integer seed-label image (label >= 0 where a pixel is a
// Cryptomatte-owned seed of that id; a sentinel < 0 where unseeded), it computes,
// for every pixel, the label of the *nearest seeded pixel* under the exact
// Euclidean metric — the Voronoi partition of the plane by the seed regions.
// Pixels whose nearest seed is farther than `max_dist` pixels are left
// unassigned (-1).
//
// The transform is an exact Euclidean distance transform (Felzenszwalb &
// Huttenlocher two-pass 1D lower-envelope EDT) that also propagates the identity
// of the nearest seed, so the recovered labels are the true nearest-seed labels.
// Distances are exact (integer-squared) up to floating round-off; ties between
// equidistant seeds may resolve to either, but the reported distance is exact.
//
// No OpenCV, no ONNX, no OFX, no threads — pure C++17. 1920x1080 runs in well
// under a second on a single core.

#pragma once

#include <vector>

namespace hastur {

// Assign every pixel the label of its nearest seeded pixel (exact Euclidean),
// clamped at `max_dist`.
//
//   seed_labels  W*H, row-major. Entry >= 0 is a seed of that label id; entry < 0
//                (e.g. -1) is unseeded.
//   W, H         grid dimensions (must be > 0 and W*H == seed_labels.size()).
//   max_dist     clamp radius in pixels: a pixel whose nearest seed is farther
//                than this is left at -1. Pass a value <= 0 to disable the clamp
//                (every pixel gets the nearest label, provided any seed exists).
//   out_labels   resized to W*H; each entry is the nearest-seed label or -1.
//   out_dist2    optional; if non-null, resized to W*H and filled with the
//                Euclidean distance (in pixels) to the assigned seed, or -1.0f
//                where the pixel is unassigned (clamped or no seed anywhere).
//
// If there are no seeds at all, every output pixel is -1.
void VoronoiLabels(const std::vector<int>& seed_labels, int W, int H,
                   float max_dist, std::vector<int>& out_labels,
                   std::vector<float>* out_dist2 = nullptr);

}  // namespace hastur
