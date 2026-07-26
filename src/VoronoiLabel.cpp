// Copyright the Hastur authors.
// SPDX-License-Identifier: LicenseRef-SAM-License
//
// VoronoiLabel — see VoronoiLabel.h. Exact Euclidean nearest-seed-label
// assignment via the Felzenszwalb & Huttenlocher two-pass 1D distance transform,
// carrying the nearest-seed identity so labels (not just distances) propagate.

#include "VoronoiLabel.h"

#include <cmath>
#include <vector>

namespace hastur {

namespace {

// A large finite sentinel standing in for +infinity. Kept well below the point
// where adding a squared pixel coordinate (<~ 8.4e6 for an 8K frame) would lose
// its low bits in a double, so mixed finite/infinite columns transform exactly.
constexpr double kInf = 1e18;

// 1D squared-distance transform of the sampled function f over [0, n) using the
// Felzenszwalb–Huttenlocher lower-envelope-of-parabolas method. Writes, for each
// q, the squared distance d[q] = min_p (q-p)^2 + f[p] and the minimizing index
// arg[q] = argmin_p. `v` and `z` are scratch buffers of length >= n+1, supplied
// by the caller so they are allocated once and reused.
void Edt1D(const double* f, int n, double* d, int* arg, int* v, double* z) {
  int k = 0;            // index of the rightmost parabola in the lower envelope
  v[0] = 0;
  z[0] = -kInf;
  z[1] = kInf;
  for (int q = 1; q < n; ++q) {
    // Intersection of the parabola from q with the current top parabola v[k].
    double s = ((f[q] + double(q) * q) - (f[v[k]] + double(v[k]) * v[k])) /
               (2.0 * q - 2.0 * v[k]);
    while (s <= z[k]) {
      --k;
      s = ((f[q] + double(q) * q) - (f[v[k]] + double(v[k]) * v[k])) /
          (2.0 * q - 2.0 * v[k]);
    }
    ++k;
    v[k] = q;
    z[k] = s;
    z[k + 1] = kInf;
  }
  k = 0;
  for (int q = 0; q < n; ++q) {
    while (z[k + 1] < q) ++k;
    const int p = v[k];
    d[q] = double(q - p) * (q - p) + f[p];
    arg[q] = p;
  }
}

}  // namespace

void VoronoiLabels(const std::vector<int>& seed_labels, int W, int H,
                   float max_dist, std::vector<int>& out_labels,
                   std::vector<float>* out_dist2) {
  const size_t n = static_cast<size_t>(W) * static_cast<size_t>(H);
  out_labels.assign(n, -1);
  if (out_dist2) out_dist2->assign(n, -1.0f);
  if (W <= 0 || H <= 0 || seed_labels.size() != n) return;

  // Pass 1 (down each column): for every pixel, the squared distance to the
  // nearest seed *within its own column* and the row index of that seed.
  std::vector<double> d1(n);      // squared distance to nearest in-column seed
  std::vector<int> arg1(n);       // row of that nearest in-column seed

  const int scratch = (W > H ? W : H) + 1;
  std::vector<int> v(scratch);
  std::vector<double> z(scratch);
  std::vector<double> f(scratch), d(scratch);
  std::vector<int> arg(scratch);

  for (int x = 0; x < W; ++x) {
    for (int y = 0; y < H; ++y)
      f[y] = (seed_labels[static_cast<size_t>(y) * W + x] >= 0) ? 0.0 : kInf;
    Edt1D(f.data(), H, d.data(), arg.data(), v.data(), z.data());
    for (int y = 0; y < H; ++y) {
      const size_t idx = static_cast<size_t>(y) * W + x;
      d1[idx] = d[y];
      arg1[idx] = arg[y];  // nearest-seed row in column x
    }
  }

  // Pass 2 (across each row): compose with the column transform to get the exact
  // 2D squared distance and the column of the nearest seed. The nearest seed is
  // then (col=j, row=arg1[y*W+j]); its label is the answer.
  const bool clamp = (max_dist > 0.0f);
  const double r2 = static_cast<double>(max_dist) * static_cast<double>(max_dist);

  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) f[x] = d1[static_cast<size_t>(y) * W + x];
    Edt1D(f.data(), W, d.data(), arg.data(), v.data(), z.data());
    for (int x = 0; x < W; ++x) {
      const size_t idx = static_cast<size_t>(y) * W + x;
      const double dist2 = d[x];
      if (dist2 >= kInf * 0.5) continue;  // no seed anywhere reachable
      if (clamp && dist2 > r2) continue;  // nearest seed beyond the clamp radius
      const int j = arg[x];                        // nearest-seed column
      const int sr = arg1[static_cast<size_t>(y) * W + j];  // nearest-seed row
      out_labels[idx] = seed_labels[static_cast<size_t>(sr) * W + j];
      if (out_dist2) (*out_dist2)[idx] = static_cast<float>(std::sqrt(dist2));
    }
  }
}

}  // namespace hastur
