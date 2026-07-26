// Copyright the Hastur authors.
// SPDX-License-Identifier: LicenseRef-SAM-License
//
// voronoi_validate — self-contained checks for VoronoiLabels: the multi-label
// nearest-seed (Voronoi) assignment that backs the Matte Partition node. Asserts
// against brute force. No models / ONNX / OFX. Exit code 0 = all pass.
//
//   (a) two seed points  -> the perpendicular bisector splits labels correctly
//   (b) random seeds 64x64 -> assigned-seed distance == brute-force min distance
//   (c) the max_dist clamp -> pixels beyond R stay unassigned (-1)
//   (d) all unseeded       -> every pixel is -1
// Plus a timing run on a 1920x1080 grid.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "VoronoiLabel.h"

using namespace hastur;

static int g_fails = 0;
static void check(bool ok, const char* msg) {
  if (!ok) {
    std::printf("FAIL: %s\n", msg);
    ++g_fails;
  }
}

// Brute-force squared distance from (x,y) to the nearest seed pixel.
static long BruteMinDist2(const std::vector<int>& seeds, int W, int H, int x,
                          int y) {
  long best = -1;
  for (int sy = 0; sy < H; ++sy)
    for (int sx = 0; sx < W; ++sx) {
      if (seeds[sy * W + sx] < 0) continue;
      const long dx = x - sx, dy = y - sy;
      const long d2 = dx * dx + dy * dy;
      if (best < 0 || d2 < best) best = d2;
    }
  return best;  // -1 if no seed at all
}

int main() {
  // --- (a) two seeds: bisector split -------------------------------------
  {
    const int W = 64, H = 64;
    std::vector<int> seeds(W * H, -1);
    const int ax = 10, ay = 32, bx = 54, by = 32;  // midline x = 32
    seeds[ay * W + ax] = 0;
    seeds[by * W + bx] = 1;
    std::vector<int> out;
    VoronoiLabels(seeds, W, H, 0.0f, out);  // no clamp
    check(out.size() == static_cast<size_t>(W * H), "(a) output sized W*H");
    bool split_ok = true;
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) {
        const long da = (long)(x - ax) * (x - ax) + (long)(y - ay) * (y - ay);
        const long db = (long)(x - bx) * (x - bx) + (long)(y - by) * (y - by);
        const int lab = out[y * W + x];
        if (da < db && lab != 0) split_ok = false;   // strictly nearer to A
        if (db < da && lab != 1) split_ok = false;   // strictly nearer to B
      }
    check(split_ok, "(a) bisector splits labels correctly");
    // Spot-check either side of the midline.
    check(out[32 * W + 5] == 0, "(a) left of bisector -> label 0");
    check(out[32 * W + 60] == 1, "(a) right of bisector -> label 1");
  }

  // --- (b) random seeds: assigned distance == brute-force min -------------
  {
    const int W = 64, H = 64;
    std::mt19937 rng(12345);
    std::uniform_int_distribution<int> cell(0, W * H - 1);
    std::vector<int> seeds(W * H, -1);
    std::vector<std::pair<int, int>> pos;  // label -> (x,y)
    int n_placed = 0;
    for (int i = 0; i < 40; ++i) {
      const int c = cell(rng);
      if (seeds[c] >= 0) continue;      // skip collisions
      seeds[c] = n_placed;              // unique label per seed pixel
      pos.emplace_back(c % W, c / W);
      ++n_placed;
    }
    std::vector<int> out;
    std::vector<float> dist;
    VoronoiLabels(seeds, W, H, 0.0f, out, &dist);
    bool ok = true, dist_ok = true;
    for (int y = 0; y < H && ok; ++y)
      for (int x = 0; x < W; ++x) {
        const int lab = out[y * W + x];
        if (lab < 0 || lab >= n_placed) { ok = false; break; }
        // Distance to the seed we were actually assigned...
        const long adx = x - pos[lab].first, ady = y - pos[lab].second;
        const long assigned_d2 = adx * adx + ady * ady;
        // ...must equal the brute-force minimum distance (ties may pick another
        // equidistant seed, but the distance is exact).
        const long best = BruteMinDist2(seeds, W, H, x, y);
        if (assigned_d2 != best) { ok = false; break; }
        // The reported Euclidean distance must match too.
        if (std::fabs(dist[y * W + x] - std::sqrt((double)best)) > 1e-3)
          dist_ok = false;
      }
    check(ok, "(b) every pixel's assigned-seed distance == brute-force min");
    check(dist_ok, "(b) reported out_dist2 matches brute-force distance");
  }

  // --- (c) clamp leaves far pixels unassigned ----------------------------
  {
    const int W = 32, H = 32;
    std::vector<int> seeds(W * H, -1);
    seeds[0 * W + 0] = 7;  // single seed at the corner
    std::vector<int> out;
    const float R = 5.0f;
    VoronoiLabels(seeds, W, H, R, out);
    bool clamp_ok = true;
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) {
        const double d = std::sqrt((double)(x * x + y * y));
        const int lab = out[y * W + x];
        if (d <= R) {
          if (lab != 7) clamp_ok = false;         // within R -> assigned
        } else {
          if (lab != -1) clamp_ok = false;        // beyond R -> unassigned
        }
      }
    check(clamp_ok, "(c) clamp: within R -> label, beyond R -> -1");
    check(out[(H - 1) * W + (W - 1)] == -1, "(c) far corner stays -1");
    check(out[0] == 7, "(c) the seed pixel itself keeps its label");
  }

  // --- (d) all unseeded -> all -1 ----------------------------------------
  {
    const int W = 40, H = 24;
    std::vector<int> seeds(W * H, -1);
    std::vector<int> out;
    std::vector<float> dist;
    VoronoiLabels(seeds, W, H, 0.0f, out, &dist);
    bool all_neg = true, all_dneg = true;
    for (int i = 0; i < W * H; ++i) {
      if (out[i] != -1) all_neg = false;
      if (dist[i] != -1.0f) all_dneg = false;
    }
    check(all_neg, "(d) all-unseeded -> every label -1");
    check(all_dneg, "(d) all-unseeded -> every distance -1");
  }

  // --- timing: 1920x1080 -------------------------------------------------
  {
    const int W = 1920, H = 1080;
    std::mt19937 rng(999);
    std::uniform_int_distribution<int> cell(0, W * H - 1);
    std::vector<int> seeds(W * H, -1);
    for (int i = 0; i < 200; ++i) seeds[cell(rng)] = i % 8;  // 8 person ids
    std::vector<int> out;
    const auto t0 = std::chrono::steady_clock::now();
    VoronoiLabels(seeds, W, H, 250.0f, out);
    const auto t1 = std::chrono::steady_clock::now();
    const double ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::printf("timing: 1920x1080 (200 seeds, R=250) -> %.1f ms\n", ms);
    check(out.size() == static_cast<size_t>(W) * H, "timing output sized");
  }

  if (g_fails == 0) std::printf("voronoi_validate: all checks passed\n");
  return g_fails == 0 ? 0 : 1;
}
