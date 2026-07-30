// Copyright the Hastur authors.
// SPDX-License-Identifier: LicenseRef-SAM-License
#include "ConfidenceOpen.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <vector>

namespace hastur {
namespace {

inline float Sigmoid(float z) {
  if (z < -30.f) return 0.f;
  if (z > 30.f) return 1.f;
  return 1.f / (1.f + std::exp(-z));
}

// Bilinear resize of a float field src(sw*sh) -> dst(dw*dh), align-corners-false (pixel centres),
// matching the tracker's ResizeBinToSoft sampling convention.
std::vector<float> ResizeBilinear(const float* src, int sw, int sh, int dw, int dh) {
  std::vector<float> out(static_cast<size_t>(dw) * dh, 0.f);
  for (int oy = 0; oy < dh; ++oy) {
    float iy = (oy + 0.5f) * sh / dh - 0.5f;
    int y0 = static_cast<int>(std::floor(iy));
    float wy = iy - y0;
    int y0c = std::min(std::max(y0, 0), sh - 1);
    int y1c = std::min(std::max(y0 + 1, 0), sh - 1);
    for (int ox = 0; ox < dw; ++ox) {
      float ix = (ox + 0.5f) * sw / dw - 0.5f;
      int x0 = static_cast<int>(std::floor(ix));
      float wx = ix - x0;
      int x0c = std::min(std::max(x0, 0), sw - 1);
      int x1c = std::min(std::max(x0 + 1, 0), sw - 1);
      float v = src[static_cast<size_t>(y0c) * sw + x0c] * (1 - wy) * (1 - wx) +
                src[static_cast<size_t>(y0c) * sw + x1c] * (1 - wy) * wx +
                src[static_cast<size_t>(y1c) * sw + x0c] * wy * (1 - wx) +
                src[static_cast<size_t>(y1c) * sw + x1c] * wy * wx;
      out[static_cast<size_t>(oy) * dw + ox] = v;
    }
  }
  return out;
}

// Chebyshev (chessboard) distance to the nearest background pixel (fg==0), two-pass. interior =
// dist > r is a square-SE erosion of the foreground by radius r. O(N), integer-capped.
std::vector<int> DistToBackground(const std::vector<uint8_t>& fg, int W, int H) {
  const int INF = W + H + 1;
  std::vector<int> d(static_cast<size_t>(W) * H, 0);
  for (size_t i = 0; i < d.size(); ++i) d[i] = fg[i] ? INF : 0;
  auto at = [&](int x, int y) -> int& { return d[static_cast<size_t>(y) * W + x]; };
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) {
      if (at(x, y) == 0) continue;
      int m = at(x, y);
      if (x > 0) m = std::min(m, at(x - 1, y) + 1);
      if (y > 0) m = std::min(m, at(x, y - 1) + 1);
      if (x > 0 && y > 0) m = std::min(m, at(x - 1, y - 1) + 1);
      if (x < W - 1 && y > 0) m = std::min(m, at(x + 1, y - 1) + 1);
      at(x, y) = m;
    }
  for (int y = H - 1; y >= 0; --y)
    for (int x = W - 1; x >= 0; --x) {
      int m = at(x, y);
      if (x < W - 1) m = std::min(m, at(x + 1, y) + 1);
      if (y < H - 1) m = std::min(m, at(x, y + 1) + 1);
      if (x < W - 1 && y < H - 1) m = std::min(m, at(x + 1, y + 1) + 1);
      if (x > 0 && y < H - 1) m = std::min(m, at(x - 1, y + 1) + 1);
      at(x, y) = m;
    }
  return d;
}

float EnvF(const char* name, float dflt) {
  const char* s = std::getenv(name);
  if (!s || !*s) return dflt;
  char* end = nullptr;
  float v = std::strtof(s, &end);
  return (end == s) ? dflt : v;
}

}  // namespace

std::vector<float> ConfidenceOpen(const float* logits, int inW, int inH, int outW, int outH,
                                  float tau, float minFrac, int edgePx) {
  if (!logits || inW <= 0 || inH <= 0 || outW <= 0 || outH <= 0)
    return std::vector<float>(static_cast<size_t>(std::max(0, outW)) * std::max(0, outH), 0.f);

  // logits -> prob at input res, then resize prob to output res.
  std::vector<float> probIn(static_cast<size_t>(inW) * inH);
  for (size_t i = 0; i < probIn.size(); ++i) probIn[i] = Sigmoid(logits[i]);
  std::vector<float> prob = ResizeBilinear(probIn.data(), inW, inH, outW, outH);

  static const bool kOff = std::getenv("HASTUR_CONFOPEN_OFF") != nullptr;
  if (kOff) return prob;

  tau = EnvF("HASTUR_CONFOPEN_TAU", tau);
  minFrac = EnvF("HASTUR_CONFOPEN_MINFRAC", minFrac);
  edgePx = static_cast<int>(EnvF("HASTUR_CONFOPEN_EDGE", static_cast<float>(edgePx)));

  const int W = outW, H = outH;
  const size_t N = static_cast<size_t>(W) * H;

  std::vector<uint8_t> fg(N, 0);
  size_t fgArea = 0;
  for (size_t i = 0; i < N; ++i)
    if (prob[i] > 0.5f) { fg[i] = 1; ++fgArea; }
  if (fgArea == 0) return prob;

  // interior = fg eroded off the silhouette edge by edgePx (protects the soft AA band).
  std::vector<int> dbg = DistToBackground(fg, W, H);

  // low-confidence deep-interior candidates = the negative space the detector is unsure about.
  std::vector<uint8_t> cand(N, 0);
  for (size_t i = 0; i < N; ++i)
    if (fg[i] && dbg[i] > edgePx && prob[i] < tau) cand[i] = 1;

  // keep only sizeable connected components (flood fill, 8-connectivity); zero them in prob.
  const size_t minArea = static_cast<size_t>(std::max(1.0, minFrac * static_cast<double>(fgArea)));
  std::vector<uint8_t> visited(N, 0);
  std::vector<int> stack;
  std::vector<int> comp;
  for (int sy = 0; sy < H; ++sy) {
    for (int sx = 0; sx < W; ++sx) {
      const size_t si = static_cast<size_t>(sy) * W + sx;
      if (!cand[si] || visited[si]) continue;
      stack.clear();
      comp.clear();
      stack.push_back(static_cast<int>(si));
      visited[si] = 1;
      while (!stack.empty()) {
        const int idx = stack.back();
        stack.pop_back();
        comp.push_back(idx);
        const int x = idx % W, y = idx / W;
        for (int dy = -1; dy <= 1; ++dy)
          for (int dx = -1; dx <= 1; ++dx) {
            if (!dx && !dy) continue;
            const int nx = x + dx, ny = y + dy;
            if (nx < 0 || ny < 0 || nx >= W || ny >= H) continue;
            const size_t ni = static_cast<size_t>(ny) * W + nx;
            if (cand[ni] && !visited[ni]) { visited[ni] = 1; stack.push_back(static_cast<int>(ni)); }
          }
      }
      if (comp.size() >= minArea)
        for (int idx : comp) prob[idx] = 0.f;  // OPEN this negative-space blob
    }
  }
  return prob;
}

std::vector<float> ErodeSoftByRadius(const std::vector<float>& cov, int W, int H, int radius) {
  std::vector<float> out = cov;
  if (radius <= 0 || W <= 0 || H <= 0 ||
      cov.size() != static_cast<size_t>(W) * H)
    return out;
  std::vector<uint8_t> fg(cov.size(), 0);
  for (size_t i = 0; i < cov.size(); ++i) fg[i] = cov[i] > 0.5f ? 1 : 0;
  std::vector<int> dbg = DistToBackground(fg, W, H);
  for (size_t i = 0; i < out.size(); ++i)
    if (dbg[i] <= radius) out[i] = 0.f;  // zero the outer `radius` band; keep deep-interior soft
  return out;
}

}  // namespace hastur
