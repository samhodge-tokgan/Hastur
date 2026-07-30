// Copyright the Hastur authors.
// SPDX-License-Identifier: LicenseRef-SAM-License
//
// Self-test for ConfidenceOpen / ErodeSoftByRadius (the inter-limb void fix). No ONNX, no models.
#include "ConfidenceOpen.h"

#include <cmath>
#include <cstdio>
#include <vector>

using hastur::ConfidenceOpen;
using hastur::ErodeSoftByRadius;

namespace {
int g_fail = 0;
void check(bool cond, const char* msg) {
  if (!cond) { std::printf("FAIL: %s\n", msg); ++g_fail; }
}
}  // namespace

int main() {
  const int W = 200, H = 200;
  auto idx = [&](int x, int y) { return static_cast<size_t>(y) * W + x; };
  auto inRect = [](int x, int y, int x0, int y0, int x1, int y1) {
    return x >= x0 && x < x1 && y >= y0 && y < y1;
  };

  // --- Case 1: a confident body [40,160)^2 with a LOW-CONFIDENCE enclosed patch [80,120)^2.
  // Expect: the patch OPENS (prob -> 0), the body stays solid (prob > 0.5).
  {
    std::vector<float> logit(static_cast<size_t>(W) * H, -12.f);  // background
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) {
        if (inRect(x, y, 40, 40, 160, 160)) logit[idx(x, y)] = 8.f;          // confident body
        if (inRect(x, y, 80, 80, 120, 120)) logit[idx(x, y)] = 0.7f;         // low-conf gap (~0.67)
      }
    auto out = ConfidenceOpen(logit.data(), W, H, W, H);  // in==out res
    check(out.size() == static_cast<size_t>(W) * H, "case1 size");
    check(out[idx(100, 100)] < 0.5f, "case1 gap should OPEN (prob<0.5 at centre)");
    check(out[idx(50, 50)] > 0.5f, "case1 body corner should stay solid");
    check(out[idx(150, 150)] > 0.5f, "case1 body opposite corner solid");
    check(out[idx(5, 5)] < 0.5f, "case1 background stays background");
  }

  // --- Case 2: a fully confident solid body, no low-conf region -> nothing opens.
  {
    std::vector<float> logit(static_cast<size_t>(W) * H, -12.f);
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x)
        if (inRect(x, y, 40, 40, 160, 160)) logit[idx(x, y)] = 8.f;
    auto out = ConfidenceOpen(logit.data(), W, H, W, H);
    check(out[idx(100, 100)] > 0.5f, "case2 solid interior must NOT open");
    check(out[idx(60, 60)] > 0.5f, "case2 solid body kept");
  }

  // --- Case 3: a SMALL low-conf speck (below minFrac) is NOT opened.
  {
    std::vector<float> logit(static_cast<size_t>(W) * H, -12.f);
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) {
        if (inRect(x, y, 40, 40, 160, 160)) logit[idx(x, y)] = 8.f;
        if (inRect(x, y, 99, 99, 103, 103)) logit[idx(x, y)] = 0.7f;  // ~16 px, tiny
      }
    auto out = ConfidenceOpen(logit.data(), W, H, W, H);
    check(out[idx(100, 100)] > 0.5f, "case3 tiny low-conf speck must NOT open (< minFrac)");
  }

  // --- Case 4: ErodeSoftByRadius zeros the outer band, keeps the deep interior.
  {
    std::vector<float> cov(static_cast<size_t>(W) * H, 0.f);
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x)
        if (inRect(x, y, 40, 40, 160, 160)) cov[idx(x, y)] = 1.f;
    auto e = ErodeSoftByRadius(cov, W, H, 15);
    check(e[idx(100, 100)] > 0.5f, "case4 deep interior kept after erode");
    check(e[idx(45, 100)] == 0.f, "case4 outer band (5 px in) zeroed by radius-15 erode");
    check(e[idx(100, 100)] == cov[idx(100, 100)], "case4 keeps soft value in interior");
  }

  if (g_fail == 0) std::printf("confidence_open_validate: PASS\n");
  else std::printf("confidence_open_validate: %d FAILURES\n", g_fail);
  return g_fail == 0 ? 0 : 1;
}
