// Copyright the Hastur authors.
// SPDX-License-Identifier: LicenseRef-SAM-License
#include "CropAffine.h"

#include <cmath>

namespace hastur {

namespace {

// Bilinear sample of an interleaved HWC image (3 ch) at continuous (x,y) with a
// constant zero border, matching cv2.warpAffine INTER_LINEAR + BORDER_CONSTANT.
inline void SampleBilinear(const float* img, int W, int H, float x, float y,
                           float* out3) {
  const int x0 = static_cast<int>(std::floor(x));
  const int y0 = static_cast<int>(std::floor(y));
  const float wx = x - x0;
  const float wy = y - y0;
  for (int k = 0; k < 3; ++k) out3[k] = 0.f;
  for (int dy = 0; dy < 2; ++dy) {
    const int yy = y0 + dy;
    if (yy < 0 || yy >= H) continue;
    const float wyk = dy ? wy : (1.f - wy);
    for (int dx = 0; dx < 2; ++dx) {
      const int xx = x0 + dx;
      if (xx < 0 || xx >= W) continue;
      const float wxk = dx ? wx : (1.f - wx);
      const float w = wxk * wyk;
      const float* p = img + (static_cast<size_t>(yy) * W + xx) * 3;
      out3[0] += w * p[0];
      out3[1] += w * p[1];
      out3[2] += w * p[2];
    }
  }
}

}  // namespace

CropInputs MakeCrop(const float* rgb, int W, int H, const BBox& box,
                    const CamInt& K, float pad) {
  CropInputs out;
  const int S = kBodyImageSize;

  BBoxCS cs = ProcessBBox(box, pad, S, S);
  out.center = cs.center;
  out.scale = cs.scale;

  std::array<float, 6> fwd, inv;
  MakeAffine(cs.center, cs.square, S, fwd, inv);
  out.affine = fwd;
  out.affine_inv = inv;

  // Warp: for each dst crop pixel, map to source via the inverse affine and
  // bilinear-sample (this is exactly what cv2.warpAffine does internally).
  out.image.assign(static_cast<size_t>(3) * S * S, 0.f);
  const size_t plane = static_cast<size_t>(S) * S;
  float px[3];
  for (int v = 0; v < S; ++v) {
    for (int u = 0; u < S; ++u) {
      const float sx = inv[0] * u + inv[1] * v + inv[2];
      const float sy = inv[3] * u + inv[4] * v + inv[5];
      SampleBilinear(rgb, W, H, sx, sy, px);
      const size_t idx = static_cast<size_t>(v) * S + u;
      // ImageNet normalize + pack CHW.
      out.image[0 * plane + idx] = (px[0] - kImageNetMean[0]) / kImageNetStd[0];
      out.image[1 * plane + idx] = (px[1] - kImageNetMean[1]) / kImageNetStd[1];
      out.image[2 * plane + idx] = (px[2] - kImageNetMean[2]) / kImageNetStd[2];
    }
  }

  out.condition_info = ConditionInfo(box, pad, K, S, S);
  RayCond(box, pad, K, out.ray_cond, S, S);
  return out;
}

}  // namespace hastur
