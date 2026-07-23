// Copyright the Hastur authors.
// SPDX-License-Identifier: LicenseRef-SAM-License

#include "SurfaceParam.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace hastur {

std::vector<float> ComputePrefNormalized(const float* base_shape_cm, int nverts) {
  std::vector<float> out;
  if (!base_shape_cm || nverts <= 0) return out;
  out.resize(static_cast<size_t>(nverts) * 3);

  // Flip Y,Z into the posed-mesh frame (matches MhrModel + the Position AOV).
  // The centimetre->metre scale is irrelevant after per-axis normalisation.
  float lo[3] = {std::numeric_limits<float>::infinity(),
                 std::numeric_limits<float>::infinity(),
                 std::numeric_limits<float>::infinity()};
  float hi[3] = {-std::numeric_limits<float>::infinity(),
                 -std::numeric_limits<float>::infinity(),
                 -std::numeric_limits<float>::infinity()};
  for (int i = 0; i < nverts; ++i) {
    const float p[3] = {base_shape_cm[3 * i + 0], -base_shape_cm[3 * i + 1],
                        -base_shape_cm[3 * i + 2]};
    out[3 * i + 0] = p[0];
    out[3 * i + 1] = p[1];
    out[3 * i + 2] = p[2];
    for (int k = 0; k < 3; ++k) {
      lo[k] = std::min(lo[k], p[k]);
      hi[k] = std::max(hi[k], p[k]);
    }
  }

  float inv[3];
  for (int k = 0; k < 3; ++k) {
    const float ext = hi[k] - lo[k];
    inv[k] = ext > 1e-6f ? 1.0f / ext : 0.0f;
  }
  for (int i = 0; i < nverts; ++i)
    for (int k = 0; k < 3; ++k) {
      float t = (out[3 * i + k] - lo[k]) * inv[k];
      out[3 * i + k] = t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
    }
  return out;
}

std::vector<float> ComputeCylindricalUV(const float* base_shape_cm, int nverts) {
  std::vector<float> out;
  if (!base_shape_cm || nverts <= 0) return out;
  out.resize(static_cast<size_t>(nverts) * 2);

  // Azimuth about the vertical axis -> U; normalised height -> V. Computed from
  // the RAW (pre-flip) base_shape to match tools/extract_mhr_assets.py.
  float hmin = std::numeric_limits<float>::infinity();
  float hmax = -std::numeric_limits<float>::infinity();
  for (int i = 0; i < nverts; ++i) {
    const float h = base_shape_cm[3 * i + 1];
    hmin = std::min(hmin, h);
    hmax = std::max(hmax, h);
  }
  const float inv_h = (hmax - hmin) > 1e-6f ? 1.0f / (hmax - hmin) : 0.0f;
  const float kTwoPi = 6.28318530717958647692f;
  for (int i = 0; i < nverts; ++i) {
    const float x = base_shape_cm[3 * i + 0];
    const float y = base_shape_cm[3 * i + 1];
    const float z = base_shape_cm[3 * i + 2];
    float u = std::atan2(x, z) / kTwoPi + 0.5f;  // [0,1), seam at the back
    float v = (y - hmin) * inv_h;                 // [0,1] head..toe
    out[2 * i + 0] = u;
    out[2 * i + 1] = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
  }
  return out;
}

std::vector<float> ComputeRestNormals(const float* verts, const int32_t* faces,
                                      int nverts, int nfaces) {
  std::vector<float> N;
  if (!verts || !faces || nverts <= 0 || nfaces <= 0) return N;
  N.assign(static_cast<size_t>(nverts) * 3, 0.0f);
  // Accumulate area-weighted face normals (unnormalised cross product) onto each
  // incident vertex, then normalise -- matches the rasterizer's vertex normals.
  for (int f = 0; f < nfaces; ++f) {
    const int i0 = faces[3 * f + 0], i1 = faces[3 * f + 1], i2 = faces[3 * f + 2];
    if (i0 < 0 || i1 < 0 || i2 < 0 || i0 >= nverts || i1 >= nverts || i2 >= nverts)
      continue;
    const float* a = &verts[3 * i0];
    const float* b = &verts[3 * i1];
    const float* c = &verts[3 * i2];
    const float e1[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
    const float e2[3] = {c[0] - a[0], c[1] - a[1], c[2] - a[2]};
    const float fn[3] = {e1[1] * e2[2] - e1[2] * e2[1],
                         e1[2] * e2[0] - e1[0] * e2[2],
                         e1[0] * e2[1] - e1[1] * e2[0]};
    for (int k = 0; k < 3; ++k) {
      N[3 * i0 + k] += fn[k];
      N[3 * i1 + k] += fn[k];
      N[3 * i2 + k] += fn[k];
    }
  }
  for (int i = 0; i < nverts; ++i) {
    float* n = &N[3 * i];
    const float len = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
    if (len > 1e-12f) {
      n[0] /= len;
      n[1] /= len;
      n[2] /= len;
    }
  }
  return N;
}

}  // namespace hastur
