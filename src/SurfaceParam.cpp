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

}  // namespace hastur
