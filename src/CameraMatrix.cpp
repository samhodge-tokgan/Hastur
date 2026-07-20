// Copyright the Hastur authors.
// SPDX-License-Identifier: LicenseRef-SAM-License
//
// CameraMatrix.cpp -- see CameraMatrix.h for the contract and conventions.

#include "CameraMatrix.h"

#include <cmath>

namespace hastur {

bool Invert4x4(const std::array<float, 16>& m, std::array<float, 16>& out) {
  double a[16];
  for (int i = 0; i < 16; ++i) a[i] = static_cast<double>(m[i]);

  double inv[16];
  inv[0] = a[5] * a[10] * a[15] - a[5] * a[11] * a[14] - a[9] * a[6] * a[15] +
           a[9] * a[7] * a[14] + a[13] * a[6] * a[11] - a[13] * a[7] * a[10];
  inv[4] = -a[4] * a[10] * a[15] + a[4] * a[11] * a[14] + a[8] * a[6] * a[15] -
           a[8] * a[7] * a[14] - a[12] * a[6] * a[11] + a[12] * a[7] * a[10];
  inv[8] = a[4] * a[9] * a[15] - a[4] * a[11] * a[13] - a[8] * a[5] * a[15] +
           a[8] * a[7] * a[13] + a[12] * a[5] * a[11] - a[12] * a[7] * a[9];
  inv[12] = -a[4] * a[9] * a[14] + a[4] * a[10] * a[13] + a[8] * a[5] * a[14] -
            a[8] * a[6] * a[13] - a[12] * a[5] * a[10] + a[12] * a[6] * a[9];
  inv[1] = -a[1] * a[10] * a[15] + a[1] * a[11] * a[14] + a[9] * a[2] * a[15] -
           a[9] * a[3] * a[14] - a[13] * a[2] * a[11] + a[13] * a[3] * a[10];
  inv[5] = a[0] * a[10] * a[15] - a[0] * a[11] * a[14] - a[8] * a[2] * a[15] +
           a[8] * a[3] * a[14] + a[12] * a[2] * a[11] - a[12] * a[3] * a[10];
  inv[9] = -a[0] * a[9] * a[15] + a[0] * a[11] * a[13] + a[8] * a[1] * a[15] -
           a[8] * a[3] * a[13] - a[12] * a[1] * a[11] + a[12] * a[3] * a[9];
  inv[13] = a[0] * a[9] * a[14] - a[0] * a[10] * a[13] - a[8] * a[1] * a[14] +
            a[8] * a[2] * a[13] + a[12] * a[1] * a[10] - a[12] * a[2] * a[9];
  inv[2] = a[1] * a[6] * a[15] - a[1] * a[7] * a[14] - a[5] * a[2] * a[15] +
           a[5] * a[3] * a[14] + a[13] * a[2] * a[7] - a[13] * a[3] * a[6];
  inv[6] = -a[0] * a[6] * a[15] + a[0] * a[7] * a[14] + a[4] * a[2] * a[15] -
           a[4] * a[3] * a[14] - a[12] * a[2] * a[7] + a[12] * a[3] * a[6];
  inv[10] = a[0] * a[5] * a[15] - a[0] * a[7] * a[13] - a[4] * a[1] * a[15] +
            a[4] * a[3] * a[13] + a[12] * a[1] * a[7] - a[12] * a[3] * a[5];
  inv[14] = -a[0] * a[5] * a[14] + a[0] * a[6] * a[13] + a[4] * a[1] * a[14] -
            a[4] * a[2] * a[13] - a[12] * a[1] * a[6] + a[12] * a[2] * a[5];
  inv[3] = -a[1] * a[6] * a[11] + a[1] * a[7] * a[10] + a[5] * a[2] * a[11] -
           a[5] * a[3] * a[10] - a[9] * a[2] * a[7] + a[9] * a[3] * a[6];
  inv[7] = a[0] * a[6] * a[11] - a[0] * a[7] * a[10] - a[4] * a[2] * a[11] +
           a[4] * a[3] * a[10] + a[8] * a[2] * a[7] - a[8] * a[3] * a[6];
  inv[11] = -a[0] * a[5] * a[11] + a[0] * a[7] * a[9] + a[4] * a[1] * a[11] -
            a[4] * a[3] * a[9] - a[8] * a[1] * a[7] + a[8] * a[3] * a[5];
  inv[15] = a[0] * a[5] * a[10] - a[0] * a[6] * a[9] - a[4] * a[1] * a[10] +
            a[4] * a[2] * a[9] + a[8] * a[1] * a[6] - a[8] * a[2] * a[5];

  double det = a[0] * inv[0] + a[1] * inv[4] + a[2] * inv[8] + a[3] * inv[12];
  if (std::fabs(det) < 1e-20) return false;
  const double inv_det = 1.0 / det;
  for (int i = 0; i < 16; ++i) out[i] = static_cast<float>(inv[i] * inv_det);
  return true;
}

CameraMatrices BuildCameraMatrices(float focal, float cx, float cy, int W, int H,
                                   float near_z, float far_z) {
  CameraMatrices m;
  m.focal = focal;
  m.principal = {cx, cy};
  m.width = W;
  m.height = H;
  m.near_z = near_z;
  m.far_z = far_z;

  const float fw = static_cast<float>(W > 0 ? W : 1);
  const float fh = static_cast<float>(H > 0 ? H : 1);
  const float fx = focal, fy = focal;
  // Depth remap: ndc_z = az + b over 1/z is linear; ndc_z(near)=-1, ndc_z(far)=+1.
  const float denom = (far_z - near_z);
  const float az = (denom != 0.f) ? (far_z + near_z) / denom : 0.f;
  const float bz = (denom != 0.f) ? (-2.f * near_z * far_z) / denom : 0.f;

  // Row-major worldToNDC. clip = M * [x,y,z,1]; ndc = clip.xyz / clip.w (=z).
  //   ndc_x = 2u/W - 1,  ndc_y = 1 - 2v/H (y UP),  ndc_z = az + bz/z.
  std::array<float, 16>& M = m.world_to_ndc;
  M = {
      2.f * fx / fw, 0.f,            2.f * cx / fw - 1.f, 0.f,
      0.f,          -2.f * fy / fh,  1.f - 2.f * cy / fh, 0.f,
      0.f,           0.f,            az,                  bz,
      0.f,           0.f,            1.f,                 0.f,
  };

  if (!Invert4x4(M, m.ndc_to_world)) m.ndc_to_world = {};
  return m;
}

}  // namespace hastur
