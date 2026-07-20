// Copyright the Hastur authors.
// SPDX-License-Identifier: LicenseRef-SAM-License
//
// LeotardMask.cpp -- see LeotardMask.h. Geometric per-vertex garment mask built
// from the rest-pose mesh + mhr70 keypoint landmarks (no joint names / no gated
// checkpoint needed).

#include "LeotardMask.h"

#include <algorithm>
#include <cmath>

namespace hastur {

std::vector<float> ComputeLeotardMask(const Mesh& rest) {
  std::vector<float> mask;
  const std::vector<float>& V = rest.verts;      // kNumVerts*3
  const std::vector<float>& K = rest.keypoints;  // kNumKeypoints*3
  if (static_cast<int>(V.size()) < kNumVerts * 3 ||
      static_cast<int>(K.size()) < kNumKeypoints * 3)
    return mask;

  auto kp = [&](int i, float o[3]) {
    o[0] = K[i * 3]; o[1] = K[i * 3 + 1]; o[2] = K[i * 3 + 2];
  };
  // mhr70 landmark indices (metadata/mhr70.py).
  float shL[3], shR[3], elL[3], elR[3], hipL[3], hipR[3], knL[3], knR[3], neck[3];
  kp(5, shL); kp(6, shR); kp(7, elL); kp(8, elR);
  kp(9, hipL); kp(10, hipR); kp(11, knL); kp(12, knR); kp(69, neck);

  auto finite3 = [](const float* a) {
    return std::isfinite(a[0]) && std::isfinite(a[1]) && std::isfinite(a[2]);
  };
  const float* pts[] = {shL, shR, elL, elR, hipL, hipR, knL, knR, neck};
  for (const float* pt : pts)
    if (!finite3(pt)) return mask;

  auto dist = [](const float* a, const float* b) {
    const float dx = a[0] - b[0], dy = a[1] - b[1], dz = a[2] - b[2];
    return std::sqrt(dx * dx + dy * dy + dz * dz);
  };
  const float span = dist(shL, shR);  // shoulder span = the body-scale unit
  if (!(span > 1e-4f)) return mask;

  const float pelvis[3] = {(hipL[0] + hipR[0]) * 0.5f,
                           (hipL[1] + hipR[1]) * 0.5f,
                           (hipL[2] + hipR[2]) * 0.5f};
  const float shMid[3] = {(shL[0] + shR[0]) * 0.5f, (shL[1] + shR[1]) * 0.5f,
                          (shL[2] + shR[2]) * 0.5f};

  // Radii as fractions of shoulder span, and a smooth-hem band (fraction of r).
  const float r_torso = 0.55f * span;
  const float r_uparm = 0.17f * span;
  const float r_upleg = 0.33f * span;
  const float band = 0.20f;

  auto toward = [](const float* a, const float* b, float t, float o[3]) {
    for (int k = 0; k < 3; ++k) o[k] = a[k] + t * (b[k] - a[k]);
  };
  // Upper-leg proximal point nudged toward the pelvis so the hip/groin fold is
  // covered (the raw hip joint sits lateral to the inguinal crease).
  float hipLc[3], hipRc[3];
  toward(hipL, pelvis, 0.45f, hipLc);
  toward(hipR, pelvis, 0.45f, hipRc);
  // Sleeve start nudged toward the neck so the shoulder/clavicle is covered
  // (raises the effective neckline from a bare boat-neck to a crew neck).
  float shLc[3], shRc[3];
  toward(shL, neck, 0.30f, shLc);
  toward(shR, neck, 0.30f, shRc);

  auto smoothstep = [](float e0, float e1, float x) {
    float t = (e1 == e0) ? 0.0f : (x - e0) / (e1 - e0);
    t = std::min(1.0f, std::max(0.0f, t));
    return t * t * (3.0f - 2.0f * t);
  };

  // Limb = capsule (rounded hem at the distal joint reads as a natural sleeve/leg).
  auto capsule = [&](const float* p, const float* A, const float* B, float r) {
    const float ab[3] = {B[0] - A[0], B[1] - A[1], B[2] - A[2]};
    const float ap[3] = {p[0] - A[0], p[1] - A[1], p[2] - A[2]};
    float ab2 = ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2];
    if (ab2 < 1e-9f) ab2 = 1e-9f;
    float t = (ap[0] * ab[0] + ap[1] * ab[1] + ap[2] * ab[2]) / ab2;
    t = std::min(1.0f, std::max(0.0f, t));
    const float c[3] = {A[0] + t * ab[0], A[1] + t * ab[1], A[2] + t * ab[2]};
    return smoothstep(r * (1.0f + band), r, dist(p, c));  // 1 inside r -> 0 at r1
  };

  // Torso = cylinder about the pelvis->collarbone axis, capped by PLANES (not
  // hemispheres) so the neckline is a flat crew neck at the collarbone, and the
  // bottom extends below the hips to fully cover the groin/seat.
  float ax[3] = {shMid[0] - pelvis[0], shMid[1] - pelvis[1], shMid[2] - pelvis[2]};
  float axlen = std::sqrt(ax[0] * ax[0] + ax[1] * ax[1] + ax[2] * ax[2]);
  if (axlen < 1e-4f) axlen = 1e-4f;
  const float u[3] = {ax[0] / axlen, ax[1] / axlen, ax[2] / axlen};
  // Crew neckline at the base of the neck (plane cap along the axis, so it is a
  // flat collar covering the collarbone/shoulders, NOT a dome over the head).
  const float top = (neck[0] - pelvis[0]) * u[0] + (neck[1] - pelvis[1]) * u[1] +
                    (neck[2] - pelvis[2]) * u[2];
  const float bot = -0.42f * span;         // below the hips -> cover groin/seat
  const float gate = 0.12f * span;         // axial cap softness
  auto torso = [&](const float* p) {
    const float apx = p[0] - pelvis[0], apy = p[1] - pelvis[1],
                apz = p[2] - pelvis[2];
    const float s = apx * u[0] + apy * u[1] + apz * u[2];  // axial coord
    const float cx = pelvis[0] + s * u[0], cy = pelvis[1] + s * u[1],
                cz = pelvis[2] + s * u[2];
    const float dx = p[0] - cx, dy = p[1] - cy, dz = p[2] - cz;
    const float d = std::sqrt(dx * dx + dy * dy + dz * dz);
    const float radial = smoothstep(r_torso * (1.0f + band), r_torso, d);
    const float axial = smoothstep(bot, bot + gate, s) *
                        (1.0f - smoothstep(top - gate, top, s));
    return radial * axial;
  };

  // Soft (probabilistic) union so overlapping soft edges accumulate to a solid
  // garment instead of leaving ~0.5 seams (as a hard max would) at the hip /
  // knee / shoulder junctions; fully-covered regions stay 1.
  auto join = [](float a, float b) { return a + b - a * b; };
  mask.assign(kNumVerts, 0.0f);
  for (int v = 0; v < kNumVerts; ++v) {
    const float* p = &V[v * 3];
    float m = torso(p);
    m = join(m, capsule(p, shLc, elL, r_uparm));
    m = join(m, capsule(p, shRc, elR, r_uparm));
    m = join(m, capsule(p, hipLc, knL, r_upleg));
    m = join(m, capsule(p, hipRc, knR, r_upleg));
    mask[v] = m;
  }
  return mask;
}

}  // namespace hastur
