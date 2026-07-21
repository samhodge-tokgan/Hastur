// Copyright the Hastur authors.
// SPDX-License-Identifier: LicenseRef-SAM-License
//
// LeotardMask.cpp -- see LeotardMask.h. Geometric per-vertex garment mask built
// from the rest-pose mesh + mhr70 keypoint landmarks (no joint names / no gated
// checkpoint needed).

#include "LeotardMask.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace hastur {

// The leotard is a CONNECTED region of the mesh surface. Rather than test each
// vertex against geometric capsules (which leave grey holes wherever the surface
// bulges past a radius -- the deltoid, the quadriceps), we FLOOD-FILL the mesh
// topology from a torso seed and stop only at the three real garment boundaries:
// the elbow (sleeve hem), the knee (leg hem) and the neck (collar). Every vertex
// reachable from the torso without crossing a boundary is leotard, so the fill is
// solid and gap-free; the only edges are the elbow/knee/neck rings.
std::vector<float> ComputeLeotardMask(const Mesh& rest) {
  std::vector<float> mask;
  const std::vector<float>& V = rest.verts;      // kNumVerts*3
  const std::vector<float>& K = rest.keypoints;  // kNumKeypoints*3
  if (static_cast<int>(V.size()) < kNumVerts * 3 ||
      static_cast<int>(K.size()) < kNumKeypoints * 3 || !rest.faces)
    return mask;
  const std::vector<int32_t>& F = *rest.faces;

  auto kp = [&](int i, float o[3]) {
    o[0] = K[i * 3]; o[1] = K[i * 3 + 1]; o[2] = K[i * 3 + 2];
  };
  // mhr70 landmark indices (metadata/mhr70.py): shoulders 5/6, elbows 7/8,
  // wrists 62/41, hips 9/10, knees 11/12, ankles 13/14, neck 69.
  float shL[3], shR[3], elL[3], elR[3], wrL[3], wrR[3];
  float hipL[3], hipR[3], knL[3], knR[3], anL[3], anR[3], neck[3];
  kp(5, shL); kp(6, shR); kp(7, elL); kp(8, elR); kp(62, wrL); kp(41, wrR);
  kp(9, hipL); kp(10, hipR); kp(11, knL); kp(12, knR);
  kp(13, anL); kp(14, anR); kp(69, neck);

  auto finite3 = [](const float* a) {
    return std::isfinite(a[0]) && std::isfinite(a[1]) && std::isfinite(a[2]);
  };
  const float* pts[] = {shL, shR, elL, elR, wrL,  wrR, hipL,
                        hipR, knL, knR, anL, anR, neck};
  for (const float* pt : pts)
    if (!finite3(pt)) return mask;

  auto sub = [](const float* a, const float* b, float o[3]) {
    o[0] = a[0] - b[0]; o[1] = a[1] - b[1]; o[2] = a[2] - b[2];
  };
  auto dot = [](const float* a, const float* b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  };
  auto len = [&](const float* a) { return std::sqrt(dot(a, a)); };
  auto dist = [&](const float* a, const float* b) {
    float d[3]; sub(a, b, d); return len(d);
  };
  const float span = dist(shL, shR);  // shoulder span = the body-scale unit
  if (!(span > 1e-4f)) return mask;
  const float pelvis[3] = {(hipL[0] + hipR[0]) * 0.5f,
                           (hipL[1] + hipR[1]) * 0.5f,
                           (hipL[2] + hipR[2]) * 0.5f};

  // Distance from p to segment AB.
  auto distSeg = [&](const float* p, const float* A, const float* B) {
    float ab[3], ap[3]; sub(B, A, ab); sub(p, A, ap);
    float ab2 = dot(ab, ab); if (ab2 < 1e-9f) ab2 = 1e-9f;
    float t = dot(ap, ab) / ab2; t = std::min(1.0f, std::max(0.0f, t));
    const float c[3] = {A[0] + t * ab[0], A[1] + t * ab[1], A[2] + t * ab[2]};
    return dist(p, c);
  };
  // True if p is beyond the distal joint J along the proximal segment S->J
  // (i.e. on the extremity side of the elbow/knee cut plane).
  auto pastJoint = [&](const float* p, const float* S, const float* J) {
    float d[3]; sub(J, S, d); float L = len(d); if (L < 1e-6f) return false;
    d[0] /= L; d[1] /= L; d[2] /= L;
    float ps[3]; sub(p, S, ps);
    return dot(ps, d) > L;
  };

  // Neck cut: a global half-space above the neck base (guarantees a complete
  // barrier so the fill can never reach the head).
  float up[3]; sub(neck, pelvis, up); float uL = len(up);
  if (uL < 1e-6f) return mask;
  up[0] /= uL; up[1] /= uL; up[2] /= uL;
  const float sNeck = uL;

  // Limb reach radii: generous enough that the elbow/knee wall forms a COMPLETE
  // ring around the limb (else the fill leaks past it).
  const float r_arm = 0.28f * span;
  const float r_leg = 0.42f * span;

  // A vertex is a boundary WALL (skin, blocks the fill) if it is above the neck,
  // or on a forearm/hand (past the elbow) or a lower leg/foot (past the knee).
  std::vector<char> wall(kNumVerts, 0);
  for (int v = 0; v < kNumVerts; ++v) {
    const float* p = &V[v * 3];
    float pv[3]; sub(p, pelvis, pv);
    bool w = dot(pv, up) > sNeck;
    if (!w && distSeg(p, shL, wrL) < r_arm && pastJoint(p, shL, elL)) w = true;
    if (!w && distSeg(p, shR, wrR) < r_arm && pastJoint(p, shR, elR)) w = true;
    if (!w && distSeg(p, hipL, anL) < r_leg && pastJoint(p, hipL, knL)) w = true;
    if (!w && distSeg(p, hipR, anR) < r_leg && pastJoint(p, hipR, knR)) w = true;
    wall[v] = w ? 1 : 0;
  }

  // Vertex adjacency from the triangle topology.
  std::vector<std::vector<int>> adj(kNumVerts);
  const int nf = static_cast<int>(F.size() / 3);
  auto addEdge = [&](int a, int b) {
    if (a >= 0 && a < kNumVerts && b >= 0 && b < kNumVerts) {
      adj[a].push_back(b); adj[b].push_back(a);
    }
  };
  for (int f = 0; f < nf; ++f) {
    const int a = F[3 * f], b = F[3 * f + 1], c = F[3 * f + 2];
    addEdge(a, b); addEdge(b, c); addEdge(c, a);
  }

  // Seeds: torso-core vertices near the pelvis (definitely leotard, not walls).
  std::vector<char> filled(kNumVerts, 0);
  std::vector<int> stack;
  const float seedR = 0.30f * span;
  for (int v = 0; v < kNumVerts; ++v)
    if (!wall[v] && dist(&V[v * 3], pelvis) < seedR) {
      filled[v] = 1; stack.push_back(v);
    }
  if (stack.empty()) {  // fallback: nearest non-wall vertex to the pelvis
    int best = -1; float bd = 1e30f;
    for (int v = 0; v < kNumVerts; ++v) {
      if (wall[v]) continue;
      float d = dist(&V[v * 3], pelvis);
      if (d < bd) { bd = d; best = v; }
    }
    if (best < 0) return mask;
    filled[best] = 1; stack.push_back(best);
  }
  // Flood the connected non-wall region.
  while (!stack.empty()) {
    const int v = stack.back(); stack.pop_back();
    for (int nb : adj[v])
      if (!filled[nb] && !wall[nb]) { filled[nb] = 1; stack.push_back(nb); }
  }

  // Soften the binary result by one round of neighbour-averaging so the hem
  // gets a smooth ~1-ring transition (the rasterizer then anti-aliases it).
  std::vector<float> m(kNumVerts);
  for (int v = 0; v < kNumVerts; ++v) m[v] = filled[v] ? 1.0f : 0.0f;
  mask.assign(kNumVerts, 0.0f);
  for (int v = 0; v < kNumVerts; ++v) {
    float acc = m[v]; int n = 1;
    for (int nb : adj[v]) { acc += m[nb]; ++n; }
    mask[v] = acc / static_cast<float>(n);
  }
  return mask;
}

}  // namespace hastur
