// Copyright the Hastur authors.
// SPDX-License-Identifier: LicenseRef-SAM-License
//
// Skeleton.cpp — see Skeleton.h. Extraction + 2D projection + hand-rolled JSON
// (no JSON library in-repo; mirrors the cryptoManifest string building).

#include "Skeleton.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace hastur {

void ProjectJoints(const std::vector<float>& j3, float focal,
                   const std::array<float, 2>& center,
                   const std::array<float, 3>& cam_t, std::vector<float>& out2d) {
  const size_t n = j3.size() / 3;
  out2d.assign(n * 2, 0.f);
  constexpr float kEps = 1e-6f;
  for (size_t i = 0; i < n; ++i) {
    const float X = j3[i * 3 + 0] + cam_t[0];
    const float Y = j3[i * 3 + 1] + cam_t[1];
    const float Z = j3[i * 3 + 2] + cam_t[2];
    if (Z > kEps) {  // in front of the camera
      out2d[i * 2 + 0] = focal * X / Z + center[0];
      out2d[i * 2 + 1] = focal * Y / Z + center[1];
    }
  }
}

SkeletonFrame BuildSkeletonFrame(const FrameResult& fr, int frame, int W, int H) {
  SkeletonFrame sk;
  sk.frame = frame;
  sk.width = W > 0 ? W : fr.render.width;
  sk.height = H > 0 ? H : fr.render.height;
  sk.people.reserve(fr.people.size());
  for (const PersonResult& p : fr.people) {
    SkeletonPerson s;
    s.track_id = p.track_id;
    s.joints3d = p.mesh.joints;
    s.joint_xforms = p.mesh.joint_xforms;
    s.keypoints3d = p.mesh.keypoints;
    s.pose.assign(p.pred.pred.begin(), p.pred.pred.end());
    s.focal = p.cam.focal;
    s.center = p.cam.center;
    s.cam_t = p.cam.cam_t;
    ProjectJoints(s.joints3d, s.focal, s.center, s.cam_t, s.joints2d);
    sk.people.push_back(std::move(s));
  }
  return sk;
}

namespace {
void AppendFloatArray(std::string& out, const std::vector<float>& v) {
  out += '[';
  char b[32];
  for (size_t i = 0; i < v.size(); ++i) {
    if (i) out += ',';
    std::snprintf(b, sizeof(b), "%.6g", static_cast<double>(v[i]));
    out += b;
  }
  out += ']';
}
}  // namespace

std::string SkeletonToJson(const SkeletonFrame& sk,
                           const std::vector<int32_t>& parents) {
  std::string o;
  o.reserve(2048 + sk.people.size() * 12288);
  char b[256];

  std::snprintf(b, sizeof(b), "{\"frame\":%d,\"width\":%d,\"height\":%d,", sk.frame,
                sk.width, sk.height);
  o += b;

  o += "\"hierarchy\":{\"joint_parents\":[";
  for (size_t i = 0; i < parents.size(); ++i) {
    if (i) o += ',';
    std::snprintf(b, sizeof(b), "%d", parents[i]);
    o += b;
  }
  o += "]},\"people\":[";

  for (size_t k = 0; k < sk.people.size(); ++k) {
    const SkeletonPerson& s = sk.people[k];
    if (k) o += ',';
    std::snprintf(
        b, sizeof(b),
        "{\"id\":%d,\"cam\":{\"focal\":%.6g,\"cx\":%.6g,\"cy\":%.6g,"
        "\"t\":[%.6g,%.6g,%.6g]},",
        s.track_id, static_cast<double>(s.focal), static_cast<double>(s.center[0]),
        static_cast<double>(s.center[1]), static_cast<double>(s.cam_t[0]),
        static_cast<double>(s.cam_t[1]), static_cast<double>(s.cam_t[2]));
    o += b;
    o += "\"joints3d\":";
    AppendFloatArray(o, s.joints3d);
    o += ",\"joints2d\":";
    AppendFloatArray(o, s.joints2d);
    o += ",\"joint_xforms\":";
    AppendFloatArray(o, s.joint_xforms);
    o += ",\"keypoints3d\":";
    AppendFloatArray(o, s.keypoints3d);
    o += ",\"pose\":";
    AppendFloatArray(o, s.pose);
    o += '}';
  }
  o += "]}";
  return o;
}

// --- Compact little-endian binary sidecar ----------------------------------
namespace {
void PutU32(std::vector<uint8_t>& b, uint32_t v) {
  b.push_back(static_cast<uint8_t>(v & 0xFF));
  b.push_back(static_cast<uint8_t>((v >> 8) & 0xFF));
  b.push_back(static_cast<uint8_t>((v >> 16) & 0xFF));
  b.push_back(static_cast<uint8_t>((v >> 24) & 0xFF));
}
void PutI32(std::vector<uint8_t>& b, int32_t v) {
  PutU32(b, static_cast<uint32_t>(v));
}
void PutF32(std::vector<uint8_t>& b, float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  PutU32(b, u);
}
void PutF32Array(std::vector<uint8_t>& b, const std::vector<float>& v, size_t n) {
  for (size_t i = 0; i < n; ++i) PutF32(b, i < v.size() ? v[i] : 0.f);
}

// Bounds-checked little-endian readers over a byte span. `ok` latches false on
// any short read; every getter no-ops once it is false.
struct Reader {
  const uint8_t* p;
  size_t n;
  size_t off = 0;
  bool ok = true;
  uint32_t u32() {
    if (!ok || off + 4 > n) { ok = false; return 0; }
    uint32_t v = static_cast<uint32_t>(p[off]) |
                 (static_cast<uint32_t>(p[off + 1]) << 8) |
                 (static_cast<uint32_t>(p[off + 2]) << 16) |
                 (static_cast<uint32_t>(p[off + 3]) << 24);
    off += 4;
    return v;
  }
  int32_t i32() { return static_cast<int32_t>(u32()); }
  float f32() {
    uint32_t u = u32();
    float f;
    std::memcpy(&f, &u, 4);
    return f;
  }
  void f32array(std::vector<float>& out, size_t count) {
    out.assign(count, 0.f);
    for (size_t i = 0; i < count && ok; ++i) out[i] = f32();
  }
};
}  // namespace

std::vector<uint8_t> SkeletonToBinary(const SkeletonFrame& sk,
                                      const std::vector<int32_t>& parents) {
  // Per-person strides: derived from the first person's arrays (uniform across a
  // frame in practice), falling back to the MHR constants when there are none.
  int32_t nj = kNumJoints, nk = kNumKeypoints, np = kParamDim;
  if (!sk.people.empty()) {
    const SkeletonPerson& s0 = sk.people.front();
    nj = static_cast<int32_t>(s0.joints3d.size() / 3);
    nk = static_cast<int32_t>(s0.keypoints3d.size() / 3);
    np = static_cast<int32_t>(s0.pose.size());
  }

  std::vector<uint8_t> b;
  b.reserve(64 + parents.size() * 4 +
            sk.people.size() * static_cast<size_t>(
                (nj * 13 + nk * 3 + np + 7) * 4 + 4));
  b.push_back('S'); b.push_back('K'); b.push_back('E'); b.push_back('L');
  PutU32(b, 1u);  // version
  PutI32(b, sk.frame);
  PutI32(b, sk.width);
  PutI32(b, sk.height);
  PutI32(b, nj);
  PutI32(b, nk);
  PutI32(b, np);
  PutI32(b, static_cast<int32_t>(parents.size()));
  for (int32_t v : parents) PutI32(b, v);
  PutI32(b, static_cast<int32_t>(sk.people.size()));
  for (const SkeletonPerson& s : sk.people) {
    PutI32(b, s.track_id);
    PutF32(b, s.focal);
    PutF32(b, s.center[0]);
    PutF32(b, s.center[1]);
    PutF32(b, s.cam_t[0]);
    PutF32(b, s.cam_t[1]);
    PutF32(b, s.cam_t[2]);
    PutF32Array(b, s.joints3d, static_cast<size_t>(nj) * 3);
    PutF32Array(b, s.joints2d, static_cast<size_t>(nj) * 2);
    PutF32Array(b, s.joint_xforms, static_cast<size_t>(nj) * 8);
    PutF32Array(b, s.keypoints3d, static_cast<size_t>(nk) * 3);
    PutF32Array(b, s.pose, static_cast<size_t>(np));
  }
  return b;
}

bool SkeletonFromBinary(const std::vector<uint8_t>& bytes, SkeletonFrame& sk,
                        std::vector<int32_t>& parents) {
  sk = SkeletonFrame{};
  parents.clear();
  Reader r{bytes.data(), bytes.size()};
  if (bytes.size() < 4 || bytes[0] != 'S' || bytes[1] != 'K' || bytes[2] != 'E' ||
      bytes[3] != 'L')
    return false;
  r.off = 4;
  if (r.u32() != 1u) return false;  // version
  sk.frame = r.i32();
  sk.width = r.i32();
  sk.height = r.i32();
  const int32_t nj = r.i32();
  const int32_t nk = r.i32();
  const int32_t np = r.i32();
  if (!r.ok || nj < 0 || nk < 0 || np < 0) return false;
  const int32_t nparents = r.i32();
  if (!r.ok || nparents < 0) return false;
  parents.resize(static_cast<size_t>(nparents));
  for (int32_t i = 0; i < nparents && r.ok; ++i) parents[i] = r.i32();
  const int32_t npeople = r.i32();
  if (!r.ok || npeople < 0) return false;
  sk.people.resize(static_cast<size_t>(npeople));
  for (int32_t i = 0; i < npeople && r.ok; ++i) {
    SkeletonPerson& s = sk.people[static_cast<size_t>(i)];
    s.track_id = r.i32();
    s.focal = r.f32();
    s.center[0] = r.f32();
    s.center[1] = r.f32();
    s.cam_t[0] = r.f32();
    s.cam_t[1] = r.f32();
    s.cam_t[2] = r.f32();
    r.f32array(s.joints3d, static_cast<size_t>(nj) * 3);
    r.f32array(s.joints2d, static_cast<size_t>(nj) * 2);
    r.f32array(s.joint_xforms, static_cast<size_t>(nj) * 8);
    r.f32array(s.keypoints3d, static_cast<size_t>(nk) * 3);
    r.f32array(s.pose, static_cast<size_t>(np));
  }
  return r.ok;
}

}  // namespace hastur
