// Copyright the Hastur authors.
// SPDX-License-Identifier: LicenseRef-SAM-License
//
// Skeleton.cpp — see Skeleton.h. Extraction + 2D projection + hand-rolled JSON
// (no JSON library in-repo; mirrors the cryptoManifest string building).

#include "Skeleton.h"

#include <cstdio>
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

}  // namespace hastur
