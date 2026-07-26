// Copyright the Hastur authors.
// SPDX-License-Identifier: LicenseRef-SAM-License
//
// skeleton_validate — self-contained checks for the Skeleton module: 2D
// projection math, FrameResult→SkeletonFrame extraction, per-joint xform
// consistency, and hand-rolled JSON well-formedness. No models / ONNX / OFX.
// Exit code 0 = all pass.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "MeshTypes.h"
#include "Skeleton.h"

using namespace hastur;

static int g_fails = 0;
static void check(bool ok, const char* msg) {
  if (!ok) {
    std::printf("FAIL: %s\n", msg);
    ++g_fails;
  }
}
static bool near(float a, float b, float e = 1e-3f) { return std::fabs(a - b) <= e; }

int main() {
  // --- ProjectJoints: origin (+cam_t) lands on the principal point ---------
  {
    std::vector<float> j = {0.f, 0.f, 0.f};
    std::vector<float> out;
    ProjectJoints(j, 1000.f, {960.f, 540.f}, {0.f, 0.f, 5.f}, out);
    check(out.size() == 2, "project size");
    check(near(out[0], 960.f) && near(out[1], 540.f), "origin -> principal point");
    // offset: P=(0.1,0.2,5) -> u=1000*0.1/5+960=980, v=1000*0.2/5+540=580
    std::vector<float> j2 = {0.1f, 0.2f, 0.f};
    ProjectJoints(j2, 1000.f, {960.f, 540.f}, {0.f, 0.f, 5.f}, out);
    check(near(out[0], 980.f) && near(out[1], 580.f), "offset projection");
    // behind camera (Z<=0) -> (0,0)
    std::vector<float> j3 = {1.f, 1.f, -10.f};
    ProjectJoints(j3, 1000.f, {960.f, 540.f}, {0.f, 0.f, 5.f}, out);
    check(out[0] == 0.f && out[1] == 0.f, "behind-camera -> (0,0)");
  }

  // --- BuildSkeletonFrame: extraction + projection + xform consistency ------
  FrameResult fr;
  fr.render.width = 1920;
  fr.render.height = 1080;
  {
    PersonResult p;
    p.track_id = 4;
    p.mesh.joints.assign(static_cast<size_t>(kNumJoints) * 3, 0.f);
    p.mesh.joint_xforms.assign(static_cast<size_t>(kNumJoints) * 8, 0.f);
    p.mesh.keypoints.assign(static_cast<size_t>(kNumKeypoints) * 3, 0.f);
    // joint 0 at (0.1,0.2,0); its rig transform = identity quat, matching t.
    p.mesh.joints[0] = 0.1f;
    p.mesh.joints[1] = 0.2f;
    p.mesh.joint_xforms[0] = 0.1f;  // tx == joint x
    p.mesh.joint_xforms[1] = 0.2f;  // ty == joint y
    p.mesh.joint_xforms[6] = 1.f;   // qw (identity)
    p.mesh.joint_xforms[7] = 1.f;   // scale
    p.pred.pred.fill(0.f);
    p.pred.pred[0] = 0.5f;
    p.cam.focal = 1000.f;
    p.cam.center = {960.f, 540.f};
    p.cam.cam_t = {0.f, 0.f, 5.f};
    fr.people.push_back(std::move(p));
  }

  SkeletonFrame sk = BuildSkeletonFrame(fr, 181);
  check(sk.frame == 181, "frame annotation");
  check(sk.width == 1920 && sk.height == 1080, "size from render");
  check(sk.people.size() == 1, "people count");
  if (!sk.people.empty()) {
    const SkeletonPerson& s = sk.people[0];
    check(s.track_id == 4, "track_id carried");
    check(s.joints3d.size() == static_cast<size_t>(kNumJoints) * 3, "joints3d size");
    check(s.joint_xforms.size() == static_cast<size_t>(kNumJoints) * 8, "xforms size");
    check(s.joints2d.size() == static_cast<size_t>(kNumJoints) * 2, "joints2d size");
    check(s.pose.size() == static_cast<size_t>(kParamDim), "pose size 519");
    // xform translation equals the joint position (rig ↔ point-cloud consistency).
    check(near(s.joint_xforms[0], s.joints3d[0]) &&
              near(s.joint_xforms[1], s.joints3d[1]),
          "xform t == joint position");
    // projected joint 0: P=(0.1,0.2,5) -> (980,580).
    check(near(s.joints2d[0], 980.f) && near(s.joints2d[1], 580.f),
          "joint0 projected to (980,580)");
  }

  // --- SkeletonToJson: well-formed + expected keys --------------------------
  {
    std::vector<int32_t> parents(kNumJoints, -1);
    parents[1] = 0;
    const std::string js = SkeletonToJson(sk, parents);
    int braces = 0, brackets = 0;
    for (char c : js) {
      if (c == '{') ++braces;
      else if (c == '}') --braces;
      else if (c == '[') ++brackets;
      else if (c == ']') --brackets;
    }
    check(braces == 0, "JSON braces balanced");
    check(brackets == 0, "JSON brackets balanced");
    auto has = [&](const char* k) { return js.find(k) != std::string::npos; };
    check(has("\"frame\":181"), "json frame");
    check(has("\"people\":["), "json people");
    check(has("\"id\":4"), "json id");
    check(has("\"joint_xforms\":"), "json joint_xforms");
    check(has("\"hierarchy\":{\"joint_parents\":["), "json hierarchy");
    check(has("\"joints2d\":") && has("\"pose\":"), "json joints2d+pose");
  }

  // --- SkeletonToBinary <-> SkeletonFromBinary: exact round-trip ------------
  {
    std::vector<int32_t> parents(kNumJoints, -1);
    parents[1] = 0;
    parents[2] = 1;
    const std::vector<uint8_t> bin = SkeletonToBinary(sk, parents);
    // Header sanity: magic "SKEL", version=1 (little-endian).
    check(bin.size() >= 8, "binary non-empty");
    check(bin.size() >= 4 && bin[0] == 'S' && bin[1] == 'K' && bin[2] == 'E' &&
              bin[3] == 'L',
          "binary magic SKEL");
    check(bin[4] == 1 && bin[5] == 0 && bin[6] == 0 && bin[7] == 0,
          "binary version 1 LE");

    SkeletonFrame rt;
    std::vector<int32_t> rtParents;
    check(SkeletonFromBinary(bin, rt, rtParents), "binary round-trip decode ok");

    // Frame annotations + hierarchy.
    check(rt.frame == sk.frame, "bin frame");
    check(rt.width == sk.width && rt.height == sk.height, "bin size");
    check(rtParents.size() == parents.size(), "bin parents size");
    bool parentsMatch = rtParents.size() == parents.size();
    for (size_t i = 0; i < rtParents.size() && parentsMatch; ++i)
      parentsMatch = (rtParents[i] == parents[i]);
    check(parentsMatch, "bin parents exact");

    // People + every field, bitwise-exact (all values fit float32 round-trip).
    check(rt.people.size() == sk.people.size(), "bin people count");
    bool peopleMatch = rt.people.size() == sk.people.size();
    for (size_t k = 0; k < rt.people.size() && peopleMatch; ++k) {
      const SkeletonPerson& a = sk.people[k];
      const SkeletonPerson& b = rt.people[k];
      peopleMatch = a.track_id == b.track_id && a.focal == b.focal &&
                    a.center == b.center && a.cam_t == b.cam_t &&
                    a.joints3d == b.joints3d && a.joints2d == b.joints2d &&
                    a.joint_xforms == b.joint_xforms &&
                    a.keypoints3d == b.keypoints3d && a.pose == b.pose;
    }
    check(peopleMatch, "bin people fields exact (bitwise)");

    // Corrupt magic -> reader rejects.
    std::vector<uint8_t> bad = bin;
    bad[0] = 'X';
    SkeletonFrame junk;
    std::vector<int32_t> junkP;
    check(!SkeletonFromBinary(bad, junk, junkP), "bad magic rejected");
    // Truncation -> reader rejects (does not read past the buffer).
    std::vector<uint8_t> trunc(bin.begin(), bin.begin() + bin.size() / 2);
    check(!SkeletonFromBinary(trunc, junk, junkP), "truncated buffer rejected");
  }

  if (g_fails == 0) std::printf("skeleton_validate: all checks passed\n");
  return g_fails == 0 ? 0 : 1;
}
