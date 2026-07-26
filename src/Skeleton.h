// Copyright the Hastur authors.
// SPDX-License-Identifier: LicenseRef-SAM-License
//
// Skeleton — a serializable per-frame, per-person record of the MHR skeleton
// (3D joints, 2D screen projections, and per-joint rig transforms), keyed by the
// stable `person_NN` track id. Everything here is already computed by the pipeline
// (`FrameResult` / `Mesh` in MeshTypes.h + `MhrModel`); this module only extracts,
// projects, and serializes it — no new geometry.
//
// The intent (per the design): `person_NN` is the single spine of the pipeline —
// the same id binds the refined matte (Cryptomatte / Matte Partition node) and the
// skeleton here, so downstream gets {matte + identity + pose} per person.
//
// All 3D quantities are in the MHR camera frame (metres, the [1,2]-flip applied —
// see MeshTypes.h). 2D is screen pixels (top-down, y down), projected with the
// person's own camera placement.

#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "MeshTypes.h"

namespace hastur {

// One person's skeleton at one frame.
struct SkeletonPerson {
  int track_id{-1};                    // person_NN identity (-1 = unassigned)
  std::vector<float> joints3d;         // kNumJoints*3  (m, camera frame)
  std::vector<float> joints2d;         // kNumJoints*2  (screen px)
  std::vector<float> joint_xforms;     // kNumJoints*8  [tx,ty,tz,qx,qy,qz,qw,s]
  std::vector<float> keypoints3d;      // kNumKeypoints*3 (m, camera frame)
  std::vector<float> pose;             // kParamDim (519-d MHR params)
  // Camera placement used for the 2D projection (per person).
  float focal{};
  std::array<float, 2> center{};       // (cx, cy) px
  std::array<float, 3> cam_t{};        // translation (m)
};

// One frame's people + frame annotations.
struct SkeletonFrame {
  int frame{};
  int width{}, height{};
  std::vector<SkeletonPerson> people;
};

// Project 3D joints (camera frame, metres) to screen pixels using a person's
// camera placement: P = joint + cam_t; u = focal*P.x/P.z + cx, v = focal*P.y/P.z +
// cy. Joints behind the camera (P.z <= eps) are emitted as (0,0). `out2d` is sized
// to joints3d.size()/3*2.
void ProjectJoints(const std::vector<float>& joints3d, float focal,
                   const std::array<float, 2>& center,
                   const std::array<float, 3>& cam_t, std::vector<float>& out2d);

// Build a per-frame skeleton record from a FrameResult: per person copies
// joints/xforms/keypoints/pose + the camera, and fills the 2D projection. `frame`
// annotates the record; W/H default to the render size when <=0.
SkeletonFrame BuildSkeletonFrame(const FrameResult& fr, int frame, int W = 0,
                                 int H = 0);

// Serialize a frame to JSON (hand-rolled — no JSON lib in-repo; mirrors the
// cryptoManifest string building). `parents` (kNumJoints, may be empty) is the
// static joint hierarchy, emitted once under "hierarchy". Compact, one object.
std::string SkeletonToJson(const SkeletonFrame& sk,
                           const std::vector<int32_t>& parents);

// Serialize a frame to a compact, dependency-free little-endian binary sidecar
// (mirrors the `.msk` / mhr_binfmt byte style). Layout (all LE):
//   magic "SKEL" (4 bytes), uint32 version=1,
//   int32 frame, width, height,
//   int32 njoints, nkeypts, nparams   (per-person array strides),
//   int32 nparents, then parents[nparents] (int32, the static joint hierarchy),
//   int32 npeople, then per person:
//     int32 track_id, float focal, float cx, cy, float t[3],
//     joints3d[njoints*3], joints2d[njoints*2], joint_xforms[njoints*8],
//     keypoints3d[nkeypts*3], pose[nparams]   (all float32).
std::vector<uint8_t> SkeletonToBinary(const SkeletonFrame& sk,
                                      const std::vector<int32_t>& parents);

// Inverse of SkeletonToBinary. Fills `sk` + `parents` from `bytes`. Returns false
// on bad magic, unsupported version, or truncation (out params left partial).
bool SkeletonFromBinary(const std::vector<uint8_t>& bytes, SkeletonFrame& sk,
                        std::vector<int32_t>& parents);

}  // namespace hastur
