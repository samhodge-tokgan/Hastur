// Copyright the Hastur authors.
// SPDX-License-Identifier: LicenseRef-SAM-License
//
// MeshTypes.h — the FROZEN inter-stage contract for the SAM 3D Body pipeline.
//
// Every pipeline stage (detector -> crop -> body regressor -> hand refiner ->
// MHR mesh -> camera -> rasterizer) communicates through the plain structs in
// this header. It is intentionally dependency-free (only <array>/<vector>/
// <cstdint>/<memory>) so each parallel track can compile against it without
// pulling in Eigen, ONNX Runtime, or the OFX SDK. Numeric layout notes come
// straight from the reference (sam-3d-body @ c259bfc): model_config.yaml,
// models/heads/mhr_head.py, models/meta_arch/sam3d_body.py,
// visualization/renderer.py.
//
// DO NOT change field meaning/layout without a coordinated bump across all
// tracks — this is the synchronization surface (plan §1.1 / §4).

#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

namespace hastur {

// ---------------------------------------------------------------------------
// Fixed dimensions (from model_config.yaml + mhr_head.py buffers).
// ---------------------------------------------------------------------------
inline constexpr int kBodyImageSize = 512;    // MODEL.IMAGE_SIZE (square)
inline constexpr int kBackbonePatch = 16;     // dinov3_vith16plus patch size
inline constexpr int kTokenGrid     = kBodyImageSize / kBackbonePatch;  // 32x32
inline constexpr int kNumVerts       = 18439; // MHR LOD1 vertices
inline constexpr int kNumFaces       = 36874; // MHR LOD1 triangles
inline constexpr int kNumJoints      = 127;   // MHR skeleton joints
inline constexpr int kNumKeypoints   = 70;    // mhr70 (first 70 of 308)
inline constexpr int kNumKptAll      = 308;   // keypoint_mapping rows
inline constexpr int kParamDim       = 519;   // MHR-head output vector

// MHR-head 519-d parameter block layout (mhr_head.py:57-64).
// [ global6d(6) | body_cont(260) | shape(45) | scale(28) | hands(108=2x54) | expr(72, zeroed) ]
inline constexpr int kGlobal6D   = 6;
inline constexpr int kBodyCont   = 260;
inline constexpr int kShapeComps = 45;
inline constexpr int kScaleComps = 28;
inline constexpr int kHandPCA    = 54;   // per hand (PCA); 108 total for both
inline constexpr int kExprComps  = 72;   // present but zeroed at inference

// ImageNet normalization for the body/hand crop (model_config.yaml).
inline constexpr std::array<float, 3> kImageNetMean{0.485f, 0.456f, 0.406f};
inline constexpr std::array<float, 3> kImageNetStd {0.229f, 0.224f, 0.225f};

// ---------------------------------------------------------------------------
// Execution-provider selection (mirrors humbaba OrtAccel).
// ---------------------------------------------------------------------------
enum class ComputeUnits { All, CpuAndGpu, CpuAndAne, CpuOnly };

// ---------------------------------------------------------------------------
// Detector stage: full-frame RGB -> person boxes.
// ---------------------------------------------------------------------------
struct BBox {
  float x0{}, y0{}, x1{}, y1{};  // xyxy, full-frame pixels
  float score{};
};
using Detections = std::vector<BBox>;

// ---------------------------------------------------------------------------
// Crop + camera conditioning for one person (CropAffine + CameraSolver).
//   image:          [3, 512, 512] row-major CHW, ImageNet-normalized (fp32;
//                   cast to fp16 at the tensor boundary by the engine).
//   ray_cond:       [2, 32, 32]  normalized ray directions (get_ray_condition).
//   condition_info: [3]          CLIFF conditioning (cx/f, cy/f, b/f) with
//                                USE_INTRIN_CENTER=true.
//   affine:         2x3 warp full-frame px -> 512x512 crop (and its inverse
//                   is needed to place results back in the frame).
// ---------------------------------------------------------------------------
struct CropInputs {
  std::vector<float> image;         // 3*512*512
  std::array<float, 2 * kTokenGrid * kTokenGrid> ray_cond{};  // 2*32*32
  std::array<float, 3> condition_info{};
  std::array<float, 6> affine{};      // forward 2x3 (frame -> crop)
  std::array<float, 6> affine_inv{};  // inverse 2x3 (crop -> frame)
  std::array<float, 2> center{};      // bbox center (px)
  std::array<float, 2> scale{};       // bbox scale (px, GetBBoxCenterScale)
};

// ---------------------------------------------------------------------------
// Body regressor stage output (Sam3dBodyEngine).
//   pred:        [519]  MHR-head parameter vector (layout above).
//   pred_cam:    [3]    weak/perspective camera params from the camera head.
//   hand_logits: [2][2] per-hand (L,R) presence logits -> sigmoid > tau gates
//                       the lazy hand-refiner sessions.
//   hand_box:    [2][4] per-hand crop boxes in the body-crop frame.
// ---------------------------------------------------------------------------
struct BodyPrediction {
  std::array<float, kParamDim> pred{};
  std::array<float, 3> pred_cam{};
  std::array<std::array<float, 2>, 2> hand_logits{};
  std::array<std::array<float, 4>, 2> hand_box{};
};

// ---------------------------------------------------------------------------
// MHR posed mesh (MhrModel C++ LBS, or the ONNX oracle).
//   verts:     [18439,3]  meters, MHR camera frame ([1,2] flipped, /100 applied).
//   joints:    [127,3]    posed joint positions (same frame/units).
//   keypoints: [70,3]     regressed via keypoint_mapping, truncated to mhr70.
//   faces:     [36874,3]  STATIC int32 topology, shared (owned by MeshAssets).
// ---------------------------------------------------------------------------
struct Mesh {
  std::vector<float> verts;      // kNumVerts*3
  std::vector<float> joints;     // kNumJoints*3
  std::vector<float> keypoints;  // kNumKeypoints*3
  std::shared_ptr<const std::vector<int32_t>> faces;  // kNumFaces*3, shared static
};

// ---------------------------------------------------------------------------
// Perspective camera for a person (CameraSolver.perspective_projection).
//   fx=fy=focal; principal point (cx,cy) defaults to frame center; the mesh is
//   translated by cam_t then viewed from the origin (renderer.py convention:
//   camera_translation[0] *= -1, plus a 180-deg mesh rotation about X).
// ---------------------------------------------------------------------------
struct Camera {
  float focal{};
  std::array<float, 2> center{};   // (cx, cy) px
  std::array<float, 3> cam_t{};    // translation (m)
};

// ---------------------------------------------------------------------------
// One person's full result, and the frame-level bundle.
// ---------------------------------------------------------------------------
struct PersonResult {
  BBox box;
  BodyPrediction pred;
  Mesh mesh;
  Camera cam;
  bool has_hands{false};
};

// RGBA render target (straight or premultiplied per plugin param), input-frame
// resolution. Layout: row-major HWC, 4 channels, float [0,1].
struct RgbaImage {
  int width{}, height{};
  std::vector<float> data;  // width*height*4
};

struct FrameResult {
  std::vector<PersonResult> people;  // depth-ordered for over-composite
  RgbaImage render;                  // grey mesh(es) + coverage alpha
};

}  // namespace hastur
