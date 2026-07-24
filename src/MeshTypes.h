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
#include <string>
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

// Source of the per-person Cryptomatte coverage matte.
//   Mesh     — the MHR mesh silhouette (rasterised beauty alpha). Classic, always
//              available; a clay-body silhouette (no hair/garment/motion-blur edge).
//   Sam3Mask — the SAM 3 instance mask (matte-quality: hair/fingers/motion blur,
//              tight silhouettes). Requires a detector ONNX that also emits masks;
//              falls back to Mesh per-person when a mask is unavailable.
//   Both     — per-pixel union (max) of the two, so the fuller silhouette wins.
enum class CryptoCoverage { Mesh, Sam3Mask, Both };

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
  // Temporally stable identity: the same physical person keeps the same id across
  // a sequence (drives the Cryptomatte person_NN name). -1 when unassigned
  // (stable ids disabled or single-frame use).
  int track_id{-1};
};

// RGBA render target (straight or premultiplied per plugin param), input-frame
// resolution. Layout: row-major HWC, 4 channels, float [0,1].
struct RgbaImage {
  int width{}, height{};
  std::vector<float> data;  // width*height*4
};

// ---------------------------------------------------------------------------
// AOV (arbitrary output variable) buffers — the geometry the rasterizer already
// computes per fragment, surfaced for downstream comp. Input-frame resolution,
// row-major HWC, top-down (same orientation as RgbaImage). Uncovered pixels are
// 0 in every buffer; `coverage` (== beauty alpha) marks validity.
//
// IMPORTANT: these are POINT-SAMPLED (nearest surface at each pixel), NOT
// anti-aliased like the beauty render. Box-averaging depth/position/normal/pref
// across a silhouette edge invents surfaces that do not exist; silhouette
// softness is represented by Cryptomatte coverage, not by blending these values.
//
//   depth:    [W*H]    camera-space Z (metres, >0 in front).
//   position: [W*H*3]  camera-space surface position (m). Camera at origin,
//                      +Z forward, so this frame IS the world (worldToCamera=I).
//   normal:   [W*H*3]  camera-space unit normal, oriented toward the camera.
//   pref:     [W*H*3]  canonical rest-pose reference position (pose/shape-
//                      invariant body atlas coord) for locator/texture pinning.
//   nref:     [W*H*3]  canonical bind-pose surface normal (orientation counterpart
//                      to pref); with pref gives a per-point canonical surface frame.
//   st:       [W*H*2]  texture (u,v); valid only when has_st (a UV set exists).
//   coverage: [W*H]    total mesh coverage (== beauty alpha), for convenience.
struct AovBuffers {
  int width{}, height{};
  bool has_st{false};
  std::vector<float> depth;     // W*H
  std::vector<float> position;  // W*H*3
  std::vector<float> normal;    // W*H*3
  std::vector<float> pref;      // W*H*3
  std::vector<float> nref;      // W*H*3
  std::vector<float> st;        // W*H*2
  std::vector<float> coverage;  // W*H
};

// Cryptomatte output (Psyop spec): per-pixel ranked (id, coverage) pairs packed
// two ranks per RGBA layer — layer[0] = (id0,cov0,id1,cov1), layer[1] = ranks
// 2,3, ... Each layer is W*H*4, top-down HWC. `manifest` is the JSON name->hash
// map; `type_name` is the layer/typename ("person"). Empty `layers` = no people.
struct CryptoResult {
  int width{}, height{};
  std::string type_name;                     // e.g. "person"
  std::string manifest;                      // JSON {"person_00":"<8hex>", ...}
  std::vector<std::vector<float>> layers;    // each W*H*4, two ranks per layer
};

// 4x4 row-major camera matrices (m44f, 16 floats). Built from the FRAME-level
// intrinsics (the single consistent camera — per-person cam_t is object
// placement, not a shared camera). worldToNDC maps camera/world -> normalized
// device coords in [-1,1]; NDCToWorld is its inverse (unproject). worldToCamera
// is identity for this canonical pinhole (camera at origin, +Z forward).
struct CameraMatrices {
  float focal{};                        // fx = fy (px)
  std::array<float, 2> principal{};     // (cx, cy) px
  int width{}, height{};
  float near_z{}, far_z{};
  std::array<float, 16> world_to_ndc{};
  std::array<float, 16> ndc_to_world{};
};

struct FrameResult {
  std::vector<PersonResult> people;  // depth-ordered for over-composite
  RgbaImage render;                  // grey mesh(es) + coverage alpha
  // AOV extensions (append-only; populated only when AOV emission is requested).
  AovBuffers aovs;
  CryptoResult crypto;
  CameraMatrices camera;
};

}  // namespace hastur
