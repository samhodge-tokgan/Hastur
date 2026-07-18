// Copyright the Hastur authors.
// SPDX-License-Identifier: LicenseRef-SAM-License
//
// CameraSolver — pure-geometry camera/conditioning math ported from the
// SAM 3D Body reference (sam-3d-body @ c259bfc). No ONNX Runtime, no OFX SDK;
// depends only on MeshTypes.h + <array>/<cmath>. This is the highest-risk
// numerical seam feeding the DINOv3 backbone (ray_cond) and the CLIFF-style
// decoder conditioning, so every function mirrors the reference bit-for-bit
// (modulo the reference's on-device fp16 rounding — we compute in fp64/fp32).
//
// Reference map:
//   * DefaultCamInt      <- data/utils/prepare_batch.py (default cam_int)
//   * GetBBoxCenterScale <- data/transforms/bbox_utils.py bbox_xyxy2cs +
//                           common.py GetBBoxCenterScale(padding)
//   * SquareScale/Affine <- common.py TopdownAffine (fix_aspect_ratio x2,
//                           get_warp_matrix, use_udp=False, input_size 512)
//   * ConditionInfo      <- meta_arch/sam3d_body.py _get_decoder_condition
//                           (CLIFF, USE_INTRIN_CENTER=true)
//   * RayCond            <- meta_arch/sam3d_body.py get_ray_condition +
//                           modules/camera_embed.py CameraEncoder (F.interpolate
//                           bilinear, align_corners=False, antialias=True, 1/16)
//   * PerspectiveProjection <- heads/camera_head.py PerspectiveHead
//                              .perspective_projection (USE_INTRIN_CENTER=true)
#pragma once

#include <array>

#include "MeshTypes.h"

namespace hastur {

// 3x3 camera intrinsics, row-major.
using CamInt = std::array<float, 9>;

inline float CamFocalX(const CamInt& K) { return K[0]; }
inline float CamCx(const CamInt& K) { return K[2]; }
inline float CamCy(const CamInt& K) { return K[5]; }

// Bounding-box -> (center, scale) exactly like the reference pipeline.
//   center:      bbox center in full-frame px (unchanged by aspect fixing)
//   scale:       padded (w,h) from GetBBoxCenterScale (padding factor applied),
//                BEFORE aspect fixing — this is CropInputs::scale
//   square:      the TopdownAffine square side s (fix_aspect_ratio applied
//                twice: prior ratio 0.75 then to 1:1 for the 512x512 input).
//                This is the "bbox_scale[0]" used by the camera/condition heads.
struct BBoxCS {
  std::array<float, 2> center{};
  std::array<float, 2> scale{};   // padded, pre-aspect (w,h)
  float square{};                 // post fix_aspect_ratio square side
};

// GetBBoxCenterScale (bbox_xyxy2cs, padding) + TopdownAffine's fix_aspect_ratio.
// `pad` = 1.25 for body, 0.9 for hands. input_size defaults to the 512x512 crop.
BBoxCS ProcessBBox(const BBox& box, float pad, int in_w = kBodyImageSize,
                   int in_h = kBodyImageSize);

// Default cam_int from frame size, matching prepare_batch.py:
//   focal = sqrt(W^2 + H^2), principal = (W/2, H/2).
// Overrides: focal_override>0 sets fx=fy directly; else fov_override>0 uses
// focal = diag / (2 tan(fov/2)) with diag = sqrt(W^2+H^2). Principal stays the
// frame center unless (px,py) given (>=0).
CamInt DefaultCamInt(int W, int H, float fov_override_deg = 0.f,
                     float focal_override = 0.f, float px = -1.f, float py = -1.f);

// Forward (frame -> 512x512 crop) 2x3 affine and its inverse (crop -> frame),
// row-major [a,b,tx, c,d,ty]. Pure scale+translation (rot=0), matching
// get_warp_matrix(center, square, rot=0, output_size=(out,out)).
void MakeAffine(const std::array<float, 2>& center, float square, int out,
                std::array<float, 6>& fwd, std::array<float, 6>& inv);

// CLIFF condition_info (cx/f, cy/f, b/f) with USE_INTRIN_CENTER=true:
//   [(cx - Kcx)/f, (cy - Kcy)/f, s/f]   (cx,cy = bbox center; s = square side).
std::array<float, 3> ConditionInfo(const BBox& box, float pad, const CamInt& K,
                                   int in_w = kBodyImageSize,
                                   int in_h = kBodyImageSize);

// get_ray_condition -> CameraEncoder interpolate -> [2, 32, 32] (kTokenGrid).
// Channel 0 = x-ray, channel 1 = y-ray; layout [2][32][32] row-major.
void RayCond(const BBox& box, float pad, const CamInt& K,
             std::array<float, 2 * kTokenGrid * kTokenGrid>& out,
             int in_w = kBodyImageSize, int in_h = kBodyImageSize);

// perspective_projection camera solve (focal + cam_t). `pred_cam` = (s,tx,ty)
// from the camera head; default_scale = 1.0 (body), DEFAULT_SCALE_FACTOR_HAND
// for hands. USE_INTRIN_CENTER=true. cam_t is the metric translation applied to
// the 3D points before projection.
Camera PerspectiveProjection(const std::array<float, 3>& pred_cam, const BBox& box,
                             float pad, const CamInt& K, float default_scale = 1.f,
                             int in_w = kBodyImageSize, int in_h = kBodyImageSize);

}  // namespace hastur
