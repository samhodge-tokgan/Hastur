// Copyright the Hastur authors.
// SPDX-License-Identifier: Apache-2.0
//
// CropAffine — build the 512x512 body/hand crop + camera conditioning for one
// person, ported from the SAM 3D Body reference (sam-3d-body @ c259bfc):
// GetBBoxCenterScale + TopdownAffine (cv2.warpAffine, INTER_LINEAR, use_udp=
// False, input_size 512x512) + ImageNet normalization to CHW, plus the camera
// conditioning (ray_cond / condition_info / affine + inverse) via CameraSolver.
#pragma once

#include "CameraSolver.h"
#include "MeshTypes.h"

namespace hastur {

// Build a CropInputs from a full-frame RGB image (row-major HWC, 3 channels,
// float in [0,1]) and a detection box. `pad` = 1.25 (body) / 0.9 (hand).
//   * image[3,512,512]  bilinear-warped crop, ImageNet-normalized CHW
//   * ray_cond[2,32,32] get_ray_condition -> CameraEncoder interpolate
//   * condition_info[3] CLIFF (USE_INTRIN_CENTER=true)
//   * affine / affine_inv  frame<->crop 2x3
//   * center / scale    GetBBoxCenterScale outputs (pre-aspect padded scale)
CropInputs MakeCrop(const float* rgb, int W, int H, const BBox& box,
                    const CamInt& K, float pad = 1.25f);

}  // namespace hastur
