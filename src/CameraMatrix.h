// Copyright the Hastur authors.
// SPDX-License-Identifier: LicenseRef-SAM-License
//
// CameraMatrix.h -- builds the m44f world<->NDC camera matrices from the pipeline's
// pinhole intrinsics, for downstream 3D reconstruction / relighting.
//
// Convention (matches SoftwareRasterizer.h): the camera sits at the origin looking
// down +Z (x right, y down in image space); a camera-space point projects to
//   u = fx*x/z + cx,   v = fy*y/z + cy   (pixels).
// For this single solved camera the camera frame IS the world (worldToCamera = I),
// so "world" below means camera space.
//
// worldToNDC maps a homogeneous camera point [x,y,z,1] to clip coords; NDC =
// clip.xyz / clip.w lands in [-1,1]^3 with NDC y UP (v flipped) and NDC z mapping
// [near,far] -> [-1,1]. NDCToWorld is the exact inverse: a downstream unprojects a
// pixel via  q = NDCToWorld * [ndc_x, ndc_y, ndc_z, 1];  P = q.xyz / q.w.
//
// Dependency-light: C++ standard library + MeshTypes.h only.

#pragma once

#include "MeshTypes.h"

namespace hastur {

// Build worldToNDC / NDCToWorld (and echo the intrinsics) for a frame. `focal`
// is fx=fy in pixels, (cx,cy) the principal point, W/H the frame size, and
// near/far the depth range mapped to NDC z = [-1, 1] (far > near > 0).
CameraMatrices BuildCameraMatrices(float focal, float cx, float cy, int W, int H,
                                   float near_z = 0.1f, float far_z = 100.0f);

// General double-precision 4x4 inverse of a row-major matrix. Returns false and
// leaves `out` untouched if `m` is singular. Exposed for tests.
bool Invert4x4(const std::array<float, 16>& m, std::array<float, 16>& out);

}  // namespace hastur
