// Copyright the Hastur authors.
// SPDX-License-Identifier: LicenseRef-SAM-License
//
// SurfaceParam.h -- static per-vertex surface parametrisation of the MHR mesh
// for the Pref and ST data AOVs, computed once at load from the canonical
// `base_shape` (so it needs no gated asset and works on every existing install,
// like LeotardMask). Dependency-light (C++ standard library only): no Eigen, no
// ONNX Runtime, no OFX.
//
// Both are derived from the SAME fixed canonical shape for every person, so they
// are shot-invariant and directly comparable frame-to-frame and across shots --
// which is what makes them usable as stable surface keys (locator/texture
// pinning, 2.5D surface-feature remapping).

#pragma once

#include <vector>

namespace hastur {

// Normalised, all-positive canonical body position, per vertex (nverts*3), each
// axis mapped to [0,1] by the canonical rest-pose bounding box. Input is the raw
// `base_shape` (nverts*3, centimetres, pre-flip); the Y/Z axes are flipped to the
// posed-mesh frame (matching MhrModel / the Position AOV) before normalising, so
// Pref and Position share an orientation. Shot-invariant (the bbox is fixed).
std::vector<float> ComputePrefNormalized(const float* base_shape_cm, int nverts);

// Per-vertex cylindrical UV unwrap, (nverts*2). Azimuth about the vertical axis
// -> U in [0,1] (seam at the back), normalised height -> V in [0,1]. Matches the
// deterministic fallback in tools/extract_mhr_assets.py:extract_uv (computed from
// the pre-flip `base_shape`), so runtime and baked assets agree. A "simple ST"
// good enough for texture projection / DensePose-style conditioning / 2.5D
// remapping -- not a seam-free artist atlas.
std::vector<float> ComputeCylindricalUV(const float* base_shape_cm, int nverts);

}  // namespace hastur
