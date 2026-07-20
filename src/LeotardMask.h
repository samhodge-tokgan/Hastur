// Copyright the Hastur authors.
// SPDX-License-Identifier: LicenseRef-SAM-License
//
// LeotardMask.h -- static per-vertex garment (leotard) mask for the MHR mesh.
// Dependency-light (MeshTypes.h + the C++ standard library only): no Eigen, no
// ONNX Runtime, no OFX — so it can be reused by tooling (tools/leotard_preview).

#pragma once

#include <vector>

#include "MeshTypes.h"

namespace hastur {

// Per-vertex garment mask (kNumVerts, 1 = leotard, 0 = skin) from a REST-pose
// mesh + its mhr70 keypoints. The garment = smooth union of five capsules: torso
// (pelvis->neck) and the upper arm (shoulder->elbow) + upper leg (hip->knee) on
// each side. Radii scale with the subject's own shoulder span (proportion-
// invariant); the smooth radial falloff gives clean fabric hems at the
// elbow/knee/neckline. Returns an empty vector on degenerate landmarks (caller
// then disables the garment -> plain neutral clay).
std::vector<float> ComputeLeotardMask(const Mesh& rest);

}  // namespace hastur
