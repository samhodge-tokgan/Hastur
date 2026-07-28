// Copyright the Hastur authors.
// SPDX-License-Identifier: LicenseRef-SAM-License
//
// AovExr — write a per-person MHR AOV EXR (companion to the refined cryptomatte).
// For each person `names[i]` (e.g. "person_00") it emits, from aovs[i]:
//   <name>.A            silhouette alpha [0,1]
//   <name>.Pref.{R,G,B} canonical reference position (Pref) XYZ in [0,1]
//   <name>.Z            camera-space body depth (metres)
// plus Color R,G,B,A = the union-of-alphas preview. Consumed by Rotobot-Next's
// topology bridge (Pref-guided island bridge) + depth work. Planar float32, ZIP.
#pragma once

#include <string>
#include <vector>

#include "CryptoExr.h"   // ExrCompression
#include "MeshTypes.h"   // PersonAov

namespace hastur {

bool WriteAovExr(const std::string& path, int W, int H,
                 const std::vector<std::string>& names,
                 const std::vector<PersonAov>& aovs,
                 ExrCompression compression = ExrCompression::kZIP);

}  // namespace hastur
