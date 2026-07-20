// Copyright the Hastur authors.
// SPDX-License-Identifier: LicenseRef-SAM-License
//
// Cryptomatte.h -- builds Psyop-standard Cryptomatte ID mattes from the pipeline's
// per-person coverage buffers. Cryptomatte encodes anti-aliased / overlapping
// object IDs as per-pixel ranked (id, coverage) pairs packed two ranks per RGBA
// layer, plus a name->hash manifest carried as metadata. This is NOT cryptographic;
// the "crypto" is a MurmurHash3 of each object's name used as a stable float ID.
//
// Reference: cryptomatte specification v1.2.1 (Friedman & Jones, Psyop) and the
// pycryptomatte `mm3hash_float` conversion.
//
// Dependency-light: C++ standard library + MeshTypes.h only.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "MeshTypes.h"

namespace hastur {

// MurmurHash3 x86 32-bit (Austin Appleby), seed 0 by default -- the exact hash
// mmh3.hash() / Cryptomatte use over the UTF-8 bytes of an object name.
uint32_t MurmurHash3_x86_32(const void* key, int len, uint32_t seed = 0);

// Cryptomatte name -> float32 ID: MurmurHash3_32(name), then clamp the exponent
// bits away from 0/255 so the reinterpreted float is never inf/NaN/denormal.
float CryptoIdFloat(const std::string& name);

// The 8-hex-digit manifest value for `name` (bit pattern of CryptoIdFloat(name)).
std::string CryptoIdHex(const std::string& name);

// The 7-hex-digit metadata key for a Cryptomatte typename (e.g. "person"),
// used to form the "cryptomatte/<key>/..." EXR metadata keys.
std::string CryptoTypeKey(const std::string& type_name);

// One person's silhouette for the current frame.
struct CryptoPerson {
  std::string name;             // e.g. "person_00"
  std::vector<float> coverage;  // W*H, fractional coverage [0,1], top-down HWC
};

// Builds the packed Cryptomatte layers + manifest. `persons` MUST be ordered
// FRONT-TO-BACK (nearest camera first) so occlusion is resolved with the same
// front-to-back "over" the beauty composite uses: per pixel, visible coverage of
// person i = coverage_i * prod_{j<i}(1 - coverage_j). Ranks are then sorted by
// visible coverage descending and packed two per RGBA layer
// (layer0 = id0,cov0,id1,cov1; layer1 = ranks 2,3; ...). `num_levels` layers.
CryptoResult BuildCryptomatte(int W, int H, const std::string& type_name,
                              const std::vector<CryptoPerson>& persons,
                              int num_levels = 2);

}  // namespace hastur
