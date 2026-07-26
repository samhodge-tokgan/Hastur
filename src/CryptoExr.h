// Copyright the Hastur authors.
// SPDX-License-Identifier: LicenseRef-SAM-License
//
// CryptoExr — writes a decodable Cryptomatte EXR straight to disk, so the Matte
// Partition node does NOT depend on the host's multi-plane Write (Natron 2.5 hangs
// assembling a plugin-produced multi-plane Write). Pure C++ + vendored tinyexr; no
// OFX / GPU here, so it is unit-testable in isolation (tests/exr_write_validate.cpp).
//
// LAYOUT (single-part, per the current chunk):
//   * Color  R,G,B,A               = the pooled matte (m,m,m,m) — beauty/preview
//   * <type>00.{R,G,B,A}           = crypto.layers[0]  (ranks 0,1)
//   * <type>01.{R,G,B,A}           = crypto.layers[1]  (ranks 2,3)  ...
//     (one Cryptomatte plane per emitted level; ids are exact float32 bit patterns)
//     <type> == the "name" metadata below (e.g. "person") — the Cryptomatte spec
//     REQUIRES the channel-layer prefix to equal `name`, or Nuke/Fusion find no
//     matte. (A decorative prefix like "hastur.CryptoObject" makes it undecodable.)
//   * float32 channels, LOSSLESS compression (ZIP default).
//   * Cryptomatte metadata attributes (Psyop spec) so it decodes in Nuke/Fusion:
//       cryptomatte/<key>/name       = type_name ("person")
//       cryptomatte/<key>/manifest   = crypto.manifest (JSON name -> 8hex)
//       cryptomatte/<key>/hash       = "MurmurHash3_32"
//       cryptomatte/<key>/conversion = "uint32_to_float32"
//     where <key> = CryptoTypeKey(type_name) (7-hex).
//   * Optional string attribute  hastur/skeleton = the per-frame skeleton JSON.
//
// COMPRESSION: the Cryptomatte channels carry exact float32 ids and MUST be lossless
// (Cryptomatte spec) — ANY lossy codec corrupts them. Every option this writer
// exposes is lossless, so a single-part file is always safe. Splitting the RGB
// (display) part off to a lossy DWAB compression requires a MULTI-PART EXR, which
// tinyexr cannot write; that is the OpenEXR/Imf follow-up (see PR notes).

#pragma once

#include <string>
#include <vector>

#include "MeshTypes.h"  // hastur::CryptoResult

namespace hastur {

// EXR compression for the whole (single-part) file. ALL choices are LOSSLESS, so
// the exact-float32 Cryptomatte ids always round-trip bit-for-bit. DWAB (lossy) is
// intentionally absent — it belongs only on a separate display part of a multi-part
// file (the OpenEXR follow-up), never on the crypto channels.
enum class ExrCompression {
  kNone = 0,  // uncompressed
  kRLE = 1,   // run-length
  kZIPS = 2,  // per-scanline zlib
  kZIP = 3,   // 16-scanline-block zlib (default)
  kPIZ = 4,   // wavelet + Huffman (lossless)
};

// Write a decodable single-part Cryptomatte EXR to `path` (directories must already
// exist). Best-effort: returns false on any failure without throwing.
//   pool          W*H pooled soft alpha, top-down HWC; written to Color as (m,m,m,m).
//   crypto        the partitioned Cryptomatte: crypto.width/height set the size,
//                 crypto.layers[i] (W*H*4) -> <type>NN.{R,G,B,A} (type = type_name),
//                 crypto.manifest + crypto.type_name feed the metadata attributes.
//   levels        number of Cryptomatte planes to emit (clamped to layers.size()).
//   compression   file compression (lossless; default ZIP).
//   skeleton_json optional; when non-empty, embedded verbatim as the EXR string
//                 attribute "hastur/skeleton" (stretch: skeleton rides in the EXR).
bool WriteCryptoExr(const std::string& path, const std::vector<float>& pool,
                    const CryptoResult& crypto, int levels,
                    ExrCompression compression = ExrCompression::kZIP,
                    const std::string& skeleton_json = std::string());

}  // namespace hastur
