// Copyright the Hastur authors.
// SPDX-License-Identifier: LicenseRef-SAM-License
//
// ExternalTracks — ingest per-frame person tracks (boxes + stable ids, and
// optional instance masks) produced by an UPSTREAM tracker (e.g. a SAM 3 video
// predictor), so Hastur can drive the MHR 3D-body pose from those boxes instead
// of running its own detector + geometric (cam_t) association.
//
// Why this exists: the internal detector + AssignTrackIds() gives stable
// person_NN ids only when one plugin instance sees frames in order and holds the
// track table between renders — which breaks in clone-per-frame hosts and swaps
// ids on dynamic footage. When external tracks are supplied the ids arrive
// PRE-ASSIGNED per frame, so every frame is independent: ids are stable by
// construction and no sequential/stateful render is required.
//
// Sidecar format (dependency-free; no JSON in-repo — mirrors the simple text
// scan in DetectorEngine.cpp and the binary fread of mhr_wrist.bin):
//
//   # hastur-tracks v1
//   # ids 0 1 2 5 7                 <- OPTIONAL: every track id in the clip
//   # frame track_id x0 y0 x1 y1 [mask_relpath]
//   1 0 320.5 88.0 512.0 940.0 masks/0001_00.msk
//   1 1 700.0 120.0 880.0 900.0
//   2 0 322.0 90.0 514.0 942.0 masks/0002_00.msk
//
//   * frame     integer, matched to lround(pipeline time) == the plate frame no.
//   * track_id  the tracker's stable id; becomes person_NN directly.
//   * x0 y0 x1 y1  full-frame pixels, xyxy, TOP-DOWN origin (y=0 at the top —
//                  the same orientation the pipeline's rgb buffer already uses;
//                  OFX is natively bottom-up, so a writer must flip).
//   * mask_relpath OPTIONAL binary coverage, relative to the sidecar's dir:
//                    magic  : 4 bytes  "MSK1"
//                    w, h   : int32 little-endian
//                    data   : w*h uint8 (0..255) full-frame stretched coverage
//                  Consumed as a DetMask (native res, resampled to the frame),
//                  i.e. it feeds the exact Sam3Mask coverage path a detector mask
//                  would (CleanDetMask + UpsampleMaskToFrame).
//
// On-disk layouts (either works): a single "tracks.txt" holding all frames
// (primary), or per-frame files via a printf pattern "tracks.%04d.txt" (an entry
// containing '%'); each per-frame file may repeat the "# ids" header.

#pragma once

#include <string>
#include <vector>

#include "DetectorEngine.h"  // hastur::DetMask
#include "MeshTypes.h"       // hastur::BBox

namespace hastur {

struct ExternalTrack {
  int id{-1};
  BBox box;      // xyxy full-frame px (top-down), score set to 1
  DetMask mask;  // empty (w==h==0) when no mask supplied / want_masks == false
};

// Load the tracks for ONE frame. `search_path` is a path-list (kPathListSep-
// separated, plus $HASTUR_TRACKS appended) of entries, each a directory holding
// tracks.txt / tracks.%04d.txt, a direct tracks file, or a '%'-printf per-frame
// pattern. Returns the records whose frame column == `frame` (empty => none for
// this frame; the caller then passes the source through). Masks are decoded only
// when `want_masks`.
std::vector<ExternalTrack> LoadExternalTracks(const std::string& search_path,
                                              long frame, bool want_masks);

// The complete set of track ids declared by the clip's "# ids ..." header, so a
// full (decodable-on-disk) Cryptomatte manifest can be baked even for frames that
// don't contain every id. Empty when no header is present.
std::vector<int> ExternalTrackIds(const std::string& search_path);

// True if any tracks sidecar can be resolved from `search_path`. Used by the
// plugin to auto-enable the external-tracks path when a sidecar is present.
bool HasExternalTracks(const std::string& search_path);

}  // namespace hastur
