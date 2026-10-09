// Copyright the Hastur authors.
// SPDX-License-Identifier: LicenseRef-SAM-License
//
// Sam3VideoTrack: the whole-clip SAM 3 multi-person pre-pass as a library call.
// It is the algorithm of Sam3TrackerPlugin::RunPrepass, which hastur_track ran
// headless: seed-frame scan, seed, forward plus backward bidirectional passes,
// det<->track IoU association, spawn and prune. Frames come from a caller-supplied
// source instead of PNG files, so an embedding application (alphagen) can feed its
// own frames and run the same tracker the node and hastur_track do.
//
// Output, as hastur_track always wrote it:
//   <out>/tracks.txt              # frame track_id x0 y0 x1 y1 mask_relpath (plate px, top-down)
//   <out>/masks/<frame>_<id>.msk  # MSK1 soft coverage, 1008^2 uint8
// A clip with no people writes a header-only tracks.txt and succeeds. An engine
// failure throws and writes no tracks.txt (an exception is not an empty clip).
#pragma once

#include <functional>
#include <string>
#include <vector>

#include "Sam3MultiTracker.h"

namespace hastur {

// Every file the tracker group needs, as names in a model dir (a sealed tree has no
// readable listing, so callers that materialise or stage it must be told).
const std::vector<std::string>& Sam3TrackerFiles();

// The engine's paths for a model dir holding G1.onnx..G5.onnx and the const_*.npy.
MultiPaths Sam3TrackerPaths(const std::string& dir);

// Fills `rgb` with plate frame `frame` as top-down interleaved RGB, W*H*3 floats in
// [0,1], in the encoding the tracker was trained on (sRGB, i.e. srgb_texture for an
// ACEScg plate). Returns false when the frame cannot be read; the frame is skipped.
using Sam3FrameSource = std::function<bool(long frame, std::vector<float>& rgb, int& W, int& H)>;

// Called as work progresses: phase is "scan", "forward" or "backward", done/total are
// frames within that phase. It may throw to abandon the run (cancellation).
using Sam3Progress = std::function<void(const char* phase, int done, int total)>;

struct Sam3VideoTrackOptions {
  long seed = -1;            // seed frame in plate numbering; -1 = auto (max visibility)
  float score = kScoreThresh;
  float nms = kDetNmsIou;
  bool bidir = true;         // backward pass from the seed frame
  Ep ep = Ep::Auto;
  ComputeUnits units = ComputeUnits::All;
  const char* log_tag = "hastur_track";  // stderr prefix
  Sam3Progress progress;     // optional
};

struct Sam3VideoTrackResult {
  size_t rows = 0;  // tracks.txt data rows
  size_t ids = 0;   // distinct track ids
};

// Tracks plate frames first..last and writes out_dir/tracks.txt + masks/. Throws
// std::exception on any failure (unreadable seed frame, no frames read, engine error).
Sam3VideoTrackResult RunSam3VideoTrack(const MultiPaths& models, const Sam3FrameSource& source,
                                       long first, long last, const std::string& out_dir,
                                       const Sam3VideoTrackOptions& opt = {});

}  // namespace hastur
