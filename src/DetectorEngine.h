// Copyright the Hastur authors.
// SPDX-License-Identifier: LicenseRef-SAM-License
//
// DetectorEngine — thin ONNX Runtime wrapper around the clean, commercial-
// licensed person detector exported by tools/export_detector.py (torchvision
// SSDLite320 MobileNetV3, BSD-3-Clause). Replaces the export-hostile Detectron2
// ViTDet-H detector for Track A / M1.
//
// The model has a FIXED square input [3,S,S] (RGB in [0,1]; ImageNet
// normalization + NMS are baked into the graph). This engine does aspect-
// preserving letterbox preprocessing on the full frame, runs the session on the
// platform accelerator EP (CoreML/CUDA, CPU fallback), then maps the returned
// boxes back to full-frame pixels and filters to the "person" class. It mirrors
// the humbaba DepthEngine structure and reuses src/OrtAccel.h.
#pragma once

#include <memory>
#include <string>

#include "MeshTypes.h"  // hastur::Detections, BBox, ComputeUnits

namespace hastur {

class DetectorEngine {
 public:
  // Loads `model_path` (an ONNX person detector with a fixed square input) and
  // creates a session. `intra_threads` <= 0 leaves the ORT default. The fixed
  // processing size is read from the model's input shape.
  DetectorEngine(const std::string& model_path, ComputeUnits units,
                 int intra_threads);
  ~DetectorEngine();

  DetectorEngine(const DetectorEngine&) = delete;
  DetectorEngine& operator=(const DetectorEngine&) = delete;

  // Runs person detection on an RGB image (interleaved, row-major, values in
  // [0,1]) of size W x H. Returns person boxes in full-frame pixels (xyxy) with
  // score >= score_thresh, sorted by descending score.
  Detections Run(const float* rgb, int W, int H, float score_thresh = 0.5f);

  // True if the session actually placed nodes on the platform accelerator EP.
  bool accelerator_active() const;

  // Whether the platform accelerator EP is compiled into this ORT build.
  static bool AcceleratorAvailable();

  const std::string& last_error() const { return last_error_; }

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  std::string last_error_;
};

}  // namespace hastur
