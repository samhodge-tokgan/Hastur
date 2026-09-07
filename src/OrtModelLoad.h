// SPDX-License-Identifier: LicenseRef-Tokgan-Proprietary
// Copyright (c) Tokgan. Proprietary — licensed under the Tokgan EULA; see NOTICE.
//
// One place that decides how an ORT session is created (#43).
//
// Header-only and ORT-aware, so hastur_modelcrypt stays free of ONNX Runtime
// while every engine gets the same rule:
//
//   real file on disk  -> open BY PATH. Required for models with external data
//                         (.onnx.data), which ORT resolves relative to the
//                         model file and refuses to resolve at all for a model
//                         loaded from a buffer. Covers plaintext trees and
//                         materialised sealed ones.
//   no file there      -> sealed single-file model; decrypt and load the bytes.
//
// Getting this wrong is not a compile error. It surfaces at runtime as
// "External data path for model loaded from bytes escapes working directory",
// which is a long way from its cause — the tracker hit exactly that when every
// site was converted to buffer loads indiscriminately.
#pragma once

#include <memory>
#include <string>

#include <onnxruntime_cxx_api.h>

#include "ModelBytes.h"
#include "OrtAccel.h"

namespace hastur {

inline std::unique_ptr<Ort::Session> MakeModelSession(
    Ort::Env& env, const std::string& model_path, const Ort::SessionOptions& so) {
  if (ModelOnDisk(model_path))
    return std::make_unique<Ort::Session>(env, OrtPath(model_path).c_str(), so);
  const std::string bytes = LoadModelBytes(model_path);
  return std::make_unique<Ort::Session>(env, bytes.data(), bytes.size(), so);
}

}  // namespace hastur
