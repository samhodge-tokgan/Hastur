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

#include <filesystem>
#include <memory>
#include <vector>
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

  // A sealed model WITH external data. The comment in ModelBytes.h says this is
  // impossible from a buffer -- "ORT resolves an .onnx.data sidecar relative to
  // the model file, and refuses outright for a model loaded from bytes". That
  // was true when it was written and stopped being true at ORT 1.17, which
  // added AddExternalInitializersFromFilesInMemory: it maps the sidecar's
  // FILENAME to a buffer, so the relative reference inside the graph resolves
  // without a file existing anywhere.
  //
  // This is what lets a sealed tracker tree load with no scratch directory --
  // on Windows, which has no tmpfs and therefore no MaterialiseModelDir at all,
  // and equally on Linux, where it means the plaintext weights never touch a
  // filesystem.
  const std::string sidecar_path = model_path + ".data";
  if (ModelExists(sidecar_path)) {
    // Must outlive the Session constructor: ORT reads through this buffer while
    // building the session. It does, because the session is constructed inside
    // this scope -- do not be tempted to hoist the return past it.
    std::string sidecar = LoadModelBytes(sidecar_path);

    // The registered name must be what the GRAPH refers to, i.e. the bare
    // filename, not the path we happened to build. A mismatch is not an error
    // at registration time; it fails later as an unresolved initializer.
    const std::filesystem::path sidecar_name =
        std::filesystem::path(sidecar_path).filename();
    std::vector<std::basic_string<ORTCHAR_T>> names{sidecar_name.native()};
    std::vector<char*> buffers{sidecar.data()};
    std::vector<size_t> lengths{sidecar.size()};

    Ort::SessionOptions opts = so.Clone();
    opts.AddExternalInitializersFromFilesInMemory(names, buffers, lengths);
    return std::make_unique<Ort::Session>(env, bytes.data(), bytes.size(), opts);
  }

  return std::make_unique<Ort::Session>(env, bytes.data(), bytes.size(), so);
}

}  // namespace hastur
