// Copyright the Hastur authors.
// SPDX-License-Identifier: LicenseRef-SAM-License
//
// Cross-platform ONNX Runtime accelerator execution-provider selection:
// CoreML or WebGPU on macOS, CUDA on Linux/Windows (with the caller's CPU fallback).
#pragma once

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <unordered_map>

#include <onnxruntime_cxx_api.h>

namespace hastur {

// Ort::Session takes a path of type const ORTCHAR_T* — wchar_t on Windows, char on
// POSIX. std::filesystem::path::c_str() yields exactly that native type, so this
// converts a UTF-8/native std::string path for the Session constructor. Keep the
// returned object alive across the c_str() use (it is, within a full expression).
inline std::filesystem::path OrtPath(const std::string& s) {
  return std::filesystem::path(s);
}

#ifdef __APPLE__
// Which provider "the accelerator" means on macOS. CoreML by default;
// HASTUR_APPLE_EP=webgpu selects ORT's WebGPU provider (Metal underneath).
//
// WebGPU is the one that runs the SAM 3 tracker graphs on the GPU: CoreML
// cannot load the fp32 graphs (external initializers) and only partitions G1
// (2051 of 2347 nodes in 89 pieces), which on an M1 is no faster than the CPU.
// WebGPU on an M1, G1-G4, 30 frames: 1.84x the CPU, masks identical.
inline bool AppleUsesWebGpu() {
  const char* e = std::getenv("HASTUR_APPLE_EP");
  return e && std::strcmp(e, "webgpu") == 0;
}
#endif

inline const char* AcceleratorSubstr() {
#ifdef __APPLE__
  return AppleUsesWebGpu() ? "WebGpu" : "CoreML";
#else
  return "CUDA";
#endif
}

// True if the platform accelerator EP is compiled into this ONNX Runtime build.
inline bool AcceleratorAvailable() {
  const std::string want = AcceleratorSubstr();
  for (const auto& p : Ort::GetAvailableProviders())
    if (p.find(want) != std::string::npos) return true;
  return false;
}

// Append the platform accelerator EP to `so`. On macOS this is CoreML (using the
// given CoreML options) or WebGPU (see AppleUsesWebGpu; the CoreML options are
// then unused); on Linux/Windows it is CUDA device `cuda_device` (the ordinal
// among the visible devices). Throws Ort::Exception on failure — callers catch
// and fall back to CPU.
inline void AppendAccelerator(Ort::SessionOptions& so, const char* coreml_units,
                              bool coreml_static, bool coreml_mlprogram = true,
                              int cuda_device = 0) {
#ifdef __APPLE__
  (void)cuda_device;
  if (AppleUsesWebGpu()) {
    so.AppendExecutionProvider("WebGPU", {});
    return;
  }
  std::unordered_map<std::string, std::string> opts;
  if (coreml_units) opts["MLComputeUnits"] = coreml_units;
  if (coreml_mlprogram) opts["ModelFormat"] = "MLProgram";
  if (coreml_static) opts["RequireStaticInputShapes"] = "1";
  so.AppendExecutionProvider("CoreML", opts);
#else
  (void)coreml_units;
  (void)coreml_static;
  (void)coreml_mlprogram;
  OrtCUDAProviderOptions cuda{};
  cuda.device_id = cuda_device;
  so.AppendExecutionProvider_CUDA(cuda);
#endif
}

}  // namespace hastur
