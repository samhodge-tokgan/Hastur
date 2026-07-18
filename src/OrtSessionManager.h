// Copyright the Hastur authors.
// SPDX-License-Identifier: LicenseRef-SAM-License
//
// OrtSessionManager — one shared Ort::Env for the whole pipeline (detector,
// body regressor, hand refiner, pose-corrective) plus a common GPU memory
// budget. Centralizes the per-engine session-creation dance humbaba did ad hoc
// (DepthEngine.cpp): append the platform accelerator EP (CoreML on macOS with
// MLComputeUnits + static shapes; CUDA on Linux/Windows with a gpu_mem_limit +
// shared arena) and fall back to a plain CPU session if the accelerator throws
// at append or session-create time. Sessions may be created LAZILY so the
// hand/pose-corrective models are only built when first needed.
#pragma once

#include <memory>
#include <string>

#include <onnxruntime_cxx_api.h>

#include "MeshTypes.h"

namespace hastur {

// Execution-provider preference.
enum class Ep {
  Auto,         // platform accelerator if available, else CPU
  Accelerator,  // force CoreML/CUDA (still CPU-fallback on failure)
  Cpu,          // CPU only
};

struct SessionConfig {
  std::string model_path;
  ComputeUnits units = ComputeUnits::All;  // CoreML MLComputeUnits
  Ep ep = Ep::Auto;
  size_t gpu_mem_limit = 0;  // CUDA per-session arena budget in bytes (0=default)
  int intra_threads = 0;     // 0 = ORT default
  bool coreml_static = true;
  bool lazy = false;         // defer creation until first Get()
};

class OrtSessionManager {
 public:
  explicit OrtSessionManager(const std::string& log_id = "Hastur");
  ~OrtSessionManager();

  OrtSessionManager(const OrtSessionManager&) = delete;
  OrtSessionManager& operator=(const OrtSessionManager&) = delete;

  Ort::Env& env();

  // A (possibly lazily-created) session bound to the shared env.
  class Handle {
   public:
    // Create-on-first-use. Throws Ort::Exception / std::runtime_error if the
    // session cannot be created even on the CPU fallback.
    Ort::Session& Get();
    // Same but returns nullptr instead of throwing.
    Ort::Session* TryGet() noexcept;
    bool created() const noexcept { return session_ != nullptr; }
    bool accelerator_active() const noexcept { return accelerator_active_; }
    const std::string& last_error() const noexcept { return last_error_; }
    const std::string& model_path() const noexcept { return cfg_.model_path; }

   private:
    friend class OrtSessionManager;
    Handle(OrtSessionManager* mgr, SessionConfig cfg) : mgr_(mgr), cfg_(std::move(cfg)) {}
    void Build();

    OrtSessionManager* mgr_;
    SessionConfig cfg_;
    std::unique_ptr<Ort::Session> session_;
    bool accelerator_active_ = false;
    std::string last_error_;
  };

  // Register a session. If cfg.lazy is false it is created immediately (any
  // failure is recorded in the handle's last_error(); TryGet() then returns
  // nullptr). If lazy, creation happens on the first Get()/TryGet().
  std::shared_ptr<Handle> MakeSession(const SessionConfig& cfg);

  // Convenience overload matching the pipeline call sites.
  std::shared_ptr<Handle> MakeSession(const std::string& path, ComputeUnits units,
                                      Ep ep = Ep::Auto, size_t gpu_mem_limit = 0,
                                      bool lazy = false);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace hastur
