// Copyright the Hastur authors.
// SPDX-License-Identifier: Apache-2.0
#include "OrtSessionManager.h"

#include <stdexcept>

#include "OrtAccel.h"

namespace hastur {

namespace {

const char* ComputeUnitsString(ComputeUnits u) {
  switch (u) {
    case ComputeUnits::All: return "ALL";
    case ComputeUnits::CpuAndGpu: return "CPUAndGPU";
    case ComputeUnits::CpuAndAne: return "CPUAndNeuralEngine";
    case ComputeUnits::CpuOnly: return "CPUOnly";
  }
  return "ALL";
}

Ort::SessionOptions BaseOptions(const SessionConfig& cfg) {
  Ort::SessionOptions so;
  so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  if (cfg.intra_threads > 0) so.SetIntraOpNumThreads(cfg.intra_threads);
  return so;
}

}  // namespace

struct OrtSessionManager::Impl {
  Ort::Env env;
#ifndef __APPLE__
  // A single CUDA arena shared across sessions so the whole pipeline lives
  // inside one GPU memory budget (registered once on the shared env).
  bool cuda_arena_registered = false;
#endif
  explicit Impl(const std::string& log_id)
      : env(ORT_LOGGING_LEVEL_WARNING, log_id.c_str()) {}
};

OrtSessionManager::OrtSessionManager(const std::string& log_id)
    : impl_(std::make_unique<Impl>(log_id)) {}
OrtSessionManager::~OrtSessionManager() = default;

Ort::Env& OrtSessionManager::env() { return impl_->env; }

// EP append. macOS: CoreML via OrtAccel (MLProgram + static shapes). Non-Apple:
// CUDA device 0 with an optional gpu_mem_limit + shared env arena.
void OrtSessionManager::Handle::Build() {
  if (session_) return;
  Impl& impl = *mgr_->impl_;

  const bool want_accel =
      (cfg_.ep == Ep::Auto || cfg_.ep == Ep::Accelerator) && AcceleratorAvailable();

  Ort::SessionOptions so = BaseOptions(cfg_);
  bool appended_accel = false;
  if (want_accel) {
    try {
#ifdef __APPLE__
      hastur::AppendAccelerator(so, ComputeUnitsString(cfg_.units),
                                /*coreml_static=*/cfg_.coreml_static,
                                /*coreml_mlprogram=*/true);
#else
      OrtCUDAProviderOptions cuda{};
      cuda.device_id = 0;
      cuda.do_copy_in_default_stream = 1;
      if (cfg_.gpu_mem_limit > 0) {
        cuda.gpu_mem_limit = cfg_.gpu_mem_limit;
        cuda.arena_extend_strategy = 1;  // kSameAsRequested — tighter budget
        if (!impl.cuda_arena_registered) {
          // One shared CUDA arena on the env so every session draws from the
          // same budget rather than each allocating its own.
          OrtArenaCfg* arena_cfg = nullptr;
          Ort::ThrowOnError(Ort::GetApi().CreateArenaCfg(
              cfg_.gpu_mem_limit, /*arena_extend_strategy=*/1,
              /*initial_chunk_size_bytes=*/-1, /*max_dead_bytes_per_chunk=*/-1,
              &arena_cfg));
          OrtMemoryInfo* mem = nullptr;
          Ort::ThrowOnError(Ort::GetApi().CreateMemoryInfo(
              "Cuda", OrtArenaAllocator, 0, OrtMemTypeDefault, &mem));
          Ort::GetApi().CreateAndRegisterAllocatorV2(
              static_cast<OrtEnv*>(impl.env), "CUDAExecutionProvider", mem,
              arena_cfg, nullptr, 0);
          Ort::GetApi().ReleaseMemoryInfo(mem);
          Ort::GetApi().ReleaseArenaCfg(arena_cfg);
          impl.cuda_arena_registered = true;
        }
        so.AddConfigEntry("session.use_env_allocators", "1");
      }
      so.AppendExecutionProvider_CUDA(cuda);
#endif
      appended_accel = true;
    } catch (const Ort::Exception& e) {
      last_error_ = std::string(AcceleratorSubstr()) + " EP append failed: " + e.what();
      appended_accel = false;
    }
  }

  try {
    session_ = std::make_unique<Ort::Session>(
        impl.env, OrtPath(cfg_.model_path).c_str(), so);
    accelerator_active_ = appended_accel;
    return;
  } catch (const Ort::Exception& e) {
    // Accelerator EP can fail at session-create time (missing CUDA/cuDNN, an
    // op CoreML can't compile, ...). Retry once on a plain CPU session.
    if (!appended_accel) {
      last_error_ = std::string("session create failed: ") + e.what();
      throw;
    }
    last_error_ = std::string(AcceleratorSubstr()) +
                  " session create failed (falling back to CPU): " + e.what();
  }

  Ort::SessionOptions cpu_so = BaseOptions(cfg_);
  session_ = std::make_unique<Ort::Session>(
      impl.env, OrtPath(cfg_.model_path).c_str(), cpu_so);
  accelerator_active_ = false;
}

Ort::Session& OrtSessionManager::Handle::Get() {
  if (!session_) Build();
  if (!session_) throw std::runtime_error("OrtSessionManager: session unavailable: " +
                                          last_error_);
  return *session_;
}

Ort::Session* OrtSessionManager::Handle::TryGet() noexcept {
  if (!session_) {
    try {
      Build();
    } catch (const std::exception& e) {
      if (last_error_.empty()) last_error_ = e.what();
      return nullptr;
    } catch (...) {
      return nullptr;
    }
  }
  return session_.get();
}

std::shared_ptr<OrtSessionManager::Handle> OrtSessionManager::MakeSession(
    const SessionConfig& cfg) {
  auto h = std::shared_ptr<Handle>(new Handle(this, cfg));
  if (!cfg.lazy) h->TryGet();  // eager build; failures recorded in last_error_
  return h;
}

std::shared_ptr<OrtSessionManager::Handle> OrtSessionManager::MakeSession(
    const std::string& path, ComputeUnits units, Ep ep, size_t gpu_mem_limit,
    bool lazy) {
  SessionConfig cfg;
  cfg.model_path = path;
  cfg.units = units;
  cfg.ep = ep;
  cfg.gpu_mem_limit = gpu_mem_limit;
  cfg.lazy = lazy;
  return MakeSession(cfg);
}

}  // namespace hastur
