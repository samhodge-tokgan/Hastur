// Copyright the Hastur authors.
// SPDX-License-Identifier: LicenseRef-SAM-License
//
// Opt-in CUDA execution-provider memory knobs, read once from the environment,
// plus the arena-statistics helpers used to measure them.
//
// Every knob is OFF by default. With none of these variables set, the provider
// options, session options and run options are exactly what they were before
// this header existed, so outputs are unchanged.
//
//   HASTUR_ORT_ARENA_EXTEND=same|pow2      arena_extend_strategy (kSameAsRequested /
//                                          kNextPowerOfTwo)
//   HASTUR_ORT_CONV_SEARCH=heuristic|default|exhaustive
//                                          cudnn_conv_algo_search
//   HASTUR_ORT_MEM_PATTERN=0               SessionOptions::DisableMemPattern
//   HASTUR_ORT_GPU_MEM_LIMIT_MB=N          gpu_mem_limit for each CUDA arena (or for the
//                                          shared arena when that is on)
//   HASTUR_ORT_SHARED_ARENA=1              the sessions of one OrtSessionManager (the
//                                          tracker's five graphs) draw from one
//                                          env-registered CUDA arena instead of five
//   HASTUR_ORT_SHRINK=g3|all               run option memory.enable_memory_arena_shrinkage
//                                          on the tracker's G3 runs, or on all of them
//   HASTUR_TRACK_MEMLOG=1|2                tracker log: per G3 call the prompt length L
//                                          (n_mem, n_ptr) and the G3 arena state; 2 adds
//                                          all five arenas once per frame
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include <onnxruntime_cxx_api.h>

namespace hastur {

struct CudaTuning {
  int arena_extend = -1;    // -1 unset, 0 kNextPowerOfTwo, 1 kSameAsRequested
  int conv_search = -1;     // -1 unset, else OrtCudnnConvAlgoSearch
  bool no_mem_pattern = false;
  size_t gpu_mem_limit = 0;  // bytes, 0 = unset
  bool shared_arena = false;
  int shrink = 0;           // 0 off, 1 G3 only, 2 every tracker graph
  int memlog = 0;
  bool any() const {
    return arena_extend >= 0 || conv_search >= 0 || no_mem_pattern || gpu_mem_limit > 0 ||
           shared_arena || shrink > 0;
  }
};

inline const CudaTuning& CudaTuningFromEnv() {
  static const CudaTuning t = [] {
    CudaTuning c;
    auto get = [](const char* k) -> const char* {
      const char* v = std::getenv(k);
      return (v && *v) ? v : nullptr;
    };
    if (const char* v = get("HASTUR_ORT_ARENA_EXTEND")) {
      if (!std::strcmp(v, "same")) c.arena_extend = 1;
      else if (!std::strcmp(v, "pow2")) c.arena_extend = 0;
    }
    if (const char* v = get("HASTUR_ORT_CONV_SEARCH")) {
      if (!std::strcmp(v, "exhaustive")) c.conv_search = OrtCudnnConvAlgoSearchExhaustive;
      else if (!std::strcmp(v, "heuristic")) c.conv_search = OrtCudnnConvAlgoSearchHeuristic;
      else if (!std::strcmp(v, "default")) c.conv_search = OrtCudnnConvAlgoSearchDefault;
    }
    if (const char* v = get("HASTUR_ORT_MEM_PATTERN")) c.no_mem_pattern = (v[0] == '0');
    if (const char* v = get("HASTUR_ORT_GPU_MEM_LIMIT_MB"))
      c.gpu_mem_limit = static_cast<size_t>(std::strtoull(v, nullptr, 10)) << 20;
    if (const char* v = get("HASTUR_ORT_SHARED_ARENA")) c.shared_arena = (v[0] == '1');
    if (const char* v = get("HASTUR_ORT_SHRINK")) {
      if (!std::strcmp(v, "g3")) c.shrink = 1;
      else if (!std::strcmp(v, "all")) c.shrink = 2;
    }
    if (const char* v = get("HASTUR_TRACK_MEMLOG")) c.memlog = std::atoi(v);
    if (c.any())
      std::fprintf(stderr,
                   "[hastur] ORT CUDA tuning: arena_extend=%d conv_search=%d mem_pattern=%d "
                   "gpu_mem_limit_mb=%zu shared_arena=%d shrink=%d\n",
                   c.arena_extend, c.conv_search, c.no_mem_pattern ? 0 : 1,
                   c.gpu_mem_limit >> 20, (int)c.shared_arena, c.shrink);
    return c;
  }();
  return t;
}

#ifndef __APPLE__
// Apply the per-session knobs. A no-op when none is set.
inline void ApplyCudaTuning(OrtCUDAProviderOptions& cuda, Ort::SessionOptions& so) {
  const CudaTuning& t = CudaTuningFromEnv();
  if (t.arena_extend >= 0) cuda.arena_extend_strategy = t.arena_extend;
  if (t.conv_search >= 0)
    cuda.cudnn_conv_algo_search = static_cast<OrtCudnnConvAlgoSearch>(t.conv_search);
  if (t.gpu_mem_limit > 0) cuda.gpu_mem_limit = t.gpu_mem_limit;
  if (t.no_mem_pattern) so.DisableMemPattern();
}
#endif

// Run options for one Run() call: the null RunOptions the callers always passed,
// unless arena shrinkage was asked for this graph.
inline Ort::RunOptions TunedRunOptions(bool is_g3, int device) {
  const CudaTuning& t = CudaTuningFromEnv();
  if (!(t.shrink == 2 || (t.shrink == 1 && is_g3))) return Ort::RunOptions{nullptr};
  Ort::RunOptions ro;
  const std::string dev = "gpu:" + std::to_string(device);
  ro.AddConfigEntry("memory.enable_memory_arena_shrinkage", dev.c_str());
  return ro;
}

// One line of CUDA-arena statistics for a session's device allocator, in MiB.
// Empty when there is no CUDA allocator or it keeps no statistics.
inline std::string CudaArenaStats(Ort::Session& s, int device) {
  try {
    Ort::MemoryInfo cuda("Cuda", OrtDeviceAllocator, device, OrtMemTypeDefault);
    Ort::Allocator a(s, cuda);
    auto kv = a.GetStats().GetKeyValuePairs();
    auto mib = [&](const char* k) -> double {
      auto it = kv.find(k);
      return it == kv.end() ? -1.0 : std::strtod(it->second.c_str(), nullptr) / 1048576.0;
    };
    auto num = [&](const char* k) -> long long {
      auto it = kv.find(k);
      return it == kv.end() ? -1 : std::strtoll(it->second.c_str(), nullptr, 10);
    };
    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "inuse=%.1f reserved=%.1f maxinuse=%.1f maxalloc=%.1f reserves=%lld "
                  "extensions=%lld shrinks=%lld",
                  mib("InUse"), mib("TotalAllocated"), mib("MaxInUse"), mib("MaxAllocSize"),
                  num("NumReserves"), num("NumArenaExtensions"), num("NumArenaShrinkages"));
    return buf;
  } catch (const std::exception&) {
    return {};
  }
}

}  // namespace hastur
