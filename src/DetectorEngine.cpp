// Copyright the Hastur authors.
// SPDX-License-Identifier: Apache-2.0
#include "DetectorEngine.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <unordered_map>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include "OrtAccel.h"

namespace hastur {

namespace {

// COCO 1-based class id for "person" (index 0 == background is dropped by the
// torchvision post-processing). See tools/person_detector.json.
constexpr int kPersonClass = 1;

const char* ComputeUnitsString(ComputeUnits u) {
  switch (u) {
    case ComputeUnits::All: return "ALL";
    case ComputeUnits::CpuAndGpu: return "CPUAndGPU";
    case ComputeUnits::CpuAndAne: return "CPUAndNeuralEngine";
    case ComputeUnits::CpuOnly: return "CPUOnly";
  }
  return "ALL";
}

// Bilinear resample of an interleaved image with `c` channels from (sw x sh) to
// (dw x dh). src/dst are row-major, channel-interleaved. (Same helper as the
// humbaba DepthEngine.)
void ResampleBilinear(const float* src, int sw, int sh, float* dst, int dw,
                      int dh, int c) {
  const float sx = sw > 1 && dw > 1 ? static_cast<float>(sw - 1) / (dw - 1) : 0.f;
  const float sy = sh > 1 && dh > 1 ? static_cast<float>(sh - 1) / (dh - 1) : 0.f;
  for (int y = 0; y < dh; ++y) {
    float fy = y * sy;
    int y0 = static_cast<int>(fy);
    int y1 = std::min(y0 + 1, sh - 1);
    float wy = fy - y0;
    for (int x = 0; x < dw; ++x) {
      float fx = x * sx;
      int x0 = static_cast<int>(fx);
      int x1 = std::min(x0 + 1, sw - 1);
      float wx = fx - x0;
      const float* p00 = src + (static_cast<size_t>(y0) * sw + x0) * c;
      const float* p01 = src + (static_cast<size_t>(y0) * sw + x1) * c;
      const float* p10 = src + (static_cast<size_t>(y1) * sw + x0) * c;
      const float* p11 = src + (static_cast<size_t>(y1) * sw + x1) * c;
      float* d = dst + (static_cast<size_t>(y) * dw + x) * c;
      for (int k = 0; k < c; ++k) {
        float top = p00[k] * (1 - wx) + p01[k] * wx;
        float bot = p10[k] * (1 - wx) + p11[k] * wx;
        d[k] = top * (1 - wy) + bot * wy;
      }
    }
  }
}

}  // namespace

struct DetectorEngine::Impl {
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "PersonDetector"};
  std::unique_ptr<Ort::Session> session;
  Ort::AllocatorWithDefaultOptions alloc;
  std::string input_name;
  std::vector<std::string> output_names;
  int procW = 320;
  int procH = 320;
  // Output index mapping, resolved from the concrete result tensors on first
  // Run(): labels == int64 ; scores == float 1-D ; boxes == float N×4.
  // (torchvision SSD emits [boxes,scores,labels]; R-CNN/RetinaNet emit
  // [boxes,labels,scores] — so we must NOT rely on positional/name order, and
  // the graph's declared output types can be underspecified.)
  int boxes_idx = -1, labels_idx = -1, scores_idx = -1;
  bool outputs_resolved = false;
  bool accel_active = false;

  void ResolveOutputs(const std::vector<Ort::Value>& r) {
    for (size_t i = 0; i < r.size(); ++i) {
      auto info = r[i].GetTensorTypeAndShapeInfo();
      ONNXTensorElementDataType dt = info.GetElementType();
      size_t rank = info.GetShape().size();
      if (dt == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
        labels_idx = static_cast<int>(i);
      } else if (rank >= 2) {
        boxes_idx = static_cast<int>(i);
      } else {
        scores_idx = static_cast<int>(i);
      }
    }
    outputs_resolved = true;
  }
};

DetectorEngine::DetectorEngine(const std::string& model_path, ComputeUnits units,
                               int intra_threads)
    : impl_(std::make_unique<Impl>()) {
  Ort::SessionOptions so;
  so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  if (intra_threads > 0) so.SetIntraOpNumThreads(intra_threads);

  bool used_accel = false;
  // HASTUR_DET_CPU=1 forces a plain CPU session (debug/repro escape hatch).
  if (AcceleratorAvailable() && !std::getenv("HASTUR_DET_CPU")) {
    try {
      // CoreML on macOS (MLProgram, static shapes so no per-frame recompiles);
      // CUDA on Linux/Windows. See src/OrtAccel.h.
      AppendAccelerator(so, ComputeUnitsString(units), /*coreml_static=*/true,
                        /*coreml_mlprogram=*/true);
      used_accel = true;
    } catch (const Ort::Exception& e) {
      last_error_ = std::string(AcceleratorSubstr()) + " EP append failed: " + e.what();
      used_accel = false;
    }
  }

  try {
    impl_->session =
        std::make_unique<Ort::Session>(impl_->env, OrtPath(model_path).c_str(), so);
  } catch (const Ort::Exception& e) {
    // The accelerator EP can fail at session-creation time (e.g. missing CUDA/
    // cuDNN on Linux). Retry once on a plain CPU session so the plugin still
    // works (slower) instead of failing outright.
    if (used_accel) {
      last_error_ = std::string(AcceleratorSubstr()) +
                    " session create failed (falling back to CPU): " + e.what();
      Ort::SessionOptions cpu_so;
      cpu_so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
      if (intra_threads > 0) cpu_so.SetIntraOpNumThreads(intra_threads);
      used_accel = false;
      try {
        impl_->session = std::make_unique<Ort::Session>(
            impl_->env, OrtPath(model_path).c_str(), cpu_so);
      } catch (const Ort::Exception& e2) {
        last_error_ = std::string("session create failed: ") + e2.what();
        impl_->session.reset();
        return;
      }
    } else {
      last_error_ = std::string("session create failed: ") + e.what();
      impl_->session.reset();
      return;
    }
  }
  impl_->accel_active = used_accel;

  // Input name + fixed processing size (last two dims of the input shape). The
  // exported graph input is rank-3 [3,S,S] (torchvision exports a single image
  // tensor with no batch dim); handle rank-4 [1,3,S,S] too.
  Ort::AllocatedStringPtr in = impl_->session->GetInputNameAllocated(0, impl_->alloc);
  impl_->input_name = in.get();
  auto ishape = impl_->session->GetInputTypeInfo(0)
                    .GetTensorTypeAndShapeInfo()
                    .GetShape();
  if (ishape.size() >= 2) {
    int64_t h = ishape[ishape.size() - 2];
    int64_t w = ishape[ishape.size() - 1];
    if (h > 0) impl_->procH = static_cast<int>(h);
    if (w > 0) impl_->procW = static_cast<int>(w);
  }

  // Output roles (boxes/labels/scores) are resolved lazily from the ACTUAL result
  // tensors on the first Run() — the graph's declared output type info can be
  // underspecified (dynamic NMS subgraph), so we classify by the concrete tensor
  // dtype/shape instead. See ResolveOutputs().
  size_t nout = impl_->session->GetOutputCount();
  for (size_t i = 0; i < nout; ++i) {
    Ort::AllocatedStringPtr on = impl_->session->GetOutputNameAllocated(i, impl_->alloc);
    impl_->output_names.emplace_back(on.get());
  }
}

DetectorEngine::~DetectorEngine() = default;

bool DetectorEngine::accelerator_active() const {
  return impl_ && impl_->accel_active;
}

bool DetectorEngine::AcceleratorAvailable() {
  return hastur::AcceleratorAvailable();
}

Detections DetectorEngine::Run(const float* rgb, int W, int H, float score_thresh) {
  Detections out;
  if (!impl_ || !impl_->session || W <= 0 || H <= 0) return out;

  const int pw = impl_->procW, ph = impl_->procH;

  // 1. Aspect-preserving letterbox of the full frame into pw×ph, centered, with
  //    zero padding. Identical geometry to tools/bench_detector.py so the ONNX
  //    boxes map back the same way.
  const float scale = std::min(static_cast<float>(pw) / W,
                               static_cast<float>(ph) / H);
  const int nw = static_cast<int>(std::lround(W * scale));
  const int nh = static_cast<int>(std::lround(H * scale));
  const int pad_x = (pw - nw) / 2;
  const int pad_y = (ph - nh) / 2;

  std::vector<float> resized(static_cast<size_t>(nw) * nh * 3);
  ResampleBilinear(rgb, W, H, resized.data(), nw, nh, 3);

  // Pack into NCHW with zero padding. Channel order is RGB (matches the export;
  // ImageNet normalization is baked into the graph, so feed values in [0,1]).
  const size_t plane = static_cast<size_t>(pw) * ph;
  std::vector<float> nchw(plane * 3, 0.f);
  for (int y = 0; y < nh; ++y) {
    for (int x = 0; x < nw; ++x) {
      const float* src = resized.data() + (static_cast<size_t>(y) * nw + x) * 3;
      size_t idx = static_cast<size_t>(y + pad_y) * pw + (x + pad_x);
      nchw[0 * plane + idx] = src[0];
      nchw[1 * plane + idx] = src[1];
      nchw[2 * plane + idx] = src[2];
    }
  }

  // 2. Run. Exported input is rank-3 [3,ph,pw] (no batch dim).
  std::array<int64_t, 3> shape{3, ph, pw};
  Ort::MemoryInfo mem =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value input = Ort::Value::CreateTensor<float>(
      mem, nchw.data(), nchw.size(), shape.data(), shape.size());

  const char* in_names[] = {impl_->input_name.c_str()};
  std::vector<const char*> out_names;
  for (auto& n : impl_->output_names) out_names.push_back(n.c_str());

  std::vector<Ort::Value> results;
  try {
    results = impl_->session->Run(Ort::RunOptions{nullptr}, in_names, &input, 1,
                                  out_names.data(), out_names.size());
  } catch (const Ort::Exception& e) {
    last_error_ = std::string("inference failed: ") + e.what();
    return out;
  }
  if (results.size() < 3) return out;
  if (!impl_->outputs_resolved) impl_->ResolveOutputs(results);
  if (impl_->boxes_idx < 0 || impl_->labels_idx < 0 || impl_->scores_idx < 0)
    return out;

  // 3. Read boxes/labels/scores (NMS already applied in-graph). Boxes are xyxy
  //    in pw×ph letterbox pixels; map back to full-frame pixels and clamp.
  const Ort::Value& vboxes = results[impl_->boxes_idx];
  const Ort::Value& vlabels = results[impl_->labels_idx];
  const Ort::Value& vscores = results[impl_->scores_idx];

  const size_t n =
      vscores.GetTensorTypeAndShapeInfo().GetElementCount();
  const float* boxes = vboxes.GetTensorData<float>();
  const int64_t* labels = vlabels.GetTensorData<int64_t>();
  const float* scores = vscores.GetTensorData<float>();

  const float inv_scale = 1.f / scale;
  for (size_t i = 0; i < n; ++i) {
    if (static_cast<int>(labels[i]) != kPersonClass) continue;
    if (scores[i] < score_thresh) continue;
    const float* b = boxes + i * 4;
    BBox box;
    box.x0 = std::clamp((b[0] - pad_x) * inv_scale, 0.f, static_cast<float>(W));
    box.y0 = std::clamp((b[1] - pad_y) * inv_scale, 0.f, static_cast<float>(H));
    box.x1 = std::clamp((b[2] - pad_x) * inv_scale, 0.f, static_cast<float>(W));
    box.y1 = std::clamp((b[3] - pad_y) * inv_scale, 0.f, static_cast<float>(H));
    box.score = scores[i];
    if (box.x1 > box.x0 && box.y1 > box.y0) out.push_back(box);
  }

  // Sort by descending score for a stable, deterministic output order.
  std::sort(out.begin(), out.end(),
            [](const BBox& a, const BBox& c) { return a.score > c.score; });
  return out;
}

}  // namespace hastur
