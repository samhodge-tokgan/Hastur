// Copyright the Hastur authors.
// SPDX-License-Identifier: LicenseRef-SAM-License
#include "DetectorEngine.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
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

// Reads a detector model's JSON sidecar for a `"resize": "stretch"` field, so a
// square-input model (SAM 3) that needs stretch-to-square + no graph-opt is
// handled AUTOMATICALLY — no host env var. Minimal dependency-free scan (we only
// need one string field). Returns false if the file/field is absent.
bool FileSaysStretch(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  const std::string s((std::istreambuf_iterator<char>(f)),
                      std::istreambuf_iterator<char>());
  const auto k = s.find("\"resize\"");
  if (k == std::string::npos) return false;
  const auto c = s.find(':', k);
  if (c == std::string::npos) return false;
  const auto q1 = s.find('"', c);
  if (q1 == std::string::npos) return false;
  const auto q2 = s.find('"', q1 + 1);
  if (q2 == std::string::npos) return false;
  return s.compare(q1 + 1, q2 - q1 - 1, "stretch") == 0;
}

// Try the model's own <name>.json first, then a <dir>/person_detector.json.
bool SidecarWantsStretch(const std::string& model_path) {
  std::string j = model_path;
  const auto dot = j.rfind(".onnx");
  if (dot != std::string::npos) {
    j.replace(dot, 5, ".json");
    if (FileSaysStretch(j)) return true;
  }
  const auto slash = model_path.find_last_of("/\\");
  const std::string dir =
      slash == std::string::npos ? std::string() : model_path.substr(0, slash + 1);
  return FileSaysStretch(dir + "person_detector.json");
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
  int boxes_idx = -1, labels_idx = -1, scores_idx = -1, masks_idx = -1;
  int mask_h = 0, mask_w = 0;  // native instance-mask resolution (e.g. 288x288)
  bool outputs_resolved = false;
  bool accel_active = false;
  bool stretch = false;  // stretch-to-square + no graph-opt (SAM 3-style models)

  void ResolveOutputs(const std::vector<Ort::Value>& r) {
    for (size_t i = 0; i < r.size(); ++i) {
      auto info = r[i].GetTensorTypeAndShapeInfo();
      ONNXTensorElementDataType dt = info.GetElementType();
      auto shape = info.GetShape();
      size_t rank = shape.size();
      if (dt == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
        labels_idx = static_cast<int>(i);        // labels [N]
      } else if (rank >= 3) {
        masks_idx = static_cast<int>(i);          // masks [N,mh,mw] or [N,1,mh,mw]
        mask_h = static_cast<int>(shape[rank - 2]);
        mask_w = static_cast<int>(shape[rank - 1]);
      } else if (rank == 2) {
        boxes_idx = static_cast<int>(i);          // boxes [N,4]
      } else {
        scores_idx = static_cast<int>(i);         // scores [N]
      }
    }
    outputs_resolved = true;
  }
};

DetectorEngine::DetectorEngine(const std::string& model_path, ComputeUnits units,
                               int intra_threads)
    : impl_(std::make_unique<Impl>()) {
  // Stretch/no-opt mode for square-input detectors (SAM 3). Auto-detected from
  // the model's JSON sidecar ("resize":"stretch") so hosts need NO env var;
  // HASTUR_DET_STRETCH remains an override for scripted/debug use. Determined
  // BEFORE session creation because it also drives the optimization level.
  // SAM 3 bakes normalization (image*2-1) into the graph; ORT_ENABLE_ALL fuses it
  // into the patch-embed conv and drifts scores/boxes, so stretch models disable
  // graph optimization for bit-exact outputs (detector speed is not a concern).
  impl_->stretch = std::getenv("HASTUR_DET_STRETCH") != nullptr ||
                   SidecarWantsStretch(model_path);
  const GraphOptimizationLevel kOptLevel =
      impl_->stretch ? GraphOptimizationLevel::ORT_DISABLE_ALL
                     : GraphOptimizationLevel::ORT_ENABLE_ALL;
  Ort::SessionOptions so;
  so.SetGraphOptimizationLevel(kOptLevel);
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
      cpu_so.SetGraphOptimizationLevel(kOptLevel);
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

bool DetectorEngine::has_masks() const {
  return impl_ && impl_->masks_idx >= 0 && impl_->mask_h > 0 && impl_->mask_w > 0;
}

Detections DetectorEngine::Run(const float* rgb, int W, int H, float score_thresh,
                               std::vector<DetMask>* out_masks) {
  Detections out;
  if (out_masks) out_masks->clear();
  if (!impl_ || !impl_->session || W <= 0 || H <= 0) return out;

  const int pw = impl_->procW, ph = impl_->procH;

  // 1. Resize the full frame into pw×ph. Two geometries:
  //    - letterbox (default): aspect-preserving, centered, zero-padded. Matches
  //      tools/bench_detector.py (torchvision FRCNN/SSDLite exports).
  //    - stretch (HASTUR_DET_STRETCH): independent x/y scale, no padding. SAM 3's
  //      processor resizes to a full square with aspect distortion, so its ONNX
  //      must be fed the same way or distant-person recall collapses.
  const bool stretch = impl_->stretch;
  float scale_x, scale_y;
  int nw, nh, pad_x, pad_y;
  if (stretch) {
    nw = pw; nh = ph; pad_x = 0; pad_y = 0;
    scale_x = static_cast<float>(pw) / W;
    scale_y = static_cast<float>(ph) / H;
  } else {
    const float scale = std::min(static_cast<float>(pw) / W,
                                 static_cast<float>(ph) / H);
    scale_x = scale_y = scale;
    nw = static_cast<int>(std::lround(W * scale));
    nh = static_cast<int>(std::lround(H * scale));
    pad_x = (pw - nw) / 2;
    pad_y = (ph - nh) / 2;
  }

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

  // Collect (box, source-index) so masks — if requested — can be gathered in the
  // SAME order after the score sort (mask i is emitted in lockstep with box i).
  struct Cand { BBox box; size_t src; };
  std::vector<Cand> cands;
  cands.reserve(n);
  const float inv_x = 1.f / scale_x, inv_y = 1.f / scale_y;
  for (size_t i = 0; i < n; ++i) {
    if (static_cast<int>(labels[i]) != kPersonClass) continue;
    if (scores[i] < score_thresh) continue;
    const float* b = boxes + i * 4;
    BBox box;
    box.x0 = std::clamp((b[0] - pad_x) * inv_x, 0.f, static_cast<float>(W));
    box.y0 = std::clamp((b[1] - pad_y) * inv_y, 0.f, static_cast<float>(H));
    box.x1 = std::clamp((b[2] - pad_x) * inv_x, 0.f, static_cast<float>(W));
    box.y1 = std::clamp((b[3] - pad_y) * inv_y, 0.f, static_cast<float>(H));
    box.score = scores[i];
    if (box.x1 > box.x0 && box.y1 > box.y0) cands.push_back({box, i});
  }

  // Sort by descending score for a stable, deterministic output order.
  std::sort(cands.begin(), cands.end(),
            [](const Cand& a, const Cand& c) { return a.box.score > c.box.score; });

  out.reserve(cands.size());
  for (const Cand& c : cands) out.push_back(c.box);

  // Gather instance masks (native resolution) aligned with the returned boxes.
  // The mask covers the same processed frame as the boxes; the caller resamples
  // native->W*H. Left empty per-detection if the model emits no masks.
  if (out_masks && impl_->masks_idx >= 0 && impl_->mask_h > 0 &&
      impl_->mask_w > 0) {
    const Ort::Value& vmasks = results[impl_->masks_idx];
    const float* masks = vmasks.GetTensorData<float>();
    const size_t mh = static_cast<size_t>(impl_->mask_h);
    const size_t mw = static_cast<size_t>(impl_->mask_w);
    const size_t stride = mh * mw;
    const size_t mcount = vmasks.GetTensorTypeAndShapeInfo().GetElementCount();
    out_masks->reserve(cands.size());
    for (const Cand& c : cands) {
      DetMask dm;
      if (stride > 0 && (c.src + 1) * stride <= mcount) {
        dm.w = static_cast<int>(mw);
        dm.h = static_cast<int>(mh);
        dm.data.assign(masks + c.src * stride, masks + (c.src + 1) * stride);
      }
      out_masks->push_back(std::move(dm));
    }
  }
  return out;
}

}  // namespace hastur
