// Copyright the Hastur authors.
// SPDX-License-Identifier: LicenseRef-SAM-License
#include "Sam3TrackerEngine.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>

#include "OrtAccel.h"

namespace hastur {

// ---------------------------------------------------------------- npy loader
Sam3TrackerEngine::Tensor Sam3TrackerEngine::LoadNpyFloat(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("npy: cannot open " + path);
  char magic[6]; f.read(magic, 6);
  if (std::memcmp(magic, "\x93NUMPY", 6) != 0) throw std::runtime_error("npy: bad magic " + path);
  unsigned char ver[2]; f.read(reinterpret_cast<char*>(ver), 2);
  uint32_t hlen;
  if (ver[0] == 1) { uint16_t h; f.read(reinterpret_cast<char*>(&h), 2); hlen = h; }
  else { f.read(reinterpret_cast<char*>(&hlen), 4); }
  std::string hdr(hlen, '\0'); f.read(&hdr[0], hlen);
  if (hdr.find("<f4") == std::string::npos)
    throw std::runtime_error("npy: expected <f4 in " + path + " hdr=" + hdr);
  if (hdr.find("'fortran_order': False") == std::string::npos && hdr.find("False") == std::string::npos)
    throw std::runtime_error("npy: expected C-order in " + path);
  // parse shape tuple
  Tensor t;
  auto sp = hdr.find("'shape':");
  auto lp = hdr.find('(', sp), rp = hdr.find(')', sp);
  std::string dims = hdr.substr(lp + 1, rp - lp - 1);
  size_t pos = 0;
  while (pos < dims.size()) {
    while (pos < dims.size() && (dims[pos] == ' ' || dims[pos] == ',')) ++pos;
    if (pos >= dims.size()) break;
    size_t e = pos; while (e < dims.size() && std::isdigit(dims[e])) ++e;
    if (e > pos) t.shape.push_back(std::stoll(dims.substr(pos, e - pos)));
    pos = e + 1;
  }
  size_t n = t.elems();
  t.data.resize(n);
  f.read(reinterpret_cast<char*>(t.data.data()), n * sizeof(float));
  if (!f) throw std::runtime_error("npy: short read " + path);
  return t;
}

// ---------------------------------------------------------------- lifecycle
Sam3TrackerEngine::Sam3TrackerEngine(const Sam3Paths& p, Ep ep, ComputeUnits units)
    : mgr_("Sam3Tracker"),
      mem_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
  InitGraph(g1_, mgr_.MakeSession(p.g1, units, ep));
  InitGraph(g3_, mgr_.MakeSession(p.g3, units, ep));
  InitGraph(g4_, mgr_.MakeSession(p.g4, units, ep));
  InitGraph(g5_, mgr_.MakeSession(p.g5, units, ep));
  accel_active_ = g1_.h->accelerator_active();
  tpos_ = LoadNpyFloat(p.tpos);       // [7,1,1,64]
  no_mem_ = LoadNpyFloat(p.no_mem);   // [1,1,256]
  proj_w_ = LoadNpyFloat(p.proj_w);   // [64,256]
  proj_b_ = LoadNpyFloat(p.proj_b);   // [64]
}

Sam3TrackerEngine::~Sam3TrackerEngine() = default;

void Sam3TrackerEngine::InitGraph(OrtGraph& g,
                                  std::shared_ptr<OrtSessionManager::Handle> h) {
  g.h = std::move(h);
  Ort::Session& s = g.h->Get();
  Ort::AllocatorWithDefaultOptions alloc;
  for (size_t i = 0; i < s.GetInputCount(); ++i)
    g.in_names.emplace_back(s.GetInputNameAllocated(i, alloc).get());
  for (size_t i = 0; i < s.GetOutputCount(); ++i)
    g.out_names.emplace_back(s.GetOutputNameAllocated(i, alloc).get());
}

std::vector<Sam3TrackerEngine::Tensor> Sam3TrackerEngine::RunFloat(
    OrtGraph& g, const std::vector<std::pair<std::string, const Tensor*>>& feeds) {
  Ort::Session& s = g.h->Get();
  std::vector<Ort::Value> ins;
  std::vector<const char*> in_names;
  for (const auto& name : g.in_names) {
    const Tensor* t = nullptr;
    for (const auto& fd : feeds) if (fd.first == name) { t = fd.second; break; }
    if (!t) throw std::runtime_error("Sam3: missing feed '" + name + "'");
    in_names.push_back(name.c_str());
    ins.push_back(Ort::Value::CreateTensor<float>(
        mem_info_, const_cast<float*>(t->data.data()), t->data.size(),
        t->shape.data(), t->shape.size()));
  }
  std::vector<const char*> out_names;
  for (const auto& n : g.out_names) out_names.push_back(n.c_str());
  auto res = s.Run(Ort::RunOptions{nullptr}, in_names.data(), ins.data(), ins.size(),
                   out_names.data(), out_names.size());
  std::vector<Tensor> out;
  for (auto& v : res) {
    Tensor t;
    t.shape = v.GetTensorTypeAndShapeInfo().GetShape();
    const float* pd = v.GetTensorData<float>();
    t.data.assign(pd, pd + t.elems());
    out.push_back(std::move(t));
  }
  return out;
}

// ---------------------------------------------------------------- graph calls
void Sam3TrackerEngine::G1(const Tensor& image, Tensor& f0, Tensor& f1, Tensor& f2,
                          Tensor& pos2) {
  auto o = RunFloat(g1_, {{"image", &image}});
  // outputs: fpn0,fpn1,fpn2,pos0,pos1,pos2,det_fpn0,det_fpn1,det_fpn2,det_pos72
  f0 = std::move(o[0]); f1 = std::move(o[1]); f2 = std::move(o[2]); pos2 = std::move(o[5]);
}

Sam3TrackerEngine::Tensor Sam3TrackerEngine::G3(const Tensor& src, const Tensor& src_pos,
                                               const Tensor& prompt, const Tensor& prompt_pos,
                                               int64_t n_optr) {
  Ort::Session& s = g3_.h->Get();
  std::vector<Ort::Value> ins;
  std::vector<const char*> names;
  auto add = [&](const char* nm, const Tensor& t) {
    names.push_back(nm);
    ins.push_back(Ort::Value::CreateTensor<float>(
        mem_info_, const_cast<float*>(t.data.data()), t.data.size(),
        t.shape.data(), t.shape.size()));
  };
  // feed in declared order
  int64_t scalar_shape[1] = {0};
  std::vector<int64_t> nopt_val = {n_optr};
  for (const auto& nm : g3_.in_names) {
    if (nm == "src") add("src", src);
    else if (nm == "src_pos") add("src_pos", src_pos);
    else if (nm == "prompt") add("prompt", prompt);
    else if (nm == "prompt_pos") add("prompt_pos", prompt_pos);
    else if (nm == "num_obj_ptr_tokens") {
      names.push_back("num_obj_ptr_tokens");
      ins.push_back(Ort::Value::CreateTensor<int64_t>(
          mem_info_, nopt_val.data(), 1, scalar_shape, 0));  // 0-d scalar
    } else throw std::runtime_error("G3 unexpected input " + nm);
  }
  const char* out_names[] = {g3_.out_names[0].c_str()};
  auto res = s.Run(Ort::RunOptions{nullptr}, names.data(), ins.data(), ins.size(),
                   out_names, 1);
  Tensor t;
  t.shape = res[0].GetTensorTypeAndShapeInfo().GetShape();
  const float* pd = res[0].GetTensorData<float>();
  t.data.assign(pd, pd + t.elems());
  return t;
}

void Sam3TrackerEngine::G4(const Tensor& bf, const Tensor& hr0, const Tensor& hr1,
                          Tensor& low, Tensor& high, Tensor& ious, Tensor& obj_ptr,
                          Tensor& osl) {
  auto o = RunFloat(g4_, {{"backbone_features", &bf}, {"high_res_0", &hr0}, {"high_res_1", &hr1}});
  low = std::move(o[0]); high = std::move(o[1]); ious = std::move(o[2]);
  obj_ptr = std::move(o[3]); osl = std::move(o[4]);
}

void Sam3TrackerEngine::G5(const Tensor& pix, const Tensor& high, const Tensor& osl,
                          Tensor& mem, Tensor& mempos) {
  auto o = RunFloat(g5_, {{"pix_feat", &pix}, {"pred_masks_high_res", &high},
                          {"object_score_logits", &osl}});
  mem = std::move(o[0]); mempos = std::move(o[1]);
}

// ---------------------------------------------------------------- host helpers
Sam3TrackerEngine::Tensor Sam3TrackerEngine::FlattenHWBC(const Tensor& bchw) {
  // [1,C,H,W] -> [H*W,1,C]
  const int C = static_cast<int>(bchw.shape[1]);
  const int HW = static_cast<int>(bchw.shape[2] * bchw.shape[3]);
  Tensor out; out.shape = {HW, 1, C}; out.data.resize(static_cast<size_t>(HW) * C);
  for (int c = 0; c < C; ++c)
    for (int p = 0; p < HW; ++p)
      out.data[static_cast<size_t>(p) * C + c] = bchw.data[static_cast<size_t>(c) * HW + p];
  return out;
}

Sam3TrackerEngine::Tensor Sam3TrackerEngine::SeqToBCHW(const Tensor& seq) {
  // [HW,1,C] -> [1,C,72,72]
  const int HW = static_cast<int>(seq.shape[0]);
  const int C = static_cast<int>(seq.shape[2]);
  Tensor out; out.shape = {1, C, kGrid, kGrid}; out.data.resize(static_cast<size_t>(C) * HW);
  for (int p = 0; p < HW; ++p)
    for (int c = 0; c < C; ++c)
      out.data[static_cast<size_t>(c) * HW + p] = seq.data[static_cast<size_t>(p) * C + c];
  return out;
}

void Sam3TrackerEngine::Sine1DPE(const std::vector<float>& pos, int dim,
                                std::vector<float>& out) {
  // matches get_1d_sine_pe: out[N,dim] = [sin(pos/dim_t), cos(pos/dim_t)]
  const int pe = dim / 2, N = static_cast<int>(pos.size());
  out.assign(static_cast<size_t>(N) * dim, 0.f);
  std::vector<float> dim_t(pe);
  for (int i = 0; i < pe; ++i)
    dim_t[i] = std::pow(10000.f, (2.f * (i / 2)) / pe);
  for (int n = 0; n < N; ++n)
    for (int i = 0; i < pe; ++i) {
      float v = pos[n] / dim_t[i];
      out[static_cast<size_t>(n) * dim + i] = std::sin(v);
      out[static_cast<size_t>(n) * dim + pe + i] = std::cos(v);
    }
}

std::pair<float, float> Sam3TrackerEngine::EffIou(float osl, const float* ious3) const {
  float obj_norm = 0.f;
  if (osl > 0.f) obj_norm = (1.f / (1.f + std::exp(-osl))) * 2.f - 1.f;
  float iou_score = std::max(std::max(ious3[0], ious3[1]), ious3[2]);
  return {obj_norm * iou_score, iou_score};  // single object -> mean == value
}

std::vector<int> Sam3TrackerEngine::FrameFilter(int frame_idx) const {
  const int max_num = std::min(total_frames_, kMaxObjPtrs);
  std::vector<int> valid;
  for (int i = frame_idx - 1; i > 0; --i) {
    const Frame& r = frames_[i];
    if (r.cond) continue;
    if (r.eff > kMfThreshold) valid.insert(valid.begin(), i);
    if (static_cast<int>(valid.size()) >= max_num - 1) break;
  }
  const int must = frame_idx - 1;
  if (std::find(valid.begin(), valid.end(), must) == valid.end()) valid.push_back(must);
  return valid;
}

// ---------------------------------------------------------------- seed / propagate
std::vector<float> Sam3TrackerEngine::Seed(const float* frame_chw1008,
                                          const float* seed_mask1008, int total_frames) {
  frames_.clear();
  total_frames_ = total_frames;
  Tensor image; image.shape = {1, 3, kImgSize, kImgSize};
  image.data.assign(frame_chw1008, frame_chw1008 + static_cast<size_t>(3) * kImgSize * kImgSize);

  Tensor f0, f1, f2, pos2;
  G1(image, f0, f1, f2, pos2);
  Tensor src = FlattenHWBC(f2);                 // [5184,1,256]
  // cond frame: memory-less features = pix_feat + no_mem_embed
  Tensor bf_seq = src;
  for (int p = 0; p < kHW; ++p)
    for (int c = 0; c < kHidden; ++c)
      bf_seq.data[static_cast<size_t>(p) * kHidden + c] += no_mem_.data[c];
  Tensor bf = SeqToBCHW(bf_seq);
  Tensor low, high4, ious, obj_ptr, osl4;
  G4(bf, f0, f1, low, high4, ious, obj_ptr, osl4);   // obj_ptr approximates mask-prompt decoder

  // seed output + memory come from the DETECTION mask
  Tensor high; high.shape = {1, 1, kImgSize, kImgSize};
  high.data.resize(static_cast<size_t>(kImgSize) * kImgSize);
  for (size_t i = 0; i < high.data.size(); ++i) high.data[i] = seed_mask1008[i] * 20.f - 10.f;
  Tensor osl; osl.shape = {1, 1}; osl.data = {10.f};   // object present
  Tensor mem, mempos;
  G5(f2, high, osl, mem, mempos);

  Frame fr; fr.cond = true; fr.mem = std::move(mem); fr.mempos = std::move(mempos);
  fr.obj_ptr = std::move(obj_ptr); fr.osl = 10.f;
  float ones3[3] = {1.f, 1.f, 1.f};
  auto ei = EffIou(10.f, ones3); fr.eff = ei.first; fr.iou = ei.second;
  frames_.push_back(std::move(fr));
  return high.data;
}

std::vector<float> Sam3TrackerEngine::Propagate(const float* frame_chw1008) {
  const int frame_idx = static_cast<int>(frames_.size());
  Tensor image; image.shape = {1, 3, kImgSize, kImgSize};
  image.data.assign(frame_chw1008, frame_chw1008 + static_cast<size_t>(3) * kImgSize * kImgSize);

  Tensor f0, f1, f2, pos2;
  G1(image, f0, f1, f2, pos2);
  Tensor src = FlattenHWBC(f2);
  Tensor src_pos = FlattenHWBC(pos2);

  std::vector<int> valid = FrameFilter(frame_idx);

  // ---- assemble prompt (maskmem tokens then obj-ptr tokens) ----
  std::vector<float> prompt, prompt_pos;   // concatenated [tokens,1,64]
  auto append_maskmem = [&](const Frame& r, int tpos_idx) {
    Tensor mtok = FlattenHWBC(r.mem);      // [5184,1,64]
    Tensor ptok = FlattenHWBC(r.mempos);   // [5184,1,64]
    const float* tp = &tpos_.data[static_cast<size_t>(tpos_idx) * kMemDim];  // [64]
    for (int p = 0; p < kHW; ++p) {
      for (int c = 0; c < kMemDim; ++c) {
        prompt.push_back(mtok.data[static_cast<size_t>(p) * kMemDim + c]);
        prompt_pos.push_back(ptok.data[static_cast<size_t>(p) * kMemDim + c] + tp[c]);
      }
    }
  };
  // cond frames (t_pos=0 -> tpos index num_maskmem-1)
  for (int ci = 0; ci < frame_idx; ++ci)
    if (frames_[ci].cond) append_maskmem(frames_[ci], kNumMaskMem - 0 - 1);
  // non-cond recent frames, t_pos = 1..num_maskmem-1
  for (int t_pos = 1; t_pos < kNumMaskMem; ++t_pos) {
    int t_rel = kNumMaskMem - t_pos;
    if (t_rel > static_cast<int>(valid.size())) continue;
    int prev = valid[valid.size() - t_rel];
    append_maskmem(frames_[prev], kNumMaskMem - t_pos - 1);
  }

  // ---- object-pointer tokens ----
  std::vector<float> pos_list;
  std::vector<const Tensor*> ptrs;
  for (int ci = 0; ci < frame_idx; ++ci)
    if (frames_[ci].cond) { pos_list.push_back(static_cast<float>(frame_idx - ci)); ptrs.push_back(&frames_[ci].obj_ptr); }
  for (int t_diff = 1; t_diff < kMaxObjPtrs; ++t_diff) {
    if (t_diff >= static_cast<int>(valid.size())) break;
    int t = valid[valid.size() - t_diff];
    pos_list.push_back(static_cast<float>(t_diff)); ptrs.push_back(&frames_[t].obj_ptr);
  }
  int64_t n_optr = 0;
  if (!ptrs.empty()) {
    const int P = static_cast<int>(ptrs.size());
    const float denom = static_cast<float>(std::min(total_frames_, kMaxObjPtrs) - 1);
    std::vector<float> posn(P);
    for (int i = 0; i < P; ++i) posn[i] = pos_list[i] / denom;
    std::vector<float> sine;   // [P,256]
    Sine1DPE(posn, kHidden, sine);
    // obj_pos = sine @ proj_w^T + proj_b   -> [P,64]
    std::vector<float> obj_pos(static_cast<size_t>(P) * kMemDim);
    for (int i = 0; i < P; ++i)
      for (int o = 0; o < kMemDim; ++o) {
        float acc = proj_b_.data[o];
        const float* w = &proj_w_.data[static_cast<size_t>(o) * kHidden];
        const float* s = &sine[static_cast<size_t>(i) * kHidden];
        for (int k = 0; k < kHidden; ++k) acc += s[k] * w[k];
        obj_pos[static_cast<size_t>(i) * kMemDim + o] = acc;
      }
    // split each [256] ptr into kPtrTokens tokens of 64; obj_pos repeat_interleave(4)
    for (int i = 0; i < P; ++i)
      for (int tk = 0; tk < kPtrTokens; ++tk) {
        const float* pv = &ptrs[i]->data[static_cast<size_t>(tk) * kMemDim];
        for (int c = 0; c < kMemDim; ++c) prompt.push_back(pv[c]);
        const float* op = &obj_pos[static_cast<size_t>(i) * kMemDim];
        for (int c = 0; c < kMemDim; ++c) prompt_pos.push_back(op[c]);
      }
    n_optr = static_cast<int64_t>(P) * kPtrTokens;
  }

  const int64_t L = static_cast<int64_t>(prompt.size()) / kMemDim;
  Tensor promptT; promptT.shape = {L, 1, kMemDim}; promptT.data = std::move(prompt);
  Tensor promptPosT; promptPosT.shape = {L, 1, kMemDim}; promptPosT.data = std::move(prompt_pos);

  Tensor memory = G3(src, src_pos, promptT, promptPosT, n_optr);   // [5184,1,256]
  Tensor bf = SeqToBCHW(memory);
  Tensor low, high, ious, obj_ptr, osl;
  G4(bf, f0, f1, low, high, ious, obj_ptr, osl);
  Tensor mem, mempos;
  G5(f2, high, osl, mem, mempos);

  Frame fr; fr.cond = false; fr.mem = std::move(mem); fr.mempos = std::move(mempos);
  fr.obj_ptr = std::move(obj_ptr); fr.osl = osl.data[0];
  auto ei = EffIou(osl.data[0], ious.data.data()); fr.eff = ei.first; fr.iou = ei.second;
  frames_.push_back(std::move(fr));
  return low.data;   // [1,1,288,288] flattened == 288*288
}

}  // namespace hastur
