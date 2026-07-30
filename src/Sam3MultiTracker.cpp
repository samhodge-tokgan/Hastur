// Copyright the Hastur authors.
// SPDX-License-Identifier: LicenseRef-SAM-License
#include "Sam3MultiTracker.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>

#include "ConfidenceOpen.h"
#include "OrtAccel.h"

namespace hastur {

// ---------------------------------------------------------------- npy loader
Sam3MultiTracker::Tensor Sam3MultiTracker::LoadNpyFloat(const std::string& path) {
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
    throw std::runtime_error("npy: expected <f4 in " + path);
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
  size_t n = t.elems(); t.data.resize(n);
  f.read(reinterpret_cast<char*>(t.data.data()), n * sizeof(float));
  if (!f) throw std::runtime_error("npy: short read " + path);
  return t;
}

// ---------------------------------------------------------------- lifecycle
Sam3MultiTracker::Sam3MultiTracker(const MultiPaths& p, Ep ep, ComputeUnits units)
    : mgr_("Sam3Multi"),
      mem_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
  InitGraph(g1_, mgr_.MakeSession(p.g1, units, ep));
  InitGraph(g2_, mgr_.MakeSession(p.g2, units, ep));
  InitGraph(g3_, mgr_.MakeSession(p.g3, units, ep));
  InitGraph(g4_, mgr_.MakeSession(p.g4, units, ep));
  InitGraph(g5_, mgr_.MakeSession(p.g5, units, ep));
  accel_active_ = g1_.h->accelerator_active();
  tpos_ = LoadNpyFloat(p.tpos);
  no_mem_ = LoadNpyFloat(p.no_mem);
  proj_w_ = LoadNpyFloat(p.proj_w);
  proj_b_ = LoadNpyFloat(p.proj_b);
  lang_feats_ = LoadNpyFloat(p.lang_feats);       // [32,1,256]
  Tensor lm = LoadNpyFloat(p.lang_mask);          // [1,32] as f32 0/1
  lang_mask_shape_ = lm.shape;
  lang_mask_bool_.resize(lm.data.size());
  for (size_t i = 0; i < lm.data.size(); ++i) lang_mask_bool_[i] = lm.data[i] > 0.5f ? 1 : 0;
}

Sam3MultiTracker::~Sam3MultiTracker() = default;

void Sam3MultiTracker::InitGraph(OrtGraph& g, std::shared_ptr<OrtSessionManager::Handle> h) {
  g.h = std::move(h);
  Ort::Session& s = g.h->Get();
  Ort::AllocatorWithDefaultOptions alloc;
  for (size_t i = 0; i < s.GetInputCount(); ++i)
    g.in_names.emplace_back(s.GetInputNameAllocated(i, alloc).get());
  for (size_t i = 0; i < s.GetOutputCount(); ++i)
    g.out_names.emplace_back(s.GetOutputNameAllocated(i, alloc).get());
}

std::vector<Sam3MultiTracker::Tensor> Sam3MultiTracker::RunFloat(
    OrtGraph& g, const std::vector<std::pair<std::string, const Tensor*>>& feeds) {
  Ort::Session& s = g.h->Get();
  std::vector<Ort::Value> ins;
  std::vector<const char*> in_names;
  for (const auto& name : g.in_names) {
    const Tensor* t = nullptr;
    for (const auto& fd : feeds) if (fd.first == name) { t = fd.second; break; }
    if (!t) throw std::runtime_error("Sam3Multi: missing feed '" + name + "'");
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

// ---------------------------------------------------------------- graphs
void Sam3MultiTracker::G1(const Tensor& image) {
  auto o = RunFloat(g1_, {{"image", &image}});
  // fpn0,fpn1,fpn2,pos0,pos1,pos2,det_fpn0,det_fpn1,det_fpn2,det_pos72
  cur_f0_ = std::move(o[0]); cur_f1_ = std::move(o[1]); cur_f2_ = std::move(o[2]);
  cur_pos2_ = std::move(o[5]);
  cur_detf0_ = std::move(o[6]); cur_detf1_ = std::move(o[7]);
  cur_detf2_ = std::move(o[8]); cur_detpos_ = std::move(o[9]);
}

std::vector<Sam3MultiTracker::Tensor> Sam3MultiTracker::G2() {
  Ort::Session& s = g2_.h->Get();
  std::vector<Ort::Value> ins;
  std::vector<const char*> names;
  auto addf = [&](const char* nm, const Tensor& t) {
    names.push_back(nm);
    ins.push_back(Ort::Value::CreateTensor<float>(
        mem_info_, const_cast<float*>(t.data.data()), t.data.size(),
        t.shape.data(), t.shape.size()));
  };
  for (const auto& nm : g2_.in_names) {
    if (nm == "fpn0") addf("fpn0", cur_detf0_);
    else if (nm == "fpn1") addf("fpn1", cur_detf1_);
    else if (nm == "fpn2") addf("fpn2", cur_detf2_);
    else if (nm == "pos72") addf("pos72", cur_detpos_);
    else if (nm == "lang_feats") addf("lang_feats", lang_feats_);
    else if (nm == "lang_mask") {
      names.push_back("lang_mask");
      ins.push_back(Ort::Value::CreateTensor<bool>(
          mem_info_, reinterpret_cast<bool*>(lang_mask_bool_.data()),
          lang_mask_bool_.size(), lang_mask_shape_.data(), lang_mask_shape_.size()));
    } else throw std::runtime_error("G2 unexpected input " + nm);
  }
  std::vector<const char*> out_names;
  for (const auto& n : g2_.out_names) out_names.push_back(n.c_str());
  auto res = s.Run(Ort::RunOptions{nullptr}, names.data(), ins.data(), ins.size(),
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

Sam3MultiTracker::Tensor Sam3MultiTracker::G3(const Tensor& src, const Tensor& src_pos,
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
          mem_info_, nopt_val.data(), 1, scalar_shape, 0));
    } else throw std::runtime_error("G3 unexpected input " + nm);
  }
  const char* out_names[] = {g3_.out_names[0].c_str()};
  auto res = s.Run(Ort::RunOptions{nullptr}, names.data(), ins.data(), ins.size(), out_names, 1);
  Tensor t;
  t.shape = res[0].GetTensorTypeAndShapeInfo().GetShape();
  const float* pd = res[0].GetTensorData<float>();
  t.data.assign(pd, pd + t.elems());
  return t;
}

void Sam3MultiTracker::G4(const Tensor& bf, const Tensor& hr0, const Tensor& hr1,
                          Tensor& low, Tensor& high, Tensor& ious, Tensor& obj_ptr, Tensor& osl) {
  auto o = RunFloat(g4_, {{"backbone_features", &bf}, {"high_res_0", &hr0}, {"high_res_1", &hr1}});
  low = std::move(o[0]); high = std::move(o[1]); ious = std::move(o[2]);
  obj_ptr = std::move(o[3]); osl = std::move(o[4]);
}

void Sam3MultiTracker::G5(const Tensor& pix, const Tensor& high, const Tensor& osl,
                          Tensor& mem, Tensor& mempos) {
  auto o = RunFloat(g5_, {{"pix_feat", &pix}, {"pred_masks_high_res", &high},
                          {"object_score_logits", &osl}});
  mem = std::move(o[0]); mempos = std::move(o[1]);
}

// ---------------------------------------------------------------- host helpers
Sam3MultiTracker::Tensor Sam3MultiTracker::FlattenHWBC(const Tensor& bchw) {
  const int C = (int)bchw.shape[1];
  const int HW = (int)(bchw.shape[2] * bchw.shape[3]);
  Tensor out; out.shape = {HW, 1, C}; out.data.resize((size_t)HW * C);
  for (int c = 0; c < C; ++c)
    for (int p = 0; p < HW; ++p)
      out.data[(size_t)p * C + c] = bchw.data[(size_t)c * HW + p];
  return out;
}

Sam3MultiTracker::Tensor Sam3MultiTracker::SeqToBCHW(const Tensor& seq) {
  const int HW = (int)seq.shape[0];
  const int C = (int)seq.shape[2];
  Tensor out; out.shape = {1, C, kMGrid, kMGrid}; out.data.resize((size_t)C * HW);
  for (int p = 0; p < HW; ++p)
    for (int c = 0; c < C; ++c)
      out.data[(size_t)c * HW + p] = seq.data[(size_t)p * C + c];
  return out;
}

void Sam3MultiTracker::Sine1DPE(const std::vector<float>& pos, int dim, std::vector<float>& out) {
  const int pe = dim / 2, N = (int)pos.size();
  out.assign((size_t)N * dim, 0.f);
  std::vector<float> dim_t(pe);
  for (int i = 0; i < pe; ++i) dim_t[i] = std::pow(10000.f, (2.f * (i / 2)) / pe);
  for (int n = 0; n < N; ++n)
    for (int i = 0; i < pe; ++i) {
      float v = pos[n] / dim_t[i];
      out[(size_t)n * dim + i] = std::sin(v);
      out[(size_t)n * dim + pe + i] = std::cos(v);
    }
}

std::pair<float, float> Sam3MultiTracker::EffIou(float osl, const float* ious3) const {
  float obj_norm = 0.f;
  if (osl > 0.f) obj_norm = (1.f / (1.f + std::exp(-osl))) * 2.f - 1.f;
  float iou_score = std::max(std::max(ious3[0], ious3[1]), ious3[2]);
  return {obj_norm * iou_score, iou_score};
}

const Sam3MultiTracker::Frame* Sam3MultiTracker::GetFrame(const ObjState& o, int idx) const {
  if (idx == o.cond_idx) return &o.cond_frame;
  for (const auto& fr : o.recent) if (fr.idx == idx) return &fr;
  return nullptr;
}

// use_memory_selection valid_indices over the object's retained non-cond frames.
// Mirrors sam3_tracker_base.frame_filter: forward selects frames strictly before the
// current frame (and after the seed); reverse (track_in_reverse) selects frames strictly
// after the current frame (and before the seed), with must-include = frame_idx+1. The
// deque may transiently hold frames from the opposite pass — those fall outside the
// [current, seed) window and are filtered here, so the ring buffer stays direction-clean.
std::vector<int> Sam3MultiTracker::FrameFilterObj(const ObjState& o, int frame_idx,
                                                  bool reverse) const {
  const int max_num = std::min(total_frames_, kMMaxObjPtrs);
  std::vector<int> valid;
  // iterate retained non-cond frames from newest-processed (back) to oldest (front);
  // for both passes the newest-processed frame is the one nearest the current frame.
  for (auto it = o.recent.rbegin(); it != o.recent.rend(); ++it) {
    if (!reverse) { if (it->idx >= frame_idx || it->idx <= o.cond_idx) continue; }
    else          { if (it->idx <= frame_idx || it->idx >= o.cond_idx) continue; }
    if (it->eff > kMMfThreshold) valid.insert(valid.begin(), it->idx);
    if ((int)valid.size() >= max_num - 1) break;
  }
  const int must = reverse ? frame_idx + 1 : frame_idx - 1;
  if (std::find(valid.begin(), valid.end(), must) == valid.end() && GetFrame(o, must))
    valid.push_back(must);
  return valid;
}

// ---------------------------------------------------------------- per-frame backbone
void Sam3MultiTracker::BeginFrame(const float* frame_chw1008, int frame_idx, int total_frames) {
  cur_frame_idx_ = frame_idx;
  total_frames_ = total_frames;
  Tensor image; image.shape = {1, 3, kMImgSize, kMImgSize};
  image.data.assign(frame_chw1008, frame_chw1008 + (size_t)3 * kMImgSize * kMImgSize);
  G1(image);
}

// ---------------------------------------------------------------- detector
[[maybe_unused]] static void ResizeBinToSoft(const std::vector<uint8_t>& src, int sh, int sw,
                            std::vector<float>& out, int H, int W) {
  out.assign((size_t)H * W, 0.f);
  for (int oy = 0; oy < H; ++oy) {
    float iy = (oy + 0.5f) * sh / H - 0.5f;
    int y0 = (int)std::floor(iy); float wy = iy - y0;
    int y0c = std::min(std::max(y0, 0), sh - 1), y1c = std::min(std::max(y0 + 1, 0), sh - 1);
    for (int ox = 0; ox < W; ++ox) {
      float ix = (ox + 0.5f) * sw / W - 0.5f;
      int x0 = (int)std::floor(ix); float wx = ix - x0;
      int x0c = std::min(std::max(x0, 0), sw - 1), x1c = std::min(std::max(x0 + 1, 0), sw - 1);
      float v = src[(size_t)y0c*sw+x0c]*(1-wy)*(1-wx) + src[(size_t)y0c*sw+x1c]*(1-wy)*wx
              + src[(size_t)y1c*sw+x0c]*wy*(1-wx) + src[(size_t)y1c*sw+x1c]*wy*wx;
      out[(size_t)oy*W+ox] = v;
    }
  }
}

std::vector<Det> Sam3MultiTracker::Detect(float score_thresh, float nms_iou) {
  auto o = G2();  // pred_logits[1,200,1], pred_boxes, pred_masks[1,200,288,288], presence[1,1]
  const Tensor& logits = o[0];
  const Tensor& masks = o[2];
  const Tensor& presence = o[3];
  const int Q = (int)logits.shape[1];
  const int MHW = kMLowRes * kMLowRes;
  const float pres_sig = 1.f / (1.f + std::exp(-presence.data[0]));

  struct Cand { int q; float prob; };
  std::vector<Cand> cands;
  for (int q = 0; q < Q; ++q) {
    float p = (1.f / (1.f + std::exp(-logits.data[q]))) * pres_sig;
    if (p > score_thresh) cands.push_back({q, p});
  }
  std::sort(cands.begin(), cands.end(), [](const Cand& a, const Cand& b){ return a.prob > b.prob; });

  // binarize candidate masks once
  auto binq = [&](int q, std::vector<uint8_t>& b) {
    b.resize(MHW);
    const float* m = masks.data.data() + (size_t)q * MHW;
    for (int i = 0; i < MHW; ++i) b[i] = m[i] > 0.f ? 1 : 0;
  };
  auto iou = [&](const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
    long inter = 0, uni = 0;
    for (int i = 0; i < MHW; ++i) { int aa = a[i], bb = b[i]; inter += (aa & bb); uni += (aa | bb); }
    return uni ? (double)inter / uni : 0.0;
  };

  std::vector<Det> kept;
  std::vector<std::vector<uint8_t>> kept_bin;
  for (const auto& c : cands) {
    std::vector<uint8_t> b; binq(c.q, b);
    bool ok = true;
    for (const auto& kb : kept_bin) if (iou(b, kb) > nms_iou) { ok = false; break; }
    if (!ok) continue;
    Det d; d.score = c.prob;
    long area = 0; for (int i = 0; i < MHW; ++i) area += b[i];
    d.area = area;
    // Confidence-aware seed coverage: keep the prob>0.5 silhouette (bin288, still used for
    // association) but RE-OPEN large low-confidence interior negative space -- the inter-limb gaps
    // the plain `m>0` binarise fills. soft1008 is the seed matte the tracker + MatAnyone2 propagate
    // from. See src/ConfidenceOpen.h.
    d.soft1008 = ConfidenceOpen(masks.data.data() + static_cast<size_t>(c.q) * MHW,
                                kMLowRes, kMLowRes, kMImgSize, kMImgSize);
    d.bin288 = b;
    kept.push_back(std::move(d));
    kept_bin.push_back(std::move(b));
  }
  return kept;
}

// ---------------------------------------------------------------- seed
void Sam3MultiTracker::SeedObj(ObjState& obj, const float* seed_mask1008) {
  obj.cond_idx = cur_frame_idx_;
  obj.born = cur_frame_idx_;
  obj.recent.clear();

  Tensor src = FlattenHWBC(cur_f2_);
  Tensor bf_seq = src;
  for (int p = 0; p < kMHW; ++p)
    for (int c = 0; c < kMHidden; ++c)
      bf_seq.data[(size_t)p * kMHidden + c] += no_mem_.data[c];
  Tensor bf = SeqToBCHW(bf_seq);
  Tensor low, high4, ious, obj_ptr, osl4;
  G4(bf, cur_f0_, cur_f1_, low, high4, ious, obj_ptr, osl4);

  Tensor high; high.shape = {1, 1, kMImgSize, kMImgSize};
  high.data.resize((size_t)kMImgSize * kMImgSize);
  for (size_t i = 0; i < high.data.size(); ++i) high.data[i] = seed_mask1008[i] * 20.f - 10.f;
  Tensor osl; osl.shape = {1, 1}; osl.data = {10.f};
  Tensor mem, mempos;
  G5(cur_f2_, high, osl, mem, mempos);

  Frame fr; fr.cond = true; fr.idx = cur_frame_idx_;
  fr.mem = std::move(mem); fr.mempos = std::move(mempos);
  fr.obj_ptr = std::move(obj_ptr); fr.osl = 10.f;
  float ones3[3] = {1.f, 1.f, 1.f};
  auto ei = EffIou(10.f, ones3); fr.eff = ei.first; fr.iou = ei.second;
  obj.cond_frame = std::move(fr);
  obj.active = true; obj.last_match = cur_frame_idx_; obj.keep_alive = kKeepAliveMax;

  // seed low-res mask (downsample 1008 high logits to 288 for association area/IoU)
  obj.last_low.assign((size_t)kMLowRes * kMLowRes, 0.f);
  long area = 0;
  for (int oy = 0; oy < kMLowRes; ++oy) {
    float iy = (oy + 0.5f) * kMImgSize / kMLowRes - 0.5f;
    int y0 = (int)std::floor(iy); float wy = iy - y0;
    int y0c = std::min(std::max(y0,0),kMImgSize-1), y1c = std::min(std::max(y0+1,0),kMImgSize-1);
    for (int ox = 0; ox < kMLowRes; ++ox) {
      float ix = (ox + 0.5f) * kMImgSize / kMLowRes - 0.5f;
      int x0 = (int)std::floor(ix); float wx = ix - x0;
      int x0c = std::min(std::max(x0,0),kMImgSize-1), x1c = std::min(std::max(x0+1,0),kMImgSize-1);
      float v = high.data[(size_t)y0c*kMImgSize+x0c]*(1-wy)*(1-wx)
              + high.data[(size_t)y0c*kMImgSize+x1c]*(1-wy)*wx
              + high.data[(size_t)y1c*kMImgSize+x0c]*wy*(1-wx)
              + high.data[(size_t)y1c*kMImgSize+x1c]*wy*wx;
      obj.last_low[(size_t)oy*kMLowRes+ox] = v;
      if (v > 0.f) ++area;
    }
  }
  obj.last_area = area;
  // seed HIGH-res output mask = the seed probability itself (1008*1008, [0,1]).
  obj.last_high.assign(seed_mask1008, seed_mask1008 + (size_t)kMImgSize * kMImgSize);
}

// ---------------------------------------------------------------- propagate
const std::vector<float>& Sam3MultiTracker::PropagateObj(ObjState& obj, bool reverse) {
  const int frame_idx = cur_frame_idx_;
  Tensor src = FlattenHWBC(cur_f2_);
  Tensor src_pos = FlattenHWBC(cur_pos2_);

  std::vector<int> valid = FrameFilterObj(obj, frame_idx, reverse);

  std::vector<float> prompt, prompt_pos;
  auto append_maskmem = [&](const Frame& r, int tpos_idx) {
    Tensor mtok = FlattenHWBC(r.mem);
    Tensor ptok = FlattenHWBC(r.mempos);
    const float* tp = &tpos_.data[(size_t)tpos_idx * kMMemDim];
    for (int p = 0; p < kMHW; ++p)
      for (int c = 0; c < kMMemDim; ++c) {
        prompt.push_back(mtok.data[(size_t)p * kMMemDim + c]);
        prompt_pos.push_back(ptok.data[(size_t)p * kMMemDim + c] + tp[c]);
      }
  };
  // single cond frame (t_pos=0 -> tpos index num_maskmem-1)
  append_maskmem(obj.cond_frame, kMNumMaskMem - 1);
  // non-cond recent, t_pos = 1..num_maskmem-1
  for (int t_pos = 1; t_pos < kMNumMaskMem; ++t_pos) {
    int t_rel = kMNumMaskMem - t_pos;
    if (t_rel > (int)valid.size()) continue;
    int prev = valid[valid.size() - t_rel];
    const Frame* fr = GetFrame(obj, prev);
    if (fr) append_maskmem(*fr, kMNumMaskMem - t_pos - 1);
  }

  // object-pointer tokens.
  // The conditioning frame's temporal position is (frame_idx - cond_idx) * tpos_sign_mul
  // (sam3_tracker_base: tpos_sign_mul = -1 when track_in_reverse). In the forward pass
  // frame_idx > cond_idx so this is a positive distance; in the reverse pass frame_idx <
  // cond_idx, and the -1 sign restores a positive distance. Non-cond obj-ptrs below use
  // t_diff (always positive) in both directions, exactly as in the reference.
  const float tpos_sign = reverse ? -1.f : 1.f;
  std::vector<float> pos_list;
  std::vector<const Tensor*> ptrs;
  pos_list.push_back((float)(frame_idx - obj.cond_idx) * tpos_sign);
  ptrs.push_back(&obj.cond_frame.obj_ptr);
  for (int t_diff = 1; t_diff < kMMaxObjPtrs; ++t_diff) {
    if (t_diff >= (int)valid.size()) break;
    int t = valid[valid.size() - t_diff];
    const Frame* fr = GetFrame(obj, t);
    if (fr) { pos_list.push_back((float)t_diff); ptrs.push_back(&fr->obj_ptr); }
  }
  int64_t n_optr = 0;
  {
    const int P = (int)ptrs.size();
    const float denom = (float)(std::min(total_frames_, kMMaxObjPtrs) - 1);
    std::vector<float> posn(P);
    for (int i = 0; i < P; ++i) posn[i] = pos_list[i] / denom;
    std::vector<float> sine;
    Sine1DPE(posn, kMHidden, sine);
    std::vector<float> obj_pos((size_t)P * kMMemDim);
    for (int i = 0; i < P; ++i)
      for (int oo = 0; oo < kMMemDim; ++oo) {
        float acc = proj_b_.data[oo];
        const float* w = &proj_w_.data[(size_t)oo * kMHidden];
        const float* sptr = &sine[(size_t)i * kMHidden];
        for (int k = 0; k < kMHidden; ++k) acc += sptr[k] * w[k];
        obj_pos[(size_t)i * kMMemDim + oo] = acc;
      }
    for (int i = 0; i < P; ++i)
      for (int tk = 0; tk < kMPtrTokens; ++tk) {
        const float* pv = &ptrs[i]->data[(size_t)tk * kMMemDim];
        for (int c = 0; c < kMMemDim; ++c) prompt.push_back(pv[c]);
        const float* op = &obj_pos[(size_t)i * kMMemDim];
        for (int c = 0; c < kMMemDim; ++c) prompt_pos.push_back(op[c]);
      }
    n_optr = (int64_t)P * kMPtrTokens;
  }

  const int64_t L = (int64_t)prompt.size() / kMMemDim;
  Tensor promptT; promptT.shape = {L, 1, kMMemDim}; promptT.data = std::move(prompt);
  Tensor promptPosT; promptPosT.shape = {L, 1, kMMemDim}; promptPosT.data = std::move(prompt_pos);

  Tensor memory = G3(src, src_pos, promptT, promptPosT, n_optr);
  Tensor bf = SeqToBCHW(memory);
  Tensor low, high, ious, obj_ptr, osl;
  G4(bf, cur_f0_, cur_f1_, low, high, ious, obj_ptr, osl);
  Tensor mem, mempos;
  G5(cur_f2_, high, osl, mem, mempos);

  Frame fr; fr.cond = false; fr.idx = frame_idx;
  fr.mem = std::move(mem); fr.mempos = std::move(mempos);
  fr.obj_ptr = std::move(obj_ptr); fr.osl = osl.data[0];
  auto ei = EffIou(osl.data[0], ious.data.data()); fr.eff = ei.first; fr.iou = ei.second;
  obj.recent.push_back(std::move(fr));
  // ---- ring-buffer / obj-ptr cap: retain only the most recent kMMaxObjPtrs non-cond frames
  while ((int)obj.recent.size() > kMMaxObjPtrs) obj.recent.pop_front();
  obj.max_recent = std::max(obj.max_recent, obj.recent.size());

  obj.last_low = low.data;
  // HIGH-res output mask: sigmoid of G4's pred_masks_high_res (1008*1008, [0,1]).
  // It is already computed above (fed to G5 for memory) — previously discarded. Keeping
  // it is the matte-resolution win: the sidecar can carry 1008² instead of the 288
  // downsample (3.5x linear), with no extra model compute.
  // Confidence-aware per-frame coverage: sigmoid + RE-OPEN large low-confidence interior negative
  // space, so PROPAGATE frames (G4) don't re-fill the inter-limb gaps the seed opened. high.data =
  // G4 pred_masks_high_res logits at 1008². See src/ConfidenceOpen.h.
  obj.last_high = ConfidenceOpen(high.data.data(), kMImgSize, kMImgSize, kMImgSize, kMImgSize);
  long area = 0; for (float v : low.data) if (v > 0.f) ++area;
  obj.last_area = area;
  return obj.last_low;
}

}  // namespace hastur
