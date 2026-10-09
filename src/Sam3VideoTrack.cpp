// Copyright the Hastur authors.
// SPDX-License-Identifier: LicenseRef-SAM-License
//
// Moved from tools/hastur_track.cpp (which now calls it) with the algorithm
// unchanged: only where frames come from (a callback instead of stbi_load) and the
// optional progress callback are new. Keep it byte-for-byte equivalent to
// Sam3TrackerPlugin::RunPrepass, as hastur_track always was.
#include "Sam3VideoTrack.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <unordered_map>

namespace fs = std::filesystem;

namespace hastur {
namespace {

// The engine's input geometry (mirrors Sam3TrackerPlugin's kIn/kLR).
constexpr int kIn = kMImgSize;  // 1008
constexpr int kLR = kMLowRes;   // 288

// Bilinear-resample a top-down W*H*3 RGB [0,1] frame to a planar CHW 3*1008*1008
// buffer, normalized 2*v-1 (mean/std 0.5) as the SAM 3 engine expects. Verbatim
// from Sam3TrackerPlugin::T3ToPlanarCHW.
void ToPlanarCHW(const std::vector<float>& rgb, int W, int H, std::vector<float>& chw) {
  chw.assign(static_cast<size_t>(3) * kIn * kIn, 0.f);
  const size_t plane = static_cast<size_t>(kIn) * kIn;
  for (int ty = 0; ty < kIn; ++ty) {
    float gy = (ty + 0.5f) * (static_cast<float>(H) / kIn) - 0.5f;
    if (gy < 0.f) gy = 0.f;
    if (gy > H - 1.f) gy = H - 1.f;
    const int y0 = static_cast<int>(gy);
    const int y1 = std::min(y0 + 1, H - 1);
    const float wy = gy - y0;
    for (int tx = 0; tx < kIn; ++tx) {
      float gx = (tx + 0.5f) * (static_cast<float>(W) / kIn) - 0.5f;
      if (gx < 0.f) gx = 0.f;
      if (gx > W - 1.f) gx = W - 1.f;
      const int x0 = static_cast<int>(gx);
      const int x1 = std::min(x0 + 1, W - 1);
      const float wx = gx - x0;
      const size_t o = static_cast<size_t>(ty) * kIn + tx;
      for (int c = 0; c < 3; ++c) {
        const float v00 = rgb[(static_cast<size_t>(y0) * W + x0) * 3 + c];
        const float v01 = rgb[(static_cast<size_t>(y0) * W + x1) * 3 + c];
        const float v10 = rgb[(static_cast<size_t>(y1) * W + x0) * 3 + c];
        const float v11 = rgb[(static_cast<size_t>(y1) * W + x1) * 3 + c];
        const float top = v00 + (v01 - v00) * wx;
        const float bot = v10 + (v11 - v10) * wx;
        const float v = top + (bot - top) * wy;
        chw[c * plane + o] = 2.f * v - 1.f;
      }
    }
  }
}

// IoU of a binary detection mask (288) vs a track's low-res logits (>0). Verbatim
// from Sam3TrackerPlugin::T3IouBinLogit.
double IouBinLogit(const std::vector<uint8_t>& det, const std::vector<float>& trk) {
  long inter = 0, uni = 0;
  for (int i = 0; i < kLR * kLR; ++i) {
    int a = det[i], bb = (i < static_cast<int>(trk.size()) && trk[i] > 0.f) ? 1 : 0;
    inter += (a & bb);
    uni += (a | bb);
  }
  return uni ? static_cast<double>(inter) / uni : 0.0;
}

}  // namespace

const std::vector<std::string>& Sam3TrackerFiles() {
  static const std::vector<std::string> f = {
      "G1.onnx", "G1.onnx.data", "G2.onnx", "G2.onnx.data",
      "G3.onnx", "G3.onnx.data", "G4.onnx", "G4.onnx.data",
      "G5.onnx", "G5.onnx.data",
      "const_maskmem_tpos_enc.npy", "const_no_mem_embed.npy",
      "const_objptr_tpos_proj_w.npy", "const_objptr_tpos_proj_b.npy",
      "const_lang_feats_person.npy", "const_lang_mask_person.npy",
  };
  return f;
}

MultiPaths Sam3TrackerPaths(const std::string& dir) {
  // Paths are names the model tree resolves (sealed or plaintext), not
  // necessarily filesystem locations: graphs and constants load from memory.
  fs::path d(dir);
  MultiPaths p;
  p.g1 = (d / "G1.onnx").string();
  p.g2 = (d / "G2.onnx").string();
  p.g3 = (d / "G3.onnx").string();
  p.g4 = (d / "G4.onnx").string();
  p.g5 = (d / "G5.onnx").string();
  p.tpos = (d / "const_maskmem_tpos_enc.npy").string();
  p.no_mem = (d / "const_no_mem_embed.npy").string();
  p.proj_w = (d / "const_objptr_tpos_proj_w.npy").string();
  p.proj_b = (d / "const_objptr_tpos_proj_b.npy").string();
  p.lang_feats = (d / "const_lang_feats_person.npy").string();
  p.lang_mask = (d / "const_lang_mask_person.npy").string();
  return p;
}

Sam3VideoTrackResult RunSam3VideoTrack(const MultiPaths& models, const Sam3FrameSource& source,
                                       long first, long last, const std::string& out_dir,
                                       const Sam3VideoTrackOptions& opt) {
  if (last < first) throw std::invalid_argument("empty range");
  const char* tag = opt.log_tag ? opt.log_tag : "hastur_track";
  const int N = static_cast<int>(last - first + 1);
  const long t0 = first;  // plate time of idx 0
  const float scoreThresh = opt.score, nmsIou = opt.nms;
  auto progress = [&](const char* phase, int done, int total) {
    if (opt.progress) opt.progress(phase, done, total);
  };

  // ---- lazily load + convert a frame (idx 0..N-1 -> plate first+idx) to planar
  // CHW, cached for the whole pre-pass (scan/forward/backward revisit frames). ----
  std::unordered_map<int, std::vector<float>> cache;
  int plateW = 0, plateH = 0;
  std::vector<float> rgb;
  auto getFrame = [&](int idx) -> const std::vector<float>* {
    auto it = cache.find(idx);
    if (it != cache.end()) return it->second.empty() ? nullptr : &it->second;
    std::vector<float>& slot = cache[idx];  // inserts empty
    int W = 0, H = 0;
    if (!source(first + idx, rgb, W, H) || W <= 0 || H <= 0 ||
        rgb.size() < static_cast<size_t>(W) * H * 3) {
      std::fprintf(stderr, "skip: cannot load frame %ld\n", first + idx);
      return nullptr;
    }
    if (plateW == 0) { plateW = W; plateH = H; }
    ToPlanarCHW(rgb, W, H, slot);
    return &slot;
  };

  // ---- build the engine (loads G1..G5 + constants; throws on missing assets) ----
  Sam3MultiTracker eng(models, opt.ep, opt.units);
  std::fprintf(stderr,
               "[%s] frames %ld..%ld (N=%d) accel=%d score=%.3f nms=%.3f "
               "bidir=%d\n",
               tag, first, last, N, static_cast<int>(eng.accelerator_active()),
               scoreThresh, nmsIou, static_cast<int>(opt.bidir));

  // ---- seed frame (idx space). -1 => auto max-visibility scan. ----
  int SEED = 0;
  if (opt.seed >= 0) {
    long s = opt.seed - first;
    if (s < 0) s = 0;
    if (s > N - 1) s = N - 1;
    SEED = static_cast<int>(s);
  } else {
    double best = -1.0;
    for (int f = 0; f < N; ++f) {
      progress("scan", f, N);
      const std::vector<float>* chw = getFrame(f);
      if (!chw) continue;
      eng.BeginFrame(chw->data(), f, N);
      auto dets = eng.Detect(scoreThresh, nmsIou);
      double vis = 0.0;
      for (auto& d : dets) vis += static_cast<double>(d.area);
      std::fprintf(stderr, "  scan %3d/%d: %zu dets, vis=%.0f\n", f + 1, N, dets.size(), vis);
      if (vis > best) { best = vis; SEED = f; }
    }
    progress("scan", N, N);
    std::fprintf(stderr, "[%s] auto seed frame = %d (plate=%ld)\n", tag, SEED, t0 + SEED);
  }

  // ---- driver state (mirrors RunPrepass) ----
  std::vector<Sam3MultiTracker::ObjState> tracks;
  int next_id = 0;
  const float NEW_DET = kNewDetThresh;   // 0.7
  const float ASSOC = kAssocIou;         // 0.1
  const float TRK_ASSOC = kTrkAssocIou;  // 0.5
  const int PRUNE_UNMATCHED = 8;

  const fs::path sdir(out_dir);
  const fs::path mdir = sdir / "masks";
  { std::error_code ec; fs::create_directories(mdir, ec); }
  struct MaskRow { long frame; int id; int x0, y0, x1, y1; std::string rel; };
  std::vector<MaskRow> track_rows;
  auto write_mask = [&](long frame, int id, const std::vector<float>& hi) {
    if (static_cast<int>(hi.size()) < kIn * kIn) return;
    int x0 = kIn, y0 = kIn, x1 = -1, y1 = -1;
    std::vector<uint8_t> cov(static_cast<size_t>(kIn) * kIn, 0);
    for (int y = 0; y < kIn; ++y)
      for (int x = 0; x < kIn; ++x) {
        const size_t idx = static_cast<size_t>(y) * kIn + x;
        float p = hi[idx];
        if (p < 0.f) p = 0.f; else if (p > 1.f) p = 1.f;
        cov[idx] = static_cast<uint8_t>(p * 255.f + 0.5f);
        if (p > 0.5f) { x0 = std::min(x0, x); y0 = std::min(y0, y);
                        x1 = std::max(x1, x); y1 = std::max(y1, y); }
      }
    if (x1 < 0) return;  // empty mask
    char rel[64];
    std::snprintf(rel, sizeof(rel), "masks/%04ld_%02d.msk", frame, id);
    std::ofstream mk(sdir / rel, std::ios::binary | std::ios::trunc);
    if (mk) {
      mk.write("MSK1", 4);
      int32_t mw = kIn, mh = kIn;
      mk.write(reinterpret_cast<const char*>(&mw), 4);
      mk.write(reinterpret_cast<const char*>(&mh), 4);
      mk.write(reinterpret_cast<const char*>(cov.data()),
               static_cast<std::streamsize>(cov.size()));
    }
    track_rows.push_back({frame, id, x0, y0, x1, y1, std::string(rel)});
  };
  auto record_frame = [&](int f) {
    const long plate = t0 + f;
    for (auto& t : tracks) {
      if (!t.active || t.last_area <= 0) continue;
      write_mask(plate, t.id, t.last_high);
    }
  };

  auto process_frame = [&](int f, bool reverse) {
    for (auto& t : tracks) if (t.active) eng.PropagateObj(t, reverse);
    auto dets = eng.Detect(scoreThresh, nmsIou);
    const int M = static_cast<int>(tracks.size());
    const int D = static_cast<int>(dets.size());
    std::vector<double> trk_best(M, 0.0), det_best(D, 0.0);
    for (int di = 0; di < D; ++di)
      for (int ti = 0; ti < M; ++ti) {
        if (!tracks[ti].active) continue;
        double j = IouBinLogit(dets[di].bin288, tracks[ti].last_low);
        if (j > det_best[di]) det_best[di] = j;
        if (j > trk_best[ti]) trk_best[ti] = j;
      }
    for (int ti = 0; ti < M; ++ti) {
      if (!tracks[ti].active) continue;
      if (trk_best[ti] >= TRK_ASSOC) tracks[ti].last_match = f;
      else if (std::abs(f - tracks[ti].last_match) >= PRUNE_UNMATCHED)
        tracks[ti].active = false;
    }
    for (int di = 0; di < D; ++di) {
      if (dets[di].score >= NEW_DET && det_best[di] < ASSOC) {
        Sam3MultiTracker::ObjState t; t.id = next_id++;
        eng.SeedObj(t, dets[di].soft1008.data());
        tracks.push_back(std::move(t));
      }
    }
    int nactive = 0; for (auto& t : tracks) if (t.active) ++nactive;
    std::fprintf(stderr, "  frame %3d (%s): %d dets, %d active\n", f,
                 reverse ? "bwd" : "fwd", D, nactive);
  };

  // ---- seed all detections at the seed frame ----
  {
    const std::vector<float>* chw = getFrame(SEED);
    if (!chw) throw std::runtime_error("seed frame unreadable");
    eng.BeginFrame(chw->data(), SEED, N);
    auto dets0 = eng.Detect(scoreThresh, nmsIou);
    std::fprintf(stderr, "[%s] seed frame %d: %zu dets\n", tag, SEED, dets0.size());
    for (auto& d : dets0) {
      Sam3MultiTracker::ObjState t; t.id = next_id++;
      eng.SeedObj(t, d.soft1008.data());
      tracks.push_back(std::move(t));
    }
    record_frame(SEED);
  }

  // snapshot seed-frame state for the backward pass (forward mutates tracks).
  std::vector<Sam3MultiTracker::ObjState> seed_snapshot = tracks;
  const int seed_next_id = next_id;

  // ---- FORWARD pass: seed+1 .. N-1 ----
  std::fprintf(stderr, "[%s] forward pass %d -> %d\n", tag, SEED + 1, N - 1);
  const int nfwd = N - 1 - SEED;
  for (int f = SEED + 1; f < N; ++f) {
    progress("forward", f - SEED - 1, nfwd);
    const std::vector<float>* chw = getFrame(f);
    if (chw) { eng.BeginFrame(chw->data(), f, N); process_frame(f, false); record_frame(f); }
  }
  progress("forward", nfwd, nfwd);

  // ---- BACKWARD pass: restore snapshot, seed-1 .. 0 ----
  if (opt.bidir) {
    tracks = seed_snapshot;
    next_id = seed_next_id;
    std::fprintf(stderr, "[%s] backward pass %d -> %d\n", tag, SEED - 1, 0);
    for (int f = SEED - 1; f >= 0; --f) {
      progress("backward", SEED - 1 - f, SEED);
      const std::vector<float>* chw = getFrame(f);
      if (chw) { eng.BeginFrame(chw->data(), f, N); process_frame(f, true); record_frame(f); }
    }
    progress("backward", SEED, SEED);
  }

  if (plateW == 0 || plateH == 0) throw std::runtime_error("no frames read; nothing written");

  // ---- write tracks.txt (masks already streamed) ----
  std::vector<int> ids;
  for (auto& r : track_rows) ids.push_back(r.id);
  std::sort(ids.begin(), ids.end());
  ids.erase(std::unique(ids.begin(), ids.end()), ids.end());

  std::ofstream tt(sdir / "tracks.txt", std::ios::binary | std::ios::trunc);
  if (!tt) throw std::runtime_error("cannot open " + (sdir / "tracks.txt").string());
  tt << "# hastur-tracks v1\n";
  tt << "# ids"; for (int id : ids) tt << " " << id; tt << "\n";
  tt << "# frame track_id x0 y0 x1 y1 mask_relpath\n";
  const float sx = static_cast<float>(plateW) / kIn;
  const float sy = static_cast<float>(plateH) / kIn;
  for (auto& r : track_rows)
    tt << r.frame << " " << r.id << " " << (r.x0 * sx) << " " << (r.y0 * sy) << " "
       << ((r.x1 + 1) * sx) << " " << ((r.y1 + 1) * sy) << " " << r.rel << "\n";
  tt.flush();
  if (!tt) throw std::runtime_error("cannot write " + (sdir / "tracks.txt").string());
  std::fprintf(stderr, "[%s] wrote %s/tracks.txt (%zu rows, %zu ids) + masks/\n",
               tag, sdir.string().c_str(), track_rows.size(), ids.size());
  // An empty-but-successfully-written clip (0 detected persons) is a SUCCESS: the
  // header-only tracks.txt is the valid empty form ExternalTracks already accepts.
  return {track_rows.size(), ids.size()};
}

}  // namespace hastur
