// Copyright the Hastur authors.
// SPDX-License-Identifier: LicenseRef-SAM-License
//
// hastur_track — headless SAM 3 video tracker (the whole-clip pre-pass of
// Sam3TrackerPlugin, with OFX fetchImage replaced by PNG reads). Produces the
// external-tracks sidecar that gives Hastur's SAM-3D-Body node/ hastur_export
// TEMPORALLY-STABLE person_NN ids (person_NN == the tracker's id, stable by
// construction — no per-frame cam_t reassignment / id flicker).
//
//   <out>/tracks.txt              # frame track_id x0 y0 x1 y1 mask_relpath (plate px, top-down)
//   <out>/masks/<frame>_<id>.msk  # soft coverage, 1008^2 uint8
//
// Feed it to hastur_export via `--tracks <out>/tracks.txt` (or Sam3dBody's
// tracksDir / $HASTUR_TRACKS). The tracking algorithm — seed-frame scan, seed,
// forward + backward bidirectional passes, det<->track IoU association, spawn/
// prune — is ported VERBATIM from src/Sam3TrackerPlugin.cpp::RunPrepass so the
// headless product is byte-for-byte the node's product. It lives in
// src/Sam3VideoTrack.cpp (RunSam3VideoTrack) so embedding apps run the same code;
// this file only reads the PNG frames and maps the flags.
//
//   hastur_track --in 'plate_%04d.png' --out <dir> --first N --last N
//                --models <dir with G1.onnx..G5.onnx + const_*.npy>
//                [--seed -1] [--score 0.5] [--nms 0.1] [--no-bidir] [--units 0]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "vendor/stb_image.h"

#include "Sam3VideoTrack.h"

namespace {
const char* arg(int c, char** v, const char* f) {
  for (int i = 1; i + 1 < c; ++i)
    if (!std::strcmp(v[i], f)) return v[i + 1];
  return nullptr;
}
bool flag(int c, char** v, const char* f) {
  for (int i = 1; i < c; ++i)
    if (!std::strcmp(v[i], f)) return true;
  return false;
}
std::string fmt(const std::string& pat, long n) {
  char b[4096];
  std::snprintf(b, sizeof(b), pat.c_str(), static_cast<int>(n));
  return std::string(b);
}

hastur::ComputeUnits ChoiceToUnits(int choice) {
  switch (choice) {
    case 1: return hastur::ComputeUnits::CpuAndGpu;
    case 2: return hastur::ComputeUnits::CpuAndAne;
    case 3: return hastur::ComputeUnits::CpuOnly;
    default: return hastur::ComputeUnits::All;
  }
}
}  // namespace

int main(int argc, char** argv) {
  const char* in_pat = arg(argc, argv, "--in");
  const char* out_dir = arg(argc, argv, "--out");
  const char* models = arg(argc, argv, "--models");
  const char* firstS = arg(argc, argv, "--first");
  const char* lastS = arg(argc, argv, "--last");
  if (!in_pat || !out_dir || !models || !firstS || !lastS) {
    std::fprintf(stderr,
                 "usage: hastur_track --in <pat> --out <dir> --first N --last N "
                 "--models <dir> [--seed -1] [--score 0.5] [--nms 0.1] "
                 "[--no-bidir] [--units 0]\n");
    return 2;
  }
  const int first = std::atoi(firstS), last = std::atoi(lastS);
  if (last < first) { std::fprintf(stderr, "empty range\n"); return 2; }
  hastur::Sam3VideoTrackOptions opt;
  opt.seed = arg(argc, argv, "--seed") ? std::atoi(arg(argc, argv, "--seed")) : -1;
  opt.score = arg(argc, argv, "--score") ? std::atof(arg(argc, argv, "--score")) : 0.5f;
  opt.nms = arg(argc, argv, "--nms") ? std::atof(arg(argc, argv, "--nms")) : 0.1f;
  opt.bidir = !flag(argc, argv, "--no-bidir");
  opt.units = ChoiceToUnits(arg(argc, argv, "--units") ? std::atoi(arg(argc, argv, "--units")) : 0);
  // HASTUR_TRACK_CPU=1 forces a plain CPU session, matching the convention the
  // other engines already use (HASTUR_DET_CPU, HASTUR_BODY_CPU, HASTUR_HAND_CPU).
  // Ep::Auto would ALSO land on CPU when no accelerator is present, but only by
  // falling back after a failed attempt; asking for Cpu outright is how an
  // explicit --cpu-only request is honoured, and it skips the provider-load
  // errors that otherwise dominate the log on a machine with no GPU.
  opt.ep = std::getenv("HASTUR_TRACK_CPU") ? hastur::Ep::Cpu : hastur::Ep::Auto;

  // Top-down 8-bit RGB PNGs (already srgb_texture), as [0,1] floats.
  auto source = [&](long frame, std::vector<float>& rgb, int& W, int& H) {
    const std::string ip = fmt(in_pat, frame);
    int nc = 0;
    unsigned char* px = stbi_load(ip.c_str(), &W, &H, &nc, 3);
    if (!px) return false;
    rgb.resize(static_cast<size_t>(W) * H * 3);
    for (size_t i = 0, n = rgb.size(); i < n; ++i) rgb[i] = px[i] / 255.f;
    stbi_image_free(px);
    return true;
  };

  // AN EXCEPTION IS NOT AN EMPTY CLIP.
  //
  // A person-free clip writes a header-only tracks.txt and returns 0 (inside
  // RunSam3VideoTrack). Anything that throws -- missing or undecryptable weights,
  // an engine error -- exits 1 and writes NO tracks.txt: a readable (even empty)
  // tracks.txt is precisely what tells the orchestrator this was a real, empty
  // result, and person_frames=0 reaching the metered .done sentinel would bill a
  // failed run as an empty shot.
  try {
    hastur::RunSam3VideoTrack(hastur::Sam3TrackerPaths(models), source, first, last, out_dir, opt);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr,
                 "[hastur_track] FAILED: engine exception (%s)\n"
                 "  This is a FAILURE, not an empty clip: no tracks were written.\n",
                 e.what());
    return 1;
  } catch (...) {
    std::fprintf(stderr,
                 "[hastur_track] FAILED: unknown engine exception\n"
                 "  This is a FAILURE, not an empty clip: no tracks were written.\n");
    return 1;
  }
}
