// Copyright the Hastur authors.
// SPDX-License-Identifier: LicenseRef-SAM-License
//
// hastur_export — headless per-frame exporter for the Rotobot-Next golden
// pipeline. Runs the SAME Sam3dBodyPipeline engine as the OFX plugin (no OFX
// host, no Natron multi-plane) and writes, for each input frame:
//     <out>/crypto_<NNNN>.exr    per-person Cryptomatte (person00/01 + manifest)
//     <out>/skeleton_<NNNN>.json MHR skeleton sidecar
// which are exactly the two inputs Rotobot.ofx / rotobot_fit consume. This
// sidesteps Natron 2.5's incomplete multi-plane routing (the CryptoObject
// planes never reach MattePartition there); the multi-plane path is only
// validated in Natron 2.6, which has no Linux build.
//
//   hastur_export --in 'plate_%04d.png' --out <dir> --first N --last N
//                 --models <dir> [--maxpeople 6] [--score 0.3]
//                 [--coverage mesh|sam3|both] [--tracks <tracks.txt>]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "vendor/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "vendor/stb_image_write.h"

#include "CryptoExr.h"
#include "Cryptomatte.h"      // CryptoIdFloat
#include "ExternalTracks.h"
#include "MattePartition.h"   // PartitionCryptomatte (fuse a refined pooled matte)
#include "MeshTypes.h"
#include "Sam3dBodyPipeline.h"
#include "Skeleton.h"

namespace {
const char* arg(int c, char** v, const char* f) {
  for (int i = 1; i + 1 < c; ++i)
    if (!std::strcmp(v[i], f)) return v[i + 1];
  return nullptr;
}
std::string fmt(const std::string& pat, int n) {
  char b[4096];
  std::snprintf(b, sizeof(b), pat.c_str(), n);
  return std::string(b);
}
std::string pad4(int n) {
  char b[16];
  std::snprintf(b, sizeof(b), "%04d", n);
  return std::string(b);
}
}  // namespace

int main(int argc, char** argv) {
  const char* in_pat = arg(argc, argv, "--in");
  const char* out_dir = arg(argc, argv, "--out");
  const char* models = arg(argc, argv, "--models");
  const char* firstS = arg(argc, argv, "--first");
  const char* lastS = arg(argc, argv, "--last");
  if (!in_pat || !out_dir || !firstS || !lastS) {
    std::fprintf(stderr,
                 "usage: hastur_export --in <pat> --out <dir> --first N --last N "
                 "--models <dir> [--maxpeople 6] [--score 0.3] "
                 "[--coverage mesh|sam3|both] [--tracks <path>]\n");
    return 2;
  }
  const int first = std::atoi(firstS), last = std::atoi(lastS);
  const int maxpeople = arg(argc, argv, "--maxpeople") ? std::atoi(arg(argc, argv, "--maxpeople")) : 6;
  const float score = arg(argc, argv, "--score") ? std::atof(arg(argc, argv, "--score")) : 0.3f;
  const char* cov = arg(argc, argv, "--coverage");
  const char* tracks = arg(argc, argv, "--tracks");
  const char* pool_dir = arg(argc, argv, "--pool");  // write pool_%04d.png (union fg)
  const char* refined_pat = arg(argc, argv, "--refined");  // read refined pooled matte
  const float max_dist = arg(argc, argv, "--max-dist")
                             ? std::atof(arg(argc, argv, "--max-dist")) : 256.0f;

  hastur::CryptoCoverage coverage = hastur::CryptoCoverage::Both;
  if (cov && !std::strcmp(cov, "mesh")) coverage = hastur::CryptoCoverage::Mesh;
  else if (cov && !std::strcmp(cov, "sam3")) coverage = hastur::CryptoCoverage::Sam3Mask;

  hastur::Sam3dBodyPipeline pipe;
  int ok = 0;
  for (int f = first; f <= last; ++f) {
    const std::string ip = fmt(in_pat, f);
    int W = 0, H = 0, nc = 0;
    unsigned char* px = stbi_load(ip.c_str(), &W, &H, &nc, 3);
    if (!px) { std::fprintf(stderr, "skip: cannot load %s\n", ip.c_str()); continue; }
    std::vector<float> rgb(static_cast<size_t>(W) * H * 3);
    for (size_t i = 0, n = rgb.size(); i < n; ++i) rgb[i] = px[i] / 255.f;
    stbi_image_free(px);

    hastur::PipelineParams p;
    p.model_dir = models ? models : "";
    p.units = hastur::ComputeUnits::All;
    p.detector_score_thresh = score;
    p.max_people = maxpeople;
    p.emit_aovs = true;              // populate fr.crypto
    p.crypto_levels = 2;
    p.crypto_coverage = coverage;
    p.premultiply = false;
    // Advance the pipeline clock EVERY frame. AssignTrackIds() keys the internal
    // cam_t track table off `time`: the association gate widens with the frame gap
    // (a person missed for a few frames still re-links to the SAME id instead of
    // spawning a new one) and stale tracks are pruned by `time - last_time`. With
    // `time` frozen (its 0.0 default) gap is always 1 (gate never widens -> every
    // flicker-out/in mints a new id) and pruning never fires -> id churn. A
    // headless in-order render is exactly the case this stable path is built for.
    p.stable_person_ids = true;
    p.time = f;
    if (tracks) {
      p.tracks_path = tracks;
      p.use_external_tracks = hastur::HasExternalTracks(tracks);
    }

    hastur::FrameResult fr = pipe.Run(rgb.data(), W, H, p);
    if (static_cast<int>(fr.render.data.size()) != W * H * 4) {
      std::fprintf(stderr, "frame %d: pipeline produced no render (%s)\n", f,
                   pipe.last_error().c_str());
      continue;
    }

    // Pooled preview matte = render alpha (W*H, top-down).
    std::vector<float> pool(static_cast<size_t>(W) * H);
    for (size_t i = 0, n = pool.size(); i < n; ++i) pool[i] = fr.render.data[i * 4 + 3];

    // Optional: dump the pooled union foreground as an 8-bit PNG (MatAnyone seed).
    if (pool_dir) {
      std::vector<unsigned char> pm(pool.size());
      for (size_t i = 0, n = pool.size(); i < n; ++i) {
        float v = pool[i]; v = v < 0 ? 0 : (v > 1 ? 1 : v);
        pm[i] = static_cast<unsigned char>(v * 255.f + 0.5f);
      }
      const std::string pp = std::string(pool_dir) + "/pool_" + pad4(f) + ".png";
      stbi_write_png(pp.c_str(), W, H, 1, pm.data(), W);
    }

    // Skeleton sidecar.
    hastur::SkeletonFrame sk = hastur::BuildSkeletonFrame(fr, f, W, H);
    const std::string js = hastur::SkeletonToJson(sk, pipe.JointParents());
    const std::string skp = std::string(out_dir) + "/skeleton_" + pad4(f) + ".json";
    if (std::FILE* sf = std::fopen(skp.c_str(), "wb")) {
      std::fwrite(js.data(), 1, js.size(), sf); std::fclose(sf);
    }

    // Per-person Cryptomatte EXR (+ embedded skeleton attribute). When a refined
    // pooled matte (MatAnyone2/Kthanid) is supplied, partition IT per-person along
    // the pipeline cryptomatte's Voronoi ownership (the MattePartition core), so
    // external edges keep MatAnyone2's soft silhouette.
    const std::string cp = std::string(out_dir) + "/crypto_" + pad4(f) + ".exr";
    bool wrote = false;
    bool refined = false;
    if (refined_pat) {
      const std::string rp = fmt(refined_pat, f);
      int rw = 0, rh = 0, rc = 0;
      unsigned char* rpx = stbi_load(rp.c_str(), &rw, &rh, &rc, 1);
      if (rpx && rw == W && rh == H) {
        std::vector<float> rpool(static_cast<size_t>(W) * H);
        for (size_t i = 0, n = rpool.size(); i < n; ++i) rpool[i] = rpx[i] / 255.f;
        // Labels from the pipeline manifest ("person_NN" keys) so the ids match
        // the packed layers.
        std::vector<hastur::CryptoLabel> ids;
        const std::string& mf = fr.crypto.manifest;
        for (size_t pp = mf.find("\"person_"); pp != std::string::npos;
             pp = mf.find("\"person_", pp)) {
          const size_t q = mf.find('"', pp + 1);
          if (q == std::string::npos) break;
          const std::string name = mf.substr(pp + 1, q - pp - 1);
          ids.push_back({name, hastur::CryptoIdFloat(name)});
          pp = q + 1;
        }
        hastur::CryptoResult ref = hastur::PartitionCryptomatte(
            fr.crypto.layers, ids, rpool, W, H, max_dist, p.crypto_levels);
        wrote = hastur::WriteCryptoExr(cp, rpool, ref, p.crypto_levels,
                                       hastur::ExrCompression::kZIP, js);
        refined = true;
      } else {
        std::fprintf(stderr, "frame %d: refined matte %s missing/mismatched; using pipeline crypto\n", f, rp.c_str());
      }
      if (rpx) stbi_image_free(rpx);
    }
    if (!refined)
      wrote = hastur::WriteCryptoExr(cp, pool, fr.crypto, p.crypto_levels,
                                     hastur::ExrCompression::kZIP, js);
    std::fprintf(stderr, "frame %d: %zu people, crypto=%s%s skel=%s\n", f,
                 fr.people.size(), wrote ? "ok" : "FAIL",
                 refined ? " (refined)" : "", js.empty() ? "-" : "ok");
    if (wrote) ++ok;
  }
  std::fprintf(stderr, "hastur_export: %d/%d frames written to %s\n", ok,
               last - first + 1, out_dir);
  return ok > 0 ? 0 : 1;
}
