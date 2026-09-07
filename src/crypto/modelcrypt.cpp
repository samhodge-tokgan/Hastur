// ============================================================================
// VENDORED FROM Rotobot-Next — DO NOT EDIT HERE.
//
// Sealing and opening must agree byte for byte. If this copy drifts from
// Rotobot-Next/src/crypto/modelcrypt.cpp, a sealed model tree written by rotobot_model_seal
// stops opening in Hastur, and the failure looks like a corrupt artifact
// rather than a source divergence.
//
// Edit the original, then re-copy. dev/sync-modelcrypt.sh checks they match,
// and the decoy-digest self-test in ModelBytes.cpp fails the build if the key
// derivation itself has diverged.
// ============================================================================
// SPDX-License-Identifier: LicenseRef-Tokgan-Proprietary
// Copyright (c) Tokgan. Proprietary — licensed under the Tokgan EULA; see NOTICE.
#include "crypto/modelcrypt.h"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <cstdlib>
#include <random>
#include <sstream>

#include "license/Crypto.h"
#include "nlohmann/json.hpp"

namespace fs = std::filesystem;
using rotobot::license::aes256gcm_decrypt;
using rotobot::license::aes256gcm_encrypt;
using rotobot::license::hex_encode;
using rotobot::license::hkdf_sha256;
using rotobot::license::hmac_sha256;
using rotobot::license::sha256;

namespace rotobot {
namespace modelcrypt {
namespace {

// Domain separation. Every derivation names its purpose so that two different
// uses of the same secret can never collide.
constexpr const char* kInfoFilename = "rbm/v1/filename";
constexpr const char* kInfoFile = "rbm/v1/file:";
constexpr const char* kInfoContent = "rbm/v1/content:";
constexpr const char* kInfoPms = "rbm/v1/pms";

// The decoy constants. Innocuous-looking strings that read as ordinary product
// vocabulary in a `strings` dump. See decoy_material() in the header for why
// this is obfuscation rather than security, and why they must never change.
constexpr const char* kD1 = "colorspace/aces-ap0-linear";
constexpr const char* kD2 = "tile:2048x2048:premul";
constexpr const char* kD3 = "sched/interleave-stride-16";
constexpr const char* kD4 = "aov/motion-vector-fwd";

std::string read_file(const std::string& p, bool* ok = nullptr) {
  std::ifstream f(p, std::ios::binary);
  if (!f) { if (ok) *ok = false; return std::string(); }
  std::ostringstream ss;
  ss << f.rdbuf();
  if (ok) *ok = true;
  return ss.str();
}

std::string h(const std::string& s) { return hex_encode(s); }

bool unhex(const std::string& s, std::string& out) {
  return rotobot::license::hex_decode(s, out);
}

}  // namespace

// --------------------------------------------------------------------------

Key decoy_material() {
  // Interleaved rather than concatenated so no single constant sits adjacent to
  // the derivation in a disassembly.
  std::string mixed;
  const std::string a(kD1), b(kD2), c(kD3), d(kD4);
  const size_t n = std::max(std::max(a.size(), b.size()), std::max(c.size(), d.size()));
  for (size_t i = 0; i < n; ++i) {
    if (i < a.size()) mixed.push_back(a[i]);
    if (i < c.size()) mixed.push_back(c[i]);
    if (i < b.size()) mixed.push_back(b[i]);
    if (i < d.size()) mixed.push_back(d[i]);
  }
  return sha256(mixed);
}

std::string decoy_digest_hex() { return h(sha256(decoy_material())); }

// --------------------------------------------------------------------------

std::string Sealed::to_json() const {
  nlohmann::json j;
  j["iv"] = h(iv);
  j["ct"] = h(ct);
  j["tag"] = h(tag);
  return j.dump();
}

bool Sealed::from_json(const std::string& json, Sealed& out) {
  try {
    const auto j = nlohmann::json::parse(json);
    return unhex(j.value("iv", ""), out.iv) && unhex(j.value("ct", ""), out.ct) &&
           unhex(j.value("tag", ""), out.tag) && out.iv.size() == 12 &&
           out.tag.size() == 16;
  } catch (...) {
    return false;
  }
}

// --------------------------------------------------------------------------

Key unwrap_pms(const std::string& licence_text, const std::string& licence_key,
               const std::string& generation) {
  // The licence payload: strip the envelope, base64, then the AES-GCM segment
  // keyed by SHA256(licence_key) — the same shape verify_offline() reads.
  const std::string body_b64 = license::strip_envelope(licence_text);
  std::string body;
  if (!license::b64_decode(body_b64, body)) return Key();
  nlohmann::json payload, doc;
  try { payload = nlohmann::json::parse(body); } catch (...) { return Key(); }
  const std::string enc = payload.value("enc", "");
  const size_t d1 = enc.find('.'), d2 = enc.rfind('.');
  if (d1 == std::string::npos || d1 == d2) return Key();
  std::string ct, iv, tag, plain;
  if (!license::b64_decode(enc.substr(0, d1), ct) ||
      !license::b64_decode(enc.substr(d1 + 1, d2 - d1 - 1), iv) ||
      !license::b64_decode(enc.substr(d2 + 1), tag) ||
      !aes256gcm_decrypt(sha256(licence_key), iv, ct, tag, plain)) {
    return Key();
  }
  try { doc = nlohmann::json::parse(plain); } catch (...) { return Key(); }

  const auto meta = doc.value("data", nlohmann::json::object())
                        .value("attributes", nlohmann::json::object())
                        .value("metadata", nlohmann::json::object());
  if (!meta.contains("model_pms")) return Key();
  const auto& gens = meta["model_pms"];
  if (!gens.is_object() || !gens.contains(generation)) return Key();

  Sealed s;
  const auto& g = gens[generation];
  if (!Sealed::from_json(g.is_string() ? g.get<std::string>() : g.dump(), s)) return Key();

  // Wrapped under SHA256(licence_key) XOR decoy_material(): a leaked .lic alone
  // does not unwrap without the constants compiled into the binary.
  std::string wrap = sha256(licence_key);
  const Key decoy = decoy_material();
  for (size_t i = 0; i < wrap.size() && i < decoy.size(); ++i) wrap[i] ^= decoy[i];

  std::string pms;
  if (!aes256gcm_decrypt(wrap, s.iv, s.ct, s.tag, pms) || pms.size() != 32) return Key();
  return pms;
}

Sealed wrap_content_key(const Key& ck, const Key& pms, const std::string& artifact) {
  Sealed s;
  const Key k = hkdf_sha256(pms, std::string(), std::string(kInfoContent) + artifact, 32);
  if (k.size() != 32) return s;
  if (!aes256gcm_encrypt(k, ck, s.iv, s.ct, s.tag)) return Sealed();
  return s;
}

Key unwrap_content_key(const Sealed& enc_ck, const Key& pms,
                       const std::string& artifact) {
  const Key k = hkdf_sha256(pms, std::string(), std::string(kInfoContent) + artifact, 32);
  if (k.size() != 32) return Key();
  std::string ck;
  if (!aes256gcm_decrypt(k, enc_ck.iv, enc_ck.ct, enc_ck.tag, ck) || ck.size() != 32)
    return Key();
  return ck;
}

Key file_subkey(const Key& ck, const std::string& logical) {
  return hkdf_sha256(ck, std::string(), std::string(kInfoFile) + logical, 32);
}

std::string derived_filename(const Key& ck, const std::string& logical) {
  const Key name_key = hkdf_sha256(ck, std::string(), kInfoFilename, 32);
  if (name_key.empty()) return std::string();
  return h(hmac_sha256(name_key, logical)).substr(0, 32) + ".rbm";
}

// --------------------------------------------------------------------------

std::string Header::to_json() const {
  nlohmann::json j;
  j["v"] = version;
  j["artifact"] = artifact;
  j["generation"] = generation;
  j["enc_ck"] = nlohmann::json::parse(enc_ck.to_json());
  return j.dump();
}

bool Header::from_json(const std::string& json, Header& out) {
  try {
    const auto j = nlohmann::json::parse(json);
    out.version = j.value("v", 0);
    out.artifact = j.value("artifact", "");
    out.generation = j.value("generation", "");
    return out.version == 1 && !out.artifact.empty() && !out.generation.empty() &&
           j.contains("enc_ck") && Sealed::from_json(j["enc_ck"].dump(), out.enc_ck);
  } catch (...) {
    return false;
  }
}

// --------------------------------------------------------------------------

std::unique_ptr<SealedTree> SealedTree::open(const std::string& dir,
                                             const std::string& licence_text,
                                             const std::string& licence_key,
                                             std::string& err) {
  std::unique_ptr<SealedTree> t(new SealedTree());
  t->dir_ = dir;

  const std::string hdr_path = dir + "/" + kHeaderName;
  std::error_code ec;
  if (!fs::exists(hdr_path, ec)) {
    t->sealed_ = false;  // plaintext tree — the pre-#43 layout, still supported
    return t;
  }

  Header hdr;
  bool ok = false;
  const std::string raw = read_file(hdr_path, &ok);
  if (!ok || !Header::from_json(raw, hdr)) {
    err = "model tree at " + dir + " has an unreadable " + kHeaderName;
    return nullptr;
  }
  if (licence_text.empty() || licence_key.empty()) {
    err = "the model tree at " + dir +
          " is encrypted and needs a licence to open.\n"
          "  Set ROTOBOT_NEXT_LICENSE_PATH (and its sibling .key), or use an "
          "unencrypted model tree.";
    return nullptr;
  }
  const Key pms = unwrap_pms(licence_text, licence_key, hdr.generation);
  if (pms.empty()) {
    err = "this licence cannot open the model tree at " + dir +
          ": it carries no key for generation '" + hdr.generation +
          "'.\n  The models were sealed for a different licence generation — ask "
          "Tokgan to re-issue.";
    return nullptr;
  }
  t->ck_ = unwrap_content_key(hdr.enc_ck, pms, hdr.artifact);
  if (t->ck_.empty()) {
    err = "model tree at " + dir + " failed to unwrap (artifact '" + hdr.artifact +
          "'). The header does not match the licence generation it names.";
    return nullptr;
  }
  t->sealed_ = true;
  return t;
}

std::unique_ptr<SealedTree> SealedTree::open_from_env(const std::string& dir,
                                                     std::string& err) {
  const char* lp = std::getenv("ROTOBOT_NEXT_LICENSE_PATH");
  std::string text, key;
  if (lp && *lp) {
    bool ok = false;
    text = read_file(lp, &ok);
    if (!ok) text.clear();
    const char* k = std::getenv("ROTOBOT_NEXT_LICENSE_KEY");
    if (k && *k) {
      key = k;
    } else {
      // Sibling <path>.key — the same fallback check_license() uses.
      bool kok = false;
      const std::string raw = read_file(std::string(lp) + ".key", &kok);
      if (kok) key = license::strip(raw);
    }
  }
  return open(dir, text, key, err);
}

bool SealedTree::has(const std::string& logical) const {
  std::error_code ec;
  if (!sealed_) return fs::exists(dir_ + "/" + logical, ec);
  return fs::exists(dir_ + "/" + derived_filename(ck_, logical), ec);
}

bool SealedTree::read(const std::string& logical, std::string& out,
                      std::string& err) const {
  if (!sealed_) {
    bool ok = false;
    out = read_file(dir_ + "/" + logical, &ok);
    if (!ok) err = "cannot read " + dir_ + "/" + logical;
    return ok;
  }
  const std::string path = dir_ + "/" + derived_filename(ck_, logical);
  bool ok = false;
  const std::string blob = read_file(path, &ok);
  if (!ok) { err = "sealed model '" + logical + "' is missing from " + dir_; return false; }
  if (blob.size() < 28) { err = "sealed model '" + logical + "' is truncated"; return false; }

  // Layout: iv(12) || tag(16) || ciphertext
  const std::string iv = blob.substr(0, 12), tag = blob.substr(12, 16),
                    ct = blob.substr(28);
  if (!aes256gcm_decrypt(file_subkey(ck_, logical), iv, ct, tag, out)) {
    err = "sealed model '" + logical + "' failed authentication — the file is "
          "corrupt or was sealed under a different key";
    return false;
  }
  return true;
}

bool SealedTree::materialise(const std::vector<std::string>& logicals,
                             std::string& dir_out, std::string& err) {
  if (!sealed_) { dir_out = dir_; return true; }
  if (!scratch_.empty()) { dir_out = scratch_; return true; }

  // RAM-backed, or not at all. The first draft of this fell back to
  // temp_directory_path() when /dev/shm was absent, which silently wrote
  // plaintext weights to a disk — the exact thing the feature exists to
  // prevent, on exactly the platforms (macOS, Windows) where the fallback
  // would fire. Refuse instead, and say why.
  //
  // /dev/shm is also only 64 MB inside a container by default, and the
  // tracker's external data is ~2 GB, so a pod without a memory-backed volume
  // fails below rather than here. That failure is deliberate too.
  std::error_code ec;
  std::string base;
  if (fs::exists("/dev/shm", ec)) {
    base = "/dev/shm";
  } else if (std::getenv("ROTOBOT_MODELS_ALLOW_DISK_SCRATCH")) {
    base = fs::temp_directory_path(ec).string();
    std::fprintf(stderr,
                 "[modelcrypt] WARNING: no /dev/shm; decrypting models to %s, "
                 "which is a DISK. Plaintext weights will exist as files until "
                 "this process exits. Set by "
                 "ROTOBOT_MODELS_ALLOW_DISK_SCRATCH.\n",
                 base.c_str());
  } else {
    err = "this model tree needs a RAM-backed scratch directory and there is no "
          "/dev/shm on this platform.\n"
          "  Models with external data cannot be loaded from memory (ORT cannot "
          "resolve an .onnx.data sidecar from a buffer), so they are decrypted "
          "to tmpfs instead.\n"
          "  Sealed model trees are a Linux feature today; use an unsealed tree "
          "here, or set ROTOBOT_MODELS_ALLOW_DISK_SCRATCH=1 to accept plaintext "
          "weights on disk for the life of the process.";
    return false;
  }
  std::random_device rd;
  scratch_ = base + "/.rbm-" + h(sha256(std::to_string(rd()) + dir_)).substr(0, 16);
  if (!fs::create_directory(scratch_, ec)) {
    err = "cannot create a scratch directory at " + scratch_ + ": " + ec.message();
    scratch_.clear();
    return false;
  }
  fs::permissions(scratch_, fs::perms::owner_all, ec);

  for (const std::string& logical : logicals) {
    std::string blob;
    if (!read(logical, blob, err)) return false;
    std::ofstream o(scratch_ + "/" + logical, std::ios::binary);
    if (!o) {
      err = "cannot write " + logical + " into " + scratch_ +
            " (is /dev/shm large enough? a container defaults to 64 MB)";
      return false;
    }
    o.write(blob.data(), (std::streamsize)blob.size());
    if (!o) {
      err = "short write of " + logical + " into " + scratch_ +
            " — /dev/shm is too small for the model set";
      return false;
    }
  }
  dir_out = scratch_;
  return true;
}

SealedTree::~SealedTree() {
  if (scratch_.empty()) return;
  // Overwrite before unlinking: tmpfs pages are RAM, but a page that is merely
  // freed can still be read by whoever gets it next under memory pressure.
  std::error_code ec;
  for (const auto& e : fs::directory_iterator(scratch_, ec)) {
    std::error_code e2;
    const auto n = fs::file_size(e.path(), e2);
    if (!e2) {
      std::ofstream z(e.path(), std::ios::binary | std::ios::in);
      const std::string zeros(1 << 20, '\0');
      for (uintmax_t w = 0; w < n; w += zeros.size())
        z.write(zeros.data(), (std::streamsize)std::min<uintmax_t>(zeros.size(), n - w));
    }
  }
  fs::remove_all(scratch_, ec);
}

// --------------------------------------------------------------------------

bool seal_tree(const std::string& src_dir, const std::string& out_dir, const Key& pms,
               const std::string& artifact, const std::string& generation, Key& ck_out,
               std::string& err) {
  std::error_code ec;
  if (!fs::is_directory(src_dir, ec)) { err = src_dir + " is not a directory"; return false; }

  std::string ck(32, '\0');
  const std::string rnd = license::random_hex(32);
  if (rnd.size() != 64 || !unhex(rnd, ck)) { err = "RNG failed"; return false; }

  fs::create_directories(out_dir, ec);
  size_t n = 0;
  for (const auto& e : fs::recursive_directory_iterator(src_dir, ec)) {
    if (!e.is_regular_file()) continue;
    const std::string logical = fs::relative(e.path(), src_dir, ec).string();
    if (logical == kHeaderName) continue;

    bool ok = false;
    const std::string pt = read_file(e.path().string(), &ok);
    if (!ok) { err = "cannot read " + e.path().string(); return false; }
    std::string iv, ct, tag;
    if (!aes256gcm_encrypt(file_subkey(ck, logical), pt, iv, ct, tag)) {
      err = "encrypt failed for " + logical;
      return false;
    }
    std::ofstream o(out_dir + "/" + derived_filename(ck, logical), std::ios::binary);
    o << iv << tag << ct;
    if (!o) { err = "cannot write the sealed form of " + logical; return false; }
    ++n;
  }
  if (n == 0) { err = "no files found under " + src_dir; return false; }

  Header hdr;
  hdr.artifact = artifact;
  hdr.generation = generation;
  hdr.enc_ck = wrap_content_key(ck, pms, artifact);
  if (hdr.enc_ck.ct.empty()) { err = "failed to wrap the content key"; return false; }
  std::ofstream h(out_dir + "/" + kHeaderName);
  h << hdr.to_json() << "\n";
  if (!h) { err = "cannot write " + std::string(kHeaderName); return false; }

  ck_out = ck;
  return true;
}

}  // namespace modelcrypt
}  // namespace rotobot
