// ============================================================================
// VENDORED FROM Rotobot-Next — DO NOT EDIT HERE.
//
// Sealing and opening must agree byte for byte. If this copy drifts from
// Rotobot-Next/src/crypto/modelcrypt.h, a sealed model tree written by rotobot_model_seal
// stops opening in Hastur, and the failure looks like a corrupt artifact
// rather than a source divergence.
//
// Edit the original, then re-copy. dev/sync-modelcrypt.sh checks they match,
// and the decoy-digest self-test in ModelBytes.cpp fails the build if the key
// derivation itself has diverged.
// ============================================================================
// SPDX-License-Identifier: LicenseRef-Tokgan-Proprietary
// Copyright (c) Tokgan. Proprietary — licensed under the Tokgan EULA; see NOTICE.
//
// Model weights at rest: encrypted, under derived filenames, decrypted
// just-in-time (issue #43).
//
// THREAT MODEL, deliberately modest. On-device inference means the weights MUST
// reach RAM, so this raises the reverse-engineering floor — it defeats casual
// copying, filename inspection, and use without a licence. It does not defeat a
// determined attacker with root who can dump process memory. Same A3 boundary
// the relay-identity spec sets out; do not describe it as more than that.
//
// KEY SCHEME — sealed once per artifact, never once per customer.
//
//   licence   ->  PMS      a stable product secret, in metadata.model_pms,
//                          keyed by generation. Issued once per customer.
//   artifact  ->  sealed.hdr carrying enc_CK = AESGCM(HKDF(PMS, artifact), CK)
//
// A new model release is sealed once and shipped; no licence is touched,
// because the artifact carries the key material needed to open itself. The
// trade is that a leaked licence exposes every artifact under that PMS
// generation rather than one version — which is why generations exist, and why
// rotating one is a deliberate act rather than routine.
//
// Filenames are derived, not mapped: hex(HMAC(name_key, logical))[:32] + ".rbm",
// computed identically at seal and load time, so there is no manifest to read
// and no plaintext model name in the binary or on disk.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace rotobot {
namespace modelcrypt {

// 32 raw bytes each.
using Key = std::string;

// The wrapped-key envelope, used for both PMS-in-licence and CK-in-artifact.
struct Sealed {
  std::string iv, ct, tag;  // raw bytes
  std::string to_json() const;
  static bool from_json(const std::string& json, Sealed& out);
};

// ---- key derivation -------------------------------------------------------

// The obfuscation layer. Assembles key material from innocuous-looking string
// constants scattered through the binary. This is OBFUSCATION, not
// cryptography: it does not add entropy an attacker cannot recover by reading
// the binary they were shipped. Its only job is that a leaked .lic plus public
// knowledge of AES-GCM is not enough to write a working unwrap script without
// reverse-engineering — the casual-copying bar this whole feature targets.
//
// DO NOT CHANGE THESE AFTER THE FIRST RELEASE. Every issued licence is wrapped
// under material derived from them; changing one invalidates all of them.
Key decoy_material();

// Digest of decoy_material(), for the cross-repo drift self-test. This file is
// vendored byte-identically into Hastur; if the two copies disagree, sealing
// and opening disagree, and the failure would surface as an unopenable model
// rather than as a build error. So make it a build error.
std::string decoy_digest_hex();

// PMS out of a licence's metadata.model_pms for `generation`. Returns empty on
// any failure — a wrong licence_key, an absent generation, a corrupt envelope.
Key unwrap_pms(const std::string& licence_text, const std::string& licence_key,
               const std::string& generation);

// CK out of a sealed.hdr. `artifact` binds the wrap to one artifact version, so
// a header lifted between releases does not open.
Key unwrap_content_key(const Sealed& enc_ck, const Key& pms,
                       const std::string& artifact);
Sealed wrap_content_key(const Key& ck, const Key& pms, const std::string& artifact);

Key file_subkey(const Key& ck, const std::string& logical);
std::string derived_filename(const Key& ck, const std::string& logical);

// ---- the artifact header --------------------------------------------------

struct Header {
  int version = 1;
  std::string artifact;    // e.g. "2026.09.1" — binds enc_ck
  std::string generation;  // which PMS generation opens it
  Sealed enc_ck;

  std::string to_json() const;
  static bool from_json(const std::string& json, Header& out);
};

constexpr const char* kHeaderName = "sealed.hdr";

// ---- reading a model tree -------------------------------------------------

// A model directory, sealed or plaintext. Which one it is depends on whether
// sealed.hdr is present, and callers do not branch on it: that is what lets the
// sealed rollout land one loader at a time without a flag day.
class SealedTree {
 public:
  // `licence_text`/`licence_key` may be empty for a plaintext tree. For a
  // sealed tree they are required, and a licence failure here is a hard stop on
  // model load rather than a warning — say so in the error.
  static std::unique_ptr<SealedTree> open(const std::string& dir,
                                          const std::string& licence_text,
                                          const std::string& licence_key,
                                          std::string& err);

  bool sealed() const { return sealed_; }
  const std::string& dir() const { return dir_; }

  // Is `logical` present? Replaces the std::filesystem::exists() probes that
  // pick fp16 variants — those cannot work against derived names.
  bool has(const std::string& logical) const;

  // Decrypt `logical` into a heap buffer. Plaintext trees read the file.
  bool read(const std::string& logical, std::string& out, std::string& err) const;

  // Resolve `logical` to a path that ORT can open directly. On a sealed tree
  // this decrypts into a RAM-backed scratch directory owned by this object and
  // shredded on destruction — needed for models with external data
  // (`.onnx.data`), which CreateSessionFromArray cannot resolve.
  bool materialise(const std::vector<std::string>& logicals, std::string& dir_out,
                   std::string& err);

  ~SealedTree();

  // Convenience for stage binaries. Resolves the licence the way check_license()
  // does — ROTOBOT_NEXT_LICENSE_PATH plus ROTOBOT_NEXT_LICENSE_KEY or a sibling
  // <path>.key — and opens `dir` with it.
  //
  // Each stage doing this for itself is deliberate: the content key never
  // crosses a process boundary, so there is nothing in argv, env or an
  // inherited fd for a `ps` or a /proc read to pick up. A plaintext tree opens
  // regardless, so a stage that never sees a licence still works until the
  // sealed-only flip.
  static std::unique_ptr<SealedTree> open_from_env(const std::string& dir,
                                                   std::string& err);

 private:
  SealedTree() = default;
  bool sealed_ = false;
  std::string dir_;
  Key ck_;
  std::string scratch_;  // /dev/shm/... when materialise() was used
};

// ---- sealing (Tokgan-side) ------------------------------------------------

// Seal `src_dir` into `out_dir` under a fresh random CK. Returns the CK so the
// caller can report it; the header is written into out_dir.
bool seal_tree(const std::string& src_dir, const std::string& out_dir,
               const Key& pms, const std::string& artifact,
               const std::string& generation, Key& ck_out, std::string& err);

}  // namespace modelcrypt
}  // namespace rotobot
