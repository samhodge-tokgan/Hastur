// ============================================================================
// VENDORED FROM Rotobot-Next — DO NOT EDIT HERE.
//
// Sealing and opening must agree byte for byte. If this copy drifts from
// Rotobot-Next/src/license/Crypto.h, a sealed model tree written by rotobot_model_seal
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
// Pure licence primitives: encoding, hashing, signature and AEAD verification,
// plus the two small parsers the licence payload needs.
//
// Split out of Keygen.cpp so it can be linked WITHOUT libcurl — that is what
// lets rotobot_selftest exercise the crypto from fixtures (see tests/selftest.cpp,
// section_license_crypto). Everything here is deterministic and side-effect free
// apart from random_hex(); no network, no getenv, no clock.

#pragma once
#include <cstdint>
#include <string>

namespace rotobot {
namespace license {

// ---- text ----
std::string strip(const std::string& s);

// ---- encoding ----
// Was decode-only until model sealing (#43) needed to WRITE an envelope.
// Standard alphabet, whitespace- and padding-tolerant.
bool b64_decode(const std::string& in, std::string& out);
// URL alphabet (-_), padding optional. Also accepts the standard alphabet: we
// always verify signatures over the received *string*, so accepting both on
// decode cannot widen what verifies. Rejects len % 4 == 1.
bool b64url_decode(const std::string& in, std::string& out);
bool hex_decode(const std::string& hex, std::string& out);
std::string hex_encode(const std::string& raw);
// Standard alphabet, padded, no line breaks.
std::string b64_encode(const std::string& raw);

// ---- crypto ----
std::string sha256(const std::string& in);
// Constant-time equality. Unequal lengths compare false without leaking where.
bool ct_eq(const std::string& a, const std::string& b);
bool ed25519_verify(const std::string& pub_raw, const std::string& msg,
                    const std::string& sig);
bool aes256gcm_decrypt(const std::string& key, const std::string& iv,
                       const std::string& ct, const std::string& tag,
                       std::string& out);
// Seals `pt` under `key` (32 bytes). `iv` is generated here rather than taken
// from the caller: a repeated GCM nonce under the same key is catastrophic, and
// the sealing tools would otherwise have to be trusted to get it right.
// Returns false if the RNG or the cipher fails; never on a short read.
bool aes256gcm_encrypt(const std::string& key, const std::string& pt,
                       std::string& iv, std::string& ct, std::string& tag);

std::string hmac_sha256(const std::string& key, const std::string& msg);
// HKDF-SHA256 (RFC 5869), `len` bytes. Used to fan one secret out into
// per-purpose subkeys so a leak of one does not imply the others.
std::string hkdf_sha256(const std::string& ikm, const std::string& salt,
                        const std::string& info, size_t len);
// n random bytes, lowercase hex. Empty string if the RNG fails — callers MUST
// treat that as fatal rather than proceeding with a predictable value.
std::string random_hex(size_t nbytes);

// ---- parsers ----
// Strip the "-----BEGIN LICENSE FILE-----" envelope, tolerating a double-base64
// outer layer (the relay wire format wraps it once more).
std::string strip_envelope(const std::string& license_file);

// Keygen expiry timestamps: "2026-08-31T00:00:00.000Z" (fractional seconds and
// the trailing Z both optional; only UTC is accepted — a licence with a local
// offset would be ambiguous, so we refuse it rather than guess).
bool parse_iso8601_utc(const std::string& s, int64_t& out_epoch);

}  // namespace license
}  // namespace rotobot
