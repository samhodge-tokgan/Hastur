// ============================================================================
// VENDORED FROM Rotobot-Next — DO NOT EDIT HERE.
//
// Sealing and opening must agree byte for byte. If this copy drifts from
// Rotobot-Next/src/license/Crypto.cpp, a sealed model tree written by rotobot_model_seal
// stops opening in Hastur, and the failure looks like a corrupt artifact
// rather than a source divergence.
//
// Edit the original, then re-copy. dev/sync-modelcrypt.sh checks they match,
// and the decoy-digest self-test in ModelBytes.cpp fails the build if the key
// derivation itself has diverged.
// ============================================================================
// SPDX-License-Identifier: LicenseRef-Tokgan-Proprietary
// Copyright (c) Tokgan. Proprietary — licensed under the Tokgan EULA; see NOTICE.
#include "license/Crypto.h"

#include <cstdio>
#include <cstring>
#include <ctime>

#include <openssl/crypto.h>
#include <openssl/hmac.h>
#include <openssl/kdf.h>
#include <openssl/evp.h>
#include <openssl/rand.h>
#include <openssl/sha.h>

namespace rotobot {
namespace license {
namespace {

// Shared decode table builder: `url` swaps 62/63 to '-' and '_'.
const int8_t* b64_table(bool url) {
  static int8_t std_t[256], url_t[256];
  static bool init = false;
  if (!init) {
    for (int i = 0; i < 256; ++i) std_t[i] = url_t[i] = -1;
    const char* a =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    for (int i = 0; i < 64; ++i) std_t[(unsigned char)a[i]] = (int8_t)i;
    std::memcpy(url_t, std_t, sizeof(url_t));
    // URL alphabet, plus the standard one kept live (see the header note).
    url_t[(unsigned char)'-'] = 62;
    url_t[(unsigned char)'_'] = 63;
    init = true;
  }
  return url ? url_t : std_t;
}

bool b64_decode_impl(const std::string& in, std::string& out, bool url) {
  const int8_t* T = b64_table(url);
  out.clear();
  int val = 0, bits = 0;
  size_t ndigits = 0;
  for (unsigned char c : in) {
    if (c == '=' || c == '\n' || c == '\r' || c == ' ' || c == '\t') continue;
    int8_t d = T[c];
    if (d < 0) return false;
    ++ndigits;
    val = (val << 6) | d;
    bits += 6;
    if (bits >= 8) {
      bits -= 8;
      out.push_back((char)((val >> bits) & 0xFF));
    }
  }
  // A single leftover digit encodes nothing and can only come from a truncated
  // or malformed segment.
  if (ndigits % 4 == 1) return false;
  return true;
}

}  // namespace

std::string strip(const std::string& s) {
  size_t a = s.find_first_not_of(" \t\r\n");
  size_t b = s.find_last_not_of(" \t\r\n");
  return (a == std::string::npos) ? "" : s.substr(a, b - a + 1);
}

bool b64_decode(const std::string& in, std::string& out) {
  return b64_decode_impl(in, out, /*url=*/false);
}

bool b64url_decode(const std::string& in, std::string& out) {
  return b64_decode_impl(in, out, /*url=*/true);
}

bool hex_decode(const std::string& hex, std::string& out) {
  if (hex.size() % 2) return false;
  out.clear();
  auto nib = [](char c) -> int {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
  };
  for (size_t i = 0; i < hex.size(); i += 2) {
    int hi = nib(hex[i]), lo = nib(hex[i + 1]);
    if (hi < 0 || lo < 0) return false;
    out.push_back((char)((hi << 4) | lo));
  }
  return true;
}

std::string hex_encode(const std::string& raw) {
  static const char* d = "0123456789abcdef";
  std::string out;
  out.reserve(raw.size() * 2);
  for (unsigned char c : raw) {
    out.push_back(d[c >> 4]);
    out.push_back(d[c & 0x0F]);
  }
  return out;
}

std::string sha256(const std::string& in) {
  unsigned char h[SHA256_DIGEST_LENGTH];
  SHA256((const unsigned char*)in.data(), in.size(), h);
  return std::string((char*)h, SHA256_DIGEST_LENGTH);
}

bool ct_eq(const std::string& a, const std::string& b) {
  if (a.size() != b.size()) return false;
  if (a.empty()) return true;
  return CRYPTO_memcmp(a.data(), b.data(), a.size()) == 0;
}

bool ed25519_verify(const std::string& pub_raw, const std::string& msg,
                    const std::string& sig) {
  if (pub_raw.size() != 32) return false;
  EVP_PKEY* pkey = EVP_PKEY_new_raw_public_key(
      EVP_PKEY_ED25519, nullptr, (const unsigned char*)pub_raw.data(), 32);
  if (!pkey) return false;
  EVP_MD_CTX* ctx = EVP_MD_CTX_new();
  bool ok = false;
  if (ctx && EVP_DigestVerifyInit(ctx, nullptr, nullptr, nullptr, pkey) == 1) {
    ok = EVP_DigestVerify(ctx, (const unsigned char*)sig.data(), sig.size(),
                          (const unsigned char*)msg.data(), msg.size()) == 1;
  }
  if (ctx) EVP_MD_CTX_free(ctx);
  EVP_PKEY_free(pkey);
  return ok;
}

std::string b64_encode(const std::string& raw) {
  if (raw.empty()) return std::string();
  std::string out(4 * ((raw.size() + 2) / 3) + 1, '\0');
  const int n = EVP_EncodeBlock(reinterpret_cast<unsigned char*>(&out[0]),
                                reinterpret_cast<const unsigned char*>(raw.data()),
                                static_cast<int>(raw.size()));
  out.resize(n > 0 ? static_cast<size_t>(n) : 0);
  return out;
}

bool aes256gcm_encrypt(const std::string& key, const std::string& pt,
                       std::string& iv, std::string& ct, std::string& tag) {
  if (key.size() != 32) return false;
  // 96-bit nonce, generated here. GCM nonce reuse under one key destroys both
  // confidentiality and authenticity, so this is not a caller's choice to make.
  iv.assign(12, '\0');
  if (RAND_bytes(reinterpret_cast<unsigned char*>(&iv[0]), 12) != 1) return false;

  EVP_CIPHER_CTX* c = EVP_CIPHER_CTX_new();
  if (!c) return false;
  bool ok = false;
  ct.assign(pt.size(), '\0');
  tag.assign(16, '\0');
  int len = 0, total = 0;
  if (EVP_EncryptInit_ex(c, EVP_aes_256_gcm(), nullptr, nullptr, nullptr) == 1 &&
      EVP_CIPHER_CTX_ctrl(c, EVP_CTRL_GCM_SET_IVLEN, 12, nullptr) == 1 &&
      EVP_EncryptInit_ex(c, nullptr, nullptr,
                         reinterpret_cast<const unsigned char*>(key.data()),
                         reinterpret_cast<const unsigned char*>(iv.data())) == 1) {
    if (pt.empty() ||
        EVP_EncryptUpdate(c, reinterpret_cast<unsigned char*>(&ct[0]), &len,
                          reinterpret_cast<const unsigned char*>(pt.data()),
                          static_cast<int>(pt.size())) == 1) {
      total = len;
      if (EVP_EncryptFinal_ex(c, reinterpret_cast<unsigned char*>(&ct[0]) + total,
                              &len) == 1) {
        total += len;
        ct.resize(static_cast<size_t>(total));
        ok = EVP_CIPHER_CTX_ctrl(c, EVP_CTRL_GCM_GET_TAG, 16, &tag[0]) == 1;
      }
    }
  }
  EVP_CIPHER_CTX_free(c);
  if (!ok) { ct.clear(); tag.clear(); iv.clear(); }
  return ok;
}

std::string hmac_sha256(const std::string& key, const std::string& msg) {
  unsigned char out[32];
  unsigned int n = 0;
  if (!HMAC(EVP_sha256(), key.data(), static_cast<int>(key.size()),
            reinterpret_cast<const unsigned char*>(msg.data()), msg.size(), out, &n) ||
      n != 32) {
    return std::string();
  }
  return std::string(reinterpret_cast<char*>(out), 32);
}

std::string hkdf_sha256(const std::string& ikm, const std::string& salt,
                        const std::string& info, size_t len) {
  // EVP_PKEY_CTX rather than EVP_KDF: the shipped dist builds against the EL8
  // OpenSSL, and this spelling works from 1.1.0 through 3.x unchanged.
  EVP_PKEY_CTX* c = EVP_PKEY_CTX_new_id(EVP_PKEY_HKDF, nullptr);
  if (!c) return std::string();
  std::string out(len, '\0');
  size_t outlen = len;
  const bool ok =
      EVP_PKEY_derive_init(c) == 1 &&
      EVP_PKEY_CTX_set_hkdf_md(c, EVP_sha256()) == 1 &&
      EVP_PKEY_CTX_set1_hkdf_salt(
          c, reinterpret_cast<const unsigned char*>(salt.data()),
          static_cast<int>(salt.size())) == 1 &&
      EVP_PKEY_CTX_set1_hkdf_key(
          c, reinterpret_cast<const unsigned char*>(ikm.data()),
          static_cast<int>(ikm.size())) == 1 &&
      EVP_PKEY_CTX_add1_hkdf_info(
          c, reinterpret_cast<const unsigned char*>(info.data()),
          static_cast<int>(info.size())) == 1 &&
      EVP_PKEY_derive(c, reinterpret_cast<unsigned char*>(&out[0]), &outlen) == 1 &&
      outlen == len;
  EVP_PKEY_CTX_free(c);
  return ok ? out : std::string();
}

bool aes256gcm_decrypt(const std::string& key, const std::string& iv,
                       const std::string& ct, const std::string& tag,
                       std::string& out) {
  if (key.size() != 32) return false;
  EVP_CIPHER_CTX* c = EVP_CIPHER_CTX_new();
  if (!c) return false;
  bool ok = false;
  out.assign(ct.size(), '\0');
  int outl = 0, finl = 0;
  if (EVP_DecryptInit_ex(c, EVP_aes_256_gcm(), nullptr, nullptr, nullptr) == 1 &&
      EVP_CIPHER_CTX_ctrl(c, EVP_CTRL_GCM_SET_IVLEN, (int)iv.size(), nullptr) == 1 &&
      EVP_DecryptInit_ex(c, nullptr, nullptr, (const unsigned char*)key.data(),
                         (const unsigned char*)iv.data()) == 1 &&
      EVP_DecryptUpdate(c, (unsigned char*)out.data(), &outl,
                        (const unsigned char*)ct.data(), (int)ct.size()) == 1 &&
      EVP_CIPHER_CTX_ctrl(c, EVP_CTRL_GCM_SET_TAG, (int)tag.size(),
                          (void*)tag.data()) == 1 &&
      EVP_DecryptFinal_ex(c, (unsigned char*)out.data() + outl, &finl) == 1) {
    out.resize(outl + finl);
    ok = true;
  }
  EVP_CIPHER_CTX_free(c);
  return ok;
}

std::string random_hex(size_t nbytes) {
  std::string raw(nbytes, '\0');
  if (RAND_bytes((unsigned char*)&raw[0], (int)nbytes) != 1) return "";
  return hex_encode(raw);
}

const std::string kHeader = "-----BEGIN LICENSE FILE-----";
const std::string kFooter = "-----END LICENSE FILE-----";

std::string strip_envelope(const std::string& license_file) {
  std::string text = strip(license_file);
  if (text.rfind(kHeader, 0) != 0) {
    std::string decoded;
    if (b64_decode(text, decoded)) {
      std::string ds = strip(decoded);
      if (ds.rfind(kHeader, 0) == 0) text = ds;
    }
  }
  if (text.rfind(kHeader, 0) == 0) text = text.substr(kHeader.size());
  if (text.size() >= kFooter.size() &&
      text.compare(text.size() - kFooter.size(), kFooter.size(), kFooter) == 0)
    text = text.substr(0, text.size() - kFooter.size());
  return strip(text);
}

bool parse_iso8601_utc(const std::string& s, int64_t& out_epoch) {
  int y = 0, mo = 0, d = 0, h = 0, mi = 0, sec = 0;
  // "%n" is not portable in scanf; read the fixed prefix and inspect the rest.
  if (std::sscanf(s.c_str(), "%4d-%2d-%2dT%2d:%2d:%2d", &y, &mo, &d, &h, &mi, &sec) != 6)
    return false;
  if (s.size() < 19) return false;
  // Whatever follows the seconds must be UTC: nothing, "Z", or ".fff[Z]".
  std::string rest = s.substr(19);
  if (!rest.empty()) {
    size_t i = 0;
    if (rest[0] == '.') {
      ++i;
      while (i < rest.size() && rest[i] >= '0' && rest[i] <= '9') ++i;
      if (i == 1) return false;  // a lone '.' with no digits
    }
    std::string z = rest.substr(i);
    if (!(z.empty() || z == "Z" || z == "z" || z == "+00:00" || z == "-00:00"))
      return false;  // a real offset — refuse rather than silently mis-read it
  }
  if (mo < 1 || mo > 12 || d < 1 || d > 31 || h > 23 || mi > 59 || sec > 60) return false;
  std::tm tm{};
  tm.tm_year = y - 1900;
  tm.tm_mon = mo - 1;
  tm.tm_mday = d;
  tm.tm_hour = h;
  tm.tm_min = mi;
  tm.tm_sec = sec;
  const time_t t = timegm(&tm);  // UTC, unaffected by TZ (unlike mktime)
  if (t == (time_t)-1) return false;
  out_epoch = (int64_t)t;
  return true;
}

}  // namespace license
}  // namespace rotobot
