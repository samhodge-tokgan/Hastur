// Copyright the Hastur authors.
// SPDX-License-Identifier: LicenseRef-SAM-License
//
// Cryptomatte.cpp -- see Cryptomatte.h for the contract and spec references.

#include "Cryptomatte.h"

#include <algorithm>
#include <cstdio>
#include <cstring>

namespace hastur {
namespace {

inline uint32_t rotl32(uint32_t x, int8_t r) {
  return (x << r) | (x >> (32 - r));
}

inline uint32_t fmix32(uint32_t h) {
  h ^= h >> 16;
  h *= 0x85ebca6bu;
  h ^= h >> 13;
  h *= 0xc2b2ae35u;
  h ^= h >> 16;
  return h;
}

}  // namespace

uint32_t MurmurHash3_x86_32(const void* key, int len, uint32_t seed) {
  const uint8_t* data = static_cast<const uint8_t*>(key);
  const int nblocks = len / 4;
  uint32_t h1 = seed;
  const uint32_t c1 = 0xcc9e2d51u, c2 = 0x1b873593u;

  // Body: read 4-byte blocks as little-endian (endian-safe byte assembly).
  for (int i = 0; i < nblocks; ++i) {
    const uint8_t* b = data + i * 4;
    uint32_t k1 = static_cast<uint32_t>(b[0]) | (static_cast<uint32_t>(b[1]) << 8) |
                  (static_cast<uint32_t>(b[2]) << 16) |
                  (static_cast<uint32_t>(b[3]) << 24);
    k1 *= c1;
    k1 = rotl32(k1, 15);
    k1 *= c2;
    h1 ^= k1;
    h1 = rotl32(h1, 13);
    h1 = h1 * 5 + 0xe6546b64u;
  }

  // Tail.
  const uint8_t* tail = data + nblocks * 4;
  uint32_t k1 = 0;
  switch (len & 3) {
    case 3:
      k1 ^= static_cast<uint32_t>(tail[2]) << 16;
      [[fallthrough]];
    case 2:
      k1 ^= static_cast<uint32_t>(tail[1]) << 8;
      [[fallthrough]];
    case 1:
      k1 ^= static_cast<uint32_t>(tail[0]);
      k1 *= c1;
      k1 = rotl32(k1, 15);
      k1 *= c2;
      h1 ^= k1;
  }

  h1 ^= static_cast<uint32_t>(len);
  h1 = fmix32(h1);
  return h1;
}

namespace {

// Cryptomatte exponent fix: keep the float away from inf/NaN/denormal.
uint32_t CryptoIdBits(const std::string& name) {
  uint32_t h = MurmurHash3_x86_32(name.data(), static_cast<int>(name.size()), 0);
  const uint32_t exponent = (h >> 23) & 0xffu;
  if (exponent == 0 || exponent == 255) h ^= (1u << 23);
  return h;
}

}  // namespace

float CryptoIdFloat(const std::string& name) {
  uint32_t bits = CryptoIdBits(name);
  float f;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

std::string CryptoIdHex(const std::string& name) {
  char buf[9];
  std::snprintf(buf, sizeof(buf), "%08x", CryptoIdBits(name));
  return std::string(buf);
}

std::string CryptoTypeKey(const std::string& type_name) {
  char buf[9];
  std::snprintf(buf, sizeof(buf), "%08x",
                MurmurHash3_x86_32(type_name.data(),
                                   static_cast<int>(type_name.size()), 0));
  return std::string(buf).substr(0, 7);
}

CryptoResult BuildCryptomatte(int W, int H, const std::string& type_name,
                              const std::vector<CryptoPerson>& persons,
                              int num_levels) {
  CryptoResult out;
  out.width = W;
  out.height = H;
  out.type_name = type_name;
  if (W <= 0 || H <= 0 || persons.empty()) {
    out.manifest = "{}";
    return out;
  }
  num_levels = std::max(1, num_levels);
  const size_t npix = static_cast<size_t>(W) * H;

  // Precompute per-person float IDs and manifest.
  std::vector<float> ids(persons.size());
  std::string manifest = "{";
  for (size_t p = 0; p < persons.size(); ++p) {
    ids[p] = CryptoIdFloat(persons[p].name);
    if (p) manifest += ",";
    manifest += "\"" + persons[p].name + "\":\"" + CryptoIdHex(persons[p].name) + "\"";
  }
  manifest += "}";
  out.manifest = std::move(manifest);

  out.layers.assign(static_cast<size_t>(num_levels),
                    std::vector<float>(npix * 4, 0.0f));
  const int max_ranks = num_levels * 2;

  // Scratch per pixel: (id, visible coverage) pairs, at most persons.size().
  std::vector<std::pair<float, float>> ranks;
  ranks.reserve(persons.size());

  for (size_t px = 0; px < npix; ++px) {
    ranks.clear();
    float trans = 1.0f;  // remaining transparency, front-to-back
    for (size_t p = 0; p < persons.size() && trans > 1e-6f; ++p) {
      const std::vector<float>& cov = persons[p].coverage;
      if (px >= cov.size()) continue;
      const float c = cov[px];
      if (c <= 0.0f) continue;
      const float vis = c * trans;  // occluded by nearer persons already processed
      if (vis > 0.0f) ranks.emplace_back(ids[p], vis);
      trans *= (1.0f - c);
    }
    if (ranks.empty()) continue;

    // Rank by visible coverage descending; keep the top max_ranks.
    std::sort(ranks.begin(), ranks.end(),
              [](const std::pair<float, float>& a, const std::pair<float, float>& b) {
                return a.second > b.second;
              });
    const int n = std::min(static_cast<int>(ranks.size()), max_ranks);
    for (int r = 0; r < n; ++r) {
      const int level = r / 2;
      const int half = r % 2;  // 0 -> (R,G) id/cov, 1 -> (B,A) id/cov
      float* dst = &out.layers[level][px * 4];
      dst[half * 2 + 0] = ranks[r].first;   // id
      dst[half * 2 + 1] = ranks[r].second;  // coverage
    }
  }
  return out;
}

}  // namespace hastur
