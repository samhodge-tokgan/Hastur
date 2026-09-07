// SPDX-License-Identifier: LicenseRef-Tokgan-Proprietary
// Copyright (c) Tokgan. Proprietary — licensed under the Tokgan EULA; see NOTICE.
#include "ModelBytes.h"

#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

#include "crypto/modelcrypt.h"

namespace fs = std::filesystem;

namespace hastur {
namespace {

// The vendored modelcrypt must agree with the one that SEALED the tree. A
// divergence in the key derivation would otherwise surface as "this artifact is
// corrupt" at a customer site rather than as a build failure here. This digest
// is the same constant compiled into Rotobot-Next; if the two copies drift,
// this stops the build.
constexpr const char* kExpectedDecoyDigest =
    "271fad169de5c6c5a4ac05e65560811ffd10e8302847e9bbd091a527e3fa0c9e";

// Checked once, before any model is opened. A mismatch means this vendored copy
// derives keys differently from the rotobot_model_seal that wrote the artifact,
// so every sealed load would fail with a misleading "corrupt" error. Fail here
// instead, naming the cause.
void assert_no_drift() {
  static const bool ok = [] {
    const std::string got = rotobot::modelcrypt::decoy_digest_hex();
    if (got != kExpectedDecoyDigest) {
      throw std::runtime_error(
          "hastur: the vendored modelcrypt has DRIFTED from Rotobot-Next "
          "(decoy digest " + got + ", expected " +
          std::string(kExpectedDecoyDigest) +
          "). Sealed model trees written by rotobot_model_seal will not open. "
          "Re-copy src/crypto and src/license from Rotobot-Next.");
    }
    return true;
  }();
  (void)ok;
}

struct TreeCache {
  std::mutex m;
  std::map<std::string, std::shared_ptr<rotobot::modelcrypt::SealedTree>> by_dir;
};
TreeCache& cache() {
  static TreeCache c;
  return c;
}

std::shared_ptr<rotobot::modelcrypt::SealedTree> tree_for(const std::string& dir) {
  assert_no_drift();
  TreeCache& c = cache();
  std::lock_guard<std::mutex> lk(c.m);
  auto it = c.by_dir.find(dir);
  if (it != c.by_dir.end()) return it->second;

  std::string err;
  auto t = rotobot::modelcrypt::SealedTree::open_from_env(dir, err);
  if (!t) throw std::runtime_error("hastur: " + err);
  std::shared_ptr<rotobot::modelcrypt::SealedTree> sp(std::move(t));
  c.by_dir.emplace(dir, sp);
  return sp;
}

void split(const std::string& model_path, std::string& dir, std::string& name) {
  const fs::path p(model_path);
  dir = p.parent_path().string();
  if (dir.empty()) dir = ".";
  name = p.filename().string();
}

}  // namespace

std::string LoadModelBytes(const std::string& model_path) {
  std::string dir, name;
  split(model_path, dir, name);
  auto t = tree_for(dir);
  std::string blob, err;
  if (!t->read(name, blob, err)) throw std::runtime_error("hastur: " + err);
  return blob;
}

bool ModelExists(const std::string& model_path) {
  std::string dir, name;
  split(model_path, dir, name);
  try {
    return tree_for(dir)->has(name);
  } catch (const std::exception&) {
    // A sealed tree we cannot open is not the same as a missing variant, but
    // for a probe the answer is the same: do not select it.
    return false;
  }
}

}  // namespace hastur
