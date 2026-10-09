// Copyright (c) Tokgan. Proprietary — licensed under the Tokgan EULA; see NOTICE.
// SPDX-License-Identifier: LicenseRef-Tokgan-Proprietary
//
// RegisterModelTree: an application-opened tree is what LoadModelBytes reads for that dir,
// other dirs still open themselves, and dir spellings normalise to one key.
//   model_tree_register_validate <scratch dir>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>

#include "ModelBytes.h"
#include "crypto/modelcrypt.h"

namespace fs = std::filesystem;

static void put(const fs::path& p, const std::string& s) {
  fs::create_directories(p.parent_path());
  std::ofstream(p, std::ios::binary) << s;
}

int main(int argc, char** argv) {
  if (argc != 2) { std::fprintf(stderr, "usage: %s <scratch>\n", argv[0]); return 2; }
  const fs::path w = fs::path(argv[1]) / "mtr";
  fs::remove_all(w);
  put(w / "a" / "x.bin", "from-a");
  put(w / "b" / "x.bin", "from-b");
  put(w / "c" / "x.bin", "from-c");
  int fails = 0;
  auto check = [&](bool ok, const char* what) { std::printf("%s: %s\n", ok ? "ok" : "FAIL", what); fails += !ok; };

  std::string err;
  auto tb = rotobot::modelcrypt::SealedTree::open_from_env((w / "b").string(), err);
  if (!tb) { std::printf("FAIL: open b: %s\n", err.c_str()); return 1; }
  hastur::RegisterModelTree((w / "a").string() + "/", std::shared_ptr<rotobot::modelcrypt::SealedTree>(std::move(tb)));

  check(hastur::LoadModelBytes((w / "a" / "x.bin").string()) == "from-b", "a registered tree serves its dir");
  check(hastur::LoadModelBytes((w / "a" / "." / "x.bin").string()) == "from-b", "dir spellings share one key");
  check(hastur::LoadModelBytes((w / "c" / "x.bin").string()) == "from-c", "unregistered dirs still open themselves");
  check(hastur::ModelExists((w / "a" / "x.bin").string()) && !hastur::ModelExists((w / "a" / "y.bin").string()),
        "ModelExists consults the registered tree");
  bool threw = false;
  try { hastur::RegisterModelTree((w / "a").string(), nullptr); } catch (const std::invalid_argument&) { threw = true; }
  check(threw, "a null tree is refused");
  fs::remove_all(w);
  return fails ? 1 : 0;
}
