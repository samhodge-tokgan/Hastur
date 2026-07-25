// Copyright the Hastur authors.
// SPDX-License-Identifier: LicenseRef-SAM-License
//
// external_tracks_validate — self-contained unit checks for the ExternalTracks
// sidecar parser + .msk decoder. Writes fixtures under a temp dir and asserts
// frame-keying, coordinate order (top-down xyxy), the "# ids" header, the .msk
// binary decode, the per-frame "%04d" layout, and comment/CRLF/malformed
// tolerance. No models, ONNX, or OFX host. Exit code 0 = all pass.

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "ExternalTracks.h"

namespace fs = std::filesystem;
using namespace hastur;

static int g_fails = 0;
static void check(bool ok, const char* msg) {
  if (!ok) {
    std::printf("FAIL: %s\n", msg);
    ++g_fails;
  }
}

// Write an "MSK1" coverage file: magic, int32 LE w, int32 LE h, w*h uint8.
static void write_msk(const fs::path& p, int w, int h, unsigned char fill) {
  std::ofstream f(p, std::ios::binary);
  f.put('M').put('S').put('K').put('1');
  auto w32 = [&](int32_t v) {
    unsigned char b[4] = {static_cast<unsigned char>(v & 0xff),
                          static_cast<unsigned char>((v >> 8) & 0xff),
                          static_cast<unsigned char>((v >> 16) & 0xff),
                          static_cast<unsigned char>((v >> 24) & 0xff)};
    f.write(reinterpret_cast<char*>(b), 4);
  };
  w32(w);
  w32(h);
  std::vector<unsigned char> data(static_cast<size_t>(w) * h, fill);
  f.write(reinterpret_cast<char*>(data.data()),
          static_cast<std::streamsize>(data.size()));
}

int main() {
  std::error_code ec;
  const fs::path dir =
      fs::temp_directory_path(ec) / "hastur_ext_tracks_test";
  fs::remove_all(dir, ec);
  fs::create_directories(dir / "masks", ec);

  // --- Single-file tracks.txt (primary layout) --------------------------
  {
    write_msk(dir / "masks" / "0001_00.msk", 4, 2, 255);
    std::ofstream f(dir / "tracks.txt");
    f << "# hastur-tracks v1\n";
    f << "# ids 0 1 7\n";
    f << "# frame track_id x0 y0 x1 y1 [mask_relpath]\n";
    f << "1 0 320.5 88 512 940 masks/0001_00.msk\n";
    f << "1 7 700 120 880 900\n";        // no mask on this row
    f << "2 0 322 90 514 942\n";
    f << "\n";                            // blank line tolerated
    f << "  # indented comment tolerated\n";
    f << "1 3 not a number row\n";        // malformed -> skipped
  }

  const std::string path = dir.string();

  check(HasExternalTracks(path), "HasExternalTracks true when sidecar present");
  check(!HasExternalTracks((dir / "nope").string()),
        "HasExternalTracks false when absent");

  // Frame 1: two well-formed rows (ids 0 and 7); the malformed id-3 row dropped.
  {
    auto t = LoadExternalTracks(path, 1, /*want_masks=*/true);
    check(t.size() == 2, "frame 1 has 2 valid tracks");
    if (t.size() == 2) {
      check(t[0].id == 0 && t[1].id == 7, "frame 1 ids 0,7 in file order");
      // Coordinate order: x0,y0,x1,y1 top-down.
      check(t[0].box.x0 == 320.5f && t[0].box.y0 == 88.f &&
                t[0].box.x1 == 512.f && t[0].box.y1 == 940.f,
            "frame 1 id0 box xyxy parsed in order");
      check(t[0].box.score == 1.f, "external box score defaults to 1");
      // Mask present on id0, absent on id7.
      check(t[0].mask.w == 4 && t[0].mask.h == 2 &&
                t[0].mask.data.size() == 8,
            "id0 .msk decoded 4x2");
      check(t[0].mask.data[0] == 1.f, "id0 mask 255 -> 1.0");
      check(t[1].mask.w == 0 && t[1].mask.h == 0, "id7 has no mask");
    }
  }

  // want_masks=false must skip mask decode (zero cost path).
  {
    auto t = LoadExternalTracks(path, 1, /*want_masks=*/false);
    check(t.size() == 2, "frame 1 count same without masks");
    if (t.size() == 2)
      check(t[0].mask.w == 0 && t[0].mask.data.empty(),
            "want_masks=false skips .msk decode");
  }

  // Frame 2: single track; frame with no rows -> empty.
  check(LoadExternalTracks(path, 2, false).size() == 1, "frame 2 has 1 track");
  check(LoadExternalTracks(path, 99, false).empty(), "absent frame -> empty");

  // "# ids" header enumerates the full (sparse) id set.
  {
    auto ids = ExternalTrackIds(path);
    check(ids.size() == 3, "ids header parsed count");
    if (ids.size() == 3)
      check(ids[0] == 0 && ids[1] == 1 && ids[2] == 7, "ids header 0,1,7");
  }

  // --- Per-frame "%04d" pattern layout ----------------------------------
  {
    const fs::path pdir = dir / "perframe";
    fs::create_directories(pdir, ec);
    {
      std::ofstream f(pdir / "tracks.0005.txt");
      f << "# ids 2\r\n";                 // CRLF tolerated
      f << "5 2 10 20 30 40\r\n";
    }
    const std::string pat = (pdir / "tracks.%04d.txt").string();
    auto t = LoadExternalTracks(pat, 5, false);
    check(t.size() == 1 && t[0].id == 2, "per-frame %04d pattern loads frame 5");
    if (t.size() == 1)
      check(t[0].box.x1 == 30.f && t[0].box.y1 == 40.f,
            "per-frame CRLF row parsed");
    auto ids = ExternalTrackIds(pat);
    check(ids.size() == 1 && ids[0] == 2, "per-frame ids header via pattern");
    check(LoadExternalTracks(pat, 6, false).empty(),
          "per-frame pattern: missing frame file -> empty");
  }

  fs::remove_all(dir, ec);
  if (g_fails == 0) std::printf("external_tracks_validate: all checks passed\n");
  return g_fails == 0 ? 0 : 1;
}
