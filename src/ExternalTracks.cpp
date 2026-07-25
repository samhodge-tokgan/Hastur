// Copyright the Hastur authors.
// SPDX-License-Identifier: LicenseRef-SAM-License
//
// ExternalTracks — see ExternalTracks.h for the sidecar format. Dependency-free
// (std streams + <filesystem>); mirrors the simple text scan in DetectorEngine.cpp
// and the binary fread of mhr_wrist.bin in Sam3dBodyPipeline.cpp.

#include "ExternalTracks.h"

#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace hastur {
namespace {

// Path-list separator: ';' on Windows (so "F:/tracks" is not split at its colon),
// ':' on POSIX. Must match the plugin's join separator (as in Sam3dBodyPipeline).
#ifdef _WIN32
constexpr char kPathListSep = ';';
#else
constexpr char kPathListSep = ':';
#endif

// Split a path-list into entries, then append $HASTUR_TRACKS (also split) so an
// env override always applies — same shape as the pipeline's SearchDirs().
std::vector<std::string> SearchEntries(const std::string& search_path) {
  std::vector<std::string> out;
  auto push_split = [&](const std::string& s) {
    size_t i = 0;
    while (i <= s.size()) {
      size_t j = s.find(kPathListSep, i);
      if (j == std::string::npos) j = s.size();
      if (j > i) out.emplace_back(s.substr(i, j - i));
      i = j + 1;
    }
  };
  push_split(search_path);
  if (const char* env = std::getenv("HASTUR_TRACKS")) push_split(env);
  return out;
}

// Expand a printf-style frame pattern ("tracks.%04d.txt") for one frame. Only a
// single integer conversion is supported; returns empty if the buffer overflows.
std::string ExpandPattern(const std::string& pat, long frame) {
  char buf[1024];
  const int n = std::snprintf(buf, sizeof(buf), pat.c_str(), frame);
  if (n <= 0 || static_cast<size_t>(n) >= sizeof(buf)) return {};
  return std::string(buf, static_cast<size_t>(n));
}

// Resolve one entry to a readable tracks file for `frame`:
//   * entry contains '%' -> printf frame pattern (per-frame file)
//   * entry is a directory -> "tracks.txt", else "tracks.%04d.txt" (expanded)
//   * entry is a file      -> used directly
// Returns the resolved path (existing + readable), else empty.
std::string ResolveTracksFile(const std::string& entry, long frame) {
  std::error_code ec;
  if (entry.find('%') != std::string::npos) {
    const std::string p = ExpandPattern(entry, frame);
    if (!p.empty() && fs::exists(p, ec)) return p;
    return {};
  }
  if (fs::is_directory(entry, ec)) {
    const fs::path single = fs::path(entry) / "tracks.txt";
    if (fs::exists(single, ec)) return single.string();
    const std::string per = ExpandPattern((fs::path(entry) / "tracks.%04d.txt").string(), frame);
    if (!per.empty() && fs::exists(per, ec)) return per;
    return {};
  }
  if (fs::exists(entry, ec)) return entry;
  return {};
}

// Split a printf integer pattern ("tracks.%04d.txt") into the literal text before
// '%' and after the conversion letter, so files can be matched frame-agnostically.
// Returns false if there is no '%...<letter>' conversion.
bool SplitPattern(const std::string& pat, std::string& prefix, std::string& suffix) {
  const size_t pc = pat.find('%');
  if (pc == std::string::npos) return false;
  size_t k = pc + 1;
  while (k < pat.size() && !std::isalpha(static_cast<unsigned char>(pat[k]))) ++k;
  if (k >= pat.size()) return false;
  prefix = pat.substr(0, pc);
  suffix = pat.substr(k + 1);
  return true;
}

// Resolve ANY existing tracks file for an entry, WITHOUT knowing a frame number
// (for the "# ids" header + presence checks): a '%'-pattern globs its directory
// for the first literal-prefix/suffix match; a directory yields tracks.txt or the
// first tracks.*.txt; a file resolves itself. Empty if nothing matches.
std::string AnyTracksFile(const std::string& entry) {
  std::error_code ec;
  if (entry.find('%') != std::string::npos) {
    std::string pre, suf;
    if (!SplitPattern(entry, pre, suf)) return {};
    const fs::path pp(pre);
    const fs::path dir = pp.has_parent_path() ? pp.parent_path() : fs::path(".");
    const std::string base = pp.filename().string();
    if (!fs::is_directory(dir, ec)) return {};
    for (const auto& de : fs::directory_iterator(dir, ec)) {
      const std::string nm = de.path().filename().string();
      if (nm.size() >= base.size() + suf.size() &&
          nm.compare(0, base.size(), base) == 0 &&
          nm.compare(nm.size() - suf.size(), suf.size(), suf) == 0)
        return de.path().string();
    }
    return {};
  }
  if (fs::is_directory(entry, ec)) {
    const fs::path single = fs::path(entry) / "tracks.txt";
    if (fs::exists(single, ec)) return single.string();
    for (const auto& de : fs::directory_iterator(entry, ec)) {
      const std::string nm = de.path().filename().string();
      if (nm.rfind("tracks.", 0) == 0 && nm.size() > 11 &&
          nm.compare(nm.size() - 4, 4, ".txt") == 0)
        return de.path().string();
    }
    return {};
  }
  if (fs::exists(entry, ec)) return entry;
  return {};
}

// Load an "MSK1" binary coverage file into a DetMask (uint8 0..255 -> float
// [0,1], full-frame stretched). Empty DetMask on any error. `path` is relative to
// the sidecar file's directory when not absolute.
DetMask LoadMsk(const fs::path& base_dir, const std::string& relpath) {
  DetMask m;
  fs::path p = relpath;
  if (p.is_relative()) p = base_dir / p;
  std::ifstream f(p, std::ios::binary);
  if (!f) return m;
  char magic[4] = {0, 0, 0, 0};
  f.read(magic, 4);
  if (!f || magic[0] != 'M' || magic[1] != 'S' || magic[2] != 'K' || magic[3] != '1')
    return m;
  auto read_i32le = [&](int32_t& v) -> bool {
    unsigned char b[4];
    f.read(reinterpret_cast<char*>(b), 4);
    if (!f) return false;
    v = static_cast<int32_t>(static_cast<uint32_t>(b[0]) |
                             (static_cast<uint32_t>(b[1]) << 8) |
                             (static_cast<uint32_t>(b[2]) << 16) |
                             (static_cast<uint32_t>(b[3]) << 24));
    return true;
  };
  int32_t w = 0, h = 0;
  if (!read_i32le(w) || !read_i32le(h)) return m;
  if (w <= 0 || h <= 0 || w > 1 << 15 || h > 1 << 15) return m;
  const size_t n = static_cast<size_t>(w) * static_cast<size_t>(h);
  std::vector<unsigned char> bytes(n);
  f.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(n));
  if (static_cast<size_t>(f.gcount()) != n) return m;
  m.w = w;
  m.h = h;
  m.data.resize(n);
  for (size_t i = 0; i < n; ++i) m.data[i] = bytes[i] * (1.f / 255.f);
  return m;
}

}  // namespace

std::vector<ExternalTrack> LoadExternalTracks(const std::string& search_path,
                                              long frame, bool want_masks) {
  std::vector<ExternalTrack> out;
  for (const std::string& entry : SearchEntries(search_path)) {
    const std::string file = ResolveTracksFile(entry, frame);
    if (file.empty()) continue;
    std::ifstream f(file);
    if (!f) continue;
    const fs::path base_dir = fs::path(file).parent_path();
    std::string line;
    while (std::getline(f, line)) {
      // Strip a trailing CR (tolerate CRLF sidecars from Windows writers).
      if (!line.empty() && line.back() == '\r') line.pop_back();
      if (line.empty() || line[0] == '#') continue;
      std::istringstream is(line);
      long fr = 0;
      ExternalTrack t;
      if (!(is >> fr >> t.id >> t.box.x0 >> t.box.y0 >> t.box.x1 >> t.box.y1))
        continue;  // malformed row -> skip
      if (fr != frame) continue;
      t.box.score = 1.f;
      std::string mask_rel;
      if (want_masks && (is >> mask_rel) && !mask_rel.empty())
        t.mask = LoadMsk(base_dir, mask_rel);
      out.push_back(std::move(t));
    }
    return out;  // first resolvable source wins (even if it has 0 rows for frame)
  }
  return out;
}

std::vector<int> ExternalTrackIds(const std::string& search_path) {
  std::vector<int> ids;
  for (const std::string& entry : SearchEntries(search_path)) {
    // The "# ids" header is frame-independent; resolve any file carrying it
    // (globs a per-frame layout without needing to guess a valid frame number).
    const std::string file = AnyTracksFile(entry);
    if (file.empty()) continue;
    std::ifstream f(file);
    if (!f) continue;
    std::string line;
    while (std::getline(f, line)) {
      if (!line.empty() && line.back() == '\r') line.pop_back();
      const auto k = line.find("# ids");
      if (k == std::string::npos) continue;
      std::istringstream is(line.substr(k + 5));
      int id;
      while (is >> id) ids.push_back(id);
      return ids;  // one header per clip
    }
    return ids;  // resolved a file but no header
  }
  return ids;
}

bool HasExternalTracks(const std::string& search_path) {
  if (search_path.empty() && std::getenv("HASTUR_TRACKS") == nullptr) return false;
  for (const std::string& entry : SearchEntries(search_path)) {
    if (!AnyTracksFile(entry).empty()) return true;  // frame-agnostic presence
  }
  return false;
}

}  // namespace hastur
