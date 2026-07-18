// Copyright the Hastur authors.
// SPDX-License-Identifier: LicenseRef-SAM-License

#include "MeshAssets.h"

#include <cstring>
#include <fstream>
#include <stdexcept>

namespace hastur {

namespace {
constexpr char kMagic[4] = {'M', 'H', 'R', 'A'};

#pragma pack(push, 1)
struct BlockHeader {
  char name[48];
  uint32_t dtype;
  uint32_t ndim;
  int64_t shape[4];
  uint64_t offset;
  uint64_t nbytes;
};
#pragma pack(pop)
static_assert(sizeof(BlockHeader) == 104, "block header must be 104 bytes");
}  // namespace

std::shared_ptr<MeshAssets> MeshAssets::Load(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) throw std::runtime_error("MeshAssets: cannot open " + path);
  const std::streamsize sz = f.tellg();
  f.seekg(0);
  auto a = std::make_shared<MeshAssets>();
  a->raw_.resize(static_cast<size_t>(sz));
  if (!f.read(reinterpret_cast<char*>(a->raw_.data()), sz))
    throw std::runtime_error("MeshAssets: short read on " + path);
  if (sz < 12 || std::memcmp(a->raw_.data(), kMagic, 4) != 0)
    throw std::runtime_error("MeshAssets: bad magic in " + path);

  uint32_t version, nblocks;
  std::memcpy(&version, a->raw_.data() + 4, 4);
  std::memcpy(&nblocks, a->raw_.data() + 8, 4);
  a->version_ = version;

  const uint8_t* base = a->raw_.data();
  size_t pos = 12;
  for (uint32_t b = 0; b < nblocks; ++b) {
    if (pos + sizeof(BlockHeader) > a->raw_.size())
      throw std::runtime_error("MeshAssets: truncated header table");
    BlockHeader bh;
    std::memcpy(&bh, base + pos, sizeof(bh));
    pos += sizeof(bh);
    if (bh.offset + bh.nbytes > a->raw_.size())
      throw std::runtime_error("MeshAssets: block data out of range");
    Block blk;
    blk.dtype = static_cast<DType>(bh.dtype);
    blk.ndim = bh.ndim;
    for (int i = 0; i < 4; ++i) blk.shape[i] = bh.shape[i];
    blk.data = base + bh.offset;
    blk.nbytes = bh.nbytes;
    a->blocks_.emplace(std::string(bh.name), blk);
  }

  // Materialize the shared faces vector (int32, kNumFaces*3) if present.
  if (a->has("faces")) {
    const Block& fb = a->block("faces");
    const int32_t* fp = reinterpret_cast<const int32_t*>(fb.data);
    a->faces_ = std::make_shared<std::vector<int32_t>>(fp, fp + fb.numel());
  }
  return a;
}

const MeshAssets::Block& MeshAssets::block(const std::string& name) const {
  auto it = blocks_.find(name);
  if (it == blocks_.end())
    throw std::runtime_error("MeshAssets: missing block '" + name + "'");
  return it->second;
}

const float* MeshAssets::f32(const std::string& name) const {
  const Block& b = block(name);
  if (b.dtype != DType::F32)
    throw std::runtime_error("MeshAssets: '" + name + "' is not float32");
  return reinterpret_cast<const float*>(b.data);
}

const int32_t* MeshAssets::i32(const std::string& name) const {
  const Block& b = block(name);
  if (b.dtype != DType::I32)
    throw std::runtime_error("MeshAssets: '" + name + "' is not int32");
  return reinterpret_cast<const int32_t*>(b.data);
}

const int64_t* MeshAssets::i64(const std::string& name) const {
  const Block& b = block(name);
  if (b.dtype != DType::I64)
    throw std::runtime_error("MeshAssets: '" + name + "' is not int64");
  return reinterpret_cast<const int64_t*>(b.data);
}

}  // namespace hastur
