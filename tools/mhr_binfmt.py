# Copyright the Hastur authors.
# SPDX-License-Identifier: Apache-2.0
#
# mhr_binfmt.py -- writer for the versioned flat binary `mhr_assets.bin`.
#
# Layout (all little-endian; matches src/MeshAssets.cpp reader):
#   magic   char[4]  = "MHRA"
#   version uint32   = 1
#   nblocks uint32
#   nblocks x block-header, each 104 bytes:
#     name    char[48]  (null-padded ASCII)
#     dtype   uint32    (0=float32, 1=int32, 2=int64)
#     ndim    uint32    (1..4)
#     shape   int64[4]  (unused trailing dims = 1)
#     offset  uint64    (absolute byte offset of the block's data)
#     nbytes  uint64
#   ... then each block's raw little-endian data at `offset` (8-byte aligned).
import json
import struct

import numpy as np

MAGIC = b"MHRA"
VERSION = 1
DT = {np.dtype("float32"): 0, np.dtype("int32"): 1, np.dtype("int64"): 2}
HDR_ALIGN = 8


def _align(x, a=HDR_ALIGN):
    return (x + a - 1) // a * a


def write_assets(path, blocks, manifest_path=None):
    """blocks: dict[name -> np.ndarray]. Names must be <=47 chars ASCII."""
    items = list(blocks.items())
    nblocks = len(items)
    HDR = 104  # struct.calcsize("<48sII4qQQ")
    hdr_size = 4 + 4 + 4 + nblocks * HDR
    # Compute offsets.
    offsets = {}
    cur = _align(hdr_size)
    for name, arr in items:
        offsets[name] = cur
        cur = _align(cur + arr.nbytes)
    total = cur

    buf = bytearray(total)
    struct.pack_into("<4sII", buf, 0, MAGIC, VERSION, nblocks)
    pos = 12
    manifest = {"version": VERSION, "blocks": []}
    for name, arr in items:
        assert arr.dtype in DT, f"{name}: unsupported dtype {arr.dtype}"
        nm = name.encode("ascii")
        assert len(nm) <= 47, f"name too long: {name}"
        shape = list(arr.shape) + [1] * (4 - arr.ndim)
        off = offsets[name]
        struct.pack_into("<48sII4qQQ", buf, pos, nm, DT[arr.dtype], arr.ndim,
                         shape[0], shape[1], shape[2], shape[3], off, arr.nbytes)
        pos += HDR
        buf[off:off + arr.nbytes] = np.ascontiguousarray(arr).tobytes()
        manifest["blocks"].append(
            {"name": name, "dtype": str(arr.dtype), "shape": list(arr.shape),
             "offset": off, "nbytes": int(arr.nbytes)})
    with open(path, "wb") as f:
        f.write(buf)
    if manifest_path:
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
    return manifest
