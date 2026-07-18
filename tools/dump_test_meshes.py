#!/usr/bin/env python
# Copyright the Hastur authors.
# SPDX-License-Identifier: Apache-2.0
#
# dump_test_meshes.py -- ground-truth fixtures for the C++ SoftwareRasterizer.
#
# Produces, for a few posed MHR meshes, a (verts, faces, cam_t, focal, W, H) tuple
# plus a REFERENCE RGBA render, packed into test-assets/raster_fixtures.bin (MHRA
# format) that tests/raster_validate.cpp reads back.
#
# REFERENCE RENDERER
# ------------------
# The reference pyrender path (visualization/renderer.py) needs a headless OpenGL
# context. On this Apple-silicon Mac NO headless GL backend is available (osmesa,
# egl and pyglet all fail to load -- see RASTER_REPORT.md). So the reference is a
# pure-NumPy software rasterizer here that implements the SAME intended semantics
# the C++ rasterizer targets: CV pinhole projection (identical to the pipeline's
# perspective_projection, which pyrender's 180-X + cam_t[0] flip reproduces), a
# z-buffer, perspective-correct barycentric interpolation, per-vertex-normal
# Lambert neutral-grey shading, and coverage alpha. This validates the C++ PORT
# against an independent implementation of the spec; the geometrically meaningful
# check is the silhouette / coverage IoU (which any correct rasterizer must match).
#
# The mesh verts come from the M2 mesh oracle fixtures (test-assets/fixtures/*.npz,
# already in metres in the MHR camera frame); faces come from mhr_assets.bin. The
# cameras are SYNTHESISED to frame each mesh (these zero/random poses have no real
# camera solve) -- the rasterizer contract only consumes (focal, center, cam_t).
import argparse
import glob
import os
import struct

import numpy as np

from mhr_binfmt import write_assets

NEAR_Z = 1e-3

# --- shading spec (MUST match src/SoftwareRasterizer.cpp) --------------------
GREY = 0.6
AMBIENT = 0.3
LIGHT_W = 0.42
LIGHTS = np.array([
    [0.5, 0.0, -0.8660254],
    [-0.25, 0.4330127, -0.8660254],
    [-0.25, -0.4330127, -0.8660254],
], dtype=np.float64)


def read_mhra_block(path, want):
    """Minimal reader for the MHRA flat binary; returns the named block ndarray."""
    with open(path, "rb") as f:
        buf = f.read()
    magic, version, nblocks = struct.unpack_from("<4sII", buf, 0)
    assert magic == b"MHRA", magic
    dtmap = {0: np.float32, 1: np.int32, 2: np.int64}
    pos = 12
    for _ in range(nblocks):
        nm, dt, ndim, s0, s1, s2, s3, off, nbytes = struct.unpack_from(
            "<48sII4qQQ", buf, pos)
        pos += 104
        name = nm.split(b"\x00", 1)[0].decode("ascii")
        if name == want:
            shape = [s0, s1, s2, s3][:ndim]
            arr = np.frombuffer(buf, dtype=dtmap[dt], count=int(np.prod(shape)),
                                offset=off)
            return arr.reshape(shape).copy()
    raise KeyError(want)


def vertex_normals(verts, faces):
    n = np.zeros_like(verts, dtype=np.float64)
    a = verts[faces[:, 0]]
    b = verts[faces[:, 1]]
    c = verts[faces[:, 2]]
    fn = np.cross(b - a, c - a)  # area-weighted (magnitude ~ 2*area)
    np.add.at(n, faces[:, 0], fn)
    np.add.at(n, faces[:, 1], fn)
    np.add.at(n, faces[:, 2], fn)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    ln[ln < 1e-20] = 1.0
    return n / ln


def shade_scalar(n, pcam, matcap=False):
    """n,pcam: (K,3). Returns (K,) grey scalar in [0,1]. Mirrors ShadeScalar."""
    n = n.copy()
    flip = np.einsum("kd,kd->k", n, pcam) > 0.0  # normal points away from camera
    n[flip] *= -1.0
    if matcap:
        vtc = -pcam / np.maximum(np.linalg.norm(pcam, axis=1, keepdims=True), 1e-20)
        d = np.maximum(0.0, np.einsum("kd,kd->k", n, vtc))
        s = AMBIENT + (1.0 - AMBIENT) * d
    else:
        s = np.full(n.shape[0], AMBIENT)
        for L in LIGHTS:
            s += LIGHT_W * np.maximum(0.0, n @ L)
    return np.clip(GREY * s, 0.0, 1.0)


def rasterize(verts, faces, focal, center, cam_t, W, H, ss=1, matcap=False):
    """NumPy reference rasterizer -> straight-alpha RGBA float32 (H,W,4)."""
    SW, SH = W * ss, H * ss
    fx = fy = focal * ss
    cx, cy = center[0] * ss, center[1] * ss
    vn = vertex_normals(verts, faces)
    vc = verts + cam_t  # camera at origin, +z forward
    z = vc[:, 2]
    ok = z > NEAR_Z
    invz = 1.0 / np.where(ok, z, 1.0)
    sx = fx * vc[:, 0] * invz + cx
    sy = fy * vc[:, 1] * invz + cy

    fb_rgb = np.zeros((SH, SW, 3), np.float64)
    fb_cov = np.zeros((SH, SW), bool)
    fb_invz = np.full((SH, SW), -np.inf)
    inv_fx, inv_fy = 1.0 / fx, 1.0 / fy

    for f in range(faces.shape[0]):
        i0, i1, i2 = faces[f]
        if not (ok[i0] and ok[i1] and ok[i2]):
            continue
        ax, ay = sx[i0], sy[i0]
        bx, by = sx[i1], sy[i1]
        cxx, cyy = sx[i2], sy[i2]
        area = (bx - ax) * (cyy - ay) - (cxx - ax) * (by - ay)
        if abs(area) < 1e-9:
            continue
        minx = max(int(np.floor(min(ax, bx, cxx))), 0)
        maxx = min(int(np.ceil(max(ax, bx, cxx))), SW - 1)
        miny = max(int(np.floor(min(ay, by, cyy))), 0)
        maxy = min(int(np.ceil(max(ay, by, cyy))), SH - 1)
        if minx > maxx or miny > maxy:
            continue
        xs = np.arange(minx, maxx + 1) + 0.5
        ys = np.arange(miny, maxy + 1) + 0.5
        gx, gy = np.meshgrid(xs, ys)  # (ny,nx)
        w0 = (bx - gx) * (cyy - gy) - (cxx - gx) * (by - gy)
        w1 = (cxx - gx) * (ay - gy) - (ax - gx) * (cyy - gy)
        w2 = (ax - gx) * (by - gy) - (bx - gx) * (ay - gy)
        inside = ((w0 >= 0) & (w1 >= 0) & (w2 >= 0)) | \
                 ((w0 <= 0) & (w1 <= 0) & (w2 <= 0))
        if not inside.any():
            continue
        b0 = w0[inside] / area
        b1 = w1[inside] / area
        b2 = w2[inside] / area
        ia, ib, ic = invz[i0], invz[i1], invz[i2]
        pinvz = b0 * ia + b1 * ib + b2 * ic
        zz = 1.0 / pinvz
        px = gx[inside]
        py = gy[inside]
        # rows/cols of the covered pixels within the framebuffer
        rr = (py - 0.5).astype(int)
        cc = (px - 0.5).astype(int)
        cur = fb_invz[rr, cc]
        win = pinvz > cur
        if not win.any():
            continue
        b0, b1, b2, zz = b0[win], b1[win], b2[win], zz[win]
        px, py, pinvz = px[win], py[win], pinvz[win]
        rr, cc = rr[win], cc[win]
        n0, n1, n2 = vn[i0], vn[i1], vn[i2]
        nn = ((b0 * ia)[:, None] * n0 + (b1 * ib)[:, None] * n1 +
              (b2 * ic)[:, None] * n2) * zz[:, None]
        nn /= np.maximum(np.linalg.norm(nn, axis=1, keepdims=True), 1e-20)
        pcam = np.stack([(px - cx) * inv_fx * zz,
                         (py - cy) * inv_fy * zz, zz], axis=1)
        g = shade_scalar(nn, pcam, matcap)
        fb_rgb[rr, cc, 0] = g
        fb_rgb[rr, cc, 1] = g
        fb_rgb[rr, cc, 2] = g
        fb_cov[rr, cc] = True
        fb_invz[rr, cc] = pinvz

    # box-downsample to (H,W)
    cov = fb_cov.reshape(H, ss, W, ss).sum(axis=(1, 3)).astype(np.float64)
    rgb_sum = fb_rgb.reshape(H, ss, W, ss, 3).sum(axis=(1, 3))
    alpha = cov / (ss * ss)
    with np.errstate(invalid="ignore", divide="ignore"):
        rgb = np.where(cov[..., None] > 0, rgb_sum / np.maximum(cov[..., None], 1), 0.0)
    out = np.concatenate([rgb, alpha[..., None]], axis=2).astype(np.float32)
    return out


def fit_camera(verts, W, H, margin=0.12, depth=3.0):
    """Synthesise (focal, center, cam_t) that frames the mesh with a margin."""
    ctr = verts.mean(0)
    cam_t = np.array([-ctr[0], -ctr[1], depth - ctr[2]], np.float64)
    vc = verts + cam_t
    rx = np.abs(vc[:, 0] / vc[:, 2]).max()
    ry = np.abs(vc[:, 1] / vc[:, 2]).max()
    fx = (0.5 - margin) * W / max(rx, 1e-6)
    fy = (0.5 - margin) * H / max(ry, 1e-6)
    focal = float(min(fx, fy))
    center = np.array([W / 2.0, H / 2.0], np.float64)
    return focal, center, cam_t


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", default=os.path.join(here, "..", "test-assets"))
    ap.add_argument("--ss", type=int, default=1, help="reference supersampling")
    args = ap.parse_args()

    faces = read_mhra_block(os.path.join(args.assets, "mhr_assets.bin"), "faces")
    faces = faces.astype(np.int64)
    fixdir = os.path.join(args.assets, "fixtures")
    npz = sorted(glob.glob(os.path.join(fixdir, "fix*.npz")))

    # A few meshes at assorted (portrait/landscape) resolutions.
    picks = [(npz[0], 512, 640),
             (npz[1], 640, 480),
             (npz[3], 480, 720)]

    blocks = {"nfix": np.asarray([len(picks)], np.int32),
              "faces": faces.astype(np.int32)}
    print(f"faces {faces.shape}, {len(picks)} fixtures, reference ss={args.ss}")
    for i, (path, W, H) in enumerate(picks):
        verts = np.load(path)["verts"].astype(np.float64)
        focal, center, cam_t = fit_camera(verts, W, H)
        rgba = rasterize(verts, faces, focal, center, cam_t, W, H, ss=args.ss)
        cov = float((rgba[:, :, 3] > 0.5).mean())
        print(f"  fix{i}: {os.path.basename(path)} {W}x{H} focal={focal:.1f} "
              f"cam_t=({cam_t[0]:.3f},{cam_t[1]:.3f},{cam_t[2]:.3f}) "
              f"coverage={cov*100:.1f}%")
        cam = np.array([focal, center[0], center[1],
                        cam_t[0], cam_t[1], cam_t[2], W, H], np.float32)
        blocks[f"verts_{i:02d}"] = verts.astype(np.float32)
        blocks[f"cam_{i:02d}"] = cam
        blocks[f"rgba_{i:02d}"] = rgba  # (H,W,4) float32, straight alpha
        # also drop an inspectable npz
        np.savez(os.path.join(fixdir, f"raster{i:02d}.npz"),
                 verts=verts.astype(np.float32), cam=cam, rgba=rgba)

    out = os.path.join(args.assets, "raster_fixtures.bin")
    write_assets(out, blocks)
    print(f"wrote {out} ({os.path.getsize(out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
