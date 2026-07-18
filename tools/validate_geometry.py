#!/usr/bin/env python3
# Copyright the Hastur authors.
# SPDX-License-Identifier: LicenseRef-SAM-License
#
# Track F geometry validation: assert the C++ ports in src/CameraSolver.{h,cpp}
# and src/CropAffine.{h,cpp} match the sam-3d-body reference (@ c259bfc) to
# within 1e-4 on real crops/bboxes.
#
#   sam-3d-body python:
#     /Users/sam/Documents/github/sam-3d-body/env/bin/python tools/validate_geometry.py
#
# Reference sources (loaded directly / copied verbatim to avoid the package
# __init__, which runs torch.hub.list at import time):
#   * get_warp_matrix / bbox_xyxy2cs / fix_aspect_ratio  <- data/transforms/bbox_utils.py
#   * condition_info (CLIFF, USE_INTRIN_CENTER)           <- meta_arch/sam3d_body.py _get_decoder_condition
#   * get_ray_condition                                   <- meta_arch/sam3d_body.py (reconstructed; verified vs .pyc bytecode)
#   * ray_cond 1/16 downsample                            <- modules/camera_embed.py CameraEncoder (F.interpolate antialias)
#   * perspective cam_t/focal                             <- heads/camera_head.py PerspectiveHead.perspective_projection (verbatim)
#   * default cam_int                                     <- data/utils/prepare_batch.py
#
# The reference runs on-device in fp16; here we validate the fp32/fp64
# mathematical result the ONNX pipeline should feed.
import importlib.util
import os
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = "/Users/sam/Documents/github/sam-3d-body"
BUILD = os.path.join(REPO, "build")
SCRATCH = os.environ.get("TMPDIR", "/tmp")
IMG = 512
PATCH = 16
GRID = IMG // PATCH  # 32


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Real reference bbox/affine math (bbox_utils.py has no package-relative imports).
bbox_utils = _load(os.path.join(REF, "sam_3d_body/data/transforms/bbox_utils.py"),
                   "ref_bbox_utils")
get_warp_matrix = bbox_utils.get_warp_matrix
bbox_xyxy2cs = bbox_utils.bbox_xyxy2cs
fix_aspect_ratio = bbox_utils.fix_aspect_ratio


# ---- verbatim from heads/camera_head.py PerspectiveHead.perspective_projection ----
def ref_perspective(points_3d, pred_cam, bbox_center, bbox_size, img_size, cam_int,
                    use_intrin_center=True, default_scale_factor=1.0):
    pred_cam = pred_cam.clone()
    pred_cam[..., [0, 2]] *= -1
    s, tx, ty = pred_cam[:, 0], pred_cam[:, 1], pred_cam[:, 2]
    bs = bbox_size * s * default_scale_factor + 1e-8
    focal_length = cam_int[:, 0, 0]
    tz = 2 * focal_length / bs
    if not use_intrin_center:
        cx = 2 * (bbox_center[:, 0] - (img_size[:, 0] / 2)) / bs
        cy = 2 * (bbox_center[:, 1] - (img_size[:, 1] / 2)) / bs
    else:
        cx = 2 * (bbox_center[:, 0] - (cam_int[:, 0, 2])) / bs
        cy = 2 * (bbox_center[:, 1] - (cam_int[:, 1, 2])) / bs
    pred_cam_t = torch.stack([tx + cx, ty + cy, tz], dim=-1)
    return pred_cam_t, focal_length


def ref_process_bbox(bbox_xyxy, pad):
    """GetBBoxCenterScale(pad) + TopdownAffine aspect fixing (prior 0.75 -> 1:1)."""
    center, scale = bbox_xyxy2cs(np.asarray(bbox_xyxy, dtype=np.float64), padding=pad)
    bbox_scale = fix_aspect_ratio(scale.copy(), aspect_ratio=0.75)
    bbox_scale = fix_aspect_ratio(bbox_scale, aspect_ratio=IMG / IMG)
    square = float(bbox_scale[0])
    return center.astype(np.float64), scale.astype(np.float64), square


def default_cam_int(W, H):
    f = (H ** 2 + W ** 2) ** 0.5
    return np.array([[f, 0, W / 2.0], [0, f, H / 2.0], [0, 0, 1]], dtype=np.float64)


def ref_condition_info(bbox_xyxy, pad, cam_int):
    center, _, square = ref_process_bbox(bbox_xyxy, pad)
    f = cam_int[0, 0]
    return np.array([(center[0] - cam_int[0, 2]) / f,
                     (center[1] - cam_int[1, 2]) / f,
                     square / f], dtype=np.float64)


def ref_ray_cond(bbox_xyxy, pad, cam_int):
    """get_ray_condition (reconstructed, verified vs .pyc) then CameraEncoder 1/16."""
    center, _, square = ref_process_bbox(bbox_xyxy, pad)
    warp = get_warp_matrix(center.astype(np.float32),
                           np.array([square, square], dtype=np.float32), 0.0,
                           output_size=(IMG, IMG))  # 2x3 forward (frame->crop)
    affine = torch.tensor(warp, dtype=torch.float64)[None, None]  # [1,1,2,3]
    K = torch.tensor(cam_int, dtype=torch.float64)[None]          # [1,3,3]
    mg = torch.stack(
        torch.meshgrid(torch.arange(IMG), torch.arange(IMG), indexing="xy"), dim=2
    )[None, None].to(torch.float64)  # [1,1,IMG,IMG,2]
    diag = affine[:, :, None, None, [0, 1], [0, 1]]
    trans = affine[:, :, None, None, [0, 1], [2, 2]]
    mg = mg / diag
    mg = mg - trans / diag
    mg = mg - K[:, None, None, None, [0, 1], [2, 2]]
    mg = mg / K[:, None, None, None, [0, 1], [0, 1]]
    mg = mg.permute(0, 1, 4, 2, 3)[0]  # [1,2,IMG,IMG]
    ray = F.interpolate(mg, scale_factor=(1.0 / PATCH, 1.0 / PATCH), mode="bilinear",
                        align_corners=False, antialias=True)  # [1,2,GRID,GRID]
    return ray[0].numpy()  # [2,GRID,GRID]


# Deterministic synthetic image (must match SynthImage in geometry_validate.cpp).
def synth_image(W, H):
    x = np.arange(W)[None, :]
    y = np.arange(H)[:, None]
    img = np.empty((H, W, 3), dtype=np.float64)
    fx = (x / (W - 1)) if W > 1 else np.zeros_like(x, dtype=np.float64)
    fy = (y / (H - 1)) if H > 1 else np.zeros_like(y, dtype=np.float64)
    img[:, :, 0] = np.broadcast_to(fx, (H, W))
    img[:, :, 1] = np.broadcast_to(fy, (H, W))
    img[:, :, 2] = 0.5 * (fx + fy)
    return img


def ref_crop_samples(bbox_xyxy, pad, W, H):
    """Reference 512 crop (cv2 inverse affine + float bilinear) -> 16x16x3 grid."""
    center, _, square = ref_process_bbox(bbox_xyxy, pad)
    inv = get_warp_matrix(center.astype(np.float32),
                          np.array([square, square], dtype=np.float32), 0.0,
                          output_size=(IMG, IMG), inv=True).astype(np.float64)  # crop->frame
    img = synth_image(W, H)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    out = []
    for gy in range(16):
        for gx in range(16):
            v = gy * (IMG // 16) + (IMG // 32)
            u = gx * (IMG // 16) + (IMG // 32)
            sx = inv[0, 0] * u + inv[0, 1] * v + inv[0, 2]
            sy = inv[1, 0] * u + inv[1, 1] * v + inv[1, 2]
            px = bilinear(img, sx, sy)
            out.extend(((px - mean) / std).tolist())
    return np.array(out)


def bilinear(img, x, y):
    H, W = img.shape[:2]
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    wx, wy = x - x0, y - y0
    acc = np.zeros(3)
    for dy in range(2):
        yy = y0 + dy
        if yy < 0 or yy >= H:
            continue
        wyk = wy if dy else 1 - wy
        for dx in range(2):
            xx = x0 + dx
            if xx < 0 or xx >= W:
                continue
            wxk = wx if dx else 1 - wx
            acc += wxk * wyk * img[yy, xx]
    return acc


CASES = [
    # W, H, bbox(x0,y0,x1,y1), pad, cam(None=default or [f,cx,cy]), pred_cam
    (1920, 1080, [600, 200, 1000, 900], 1.25, None, [0.9, 0.05, -0.1]),
    (1280, 720, [100, 50, 300, 600], 1.25, None, [1.1, -0.2, 0.15]),
    (1000, 1000, [400, 400, 700, 850], 1.25, [1200.0, 500.0, 500.0], [0.8, 0.0, 0.0]),
    (1920, 1080, [50, 50, 250, 450], 0.9, None, [1.3, 0.1, 0.2]),
    (800, 1200, [200, 300, 600, 1100], 1.25, [1500.0, 390.0, 610.0], [0.7, -0.05, 0.3]),
    (1500, 1500, [10, 10, 1490, 1490], 1.25, None, [1.0, 0.0, 0.0]),
]


def build_harness():
    os.makedirs(BUILD, exist_ok=True)
    binpath = os.path.join(BUILD, "geometry_validate")
    cmd = [
        "clang++", "-std=c++17", "-O2", "-I", os.path.join(REPO, "src"),
        os.path.join(REPO, "tests/geometry_validate.cpp"),
        os.path.join(REPO, "src/CameraSolver.cpp"),
        os.path.join(REPO, "src/CropAffine.cpp"),
        "-o", binpath,
    ]
    print("compiling:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return binpath


def parse_cpp(path):
    cases = {}
    cur = None
    with open(path) as f:
        for line in f:
            t = line.split()
            if not t:
                continue
            if t[0] == "CASE":
                cur = {}
                cases[int(t[1])] = cur
            else:
                cur[t[0]] = np.array([float(x) for x in t[1:]], dtype=np.float64)
    return cases


def main():
    binpath = build_harness()
    cases_txt = os.path.join(SCRATCH, "geom_cases.txt")
    out_txt = os.path.join(SCRATCH, "geom_out.txt")
    with open(cases_txt, "w") as f:
        for (W, H, bbox, pad, cam, pc) in CASES:
            camflag = 0 if cam is None else 1
            f_, cx, cy = (cam if cam is not None else (0, 0, 0))
            f.write(f"{W} {H} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]} {pad} "
                    f"{camflag} {f_} {cx} {cy} {pc[0]} {pc[1]} {pc[2]} 1.0\n")
    subprocess.run([binpath, cases_txt, out_txt], check=True)
    cpp = parse_cpp(out_txt)

    fields = ["center", "scale", "square", "aff_fwd", "aff_inv", "cond", "perspective",
              "ray_cond", "crop"]
    maxdiff = {k: 0.0 for k in fields}

    for i, (W, H, bbox, pad, cam, pc) in enumerate(CASES):
        # The pipeline carries cam_int as float32 tensors and the contract stores
        # Camera.focal as float32, so mirror that precision in the reference.
        K = default_cam_int(W, H) if cam is None else np.array(
            [[cam[0], 0, cam[1]], [0, cam[0], cam[2]], [0, 0, 1]], dtype=np.float64)
        K = K.astype(np.float32).astype(np.float64)
        center, scale, square = ref_process_bbox(bbox, pad)
        warp_f = get_warp_matrix(center.astype(np.float32),
                                 np.array([square, square], dtype=np.float32), 0.0,
                                 output_size=(IMG, IMG)).reshape(-1).astype(np.float64)
        warp_i = get_warp_matrix(center.astype(np.float32),
                                 np.array([square, square], dtype=np.float32), 0.0,
                                 output_size=(IMG, IMG), inv=True).reshape(-1).astype(np.float64)
        cond = ref_condition_info(bbox, pad, K)
        ray = ref_ray_cond(bbox, pad, K).reshape(-1)
        pct, focal = ref_perspective(
            torch.zeros(1, 1, 3, dtype=torch.float64),
            torch.tensor(np.array([pc]), dtype=torch.float64),
            torch.tensor(np.array([center]), dtype=torch.float64),
            torch.tensor(np.array([square]), dtype=torch.float64),
            torch.tensor(np.array([[W, H]]), dtype=torch.float64),
            torch.tensor(np.array([K]), dtype=torch.float64),
            use_intrin_center=True)
        persp = np.array([focal[0].item(), pct[0, 0].item(), pct[0, 1].item(),
                          pct[0, 2].item()])
        crop = ref_crop_samples(bbox, pad, W, H)

        c = cpp[i]
        diffs = {
            "center": np.abs(c["CENTER"] - center).max(),
            "scale": np.abs(c["SCALE"] - scale).max(),
            "square": abs(c["SQUARE"][0] - square),
            "aff_fwd": np.abs(c["AFF"] - warp_f).max(),
            "aff_inv": np.abs(c["AFFINV"] - warp_i).max(),
            "cond": np.abs(c["COND"] - cond).max(),
            "perspective": np.abs(c["PERSP"] - persp).max(),
            "ray_cond": np.abs(c["RAY"] - ray).max(),
            "crop": np.abs(c["CROP"] - crop).max(),
        }
        for k, v in diffs.items():
            maxdiff[k] = max(maxdiff[k], float(v))
        print(f"case {i} (W={W} H={H} pad={pad} cam={'default' if cam is None else 'given'}): "
              + " ".join(f"{k}={v:.2e}" for k, v in diffs.items()))

    print("\n=== max abs diff vs reference ===")
    for k in fields:
        print(f"  {k:14s} {maxdiff[k]:.3e}")

    tol = 1e-4
    ok = all(v < tol for v in maxdiff.values())
    write_report(maxdiff, tol, ok)
    if not ok:
        print("\nFAIL: some diffs exceed 1e-4")
        sys.exit(1)
    print(f"\nPASS: all diffs < {tol}")


def write_report(maxdiff, tol, ok):
    path = os.path.join(REPO, "tools/GEOMETRY_REPORT.md")
    lines = [
        "# Track F geometry validation report",
        "",
        f"Status: {'PASS' if ok else 'FAIL'} (tolerance {tol:g}, {len(CASES)} cases)",
        "",
        "C++ (`src/CameraSolver.{h,cpp}`, `src/CropAffine.{h,cpp}`) vs the",
        "sam-3d-body reference (@ c259bfc). Reference runs on-device in fp16; we",
        "validate the fp32/fp64 mathematical result the ONNX pipeline feeds.",
        "",
        "| quantity | reference | max abs diff |",
        "|---|---|---|",
        f"| GetBBoxCenterScale center | bbox_xyxy2cs | {maxdiff['center']:.3e} |",
        f"| GetBBoxCenterScale scale (padded) | bbox_xyxy2cs | {maxdiff['scale']:.3e} |",
        f"| TopdownAffine square side | fix_aspect_ratio x2 | {maxdiff['square']:.3e} |",
        f"| TopdownAffine affine (forward) | get_warp_matrix | {maxdiff['aff_fwd']:.3e} |",
        f"| TopdownAffine affine (inverse) | get_warp_matrix inv | {maxdiff['aff_inv']:.3e} |",
        f"| condition_info (CLIFF, intrin center) | _get_decoder_condition | {maxdiff['cond']:.3e} |",
        f"| ray_cond [2,32,32] | get_ray_condition + CameraEncoder 1/16 | {maxdiff['ray_cond']:.3e} |",
        f"| perspective_projection (focal, cam_t) | camera_head | {maxdiff['perspective']:.3e} |",
        f"| crop warp (ImageNet CHW, 16x16x3 samples) | inverse affine + bilinear | {maxdiff['crop']:.3e} |",
        "",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", path)


if __name__ == "__main__":
    main()
