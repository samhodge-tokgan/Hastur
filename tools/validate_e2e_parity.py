#!/usr/bin/env python
# Copyright the Hastur authors.
# SPDX-License-Identifier: LicenseRef-SAM-License
#
# validate_e2e_parity.py -- M5 end-to-end numeric parity: the C++ Sam3dBodyPipeline
# vs the Python SAM-3D-Body reference, per detected person.
#
# The C++ side is produced by tools/parity_dump (a standalone harness that runs the
# REAL Sam3dBodyPipeline and dumps pred[519] / solved camera / verts / keypoints /
# silhouette to <prefix>.bin, plus the exact RGB frame to <prefix>.png).
#
# The reference here is the SAM-3D-Body body decoder driven through its own
# `forward_decoder` with the REAL momentum-MHR mesh + the reference `camera_project`
# (== sam_3d_body_estimator.process_one_image's body branch, minus the clone's
# broken `.to("mps")` device hacks that make process_one_image itself un-runnable;
# this is the same faithful CPU/fp32 reference tools/export_sam3dbody.py validates
# against). It yields the reference pred_vertices / pred_keypoints_3d /
# pred_keypoints_2d(full-frame px) / pred_cam_t / focal_length / mhr_model_params.
#
# TWO comparisons per image (see the M5 task):
#   (a) FIXED-BBOX  -- feed the C++ pipeline's detected box to the reference too, so
#       BOTH run the identical person box. Isolates true PIPELINE parity (fp16 body
#       ONNX vs torch, C++ MHR LBS vs momentum MHR, C++ camera solver vs reference)
#       from the detector swap.
#   (b) END-TO-END  -- the reference runs its OWN ViTDet-H detector; the C++ used its
#       ssdlite320 detector. The realistic delta incl. the different crop.
#
# Metrics: pred max_abs + Pearson; verts per-vertex RMSE (mm); cam_t / focal rel err;
# keypoint 2D reprojection error (px); silhouette IoU (reference mesh projected with
# the reference camera vs the C++ silhouette, and a same-rasterizer geometry IoU).
#
# NOTE: pyrender has no headless-GL backend on this Mac, so a pyrender-exact PIXEL
# comparison of the shaded render is deferred to the Linux VM (M9). Here the render
# parity is measured geometrically: projected-silhouette IoU + keypoint reprojection
# + metric cam_t, which is what actually drives the composited coverage.
import argparse
import os
import struct
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import export_sam3dbody as E  # noqa: E402  (load_model/build_batch/clean helpers)

REF_ROOT = E.REF_ROOT
sys.path.insert(0, REF_ROOT)
REPORT = os.path.join(HERE, "E2E_PARITY_REPORT.md")


# --------------------------------------------------------------------------- #
#  C++ dump reader (HPDUMP2, see tools/parity_dump.cpp)
# --------------------------------------------------------------------------- #
def read_dump(path):
    with open(path, "rb") as f:
        buf = f.read()
    off = 0

    def take(fmt):
        nonlocal off
        n = struct.calcsize(fmt)
        v = struct.unpack_from(fmt, buf, off)
        off += n
        return v

    magic = buf[:8]
    off = 8
    assert magic[:7] == b"HPDUMP2", f"bad magic {magic!r}"
    W, H, npeople = take("<3i")
    people = []
    for _ in range(npeople):
        box = np.array(take("<5f"), np.float64)
        pred = np.array(take("<519f"), np.float64)
        pred_cam = np.array(take("<3f"), np.float64)
        cam = np.array(take("<6f"), np.float64)  # focal, cx, cy, tx, ty, tz
        (nv,) = take("<i")
        verts = np.array(take(f"<{nv * 3}f"), np.float64).reshape(nv, 3)
        (nk,) = take("<i")
        kpts = np.array(take(f"<{nk * 3}f"), np.float64).reshape(nk, 3)
        people.append(dict(box=box, pred=pred, pred_cam=pred_cam, cam=cam,
                           verts=verts, kpts=kpts))
    alpha = np.array(take(f"<{W * H}f"), np.float64).reshape(H, W)
    return dict(W=W, H=H, people=people, alpha=alpha)


# --------------------------------------------------------------------------- #
#  Reference forward (real momentum MHR + reference camera_project), body-only.
#  Mirrors export_sam3dbody.run_reference but returns the FULL final pose_output.
# --------------------------------------------------------------------------- #
@torch.no_grad()
def ref_forward(model, cfg, box_xyxy, img_path):
    batch = E.build_batch(box=np.asarray(box_xyxy, np.float32).reshape(1, 4),
                          img_path=img_path)
    model._initialize_batch(batch)
    B, N = batch["img"].shape[:2]
    model.body_batch_idx = list(range(B * N))
    model.hand_batch_idx = []
    x = E.data_preprocess_clean(model, model._flatten_person(batch["img"]))
    ray = model._flatten_person(E.get_ray_condition_clean(model, batch))
    batch["ray_cond"] = ray[model.body_batch_idx].clone()
    ie = model.backbone(x.type(model.backbone_dtype), extra_embed=None).type(x.dtype)
    if cfg.MODEL.PROMPT_ENCODER.get("MASK_EMBED_TYPE", None) is not None:
        ie = ie + model._get_mask_prompt(batch, ie)
    cond = model._get_decoder_condition(batch)
    kp = torch.zeros((B * N, 1, 3))
    kp[:, :, -1] = -2
    _tokens, pose_outputs = model.forward_decoder(
        ie[model.body_batch_idx], init_estimate=None,
        keypoints=kp[model.body_batch_idx], prev_estimate=None,
        condition_info=cond[model.body_batch_idx], batch=batch)
    po = pose_outputs[-1]
    pred = torch.cat([po["pred_pose_raw"], po["shape"], po["scale"], po["hand"],
                      po["face"] * 0.0], dim=1)
    g = lambda k: np.asarray(po[k].detach().cpu().float().numpy())
    return dict(
        pred=pred.detach().cpu().float().numpy()[0],           # (519,)
        verts=g("pred_vertices")[0],                            # (V,3) m, flipped
        kp3d=g("pred_keypoints_3d")[0],                         # (70,3)
        kp2d=g("pred_keypoints_2d")[0],                         # (70,2) full-frame px
        cam_t=g("pred_cam_t")[0],                               # (3,)
        focal=float(g("focal_length").ravel()[0]),
        box=np.asarray(batch["bbox"][0, 0].cpu().numpy(), np.float64),
    )


# --------------------------------------------------------------------------- #
#  Metrics
# --------------------------------------------------------------------------- #
def pearson(a, b):
    a = a.ravel(); b = b.ravel()
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 1.0
    return float(np.corrcoef(a, b)[0, 1])


def project(pts3d, focal, cx, cy, cam_t):
    """CV pinhole: v_cam = pts + cam_t; u = f*x/z + cx; v = f*y/z + cy."""
    p = pts3d + cam_t[None, :]
    z = np.clip(p[:, 2], 1e-6, None)
    u = focal * p[:, 0] / z + cx
    v = focal * p[:, 1] / z + cy
    return np.stack([u, v], axis=1)


def rasterize_mask(verts, faces, focal, cx, cy, cam_t, W, H):
    """Silhouette = UNION of all forward-projected triangles. Each triangle is
    filled separately with fillConvexPoly (accumulating OR); cv2.fillPoly on the
    whole triangle list would apply an even-odd rule and punch holes where an even
    number of overlapping triangles cover a pixel."""
    import cv2
    uv = project(verts, focal, cx, cy, cam_t)
    tri = uv[faces].astype(np.int32)  # (F,3,2)
    mask = np.zeros((H, W), np.uint8)
    for t in tri:
        cv2.fillConvexPoly(mask, t, 1)
    return mask.astype(bool)


def iou(a, b):
    a = a.astype(bool); b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    uni = np.logical_or(a, b).sum()
    return float(inter) / float(uni) if uni > 0 else float("nan")


def compare(cpp, ref, faces, W, H, alpha_cpp):
    m = {}
    # pred[519]
    m["pred_maxabs"] = float(np.max(np.abs(cpp["pred"] - ref["pred"])))
    m["pred_pearson"] = pearson(cpp["pred"], ref["pred"])
    # verts per-vertex RMSE (mm)
    d = np.linalg.norm(cpp["verts"] - ref["verts"], axis=1)  # (V,) m
    m["verts_rmse_mm"] = float(np.sqrt(np.mean(d ** 2)) * 1000.0)
    m["verts_mean_mm"] = float(np.mean(d) * 1000.0)
    m["verts_max_mm"] = float(np.max(d) * 1000.0)
    # camera
    ct_c, ct_r = cpp["cam"][3:6], ref["cam_t"]
    m["camt_relerr"] = float(np.linalg.norm(ct_c - ct_r) / (np.linalg.norm(ct_r) + 1e-9))
    m["camt_cpp"] = ct_c.tolist()
    m["camt_ref"] = ct_r.tolist()
    f_c, f_r = float(cpp["cam"][0]), float(ref["focal"])
    m["focal_relerr"] = abs(f_c - f_r) / (abs(f_r) + 1e-9)
    m["focal_cpp"], m["focal_ref"] = f_c, f_r
    # keypoint 2D reprojection error (px): each side projects its own 3D kpts with
    # its own solved camera; compare in full-frame pixels.
    cx, cy = W / 2.0, H / 2.0
    kp2d_cpp = project(cpp["kpts"], f_c, cx, cy, ct_c)
    kp2d_ref = ref["kp2d"]  # reference already full-frame px
    e = np.linalg.norm(kp2d_cpp - kp2d_ref, axis=1)
    m["kp2d_mean_px"] = float(np.mean(e))
    m["kp2d_med_px"] = float(np.median(e))
    m["kp2d_max_px"] = float(np.max(e))
    # silhouette IoU
    mask_ref = rasterize_mask(ref["verts"], faces, f_r, cx, cy, ct_r, W, H)
    mask_cpp_geom = rasterize_mask(cpp["verts"], faces, f_c, cx, cy, ct_c, W, H)
    m["iou_ref_vs_cpprender"] = iou(mask_ref, alpha_cpp > 0.5)
    m["iou_ref_vs_cppgeom"] = iou(mask_ref, mask_cpp_geom)
    m["mask_ref_px"] = int(mask_ref.sum())
    m["mask_cpprender_px"] = int((alpha_cpp > 0.5).sum())
    return m


def fmt_block(title, m):
    L = [f"### {title}"]
    L.append(f"- pred[519]: max_abs = {m['pred_maxabs']:.4e}, Pearson = {m['pred_pearson']:.6f}")
    L.append(f"- verts per-vertex RMSE = {m['verts_rmse_mm']:.4f} mm "
             f"(mean {m['verts_mean_mm']:.4f} mm, max {m['verts_max_mm']:.4f} mm)")
    L.append(f"- cam_t rel err = {m['camt_relerr']:.4e}  "
             f"(cpp {np.array(m['camt_cpp']).round(4).tolist()} vs "
             f"ref {np.array(m['camt_ref']).round(4).tolist()})")
    L.append(f"- focal rel err = {m['focal_relerr']:.4e}  "
             f"(cpp {m['focal_cpp']:.2f} vs ref {m['focal_ref']:.2f})")
    L.append(f"- keypoint 2D reproj err = {m['kp2d_mean_px']:.3f} px mean "
             f"(median {m['kp2d_med_px']:.3f}, max {m['kp2d_max_px']:.3f})")
    L.append(f"- silhouette IoU (ref proj vs C++ render) = {m['iou_ref_vs_cpprender']:.4f}; "
             f"same-rasterizer geom IoU = {m['iou_ref_vs_cppgeom']:.4f}")
    return "\n".join(L)


# --------------------------------------------------------------------------- #
def build_vitdet(device):
    # Python 3.12 removed pkgutil.ImpImporter, which detectron2's (old) pkg_resources
    # import chain still references. Shim it so the cached ViTDet weights can load.
    import pkgutil
    if not hasattr(pkgutil, "ImpImporter"):
        class _Imp:  # minimal no-op finder
            def __init__(self, *a, **k):
                pass

            def find_module(self, *a, **k):
                return None
        pkgutil.ImpImporter = _Imp
    from tools.build_detector import HumanDetector
    return HumanDetector(name="vitdet", device=device,
                         path=os.environ.get("SAM3D_DETECTOR_PATH", ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", action="append", required=True,
                    help="C++ dump prefix (expects <prefix>.bin + <prefix>.png). "
                         "Repeatable; label with name=prefix.")
    ap.add_argument("--no-e2e", action="store_true", help="skip the ViTDet end-to-end pass")
    args = ap.parse_args()

    print("[ref] loading SAM-3D-Body model (cpu/fp32, real momentum MHR) ...")
    model, cfg = E.load_model(device="cpu")
    faces = model.head_pose.faces.detach().cpu().numpy().astype(np.int32)
    print(f"[ref] faces {faces.shape}")

    detector = None
    if not args.no_e2e:
        try:
            print("[ref] loading ViTDet-H detector (cached weights) ...")
            detector = build_vitdet("cpu")
        except Exception as ex:
            print(f"[ref] ViTDet unavailable ({repr(ex)[:160]}); end-to-end skipped")
            detector = None

    import cv2
    out = ["# M5 -- C++ Sam3dBodyPipeline vs Python SAM-3D-Body reference (numeric parity)\n"]
    results = {}
    for spec in args.dump:
        name, prefix = spec.split("=", 1) if "=" in spec else (os.path.basename(spec), spec)
        binp, pngp = prefix + ".bin", prefix + ".png"
        print(f"\n===== {name} =====")
        d = read_dump(binp)
        W, H = d["W"], d["H"]
        cpp = d["people"][0]
        print(f"image {W}x{H}; C++ box {cpp['box'][:4].round(1).tolist()} score {cpp['box'][4]:.3f}")

        # (a) fixed-bbox: reference uses the C++ detected box.
        refa = ref_forward(model, cfg, cpp["box"][:4], pngp)
        ma = compare(cpp, refa, faces, W, H, d["alpha"])
        print(fmt_block("(a) fixed-bbox pipeline parity", ma))

        blocks = [f"## {name}  ({W}x{H})",
                  f"C++ detector box: {cpp['box'][:4].round(1).tolist()} (score {cpp['box'][4]:.3f})",
                  fmt_block("(a) FIXED-BBOX -- pipeline parity (same box to both)", ma)]

        mb = None
        if detector is not None:
            img_bgr = cv2.imread(pngp)  # detector expects BGR
            boxes = detector.run_human_detection(
                img_bgr, det_cat_id=0, bbox_thr=0.5, nms_thr=0.3,
                default_to_full_image=False)
            if len(boxes):
                # pick the box closest to the C++ box (same person)
                cx0 = (cpp["box"][0] + cpp["box"][2]) / 2
                cy0 = (cpp["box"][1] + cpp["box"][3]) / 2
                cc = np.array([[ (b[0]+b[2])/2, (b[1]+b[3])/2 ] for b in boxes])
                j = int(np.argmin(np.linalg.norm(cc - [cx0, cy0], axis=1)))
                vbox = boxes[j]
                refb = ref_forward(model, cfg, vbox, pngp)
                mb = compare(cpp, refb, faces, W, H, d["alpha"])
                mb["_vitdet_box"] = np.asarray(vbox, np.float64).round(1).tolist()
                print(fmt_block("(b) end-to-end (ViTDet ref box)", mb))
                blocks.append(f"ViTDet-H box: {mb['_vitdet_box']}  (C++ ssdlite box above)")
                blocks.append(fmt_block("(b) END-TO-END -- each with its own detector", mb))
            else:
                blocks.append("(b) END-TO-END: ViTDet found no person; skipped.")
        else:
            blocks.append("(b) END-TO-END: ViTDet detector unavailable; skipped.")

        out.append("\n\n".join(blocks))
        results[name] = dict(fixed=ma, e2e=mb, W=W, H=H, box=cpp["box"].tolist())

    with open(REPORT, "w") as f:
        f.write("\n\n---\n\n".join(out) + "\n")
    print(f"\n[parity] wrote {REPORT}")

    # machine-readable summary line for quick scanning
    print("\n[SUMMARY]")
    for name, r in results.items():
        a = r["fixed"]
        print(f"  {name:18s} fixed: pred_maxabs={a['pred_maxabs']:.2e} "
              f"verts_rmse={a['verts_rmse_mm']:.3f}mm camt_rel={a['camt_relerr']:.1e} "
              f"kp2d={a['kp2d_mean_px']:.2f}px IoU={a['iou_ref_vs_cpprender']:.3f}")


if __name__ == "__main__":
    main()
