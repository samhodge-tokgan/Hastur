#!/usr/bin/env python
# Copyright the Hastur authors.
# SPDX-License-Identifier: LicenseRef-SAM-License
#
# validate_prompt.py -- parity for the keypoint-prompted body re-run graph.
#
#   1. reference (unmodified forward_decoder + real MHR head, the exact
#      run_keypoint_prompt call) vs the PromptWrapper torch module, fed the
#      reference's own first-pass image_embeddings / prev_estimate;
#   2. the body graph's new outputs (image_embeddings, kp2d) vs the reference;
#   3. onnxruntime (fp16) vs the wrapper, for sam3dbody_prompt.onnx and, when
#      given, sam3dbody_body.onnx / sam3dbody_hand.onnx's new outputs;
#   4. fixed-K dummies vs the single-person reference dropping invalid slots.
import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import export_sam3dbody as B  # noqa: E402
import export_prompt as P  # noqa: E402


def stats(a, b):
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    d = np.abs(a - b)
    p = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")
    return float(d.max()), float(d.mean()), p


@torch.no_grad()
def reference_chain(box, kp_fn):
    """First pass + run_keypoint_prompt exactly as the reference runs them."""
    model, cfg = B.load_model()
    batch = B.build_batch(box=box)
    first = B.run_reference(model, cfg, batch)        # sets batch["ray_cond"] etc.
    cond = model._get_decoder_condition(batch)
    prev = torch.cat([first["pred"], first["pred_cam"]], dim=1).unsqueeze(1)
    kp = kp_fn(first["kp2d"])
    _, po = model.forward_decoder(first["image_embeddings"], init_estimate=None,
                                  keypoints=kp, prev_estimate=prev,
                                  condition_info=cond, batch=batch)
    po = po[-1]
    pred = torch.cat([po["pred_pose_raw"], po["shape"], po["scale"], po["hand"],
                      po["face"] * 0.0], dim=1)
    return first, kp, pred, po["pred_cam"]


def prompts(kp2d, dummy=None, drop=None):
    kp = torch.zeros(1, P.NUM_PROMPTS, 3)
    for i, j in enumerate(P.PROMPT_LABELS):
        kp[0, i, :2] = (kp2d[0, j] + 0.5 + 0.02).clamp(0, 1)   # displaced prompt
        kp[0, i, 2] = j
    if dummy is not None:
        kp[0, dummy] = torch.tensor([0.0, 0.0, -2.0])
    if drop is not None:
        kp = kp[:, [i for i in range(P.NUM_PROMPTS) if i != drop]]
    return kp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-onnx", default=P.DEFAULT_OUT)
    ap.add_argument("--body-onnx", default=None)
    args = ap.parse_args()
    torch.manual_seed(0)
    box = np.array([[60, 40, 520, 760]], np.float32)  # off-centre crop of the demo image

    first, kp, ref_pred, ref_cam = reference_chain(box, prompts)
    model, cfg = B.load_model()
    image, ray32, cond = B.make_onnx_inputs(model, cfg, B.build_batch(box=box))

    # 2: body wrapper new outputs vs reference
    body = B.BodyWrapper(model, cfg, mesh=B.build_mesh(use_double=False)).eval()
    with torch.no_grad():
        bo = dict(zip(B.OUTPUT_NAMES, body(image, ray32, cond)))
    for n in ("pred", "image_embeddings", "kp2d"):
        print(f"body wrapper {n:17s} vs reference: max_abs=%.3e mean=%.3e r=%.7f"
              % stats(bo[n], first[n]))

    # 1: prompt wrapper vs reference re-run
    model, cfg = B.load_model()
    wrap = P.PromptWrapper(model, cfg, mesh=B.build_mesh(use_double=False)).eval()
    prev = torch.cat([first["pred"], first["pred_cam"]], dim=1)
    feeds_t = (first["image_embeddings"], ray32, cond, kp, prev)
    with torch.no_grad():
        wp, wcam, wkp = wrap(*feeds_t)
    print("prompt wrapper pred     vs reference: max_abs=%.3e mean=%.3e r=%.7f"
          % stats(wp, ref_pred))
    print("prompt wrapper pred_cam vs reference: max_abs=%.3e mean=%.3e r=%.7f"
          % stats(wcam, ref_cam))
    print("re-run moved pred by: max_abs=%.3e mean=%.3e" % stats(ref_pred, first["pred"])[:2])

    # 4: dummies vs dropping a slot (single-person reference semantics)
    for slot in range(P.NUM_PROMPTS):
        _, _, pd, _ = reference_chain(box, lambda k, s=slot: prompts(k, dummy=s))
        _, _, pr, _ = reference_chain(box, lambda k, s=slot: prompts(k, drop=s))
        print(f"slot {slot} invalid: dummy vs drop pred max_abs=%.3e mean=%.3e" % stats(pd, pr)[:2])

    # 3: onnxruntime
    import onnxruntime as ort
    h = lambda t: t.numpy().astype(np.float16)
    if args.prompt_onnx and os.path.exists(args.prompt_onnx):
        s = ort.InferenceSession(args.prompt_onnx, providers=["CPUExecutionProvider"])
        o = s.run(P.OUTPUT_NAMES, dict(zip(P.INPUT_NAMES, map(h, feeds_t))))
        print("prompt onnx pred vs wrapper: max_abs=%.3e mean=%.3e r=%.7f" % stats(o[0], wp))
        print("prompt onnx pred vs reference: max_abs=%.3e mean=%.3e r=%.7f" % stats(o[0], ref_pred))
        print("prompt onnx pred_cam vs wrapper: max_abs=%.3e mean=%.3e r=%.7f" % stats(o[1], wcam))
    if args.body_onnx and os.path.exists(args.body_onnx):
        s = ort.InferenceSession(args.body_onnx, providers=["CPUExecutionProvider"])
        o = dict(zip(B.OUTPUT_NAMES, s.run(B.OUTPUT_NAMES, {
            "image": h(image), "ray_cond": h(ray32), "condition_info": h(cond)})))
        for n in ("pred", "pred_cam", "image_embeddings", "kp2d"):
            print(f"body onnx {n:17s} vs wrapper: max_abs=%.3e mean=%.3e r=%.7f" % stats(o[n], bo[n]))


if __name__ == "__main__":
    main()
