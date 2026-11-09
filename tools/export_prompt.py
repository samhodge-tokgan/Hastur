#!/usr/bin/env python
# Copyright the Hastur authors.
# SPDX-License-Identifier: LicenseRef-SAM-License
#
# export_prompt.py -- export the SAM-3D-Body *keypoint-prompted body re-run*
# (run_inference -> run_keypoint_prompt) to a fixed-shape fp16 ONNX.
#
#   inputs : image_embeddings[1,1280,32,32]  (sam3dbody_body.onnx output: backbone
#                                             + no-mask embed, pre ray conditioning)
#            ray_cond[1,2,32,32]   (the body crop's token-grid rays, as the body graph)
#            condition_info[1,3]   (the body crop's CLIFF condition, as the body graph)
#            keypoints[1,4,3]      (x, y in [0,1] of the body crop, label) for
#                                  [right wrist 41, left wrist 62, right elbow 8,
#                                   left elbow 7]; an unused slot is the dummy
#                                  prompt [0, 0, -2]
#            prev_estimate[1,522]  (the first pass: pred[519] ++ pred_cam[3])
#   outputs: pred[1,519]  pred_cam[1,3]  kp2d[1,70,2]
#
# This is the body decoder only (no backbone), so the graph is small. It reuses
# export_sam3dbody's wrapper and op replacements verbatim; the only change is the
# prompt encoder's keypoint embedding, rewritten without boolean-mask in-place
# writes (one-hot matmul, numerically identical) so it traces to a compact graph.
#
# Fixed K = 4 with dummies for unused slots is the reference's own batched
# (multi-person) semantics in run_inference. The single-person reference instead
# drops invalid slots; measured on the labelled hands by tools/validate_prompt.py.
import argparse
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import export_sam3dbody as B  # noqa: E402

DEFAULT_OUT = os.path.join(os.path.dirname(HERE), "models", "sam3dbody_prompt.onnx")
INPUT_NAMES = ["image_embeddings", "ray_cond", "condition_info", "keypoints",
               "prev_estimate"]
OUTPUT_NAMES = ["pred", "pred_cam", "kp2d"]
# run_inference: hand-decoder wrists, body elbows (mhr70 indices = prompt labels).
PROMPT_LABELS = [41, 62, 8, 7]
NUM_PROMPTS = len(PROMPT_LABELS)


def embed_keypoints_onehot(self, points, labels):
    """PromptEncoder._embed_keypoints without masked in-place writes."""
    pe = self.pe_layer._pe_encoding(points.to(torch.float))
    lab = labels.unsqueeze(-1)
    pe = pe * (lab >= 0).to(pe.dtype)                       # -1/-2 rows zeroed
    pe = pe + (lab == -2).to(pe.dtype) * self.invalid_point_embed.weight[0]
    pe = pe + (lab == -1).to(pe.dtype) * self.not_a_point_embed.weight[0]
    W = torch.cat([e.weight for e in self.point_embeddings], dim=0)  # (J,C)
    ids = torch.arange(W.shape[0], dtype=labels.dtype, device=labels.device)
    onehot = (lab == ids).to(pe.dtype)                      # (B,N,J)
    pe = pe + onehot @ W
    return pe, labels > -2


class PromptWrapper(B.BodyWrapper):
    """Body decoder re-run with keypoint prompts and the first-pass estimate."""

    def __init__(self, model, cfg, mesh=None):
        super().__init__(model, cfg, mesh=mesh, mesh_free=False)
        pe = model.prompt_encoder
        pe._embed_keypoints = embed_keypoints_onehot.__get__(pe, type(pe))

    def forward(self, image_embeddings, ray_cond, condition_info, keypoints,
                prev_estimate):
        model = self.model
        model._cond_body = condition_info
        self._fake_batch["ray_cond"] = ray_cond
        _, pose_outputs = model.forward_decoder(
            image_embeddings, init_estimate=None, keypoints=keypoints,
            prev_estimate=prev_estimate.unsqueeze(1), condition_info=condition_info,
            batch=self._fake_batch)
        po = pose_outputs[-1]
        pred = torch.cat([po["pred_pose_raw"], po["shape"], po["scale"],
                          po["hand"], po["face"] * 0.0], dim=1)
        return pred, po["pred_cam"], po["pred_keypoints_2d"][..., :2]


def example_inputs(model, cfg):
    """A real first pass on the demo image (body wrapper outputs), plus a
    representative prompt set (its own wrists/elbows, nudged)."""
    mesh = B.build_mesh(use_double=False)
    body = B.BodyWrapper(model, cfg, mesh=mesh).eval()
    batch = B.build_batch()
    image, ray32, cond = B.make_onnx_inputs(model, cfg, batch)
    with torch.no_grad():
        pred, pred_cam, _, _, ie, kp2d = body(image, ray32, cond)
    kp = torch.zeros(1, NUM_PROMPTS, 3)
    for i, j in enumerate(PROMPT_LABELS):
        kp[0, i, :2] = (kp2d[0, j] + 0.5 + 0.01).clamp(0, 1)
        kp[0, i, 2] = j
    prev = torch.cat([pred, pred_cam], dim=1)
    return ie, ray32, cond, kp, prev


def export(out=DEFAULT_OUT, keep_fp32=False):
    import shutil
    import tempfile

    os.makedirs(os.path.dirname(out), exist_ok=True)
    print("[export-prompt] loading model ...")
    model, cfg = B.load_model()
    feeds = example_inputs(model, cfg)
    model, cfg = B.load_model()  # fresh: example_inputs patched op-methods
    wrap = PromptWrapper(model, cfg, mesh=B.build_mesh(use_double=False)).eval()
    with torch.no_grad():
        wrap(*feeds)  # smoke run

    keepdir = out.replace(".onnx", "_fp32")
    tmpdir = keepdir if keep_fp32 else tempfile.mkdtemp(prefix="sam3dbody_prompt_export_")
    os.makedirs(tmpdir, exist_ok=True)
    fp32 = os.path.join(tmpdir, "prompt_fp32.onnx")
    print("[export-prompt] tracing + ONNX export (fp32) ...")
    torch.onnx.export(wrap, feeds, fp32, input_names=INPUT_NAMES,
                      output_names=OUTPUT_NAMES, opset_version=17, dynamo=False,
                      do_constant_folding=True)
    B._convert_and_save_fp16(fp32, out)
    if not keep_fp32:
        shutil.rmtree(tmpdir, ignore_errors=True)
    write_sidecar(out)
    return out


def write_sidecar(out):
    import json
    meta = {
        "name": "sam3dbody_prompt",
        "kind": "keypoint-prompted body decoder re-run (run_keypoint_prompt)",
        "precision": "fp16 (fp16 I/O)",
        "inputs": {"image_embeddings": [1, 1280, 32, 32], "ray_cond": [1, 2, 32, 32],
                   "condition_info": [1, 3], "keypoints": [1, NUM_PROMPTS, 3],
                   "prev_estimate": [1, 522]},
        "outputs": {"pred": [1, 519], "pred_cam": [1, 3], "kp2d": [1, 70, 2]},
        "prompt_labels": PROMPT_LABELS,
        "dummy_prompt": [0, 0, -2],
        "size_bytes": os.path.getsize(out),
    }
    jpath = out.rsplit(".", 1)[0] + ".json"
    with open(jpath, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[export-prompt] wrote {jpath}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--keep-fp32", action="store_true")
    args = ap.parse_args()
    export(args.out, args.keep_fp32)


if __name__ == "__main__":
    main()
