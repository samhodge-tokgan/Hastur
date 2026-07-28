#!/usr/bin/env python3
# Copyright the Hastur authors.
# SPDX-License-Identifier: LicenseRef-SAM-License
#
# export_sam3_tracker.py — carve Meta's SAM 3 video tracker (sam3.pt) into the five
# per-frame ONNX graphs + learned .npy constants that Hastur's C++ tracker engine
# (src/Sam3MultiTracker.cpp / Sam3TrackerEngine.cpp) drives. This is the tracker
# analogue of export_sam3dbody.py: it produces the whole-clip video-tracker model
# set that gives temporally-STABLE person_NN ids (external tracks), replacing the
# per-frame cam_t association that flickers.
#
# Outputs (into --out), matching MultiPaths in Sam3MultiTracker.cpp / T3BuildPaths:
#   G1.onnx  image[1,3,1008,1008] ->
#            fpn0,fpn1,fpn2, pos0,pos1,pos2, det_fpn0,det_fpn1,det_fpn2, det_pos72
#   G2.onnx  fpn0,fpn1,fpn2,pos72,lang_feats,lang_mask(bool) ->
#            detector pred_logits, pred_boxes, pred_masks, presence
#   G3.onnx  src,src_pos,prompt,prompt_pos,num_obj_ptr_tokens(int64) -> attended
#   G4.onnx  backbone_features,high_res_0,high_res_1 ->
#            low_res(288^2), pred_masks_high_res(1008^2), ious, obj_ptr[1,256], object_score_logits
#   G5.onnx  pix_feat,pred_masks_high_res,object_score_logits ->
#            maskmem_features[1,64,72,72], maskmem_pos_enc[1,64,72,72]
#   const_maskmem_tpos_enc.npy    [7,1,1,64]   tracker.maskmem_tpos_enc
#   const_no_mem_embed.npy        [1,1,256]    tracker.no_mem_embed
#   const_objptr_tpos_proj_w.npy  [64,256]     tracker.obj_ptr_tpos_proj.weight
#   const_objptr_tpos_proj_b.npy  [64]         tracker.obj_ptr_tpos_proj.bias
#   const_lang_feats_person.npy   [1,32,1024?] detector text-encoder("person")
#   const_lang_mask_person.npy    [1,32] bool  text-encoder padding mask for "person"
#
# Run with the sam-3d-body env that has torch + onnx + the `sam3` package:
#   .../sam-3d-body/env/bin/python tools/export_sam3_tracker.py \
#       --ckpt <sam3.pt> --out <model dir> [--device cuda] [--opset 17]
import argparse
import os

import numpy as np
import torch

IMG = 1008
GRID = 72          # top-level feature grid (1008/14 for a /14 backbone)
LOWRES = 288
MEMDIM = 64
HIDDEN = 256


def realify_vitdet_rope(model):
    """The ViT trunk (vitdet.py) stores each attention block's rotary table as a
    *complex* buffer `freqs_cis` (torch.polar) and rotates via view_as_complex — ONNX/
    ORT cannot represent complex. Convert every such buffer to two REAL buffers
    (rope_cos/rope_sin) BEFORE tracing (so no complex enters the graph) and replace
    Attention._apply_rope with the algebraically-identical real rotation (the complex
    multiply over (even,odd) pairs). G3's memory attention already uses the real
    apply_rotary_enc, so only the ViT needs this."""
    from sam3.model import vitdet

    n = 0
    for m in model.modules():
        if isinstance(m, vitdet.Attention) and getattr(m, "use_rope", False):
            fc = getattr(m, "freqs_cis", None)
            if fc is not None and fc.is_complex():
                del m._buffers["freqs_cis"]
                m.register_buffer("rope_cos", fc.real.contiguous(), persistent=False)
                m.register_buffer("rope_sin", fc.imag.contiguous(), persistent=False)
                n += 1

    def _apply_rope_real(self, q, k):
        if not self.use_rope:
            return q, k
        fr, fi = self.rope_cos, self.rope_sin

        def rot(x):
            xr = x.float().reshape(*x.shape[:-1], -1, 2)
            a, b = xr[..., 0], xr[..., 1]
            frb = vitdet.reshape_for_broadcast(fr, a)
            fib = vitdet.reshape_for_broadcast(fi, b)
            return torch.stack([a * frb - b * fib, a * fib + b * frb], dim=-1).flatten(3).type_as(x)

        return rot(q), rot(k)

    vitdet.Attention._apply_rope = _apply_rope_real
    print(f"  [patch] realified {n} ViT rope buffers -> ONNX-safe cos/sin")


def savenpy(path, t):
    a = t.detach().cpu().float().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)
    np.save(path, a)
    print(f"  wrote {os.path.basename(path)} {a.shape} {a.dtype}")
    return a


def export_constants(model, out):
    """The 4 learned scalar/proj constants the C++ engine loads verbatim (they are
    weights, not graphs). Attribute paths confirmed against the loaded module tree."""
    tr = model.tracker
    savenpy(os.path.join(out, "const_maskmem_tpos_enc.npy"), tr.maskmem_tpos_enc)   # [7,1,1,64]
    savenpy(os.path.join(out, "const_no_mem_embed.npy"), tr.no_mem_embed)           # [1,1,256]
    savenpy(os.path.join(out, "const_objptr_tpos_proj_w.npy"), tr.obj_ptr_tpos_proj.weight)  # [64,256]
    savenpy(os.path.join(out, "const_objptr_tpos_proj_b.npy"), tr.obj_ptr_tpos_proj.bias)     # [64]


def _check_onnx(path, feeds, ref_outs, names, atol=2e-3):
    """Load the exported graph in ORT and compare to the eager reference outputs."""
    import onnxruntime as ort
    so = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    got = so.run(None, feeds)
    print(f"  [check] {os.path.basename(path)} in={list(feeds)} out={[g.shape for g in got]}")
    for nm, g, r in zip(names, got, ref_outs):
        r = r.detach().cpu().float().numpy() if isinstance(r, torch.Tensor) else np.asarray(r)
        if tuple(g.shape) != tuple(r.shape):
            print(f"    !! {nm} shape {g.shape} != ref {r.shape}"); continue
        d = float(np.abs(g.astype(np.float32) - r.astype(np.float32)).max())
        print(f"    {nm:22s} shape {tuple(g.shape)} maxabs-diff {d:.2e} {'OK' if d < atol else '<-- HIGH'}")


# --- language constants for the fixed "person" prompt -----------------------------
# The detector's SAM3VLBackbone text encoder on "person" -> lang_feats [32,1,256]
# (seq,B,d) + lang_mask [1,32] bool (True = padding, the detector's key-pad mask).
# Fixed prompt -> cache as constants; the 24-layer text transformer never exports.
def export_person_language(model, out, device):
    tout = model.detector.backbone.forward_text(["person"], device=device)
    lf = tout["language_features"]          # [32,1,256]
    lm = tout["language_mask"]              # [1,32] bool
    savenpy(os.path.join(out, "const_lang_feats_person.npy"), lf.float())
    # bool mask -> save as bool
    lm_np = lm.detach().cpu().numpy().astype(bool)
    np.save(os.path.join(out, "const_lang_mask_person.npy"), lm_np)
    print(f"  wrote const_lang_mask_person.npy {lm_np.shape} bool (True-count {int(lm_np.sum())})")


# --- the five graphs --------------------------------------------------------------
# Each wrapper is an nn.Module whose forward() reproduces, in the engine's exact
# input/output order + shapes, the slice of the SAM 3 forward that the corresponding
# C++ G-method feeds. torch.onnx.export traces the wrapper. TODO(map): fill bodies
# from the forward-mapping (module methods + tensor layouts), then validate each
# graph's outputs against the eager model (parity harness below).
class G1Wrap(torch.nn.Module):
    """image -> fpn0,fpn1,fpn2, pos0,pos1,pos2, det_fpn0,det_fpn1,det_fpn2, det_pos72.
    The dual ViTDet neck runs once; the tracker (sam2) branch's L0/L1 get the mask
    decoder's conv_s0/conv_s1 projections (32/64-ch); the detector (sam3) branch is
    passed through. Mirrors forward_image (sam3_tracker_base.py:443-455) +
    sam3_video_base.py:381-391."""
    def __init__(self, model):
        super().__init__()
        self.backbone = model.detector.backbone
        self.conv_s0 = model.tracker.sam_mask_decoder.conv_s0
        self.conv_s1 = model.tracker.sam_mask_decoder.conv_s1

    def forward(self, image):
        bo = self.backbone.forward_image(image)
        trk = bo["sam2_backbone_out"]
        tf, tp = trk["backbone_fpn"], trk["vision_pos_enc"]   # 3 levels each
        fpn0 = self.conv_s0(tf[0]); fpn1 = self.conv_s1(tf[1]); fpn2 = tf[2]
        pos0, pos1, pos2 = tp[0], tp[1], tp[2]
        df, dp = bo["backbone_fpn"], bo["vision_pos_enc"]     # detector branch
        return (fpn0, fpn1, fpn2, pos0, pos1, pos2, df[0], df[1], df[2], dp[2])


G1_OUT = ["fpn0", "fpn1", "fpn2", "pos0", "pos1", "pos2",
          "det_fpn0", "det_fpn1", "det_fpn2", "det_pos72"]


def export_g1(model, out, device, opset):
    w = G1Wrap(model).to(device).eval()
    image = torch.zeros(1, 3, IMG, IMG, device=device)
    with torch.inference_mode():
        ref = w(image)
    p = os.path.join(out, "G1.onnx")
    torch.onnx.export(w, (image,), p, opset_version=opset,
                      input_names=["image"], output_names=G1_OUT, dynamo=True)
    _check_onnx(p, {"image": image.cpu().numpy()}, ref, G1_OUT)


def export_g2(model, out, device, opset):
    raise NotImplementedError("filled from the forward-mapping")


def export_g3(model, out, device, opset):
    raise NotImplementedError("filled from the forward-mapping")


def export_g4(model, out, device, opset):
    raise NotImplementedError("filled from the forward-mapping")


class G5Wrap(torch.nn.Module):
    """pix_feat[1,256,72,72], pred_masks_high_res[1,1,1008,1008], object_score_logits[1,1]
    -> maskmem_features[1,64,72,72], maskmem_pos_enc[1,64,72,72].
    Mirrors _encode_new_memory (sam3_tracker_base.py:797-848) on the is_mask_from_pts
    =False path (mask_for_mem = sigmoid(x)*20-10): the engine feeds ONE G5 for both
    seed and track, pre-scaling the seed mask through this same formula."""
    def __init__(self, model):
        super().__init__()
        self.tr = model.tracker
        self.scale = float(model.tracker.sigmoid_scale_for_mem_enc)   # 20
        self.bias = float(model.tracker.sigmoid_bias_for_mem_enc)     # -10

    def forward(self, pix_feat, pred_masks_high_res, object_score_logits):
        mask_for_mem = torch.sigmoid(pred_masks_high_res) * self.scale + self.bias
        mm = self.tr.maskmem_backbone(pix_feat, mask_for_mem, skip_mask_sigmoid=True)
        feats = mm["vision_features"]
        pos = mm["vision_pos_enc"][0]
        is_obj = (object_score_logits > 0).float()
        feats = feats + (1 - is_obj[..., None, None]) * \
            self.tr.no_obj_embed_spatial[..., None, None].expand_as(feats)
        return feats, pos


G5_OUT = ["maskmem_features", "maskmem_pos_enc"]


def export_g5(model, out, device, opset):
    w = G5Wrap(model).to(device).eval()
    pix = torch.zeros(1, 256, GRID, GRID, device=device)
    high = torch.zeros(1, 1, IMG, IMG, device=device)
    osl = torch.zeros(1, 1, device=device)
    with torch.inference_mode():
        ref = w(pix, high, osl)
    p = os.path.join(out, "G5.onnx")
    torch.onnx.export(w, (pix, high, osl), p, opset_version=opset,
                      input_names=["pix_feat", "pred_masks_high_res", "object_score_logits"],
                      output_names=G5_OUT, dynamo=True)
    _check_onnx(p, {"pix_feat": pix.cpu().numpy(), "pred_masks_high_res": high.cpu().numpy(),
                    "object_score_logits": osl.cpu().numpy()}, ref, G5_OUT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to sam3.pt")
    ap.add_argument("--out", required=True, help="output model dir (G1..G5 + const_*.npy)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--opset", type=int, default=18)  # >=18 for Resize antialias
    ap.add_argument("--only", default="", help="comma list subset: const,lang,g1,g2,g3,g4,g5")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    from sam3.model_builder import build_sam3_video_model
    print(f"building SAM 3 video model from {a.ckpt} on {a.device} ...")
    model = build_sam3_video_model(checkpoint_path=a.ckpt, load_from_HF=False,
                                   strict_state_dict_loading=True, device=a.device)
    model.eval()
    realify_vitdet_rope(model)

    want = set(s for s in a.only.split(",") if s) or {"const", "lang", "g1", "g2", "g3", "g4", "g5"}
    with torch.inference_mode():
        if "const" in want: print("constants:");  export_constants(model, a.out)
        if "lang" in want:  print("person language:"); export_person_language(model, a.out, a.device)
        if "g1" in want: print("G1:"); export_g1(model, a.out, a.device, a.opset)
        if "g2" in want: print("G2:"); export_g2(model, a.out, a.device, a.opset)
        if "g3" in want: print("G3:"); export_g3(model, a.out, a.device, a.opset)
        if "g4" in want: print("G4:"); export_g4(model, a.out, a.device, a.opset)
        if "g5" in want: print("G5:"); export_g5(model, a.out, a.device, a.opset)
    print("done ->", a.out)


if __name__ == "__main__":
    main()
