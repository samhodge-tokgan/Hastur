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
    # The C++ engine loads this via LoadNpyFloat (float32 0/1) and thresholds >0.5 to
    # the bool key-padding mask, so save float32 (NOT bool). True/1 = padding.
    lm_np = lm.detach().cpu().numpy().astype(np.float32)
    np.save(os.path.join(out, "const_lang_mask_person.npy"), lm_np)
    print(f"  wrote const_lang_mask_person.npy {lm_np.shape} float32 (pad-count {int(lm_np.sum())})")


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


# --- G2 export shims (mirror the validated reference tools/sam3_tracker_export/export_g2.py)
# The detector's forward_grounding is reused verbatim; three surgical method shims make it
# torch.export-clean for the empty ("person" text-only) grounding prompt:
def _encode_points_shim(self, points, points_mask, points_labels, img_feats):
    """Skip the grid_sample/pool-project branch when there are 0 point prompts (it runs
    unconditionally, gated on module presence not prompt count, and emits ONNX-hostile
    0-dim ops). Static-0 count -> return a clean empty (0,bs,C). Numerically identical."""
    import torch.nn.functional as F
    points_embed = None
    n_points, bs = points.shape[:2]
    if n_points == 0:
        return self.label_embed(points_labels.long()), points_mask
    if self.points_direct_project is not None:
        points_embed = self.points_direct_project(points)
    if self.points_pool_project is not None:
        grid = points.transpose(0, 1).unsqueeze(2); grid = (grid * 2) - 1
        sampled = F.grid_sample(img_feats, grid, align_corners=False)
        sampled = sampled.squeeze(-1).permute(2, 0, 1)
        proj = self.points_pool_project(sampled)
        points_embed = proj if points_embed is None else points_embed + proj
    if self.points_pos_enc_project is not None:
        x, y = points.unbind(-1)
        enc_x, enc_y = self.pos_enc._encode_xy(x.flatten(), y.flatten())
        enc_x = enc_x.view(n_points, bs, enc_x.shape[-1]); enc_y = enc_y.view(n_points, bs, enc_y.shape[-1])
        enc = torch.cat([enc_x, enc_y], -1); proj = self.points_pos_enc_project(enc)
        points_embed = proj if points_embed is None else points_embed + proj
    type_embed = self.label_embed(points_labels.long())
    if points_embed is None: points_embed = 0.0
    return type_embed + points_embed, points_mask


def _encode_boxes_shim(self, boxes, boxes_mask, boxes_labels, img_feats):
    """Same as _encode_points_shim for boxes: skip roi_align/pool-project/pin_memory when
    there are 0 box prompts. Returns a clean empty (0,bs,C). Numerically identical."""
    from sam3.model.box_ops import box_cxcywh_to_xyxy
    boxes_embed = None
    n_boxes, bs = boxes.shape[:2]
    if n_boxes == 0:
        return self.label_embed(boxes_labels.long()), boxes_mask
    if self.boxes_direct_project is not None:
        boxes_embed = self.boxes_direct_project(boxes)
    if self.boxes_pool_project is not None:
        import torchvision
        H, W = img_feats.shape[-2:]
        boxes_xyxy = box_cxcywh_to_xyxy(boxes)
        scale = torch.tensor([W, H, W, H], dtype=boxes_xyxy.dtype, device=boxes_xyxy.device).view(1, 1, 4)
        boxes_xyxy = boxes_xyxy * scale
        sampled = torchvision.ops.roi_align(img_feats, boxes_xyxy.float().transpose(0, 1).unbind(0), self.roi_size)
        proj = self.boxes_pool_project(sampled).view(bs, n_boxes, self.d_model).transpose(0, 1)
        boxes_embed = proj if boxes_embed is None else boxes_embed + proj
    if self.boxes_pos_enc_project is not None:
        cx, cy, w, h = boxes.unbind(-1)
        enc = self.pos_enc.encode_boxes(cx.flatten(), cy.flatten(), w.flatten(), h.flatten())
        enc = enc.view(boxes.shape[0], boxes.shape[1], enc.shape[-1])
        proj = self.boxes_pos_enc_project(enc)
        boxes_embed = proj if boxes_embed is None else boxes_embed + proj
    type_embed = self.label_embed(boxes_labels.long())
    if boxes_embed is None: boxes_embed = 0.0
    return type_embed + boxes_embed, boxes_mask


def _rpb_shim(self, reference_boxes, feat_size):
    """decoder box-relative-position-bias: drop the tensor-valued size-cache '==' guard
    (decoder.py:339) that forces a data-dependent aten.item(), and reuse the concrete coord
    cache (populated on the export device by the eager warm-up) so we never run arange() on
    the tensor-valued H/W (which would mint unbacked symints)."""
    from sam3.model.box_ops import box_cxcywh_to_xyxy
    H, W = feat_size
    boxes_xyxy = box_cxcywh_to_xyxy(reference_boxes).transpose(0, 1)
    bs, num_queries, _ = boxes_xyxy.shape
    if self.compilable_cord_cache is None:
        self.compilable_cord_cache = self._get_coords(H, W, reference_boxes.device)
    coords_h, coords_w = self.compilable_cord_cache
    deltas_y = coords_h.view(1, -1, 1) - boxes_xyxy.reshape(-1, 1, 4)[:, :, 1:4:2]
    deltas_y = deltas_y.view(bs, num_queries, -1, 2)
    deltas_x = coords_w.view(1, -1, 1) - boxes_xyxy.reshape(-1, 1, 4)[:, :, 0:3:2]
    deltas_x = deltas_x.view(bs, num_queries, -1, 2)
    if self.boxRPB in ["log", "both"]:
        dxl = deltas_x * 8; dxl = torch.sign(dxl) * torch.log2(torch.abs(dxl) + 1.0) / np.log2(8)
        dyl = deltas_y * 8; dyl = torch.sign(dyl) * torch.log2(torch.abs(dyl) + 1.0) / np.log2(8)
        if self.boxRPB == "log": deltas_x, deltas_y = dxl, dyl
        else: deltas_x = torch.cat([deltas_x, dxl], -1); deltas_y = torch.cat([deltas_y, dyl], -1)
    deltas_x = self.boxRPB_embed_x(x=deltas_x)
    deltas_y = self.boxRPB_embed_y(x=deltas_y)
    B = deltas_y.unsqueeze(3) + deltas_x.unsqueeze(2)
    B = B.flatten(2, 3).permute(0, 3, 1, 2).contiguous()
    return B


class G2Wrap(torch.nn.Module):
    """det_fpn0[1,256,288,288], det_fpn1[1,256,144,144], det_fpn2[1,256,72,72],
    pos72[1,256,72,72], lang_feats[32,1,256] (seq,B,d), lang_mask[1,32] bool ->
    pred_logits[1,200,1], pred_boxes[1,200,4] (normalized cxcywh), pred_masks[1,200,288,288]
    (288^2 logits), presence[1,1].

    Bypasses the ViT + text encoder: assembles a `backbone_out` carrying the 3 detector
    FPN levels + the grid-72 pos-enc + the cached "person" language features/mask, then
    drives the detector's own forward_grounding (_encode_prompt -> _run_encoder ->
    _run_decoder -> _run_segmentation_heads) on the TRUE empty grounding prompt
    (det._get_dummy_prompt(), matching sam3_video_inference.py:155 `empty_geometric_prompt`).
    num_feature_levels=1 so the DETR encoder consumes only fpn2/pos72; the pixel decoder /
    seg head uses all 3 fpn levels. presence == the decoder presence-token logit
    `presence_logit_dec` (last decoder layer, [bs,1]). The three module-method shims above
    are bound here so the empty-prompt trace is torch.export-clean (no 0-dim ops, no
    data-dependent guards)."""
    def __init__(self, model):
        super().__init__()
        import types
        from sam3.model.data_misc import FindStage
        self.det = model.detector
        self._FindStage = FindStage
        ge = self.det.geometry_encoder
        ge._encode_points = types.MethodType(_encode_points_shim, ge)
        ge._encode_boxes = types.MethodType(_encode_boxes_shim, ge)
        dec = self.det.transformer.decoder
        dec._get_rpb_matrix = types.MethodType(_rpb_shim, dec)

    def forward(self, fpn0, fpn1, fpn2, pos72, lang_feats, lang_mask):
        dev = fpn2.device
        backbone_out = {
            "backbone_fpn": [fpn0, fpn1, fpn2],   # seg head / pixel decoder uses all 3
            "vision_pos_enc": [pos72],            # only the last level (grid 72) is used
            "vision_features": fpn2,
            "language_features": lang_feats,      # [32,1,256]
            "language_mask": lang_mask,           # [1,32] bool, True = padding
        }
        find_input = self._FindStage(
            img_ids=torch.zeros(1, dtype=torch.long, device=dev),
            text_ids=torch.zeros(1, dtype=torch.long, device=dev),
            input_boxes=None, input_boxes_mask=None, input_boxes_label=None,
            input_points=None, input_points_mask=None)
        geometric_prompt = self.det._get_dummy_prompt()
        out = self.det.forward_grounding(
            backbone_out=backbone_out,
            find_input=find_input,
            find_target=None,
            geometric_prompt=geometric_prompt,
        )
        return (out["pred_logits"], out["pred_boxes"],
                out["pred_masks"], out["presence_logit_dec"])


G2_OUT = ["pred_logits", "pred_boxes", "pred_masks", "presence"]


def export_g2(model, out, device, opset):
    # The detector decoder's boxRPB relative-position cache is precomputed on "cuda" at
    # model-init (decoder.py hardcodes device="cuda"); reset it so the _rpb_shim rebuilds it
    # lazily on the export device (CPU) during the eager warm-up (the GPU reference doesn't
    # need this, but we export on CPU to avoid bf16 autocast).
    dec = model.detector.transformer.decoder
    dec.compilable_cord_cache = None
    dec.compilable_stored_size = None
    dec.coord_cache = {}
    w = G2Wrap(model).to(device).eval()
    torch.manual_seed(0)
    fpn0 = torch.randn(1, HIDDEN, 4 * GRID, 4 * GRID, device=device)  # [1,256,288,288]
    fpn1 = torch.randn(1, HIDDEN, 2 * GRID, 2 * GRID, device=device)  # [1,256,144,144]
    fpn2 = torch.randn(1, HIDDEN, GRID, GRID, device=device)          # [1,256,72,72]
    pos72 = torch.randn(1, HIDDEN, GRID, GRID, device=device)         # [1,256,72,72]
    lang_feats = torch.randn(32, 1, HIDDEN, device=device)            # [32,1,256]
    lang_mask = torch.zeros(1, 32, dtype=torch.bool, device=device)   # [1,32], no padding
    ins = (fpn0, fpn1, fpn2, pos72, lang_feats, lang_mask)
    with torch.inference_mode():
        ref = w(*ins)   # eager warm-up also populates the CPU boxRPB coord cache
    print(f"  [g2] eager out shapes: {[tuple(r.shape) for r in ref]}")
    p = os.path.join(out, "G2.onnx")
    torch.onnx.export(
        w, ins, p, opset_version=opset,
        input_names=["fpn0", "fpn1", "fpn2", "pos72", "lang_feats", "lang_mask"],
        output_names=G2_OUT, dynamo=True)
    # pred_masks are 288^2 raw logits of magnitude ~O(40) here (randn inputs), so a ~2e-3
    # abs diff is ~5e-5 relative -- float32 accumulation in the 256-dim mask einsum, and
    # irrelevant to the C++ masks>0 binarization; logits/boxes/presence match to ~1e-6.
    _check_onnx(p, {"fpn0": fpn0.cpu().numpy(), "fpn1": fpn1.cpu().numpy(),
                    "fpn2": fpn2.cpu().numpy(), "pos72": pos72.cpu().numpy(),
                    "lang_feats": lang_feats.cpu().numpy(),
                    "lang_mask": lang_mask.cpu().numpy()}, ref, G2_OUT, atol=3e-3)


def realify_memattn_rope(model):
    """The tracker memory-attention (tracker.transformer) RoPEAttention blocks also
    carry complex freqs_cis. Realify them (cos/sin buffers, use_rope_real=True) and
    replace forward with an export-safe version: (a) no in-place slice-assignment
    (rebuild k via cat) so num_k_exclude_rope can be a DYNAMIC int64 the C++ varies
    per frame; (b) drop the fa3 / freqs-recompute / shape-assert branches (src is a
    fixed 72x72=5184 grid). Also force _forward_ca to pass num_k_exclude_rope
    unconditionally (the tracking path always has object pointers)."""
    import torch.nn.functional as F
    from sam3.sam import transformer as T
    from sam3.sam.rope import apply_rotary_enc_real

    # freqs_cis here is a plain attribute (not a registered buffer), so model.to()
    # never moved it off the cuda device it was built on -> pin to the model's device.
    dev = next(model.parameters()).device
    n = 0
    for m in model.tracker.transformer.modules():
        if isinstance(m, T.RoPEAttention):
            fc = m.freqs_cis.to(dev)
            m.use_rope_real = True
            m.freqs_cis_real = fc.real.contiguous().to(dev)
            m.freqs_cis_imag = fc.imag.contiguous().to(dev)
            n += 1

    def _rope_pairs(x, fr, fi):
        # x[...,D]; fr,fi broadcastable to [...,D/2]. Complex rotate over (even,odd).
        lead = x.shape[:-1]
        xr = x.reshape(*lead, x.shape[-1] // 2, 2)
        a, b = xr[..., 0], xr[..., 1]
        return torch.stack([a * fr - b * fi, a * fi + b * fr], dim=-1).reshape(*lead, x.shape[-1])

    def rope_forward(self, q, k, v, num_k_exclude_rope=0):
        q = self._separate_heads(self.q_proj(q), self.num_heads)   # [B,H,Lq,D]
        k = self._separate_heads(self.k_proj(k), self.num_heads)   # [B,H,Lk,D]
        v = self._separate_heads(self.v_proj(v), self.num_heads)
        fr, fi = self.freqs_cis_real, self.freqs_cis_imag          # [Lq, D/2]
        Lq, Dh = fr.shape[0], fr.shape[1]
        # num_k_exclude_rope arrives as a 1-D tensor whose LENGTH is the pointer-token
        # count; nx = .shape[0] is a BACKED dynamic dim (unlike an unbacked .item()
        # value), so the k split and the memory-frame reshape guard cleanly. The C++
        # engine feeds a length-nx int64 tensor (values unused). Self-attn passes 0.
        if torch.is_tensor(num_k_exclude_rope) and num_k_exclude_rope.dim() > 0:
            nx = num_k_exclude_rope.shape[0]
        else:
            nx = int(num_k_exclude_rope)
        nk = k.shape[-2]
        num_k_rope = nk - nx
        k_rope, k_rest = k[:, :, :num_k_rope], k[:, :, num_k_rope:]
        q = _rope_pairs(q, fr.view(1, 1, Lq, Dh), fi.view(1, 1, Lq, Dh))
        if self.rope_k_repeat:
            # k_rope spans n_mem memory frames of Lq tokens; apply the SAME per-position
            # freqs to each frame via a reshape-broadcast ([B,H,-1,Lq,D]) instead of a
            # data-dependent .repeat of the freqs table (which torch.export can't guard).
            B, H = k_rope.shape[0], k_rope.shape[1]
            kr = k_rope.reshape(B, H, -1, Lq, k_rope.shape[-1])
            kr = _rope_pairs(kr, fr.view(1, 1, 1, Lq, Dh), fi.view(1, 1, 1, Lq, Dh))
            k_rope = kr.reshape(B, H, -1, k_rope.shape[-1])
        else:
            k_rope = _rope_pairs(k_rope, fr.view(1, 1, Lq, Dh), fi.view(1, 1, Lq, Dh))
        k = torch.cat([k_rope, k_rest], dim=2)
        out = F.scaled_dot_product_attention(q, k, v)
        return self.out_proj(self._recombine_heads(out))

    T.RoPEAttention.forward = rope_forward

    from sam3.model import decoder as D

    def _forward_ca(self, tgt, memory, query_pos, pos, num_k_exclude_rope=0):
        if self.cross_attn_image is None:
            return tgt
        kwds = {}
        if isinstance(self.cross_attn_image, T.RoPEAttention):
            kwds = {"num_k_exclude_rope": num_k_exclude_rope}
        tgt2 = self.norm2(tgt)
        tgt2 = self.cross_attn_image(
            q=tgt2 + query_pos if self.pos_enc_at_cross_attn_queries else tgt2,
            k=memory + pos if self.pos_enc_at_cross_attn_keys else memory,
            v=memory, **kwds)
        return tgt + self.dropout2(tgt2)

    D.TransformerDecoderLayerv2._forward_ca = _forward_ca
    print(f"  [patch] realified {n} mem-attn rope blocks + dynamic num_k_exclude_rope")


class G3Wrap(torch.nn.Module):
    """src[5184,1,256], src_pos[5184,1,256], prompt[L,1,64], prompt_pos[L,1,64],
    num_obj_ptr_tokens(int64 scalar) -> memory-conditioned pix_feat [1,256,72,72].
    Just the transformer encoder call (sam3_tracker_base.py:782-794); the memory-bank
    assembly that builds prompt/prompt_pos lives in the C++ engine."""
    def __init__(self, model):
        super().__init__()
        self.enc = model.tracker.transformer.encoder

    def forward(self, src, src_pos, prompt, prompt_pos, num_obj_ptr_tokens):
        out = self.enc(src=[src], src_key_padding_mask=[None], src_pos=[src_pos],
                       prompt=prompt, prompt_pos=prompt_pos, prompt_key_padding_mask=None,
                       feat_sizes=[(GRID, GRID)], num_obj_ptr_tokens=num_obj_ptr_tokens)
        # Return the encoder memory SEQ-FIRST [5184,1,256]; the C++ engine's SeqToBCHW
        # does the ->[1,256,72,72] reshape itself (don't double-convert here).
        return out["memory"]


def export_g3(model, out, device, opset):
    # The variable-length memory RoPE (dynamic num_k_rope) trips torch.export's
    # should_swap/stride guards on unbacked sizes; size-oblivious guarding resolves
    # them (the branches are stride-order choices, correct either way).
    try:
        from torch.fx.experimental import _config as fx_config
        fx_config.backed_size_oblivious = True
    except Exception:
        pass
    realify_memattn_rope(model)
    w = G3Wrap(model).to(device).eval()
    HW = GRID * GRID
    n_mem, n_ptr = 3, 5                            # representative memory bank
    L = n_mem * HW + n_ptr * 4
    src = torch.randn(HW, 1, HIDDEN, device=device)
    src_pos = torch.randn(HW, 1, HIDDEN, device=device)
    prompt = torch.randn(L, 1, MEMDIM, device=device)
    prompt_pos = torch.randn(L, 1, MEMDIM, device=device)
    nopt = torch.zeros(n_ptr * 4, dtype=torch.int64, device=device)  # LENGTH = #ptr tokens
    ins = (src, src_pos, prompt, prompt_pos, nopt)
    with torch.inference_mode():
        ref = w(*ins)
    p = os.path.join(out, "G3.onnx")
    Ldim = torch.export.Dim("L", min=HW, max=8 * HW)
    Pdim = torch.export.Dim("P", min=4, max=4 * 16)
    torch.onnx.export(
        w, ins, p, opset_version=opset,
        input_names=["src", "src_pos", "prompt", "prompt_pos", "num_obj_ptr_tokens"],
        output_names=["memory"],
        dynamic_shapes=({}, {}, {0: Ldim}, {0: Ldim}, {0: Pdim}), dynamo=True)
    _check_onnx(p, {"src": src.cpu().numpy(), "src_pos": src_pos.cpu().numpy(),
                    "prompt": prompt.cpu().numpy(), "prompt_pos": prompt_pos.cpu().numpy(),
                    "num_obj_ptr_tokens": nopt.cpu().numpy()}, [ref], ["memory"])


class G4Wrap(torch.nn.Module):
    """backbone_features[1,256,72,72], high_res_0=fpn0[1,32,288,288],
    high_res_1=fpn1[1,64,144,144] -> low_res(best,[1,1,288,288]),
    pred_masks_high_res(best,[1,1,1008,1008]), ious[1,3], obj_ptr[1,256],
    object_score_logits[1,1]. The no-prompt tracking decode: _forward_sam_heads with
    point_inputs=mask_inputs=None, multimask_output=True (sam3_tracker_base.py:218-387).
    Returns the C++ order (best low, best high, ious, obj_ptr, osl)."""
    def __init__(self, model):
        super().__init__()
        self.tr = model.tracker

    def forward(self, backbone_features, high_res_0, high_res_1):
        (low_mm, high_mm, ious, low_best, high_best, obj_ptr, osl) = \
            self.tr._forward_sam_heads(
                backbone_features, point_inputs=None, mask_inputs=None,
                high_res_features=[high_res_0, high_res_1], multimask_output=True)
        return low_best, high_best, ious, obj_ptr, osl


G4_OUT = ["low_res_masks", "pred_masks_high_res", "ious", "obj_ptr", "object_score_logits"]


def export_g4(model, out, device, opset):
    w = G4Wrap(model).to(device).eval()
    bf = torch.zeros(1, 256, GRID, GRID, device=device)
    h0 = torch.zeros(1, 32, 4 * GRID, 4 * GRID, device=device)   # [1,32,288,288]
    h1 = torch.zeros(1, 64, 2 * GRID, 2 * GRID, device=device)   # [1,64,144,144]
    with torch.inference_mode():
        ref = w(bf, h0, h1)
    p = os.path.join(out, "G4.onnx")
    torch.onnx.export(w, (bf, h0, h1), p, opset_version=opset,
                      input_names=["backbone_features", "high_res_0", "high_res_1"],
                      output_names=G4_OUT, dynamo=True)
    _check_onnx(p, {"backbone_features": bf.cpu().numpy(), "high_res_0": h0.cpu().numpy(),
                    "high_res_1": h1.cpu().numpy()}, ref, G4_OUT)


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
