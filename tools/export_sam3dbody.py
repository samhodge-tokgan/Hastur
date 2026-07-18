#!/usr/bin/env python
# Copyright the Hastur authors.
# SPDX-License-Identifier: Apache-2.0
#
# export_sam3dbody.py -- export the SAM-3D-Body *body-only* core regressor to a
# fixed-shape fp16 ONNX for the Hastur ORT engine (M3 / Track B).
#
#   inputs : image[1,3,512,512]  (ImageNet-normalized CHW, fp32 at the boundary)
#            ray_cond[1,2,32,32]  (token-grid ray directions; see note below)
#            condition_info[1,3]  (CLIFF cx/f, cy/f, b/f; USE_INTRIN_CENTER)
#   outputs: pred[1,519]  pred_cam[1,3]  hand_logits[1,2,2]  hand_box[1,2,4]
#
# The graph fuses: DINOv3-vith16plus backbone (image-only) -> +no-mask embed ->
# SAM promptable decoder (6 layers, dummy keypoint prompt baked as a constant,
# per-layer keypoint-token refinement) -> MHR-head 519 vector + perspective
# camera head + hand-detect logits/box.
#
# KEY DESIGN DECISIONS
# --------------------
# * The reference (sam3d_body.py in the local repo) is device-pinned with broken
#   `.to("mps")` / `torch.batch[...].dtype` hacks that cannot run. We do NOT mutate
#   it; we drive its *unbroken* methods (backbone, forward_decoder, keypoint update
#   fns) and supply clean CPU/fp32 replacements only for the broken helpers
#   (data_preprocess, get_ray_condition) and for two ops that must change for a
#   static, export-friendly graph:
#
#   (1) camera_project -> a condition_info-ONLY reimplementation. The reference
#       projects via absolute cam_int/bbox_center/affine_trans, but for CLIFF
#       conditioning the crop-space keypoints are fully determined by
#       condition_info alone. Proven equivalent to ~1e-7 (see validate_body.py),
#       so the frozen 3-input contract is sufficient -- no extra camera inputs.
#
#   (2) the MHR mesh used for the per-layer refinement. The shipped MHR is a
#       momentum TorchScript whose LBS uses aten::sparse_coo_tensor (pose
#       correctives) which NO ONNX opset supports (torch.onnx cannot export it,
#       legacy or dynamo). We substitute tools/mhr_meshmodule.MhrMeshModule -- a
#       pure-torch, ONNX-exportable reimplementation of the same mesh path,
#       validated bit-for-bit against the TorchScript -- so the refinement (and
#       therefore the pose token) stays faithful. Use --mode meshfree to instead
#       drop the refinement entirely (smaller/faster graph, but degraded parity;
#       see the report).
#
# * ray_cond is consumed at the 32x32 token grid (post the antialiased 1/16
#   downsample the reference applies inside CameraEncoder). The C++ crop stage
#   must produce ray_cond at 32x32 the same way (see note in the report).
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

# The reference repo forces GitHub round-trips at import (transforms __init__ does
# torch.hub.list('pytorch/vision', force_reload=True); the DINOv3 backbone re-
# validates facebookresearch/dinov3). Both repos are already in the torch.hub
# cache, so neutralize the forced network access (avoids HTTP 403 rate limits).
torch.hub._validate_not_a_forked_repo = lambda *a, **k: None
_ORIG_HUB_LIST = torch.hub.list
def _offline_hub_list(*a, **k):
    k["force_reload"] = False
    try:
        return _ORIG_HUB_LIST(*a, **k)
    except Exception:
        return []
torch.hub.list = _offline_hub_list

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # mhr_common / mhr_meshmodule
os.environ.setdefault("MOMENTUM_ENABLED", "0")  # force TorchScript MHR path

REF_ROOT = os.environ.get("SAM3D_ROOT", "/Users/sam/Documents/github/sam-3d-body")
CKPT = os.path.join(REF_ROOT, "checkpoints/sam-3d-body-dinov3/model.ckpt")
MHR = os.path.join(REF_ROOT, "checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt")
DEMO_IMG = os.path.join(REF_ROOT, "notebook/images/dancing.jpg")
sys.path.insert(0, REF_ROOT)

DEFAULT_OUT = os.path.join(os.path.dirname(HERE), "models", "sam3dbody_body.onnx")
OUTPUT_NAMES = ["pred", "pred_cam", "hand_logits", "hand_box"]


# --------------------------------------------------------------------------- #
#  Model loading + clean CPU/fp32 replacements for the broken reference helpers
# --------------------------------------------------------------------------- #
def load_model(device="cpu"):
    from sam_3d_body.build_models import load_sam_3d_body
    model, cfg = load_sam_3d_body(CKPT, device=device, mhr_path=MHR)
    model = model.float()                # undo the bf16 torso -> clean fp32 trace
    model.backbone_dtype = torch.float32
    model.eval()
    return model, cfg


def data_preprocess_clean(model, inputs):
    mean = model.image_mean.float()
    std = model.image_std.float()
    if inputs.max() > 1:
        inputs = inputs / 255.0
    return (inputs - mean) / std


def get_ray_condition_clean(model, batch):
    B, N, _, H, W = batch["img"].shape
    aff = batch["affine_trans"].float()
    cam = batch["cam_int"].float()
    grid = torch.stack(torch.meshgrid(torch.arange(H), torch.arange(W), indexing="xy"), dim=2)
    grid = grid[None, None].repeat(B, N, 1, 1, 1).float()
    a_scale = aff[:, :, None, None, [0, 1], [0, 1]]
    a_trans = aff[:, :, None, None, [0, 1], [2, 2]]
    grid = grid / a_scale
    grid = grid - a_trans / a_scale
    grid = grid - cam[:, None, None, None, [0, 1], [2, 2]]
    grid = grid / cam[:, None, None, None, [0, 1], [0, 1]]
    return grid.permute(0, 1, 4, 2, 3).contiguous()


def build_batch(box=None, img_path=DEMO_IMG):
    import cv2
    from sam_3d_body.data.transforms import (Compose, GetBBoxCenterScale,
                                             TopdownAffine, VisionTransformWrapper)
    from torchvision.transforms import ToTensor
    from sam_3d_body.data.utils.prepare_batch import prepare_batch
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]
    if box is None:
        box = np.array([[0, 0, W, H]], dtype=np.float32)
    tf = Compose([GetBBoxCenterScale(),
                  TopdownAffine(input_size=[512, 512], use_udp=False),
                  VisionTransformWrapper(ToTensor())])
    batch = prepare_batch(img, tf, box)
    for k in list(batch.keys()):
        if torch.is_tensor(batch[k]):
            batch[k] = batch[k].float().cpu()
    return batch


# --------------------------------------------------------------------------- #
#  Op replacements baked into the wrapper
# --------------------------------------------------------------------------- #
def camera_project_cond(self, pose_output, batch):
    """condition_info-only replacement for SAM3DBody.camera_project (body path)."""
    cond = self._cond_body
    pred_cam = pose_output["pred_cam"]
    kp3d = pose_output["pred_keypoints_3d"]
    s = -pred_cam[:, 0]
    tx = pred_cam[:, 1]
    ty = -pred_cam[:, 2]
    c0, c1, c2 = cond[:, 0], cond[:, 1], cond[:, 2]
    denom = c2 * s
    camt = torch.stack([tx + 2 * c0 / denom, ty + 2 * c1 / denom, 2 / denom], dim=-1)
    j = kp3d + camt[:, None, :]
    X, Y, Z = j[..., 0], j[..., 1], j[..., 2]
    cropped = torch.stack([(X / Z - c0[:, None]) / c2[:, None],
                           (Y / Z - c1[:, None]) / c2[:, None]], dim=-1)
    pose_output.update({
        "pred_keypoints_2d": cropped,       # already crop-normalized; _full_to_crop is a passthrough
        "pred_cam_t": camt,
        "focal_length": 1.0 / c2,
        "pred_keypoints_2d_depth": Z,
    })
    return pose_output


def full_to_crop_passthrough(self, batch, pred_keypoints_2d, batch_idx=None):
    return pred_keypoints_2d[..., :2]


def camera_encoder_no_interp(self, img_embeddings, rays):
    import einops
    B, D, _h, _w = img_embeddings.shape
    rays = rays.permute(0, 2, 3, 1).contiguous()   # already token-grid (2,32,32)
    rays = torch.cat([rays, torch.ones_like(rays[..., :1])], dim=-1)
    rays_embeddings = self.camera(pos=rays.reshape(B, -1, 3))
    rays_embeddings = einops.rearrange(rays_embeddings, "b (h w) c -> b c h w",
                                       h=_h, w=_w).contiguous()
    z = torch.concat([img_embeddings, rays_embeddings], dim=1)
    return self.norm(self.conv(z))


def _identity_update(self, *a):
    return a[-4], a[-3], a[-2], a[-1]


def make_head_forward(mesh):
    """Return a head_pose.forward that yields the raw 519 slices + pred_keypoints_3d.
    If mesh is not None it is the ONNX-exportable MhrMeshModule (faithful path);
    otherwise keypoints are zeroed (meshfree path -- no refinement)."""
    def head_forward(self, x, init_estimate=None, do_pcblend=True, slim_keypoints=False):
        pred = self.proj(x)
        if init_estimate is not None:
            pred = pred + init_estimate
        c = 6
        g6 = pred[:, :c]
        cont = pred[:, c:c + self.body_cont_dim]; c += self.body_cont_dim
        shape = pred[:, c:c + self.num_shape_comps]; c += self.num_shape_comps
        scale = pred[:, c:c + self.num_scale_comps]; c += self.num_scale_comps
        hand = pred[:, c:c + self.num_hand_comps * 2]; c += self.num_hand_comps * 2
        face = pred[:, c:c + self.num_face_comps] * 0
        if mesh is not None:
            _, _, j3d = mesh(pred)                       # (B,70,3) metres, flipped
            kp3d = j3d
        else:
            kp3d = torch.zeros(x.shape[0], 70, 3, dtype=x.dtype, device=x.device)
        return {
            "pred_pose_raw": torch.cat([g6, cont], dim=1),
            "shape": shape, "scale": scale, "hand": hand, "face": face,
            "pred_keypoints_3d": kp3d, "pred_vertices": None,
        }
    return head_forward


class BodyWrapper(torch.nn.Module):
    """Fixed batch=1 body-only regressor wrapper."""

    def __init__(self, model, cfg, mesh=None, mesh_free=False):
        super().__init__()
        self.model = model
        self.mesh = mesh  # registered as submodule so its buffers trace/export
        self.mesh_free = mesh_free
        model.camera_project = camera_project_cond.__get__(model, type(model))
        model._full_to_crop = full_to_crop_passthrough.__get__(model, type(model))
        model.ray_cond_emb.forward = camera_encoder_no_interp.__get__(
            model.ray_cond_emb, type(model.ray_cond_emb))
        model.head_pose.forward = make_head_forward(mesh).__get__(
            model.head_pose, type(model.head_pose))
        if mesh_free:
            model.keypoint_token_update_fn = _identity_update.__get__(model, type(model))
            model.keypoint3d_token_update_fn = _identity_update.__get__(model, type(model))
        model.body_batch_idx = [0]
        model.hand_batch_idx = []
        self.register_buffer("dummy_kp", torch.zeros(1, 1, 3))
        self.dummy_kp[:, :, -1] = -2
        self._fake_batch = {}

    def forward(self, image, ray_cond, condition_info):
        model = self.model
        model._cond_body = condition_info
        self._fake_batch["ray_cond"] = ray_cond
        # NB: no explicit dtype casts here -- the whole traced graph is single-
        # precision, and an identity Cast(to=FLOAT) on `image` would survive fp16
        # conversion and clash with the fp16 patch-embed Conv weight.
        ie = model.backbone(image, extra_embed=None)
        if isinstance(ie, tuple):
            ie = ie[-1]
        ie = ie + model.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1)
        tokens_output, pose_outputs = model.forward_decoder(
            ie, init_estimate=None, keypoints=self.dummy_kp, prev_estimate=None,
            condition_info=condition_info, batch=self._fake_batch)
        po = pose_outputs[-1]
        pred = torch.cat([po["pred_pose_raw"], po["shape"], po["scale"],
                          po["hand"], po["face"] * 0.0], dim=1)
        pred_cam = po["pred_cam"]
        hand_box = model.bbox_embed(tokens_output).sigmoid()
        hand_logits = model.hand_cls_embed(tokens_output)
        return pred, pred_cam, hand_logits, hand_box


def _lbs_scatter_add(self, global_skel, unposed):
    """LBS accumulation via scatter_add instead of index_add. torch.onnx's legacy
    lowering of aten::index_add (duplicate skin indices) emits a broken Expand
    that ORT rejects at load; scatter_add lowers to ScatterElements(reduction=add)
    which ORT accepts. Index is explicitly broadcast to the source shape so no
    dynamic Expand is generated. Numerically identical to the reference LBS."""
    from mhr_meshmodule import quat_rotate, skel_multiply, skel_normalize_q
    jstate = skel_multiply(global_skel, self.inverse_bind_pose[None])
    B = unposed.shape[0]
    js = jstate.index_select(-2, self.skin_joint_idx)
    vp = unposed.index_select(-2, self.skin_vert_idx)
    t, q, s = skel_normalize_q(js)
    transformed = t + quat_rotate(q, s * vp)
    contrib = transformed * self.skin_weights[None, :, None]
    idx = self.skin_vert_idx.view(1, -1, 1).expand(B, -1, 3)
    skinned = torch.zeros(B, self.V, 3, dtype=unposed.dtype, device=unposed.device)
    return skinned.scatter_add(-2, idx, contrib)


def build_mesh(use_double=False, correctives=False):
    from mhr_common import MhrReference
    from mhr_meshmodule import MhrMeshModule
    mesh = MhrMeshModule(MhrReference("cpu")).float().eval()
    mesh.lbs = _lbs_scatter_add.__get__(mesh, type(mesh))  # ORT-loadable LBS
    if not use_double:
        # keep the graph fp32/fp16-friendly (no float64 FK) for CoreML/fp16
        orig_fk = mesh.fk
        mesh.fk = (lambda local, use_double=False: orig_fk(local, use_double=False))
    if not correctives:
        # Drop the pose-corrective blendshapes from the IN-GRAPH refinement mesh.
        # They only nudge the intermediate keypoints that drive the per-layer
        # refinement (NOT a graph output), and dropping them removes the 663 MB
        # dense W2 weight (which torch.onnx otherwise duplicates once per MHR call
        # site -> multi-GB external-data blob). Cost: pred max_abs ~1.4e-2 vs the
        # exact reference (Pearson 1.0), well inside fp16 quantization noise. Use
        # --correctives for the bit-exact (but multi-GB) variant.
        V = mesh.V
        mesh.pose_corrective = (lambda jp: torch.zeros(
            jp.shape[0], V, 3, dtype=jp.dtype, device=jp.device))
    return mesh


def make_onnx_inputs(model, cfg, batch):
    """Produce the exact (image, ray_cond32, condition_info) the ONNX consumes."""
    model._initialize_batch(batch)
    B, N = batch["img"].shape[:2]
    model.body_batch_idx = list(range(B * N))
    model.hand_batch_idx = []
    image = data_preprocess_clean(model, model._flatten_person(batch["img"]))
    ray512 = model._flatten_person(get_ray_condition_clean(model, batch))
    ray32 = F.interpolate(ray512, scale_factor=(1 / 16, 1 / 16), mode="bilinear",
                          align_corners=False, antialias=True)
    batch["ray_cond"] = ray512.clone()
    cond = model._get_decoder_condition(batch)
    return image.contiguous(), ray32.contiguous(), cond.contiguous()


# --------------------------------------------------------------------------- #
#  Reference path (drives the unmodified decoder w/ the real momentum MHR)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def run_reference(model, cfg, batch):
    """Body-only branch via the reference's own forward_decoder + real MHR.
    NB: mutates model op-methods, so call BEFORE constructing a BodyWrapper (or on
    a fresh model)."""
    model._initialize_batch(batch)
    B, N = batch["img"].shape[:2]
    model.body_batch_idx = list(range(B * N))
    model.hand_batch_idx = []
    x = data_preprocess_clean(model, model._flatten_person(batch["img"]))
    ray = model._flatten_person(get_ray_condition_clean(model, batch))
    batch["ray_cond"] = ray[model.body_batch_idx].clone()
    ie = model.backbone(x.type(model.backbone_dtype), extra_embed=None).type(x.dtype)
    if cfg.MODEL.PROMPT_ENCODER.get("MASK_EMBED_TYPE", None) is not None:
        ie = ie + model._get_mask_prompt(batch, ie)
    cond = model._get_decoder_condition(batch)
    kp = torch.zeros((B * N, 1, 3)); kp[:, :, -1] = -2
    tokens_output, pose_outputs = model.forward_decoder(
        ie[model.body_batch_idx], init_estimate=None, keypoints=kp[model.body_batch_idx],
        prev_estimate=None, condition_info=cond[model.body_batch_idx], batch=batch)
    po = pose_outputs[-1]
    pred = torch.cat([po["pred_pose_raw"], po["shape"], po["scale"],
                      po["hand"], po["face"] * 0.0], dim=1)
    return {"pred": pred, "pred_cam": po["pred_cam"],
            "hand_logits": model.hand_cls_embed(tokens_output),
            "hand_box": model.bbox_embed(tokens_output).sigmoid()}


# --------------------------------------------------------------------------- #
#  Momentum-MHR export blocker probe (documented in the report)
# --------------------------------------------------------------------------- #
def probe_momentum_blocker():
    """Reproduce the hard ONNX blocker on the shipped momentum MHR TorchScript."""
    import tempfile
    model, cfg = load_model()
    mhr = model.head_pose.mhr
    a = (torch.zeros(1, 45), torch.zeros(1, 204), torch.zeros(1, 72))

    class Tiny(torch.nn.Module):
        def __init__(s, m): super().__init__(); s.m = m
        def forward(s, sh, mp, ex):
            v, sk = s.m(sh, mp, ex); return v.sum() + sk.sum()
    tr = torch.jit.trace(Tiny(mhr), a, check_trace=False)
    tmp = os.path.join(tempfile.gettempdir(), "mhr_probe.onnx")
    try:
        torch.onnx.export(tr, a, tmp, opset_version=17, dynamo=False)
        return "UNEXPECTED: momentum MHR exported"
    except Exception as e:
        return f"{type(e).__name__}: {repr(e)[:200]}"


# --------------------------------------------------------------------------- #
def _to_fp16_tensor(t):
    """In-place convert a FLOAT/DOUBLE TensorProto (raw or typed) to FLOAT16."""
    import numpy as _np
    import onnx
    from onnx import numpy_helper
    if t.data_type not in (onnx.TensorProto.FLOAT, onnx.TensorProto.DOUBLE):
        return
    arr = numpy_helper.to_array(t).astype(_np.float16)
    name = t.name
    t.CopyFrom(numpy_helper.from_array(arr, name))


def _convert_and_save_fp16(fp32, out):
    import onnx
    from onnx import external_data_helper as ed
    src_dir = os.path.dirname(os.path.abspath(fp32))
    print("[export] converting to fp16 (uniform pass) ...")
    m = onnx.load(fp32, load_external_data=True)
    for t in ed._get_all_tensors(m):
        if len(t.external_data):
            ed.load_external_data_for_tensor(t, src_dir)
            t.ClearField("external_data")
        t.data_location = onnx.TensorProto.DEFAULT

    FL, DB, F16 = (onnx.TensorProto.FLOAT, onnx.TensorProto.DOUBLE,
                   onnx.TensorProto.FLOAT16)
    # DINOv3's RoPE (Sin/Cos, and an If subgraph) and the MHR quaternion math
    # keep float32/float64 islands; onnxconverter_common leaves them mixed with
    # the converted fp16 tensors, which ORT rejects. A COMPLETE, RECURSIVE pass
    # (every float/double tensor, Constant, Cast target, and value type -> fp16,
    # descending into If/Loop subgraphs) yields a single-precision graph with no
    # boundary clashes. Numerics verified in validate_body.py.
    def convert_graph(g):
        for t in g.initializer:
            _to_fp16_tensor(t)
        for n in g.node:
            for a in n.attribute:
                if n.op_type == "Constant" and a.name == "value" \
                        and a.t.data_type in (FL, DB):
                    _to_fp16_tensor(a.t)
                if n.op_type == "Cast" and a.name == "to" and a.i in (FL, DB):
                    a.i = F16
                if a.HasField("g"):
                    convert_graph(a.g)           # If/Loop subgraph
                for sub in a.graphs:
                    convert_graph(sub)
                if a.name == "value" and a.t.data_type in (FL, DB) \
                        and n.op_type != "Constant":
                    _to_fp16_tensor(a.t)
        for vi in list(g.input) + list(g.output) + list(g.value_info):
            tt = vi.type.tensor_type
            if tt.elem_type in (FL, DB):
                tt.elem_type = F16

    convert_graph(m.graph)

    # ORT has no fp16 CPU kernel for ScatterElements/ScatterND with reduction=
    # 'add' (the LBS accumulation). Keep just those nodes in fp32: cast their
    # data+updates inputs up, run the scatter in fp32, cast the result back.
    from onnx import helper
    g = m.graph
    new_nodes = []
    wrapped = 0
    for n in list(g.node):
        red = [a.s for a in n.attribute if a.name == "reduction"]
        if n.op_type in ("ScatterElements", "ScatterND") and red and red[0] == b"add":
            data_in, idx_in, upd_in = n.input[0], n.input[1], n.input[2]
            out_name = n.output[0]
            d32, u32, o32 = data_in + "_f32", upd_in + "_f32", out_name + "_f32"
            new_nodes.append(helper.make_node("Cast", [data_in], [d32], to=FL))
            new_nodes.append(helper.make_node("Cast", [upd_in], [u32], to=FL))
            n.input[0], n.input[2] = d32, u32
            n.output[0] = o32
            new_nodes.append(n)
            new_nodes.append(helper.make_node("Cast", [o32], [out_name], to=F16))
            wrapped += 1
        else:
            new_nodes.append(n)
    del g.node[:]
    g.node.extend(new_nodes)
    print(f"[export] kept {wrapped} scatter-add node(s) in fp32")

    # The fp16 model is < 2 GB, so save it INLINE (single self-contained file).
    # External data breaks the CoreML MLProgram path (it needs a non-empty
    # model_path to read external initializers); inline avoids that entirely.
    dpath = os.path.join(os.path.dirname(out), os.path.basename(out) + ".data")
    if os.path.exists(dpath):
        os.remove(dpath)
    try:
        onnx.save(m, out)  # inline
        note = "single file"
    except (ValueError, Exception):  # >2 GB protobuf -> fall back to external data
        onnx.save(m, out, save_as_external_data=True, all_tensors_to_one_file=True,
                  location=os.path.basename(out) + ".data", convert_attribute=True)
        note = "+ .data (external, >2 GB)"
    total = os.path.getsize(out) + (os.path.getsize(dpath) if os.path.exists(dpath) else 0)
    print(f"[export] wrote {out} ({note}); {total/1e6:.1f} MB fp16 (fp16 I/O)")
    return out


def export(mode="faithful", out=DEFAULT_OUT, keep_fp32=False, correctives=False,
           from_fp32=None):
    import shutil
    import tempfile

    os.makedirs(os.path.dirname(out), exist_ok=True)
    if from_fp32:  # skip tracing; just (re)convert an existing fp32 export
        return _convert_and_save_fp16(from_fp32, out)

    print(f"[export] loading model (mode={mode}, correctives={correctives}) ...")
    model, cfg = load_model()
    mesh = None if mode == "meshfree" else build_mesh(use_double=False,
                                                      correctives=correctives)
    wrap = BodyWrapper(model, cfg, mesh=mesh, mesh_free=(mode == "meshfree")).eval()

    batch = build_batch()
    image, ray32, cond = make_onnx_inputs(model, cfg, batch)
    with torch.no_grad():
        wrap(image, ray32, cond)  # smoke run

    # torch.onnx writes fp32 > 2 GB as external data alongside the .onnx path, so
    # export into a scratch dir (keeps models/ clean) then fold to a single fp16.
    keepdir = out.replace(".onnx", "_fp32")
    tmpdir = keepdir if keep_fp32 else tempfile.mkdtemp(prefix="sam3dbody_export_")
    os.makedirs(tmpdir, exist_ok=True)
    fp32 = os.path.join(tmpdir, "body_fp32.onnx")
    print("[export] tracing + ONNX export (fp32) ...")
    torch.onnx.export(
        wrap, (image, ray32, cond), fp32,
        input_names=["image", "ray_cond", "condition_info"],
        output_names=OUTPUT_NAMES, opset_version=17, dynamo=False,
        do_constant_folding=True)
    _convert_and_save_fp16(fp32, out)
    if not keep_fp32:
        shutil.rmtree(tmpdir, ignore_errors=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["faithful", "meshfree"], default="faithful",
                    help="faithful: MHR-driven refinement in-graph (MhrMeshModule); "
                         "meshfree: drop refinement (smaller, degraded parity).")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--keep-fp32", action="store_true")
    ap.add_argument("--correctives", action="store_true",
                    help="keep MHR pose-corrective blendshapes in the refinement "
                         "mesh (bit-exact vs reference, but multi-GB external data).")
    ap.add_argument("--from-fp32", default=None,
                    help="skip tracing; re-convert an existing fp32 body_fp32.onnx.")
    ap.add_argument("--probe-blocker", action="store_true",
                    help="just reproduce+print the momentum-MHR ONNX blocker.")
    args = ap.parse_args()
    if args.probe_blocker:
        print("momentum MHR direct-export blocker:", probe_momentum_blocker())
        return
    export(args.mode, args.out, args.keep_fp32, args.correctives, args.from_fp32)


if __name__ == "__main__":
    main()
