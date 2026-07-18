#!/usr/bin/env python
# Copyright the Hastur authors.
# SPDX-License-Identifier: Apache-2.0
#
# export_hand.py -- export the SAM-3D-Body *hand-decoder* 2nd pass to a
# fixed-shape fp16 ONNX for the Hastur HandRefinerEngine (M7).
#
#   inputs : image[1,3,512,512]  (ImageNet-normalized CHW, a HAND crop, pad 0.9)
#            ray_cond[1,2,32,32]  (token-grid ray directions of the hand crop)
#            condition_info[1,3]  (CLIFF cx/f, cy/f, b/f; USE_INTRIN_CENTER)
#   outputs: pred[1,519]         (hand-decoder MHR-head param vector; the hand
#                                 PCA[108]/scale[28]/shape[45] slices are merged
#                                 into the body pred by the C++ refiner)
#            wrist_global[1,2,3,3] (GLOBAL rotation matrices of the wrist joints
#                                 [78=left, 42=right] from the in-graph MHR FK --
#                                 == pose_output["mhr_hand"]["joint_global_rots"]
#                                 at those joints; drives the wrist-angle gate.)
#
# This is the direct analogue of tools/export_sam3dbody.py (the body pass). It
# shares that tool's crop contract (image / ray_cond / condition_info) and all of
# its export mechanics (clean CPU/fp32 replacements for the mps-pinned helpers,
# condition_info-only camera projection, the ONNX-exportable MhrMeshModule for the
# in-graph per-layer refinement, scatter-add LBS, the recursive fp16 fold). The
# ONLY new pieces are:
#
#   (1) it drives model.forward_decoder_hand (head_pose_hand / decoder_hand /
#       ray_cond_emb_hand / keypoint*_token_update_fn_hand) instead of the body
#       forward_decoder; and
#   (2) a HandMeshModule that faithfully reproduces the hand-head mhr_forward
#       (enable_hand_model=True): it applies the local_to_world_wrist transform to
#       the global rotation, zeros the non-hand model params, keeps only the
#       right-hand keypoints (21:42) for the refinement, and additionally emits the
#       wrist global rotmats needed for the merge's wrist-angle criterion.
#
# The reference "full" hand pass (parity target) is model.forward_decoder_hand
# driven with the real shipped TorchScript MHR head -- exactly the per-crop hand
# building block run_inference() invokes for each of the two hand crops.
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# Reuse the body export tool verbatim for the shared machinery.
import export_sam3dbody as B  # noqa: E402
from export_sam3dbody import (  # noqa: E402
    load_model, data_preprocess_clean, get_ray_condition_clean,
    camera_project_cond, full_to_crop_passthrough, camera_encoder_no_interp,
    build_mesh, _convert_and_save_fp16, REF_ROOT,
)

DEFAULT_OUT = os.path.join(os.path.dirname(HERE), "models", "sam3dbody_hand.onnx")
OUTPUT_NAMES = ["pred", "wrist_global"]

# MHR skeleton joint indices for the two wrists (sam3d_body.run_inference):
#   left wrist = 78, right wrist = 42.
WRIST_JOINT_IDXS = [78, 42]


# --------------------------------------------------------------------------- #
#  euler<->rotmat for the enable_hand_model wrist transform (roma "xyz"),
#  implemented with plain trig so the graph stays ONNX-exportable.
# --------------------------------------------------------------------------- #
def euler_xyz_to_rotmat(e):
    # roma "xyz" == Rz(a2) @ Ry(a1) @ Rx(a0)  (verified vs roma AND the FK's
    # euler_xyz_to_quat, which agree to ~6e-8). e: (...,3)
    x, y, z = e[..., 0], e[..., 1], e[..., 2]
    cx, sx = torch.cos(x), torch.sin(x)
    cy, sy = torch.cos(y), torch.sin(y)
    cz, sz = torch.cos(z), torch.sin(z)
    r00 = cz * cy
    r01 = -sz * cx + cz * sy * sx
    r02 = sz * sx + cz * sy * cx
    r10 = sz * cy
    r11 = cz * cx + sz * sy * sx
    r12 = -cz * sx + sz * sy * cx
    r20 = -sy
    r21 = cy * sx
    r22 = cy * cx
    row0 = torch.stack([r00, r01, r02], dim=-1)
    row1 = torch.stack([r10, r11, r12], dim=-1)
    row2 = torch.stack([r20, r21, r22], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)  # (...,3,3)


def rotmat_to_euler_xyz(R):
    # inverse of euler_xyz_to_rotmat (roma "xyz" == Rz@Ry@Rx).
    r00, r10, r20 = R[..., 0, 0], R[..., 1, 0], R[..., 2, 0]
    r21, r22 = R[..., 2, 1], R[..., 2, 2]
    a0 = torch.atan2(r21, r22)
    a1 = torch.atan2(-r20, torch.sqrt(r00 * r00 + r10 * r10))
    a2 = torch.atan2(r10, r00)
    return torch.stack([a0, a1, a2], dim=-1)


def unitquat_to_rotmat(q):
    # q xyzw (B,...,4) -> rotmat (B,...,3,3). Standard formula.
    q = F.normalize(q, dim=-1)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    r00 = 1 - 2 * (y * y + z * z)
    r01 = 2 * (x * y - z * w)
    r02 = 2 * (x * z + y * w)
    r10 = 2 * (x * y + z * w)
    r11 = 1 - 2 * (x * x + z * z)
    r12 = 2 * (y * z - x * w)
    r20 = 2 * (x * z - y * w)
    r21 = 2 * (y * z + x * w)
    r22 = 1 - 2 * (x * x + y * y)
    row0 = torch.stack([r00, r01, r02], dim=-1)
    row1 = torch.stack([r10, r11, r12], dim=-1)
    row2 = torch.stack([r20, r21, r22], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


# --------------------------------------------------------------------------- #
#  HandMeshModule -- MhrMeshModule + the hand-head (enable_hand_model) semantics.
# --------------------------------------------------------------------------- #
def build_hand_mesh(model, correctives=False):
    from mhr_meshmodule import MhrMeshModule
    from mhr_common import MhrReference

    base = build_mesh(use_double=False, correctives=correctives)  # ORT-loadable LBS
    head = model.head_pose_hand

    # Extra hand-head buffers (rig geometry constants).
    base.register_buffer("local_to_world_wrist",
                         head.local_to_world_wrist.detach().float().clone(),
                         persistent=False)
    base.register_buffer("right_wrist_coords",
                         head.right_wrist_coords.detach().float().clone(),
                         persistent=False)
    base.register_buffer("root_coords",
                         head.root_coords.detach().float().clone(),
                         persistent=False)
    base.register_buffer("nonhand_param_idxs",
                         head.nonhand_param_idxs.detach().long().clone(),
                         persistent=False)
    base.register_buffer("wrist_joint_idxs",
                         torch.tensor(WRIST_JOINT_IDXS, dtype=torch.long),
                         persistent=False)
    base.register_buffer("wrist_zero_mask",
                         torch.ones(204), persistent=False)
    # float mask that zeroes the non-hand model params (index_copy-friendly).
    m = torch.ones(204)
    m[head.nonhand_param_idxs.long()] = 0.0
    base.wrist_zero_mask.data = m

    orig_decode = base.decode
    orig_forward = base.forward

    def decode_hand(pred):
        # Reproduce MhrMeshModule.decode but with the enable_hand_model global-rot
        # transform, then zero the non-hand model params.
        from mhr_meshmodule import (rot6d_to_rotmat, quat_to_euler_zyx,
                                     rotmat_to_unitquat, compact_cont_body_exp,
                                     compact_cont_hand_exp)
        G6D, BODY_CONT, SHAPE, SCALE, HANDPCA = 6, 260, 45, 28, 54
        c = 0
        g6 = pred[:, c:c + G6D]; c += G6D
        grm = rot6d_to_rotmat(g6)
        grot0 = quat_to_euler_zyx(rotmat_to_unitquat(grm))          # "ZYX" euler
        # enable_hand_model: transfer wrist-centric prediction to the body root.
        R0 = euler_xyz_to_rotmat(grot0)
        Rw = torch.matmul(R0, base.local_to_world_wrist[None])
        grot = rotmat_to_euler_xyz(Rw)
        Rg = euler_xyz_to_rotmat(grot)
        offs = base.right_wrist_coords - base.root_coords
        gtrans = -(torch.matmul(Rg, offs[None, :, None]).squeeze(-1)
                   + base.root_coords[None])
        cont = pred[:, c:c + BODY_CONT]; c += BODY_CONT
        pe = compact_cont_body_exp(cont) * base.body_zero_mask[None]
        shape = pred[:, c:c + SHAPE]; c += SHAPE
        scale = pred[:, c:c + SCALE]; c += SCALE
        hand = pred[:, c:c + 2 * HANDPCA]; c += 2 * HANDPCA
        body = pe[:, :130]
        scales = base.scale_mean[None] + scale @ base.scale_comps
        full = torch.cat([gtrans * 10, grot, body], dim=1)  # (B,136)
        lh, rh = torch.split(hand, [HANDPCA, HANDPCA], dim=1)
        lh_mp = compact_cont_hand_exp(base.hand_pose_mean
                                      + torch.einsum("da,ab->db", lh, base.hand_pose_comps))
        rh_mp = compact_cont_hand_exp(base.hand_pose_mean
                                      + torch.einsum("da,ab->db", rh, base.hand_pose_comps))
        full = full.index_copy(1, base.hand_idx_left, lh_mp)
        full = full.index_copy(1, base.hand_idx_right, rh_mp)
        model_params = torch.cat([full, scales], dim=1)         # (B,204)
        # enable_hand_model: zero the non-hand model params.
        model_params = model_params * base.wrist_zero_mask[None]
        return model_params, shape

    base.decode = decode_hand

    def forward_hand(pred):
        # Mirror MhrMeshModule.forward but keep only the right-hand kps (21:42) and
        # also return the wrist global rotation matrices.
        model_params, shape = base.decode(pred)
        Bn = pred.shape[0]
        identity_rest = base.base_shape[None] + torch.einsum(
            "nvd,bn->bvd", base.shape_vectors, shape)
        full249 = torch.cat(
            [model_params, torch.zeros(Bn, 45, dtype=pred.dtype, device=pred.device)],
            dim=1)
        jp = torch.einsum("dn,bn->bd", base.parameter_transform, full249)
        local = base.local_skel(jp)
        glob = base.fk(local)                       # (B,J,8): t,q(xyzw),s
        corr = base.pose_corrective(jp)
        unposed = identity_rest + corr
        verts_cm = base.lbs(glob, unposed)
        jcoords_cm = glob[..., :3]
        verts = verts_cm / 100.0
        jcoords = jcoords_cm / 100.0
        mvj = torch.cat([verts, jcoords], dim=1)
        kpts = torch.einsum("kn,bnd->bkd", base.keypoint_mapping, mvj)  # (B,308,3)
        j3d = kpts[:, :70].clone()
        # enable_hand_model: keep only right-hand keypoints (21:42).
        mask = torch.zeros(70, 1, dtype=j3d.dtype, device=j3d.device)
        mask[21:42] = 1.0
        j3d = j3d * mask[None]
        flip = torch.tensor([1.0, -1.0, -1.0], dtype=verts.dtype, device=verts.device)
        j3d = j3d * flip
        # wrist global rotmats (joints [78,42]).
        wq = glob.index_select(1, base.wrist_joint_idxs)[..., 3:7]  # (B,2,4) xyzw
        wrist_global = unitquat_to_rotmat(wq)                       # (B,2,3,3)
        return verts, jcoords, j3d, wrist_global

    base.forward_hand = forward_hand
    return base


# --------------------------------------------------------------------------- #
#  head_pose_hand.forward -> raw 519 slices + right-hand j3d (from HandMeshModule)
# --------------------------------------------------------------------------- #
def make_hand_head_forward(mesh):
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
        pred519 = torch.cat([g6, cont, shape, scale, hand, face], dim=1)
        _, _, j3d, _ = mesh.forward_hand(pred519)
        return {
            "pred_pose_raw": torch.cat([g6, cont], dim=1),
            "shape": shape, "scale": scale, "hand": hand, "face": face,
            "pred_keypoints_3d": j3d, "pred_vertices": None,
        }
    return head_forward


class HandWrapper(torch.nn.Module):
    """Fixed batch=1 hand-decoder wrapper (mirrors BodyWrapper)."""

    def __init__(self, model, cfg, mesh):
        super().__init__()
        self.model = model
        self.mesh = mesh  # submodule so its buffers trace/export
        # cond-only camera projection (validated ~1e-7 vs the absolute path).
        model.camera_project_hand = camera_project_cond.__get__(model, type(model))
        model._full_to_crop = full_to_crop_passthrough.__get__(model, type(model))
        model.ray_cond_emb_hand.forward = camera_encoder_no_interp.__get__(
            model.ray_cond_emb_hand, type(model.ray_cond_emb_hand))
        model.head_pose_hand.forward = make_hand_head_forward(mesh).__get__(
            model.head_pose_hand, type(model.head_pose_hand))
        model.body_batch_idx = []
        model.hand_batch_idx = [0]
        self.register_buffer("dummy_kp", torch.zeros(1, 1, 3))
        self.dummy_kp[:, :, -1] = -2
        self._fake_batch = {}

    def forward(self, image, ray_cond, condition_info):
        model = self.model
        model._cond_body = condition_info   # camera_project_cond reads this
        model.hand_batch_idx = [0]
        model.body_batch_idx = []
        self._fake_batch["ray_cond_hand"] = ray_cond
        ie = model.backbone(image, extra_embed=None)
        if isinstance(ie, tuple):
            ie = ie[-1]
        ie = ie + model.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1)
        tokens_output, pose_outputs = model.forward_decoder_hand(
            ie, init_estimate=None, keypoints=self.dummy_kp, prev_estimate=None,
            condition_info=condition_info, batch=self._fake_batch)
        po = pose_outputs[-1]
        pred = torch.cat([po["pred_pose_raw"], po["shape"], po["scale"],
                          po["hand"], po["face"] * 0.0], dim=1)
        # wrist global rotmats from the final refined pred.
        _, _, _, wrist_global = self.mesh.forward_hand(pred)
        return pred, wrist_global


# --------------------------------------------------------------------------- #
#  Hand-crop batch + ONNX inputs
# --------------------------------------------------------------------------- #
def build_hand_batch(box, img_path=None):
    import cv2
    from sam_3d_body.data.transforms import (Compose, GetBBoxCenterScale,
                                             TopdownAffine, VisionTransformWrapper)
    from torchvision.transforms import ToTensor
    from sam_3d_body.data.utils.prepare_batch import prepare_batch
    if img_path is None:
        img_path = os.path.join(REF_ROOT, "notebook/images/dancing.jpg")
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]
    tf = Compose([GetBBoxCenterScale(padding=0.9),
                  TopdownAffine(input_size=[512, 512], use_udp=False),
                  VisionTransformWrapper(ToTensor())])
    batch = prepare_batch(img, tf, box)
    for k in list(batch.keys()):
        if torch.is_tensor(batch[k]):
            batch[k] = batch[k].float().cpu()
    return batch, img


def make_onnx_inputs(model, cfg, batch):
    model._initialize_batch(batch)
    Bn, N = batch["img"].shape[:2]
    model.body_batch_idx = []
    model.hand_batch_idx = list(range(Bn * N))
    image = data_preprocess_clean(model, model._flatten_person(batch["img"]))
    ray512 = model._flatten_person(get_ray_condition_clean(model, batch))
    ray32 = F.interpolate(ray512, scale_factor=(1 / 16, 1 / 16), mode="bilinear",
                          align_corners=False, antialias=True)
    batch["ray_cond_hand"] = ray512.clone()
    cond = model._get_decoder_condition(batch)
    return image.contiguous(), ray32.contiguous(), cond.contiguous()


# --------------------------------------------------------------------------- #
#  Reference "full" hand pass (unmodified decoder + real TorchScript MHR head)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def run_reference_hand(model, cfg, batch):
    """The per-crop hand building block run_inference() uses: forward_decoder_hand
    driven with the real shipped head_pose_hand (momentum-free TorchScript MHR).
    Uses the real camera_project_hand (absolute) so the crop-space keypoints that
    drive the refinement are exactly the reference's."""
    model._initialize_batch(batch)
    Bn, N = batch["img"].shape[:2]
    model.body_batch_idx = []
    model.hand_batch_idx = list(range(Bn * N))
    x = data_preprocess_clean(model, model._flatten_person(batch["img"]))
    ray = model._flatten_person(get_ray_condition_clean(model, batch))
    batch["ray_cond_hand"] = ray[model.hand_batch_idx].clone()
    ie = model.backbone(x.type(model.backbone_dtype), extra_embed=None).type(x.dtype)
    ie = ie + model.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1)
    cond = model._get_decoder_condition(batch)
    kp = torch.zeros((Bn * N, 1, 3)); kp[:, :, -1] = -2
    tokens_output, pose_outputs = model.forward_decoder_hand(
        ie[model.hand_batch_idx], init_estimate=None, keypoints=kp[model.hand_batch_idx],
        prev_estimate=None, condition_info=cond[model.hand_batch_idx], batch=batch)
    po = pose_outputs[-1]
    pred = torch.cat([po["pred_pose_raw"], po["shape"], po["scale"],
                      po["hand"], po["face"] * 0.0], dim=1)
    wrist = po["joint_global_rots"][:, WRIST_JOINT_IDXS]  # (B,2,3,3)
    return {"pred": pred, "wrist_global": wrist,
            "hand": po["hand"], "scale": po["scale"], "shape": po["shape"]}


# --------------------------------------------------------------------------- #
def _hand_box_from_body(model, cfg, img_path):
    """Run the body decoder once to get a realistic hand box (crop-frame cx,cy,w,h)
    -> full-frame xyxy, for building a representative hand crop."""
    from export_sam3dbody import build_batch, run_reference
    batch = build_batch(img_path=img_path)
    out = run_reference(model, cfg, batch)
    # hand_box[:, 1] = right hand (cx,cy,w,h) in [0,1] of the 512 body crop.
    hb = out["hand_box"][0]  # (2,4)
    aff = batch["affine_trans"][0, 0]  # (2,3) full->crop
    S = cfg.MODEL.IMAGE_SIZE[0]
    boxes = []
    for side in range(2):
        cx, cy, w, h = (hb[side] * S).tolist()
        s = max(w, h)
        # crop-space center/scale -> full-frame
        a00 = aff[0, 0].item()
        fx = (cx - aff[0, 2].item()) / a00
        fy = (cy - aff[1, 2].item()) / a00
        fs = s / a00
        boxes.append([fx - fs / 2, fy - fs / 2, fx + fs / 2, fy + fs / 2])
    return np.array(boxes, dtype=np.float32)


# --------------------------------------------------------------------------- #
def export(out=DEFAULT_OUT, keep_fp32=False, correctives=False, img_path=None):
    import shutil
    import tempfile

    os.makedirs(os.path.dirname(out), exist_ok=True)
    print("[export-hand] loading model ...")
    model, cfg = load_model()
    if img_path is None:
        img_path = os.path.join(REF_ROOT, "notebook/images/dancing.jpg")

    # A representative hand crop: derive the right-hand box from the body decoder.
    try:
        boxes = _hand_box_from_body(model, cfg, img_path)
        box = boxes[1:2]  # right hand
    except Exception as e:
        print(f"[export-hand] body hand-box probe failed ({e}); using center box")
        import cv2
        im = cv2.imread(img_path)
        H, W = im.shape[:2]
        box = np.array([[W * 0.4, H * 0.4, W * 0.6, H * 0.6]], dtype=np.float32)

    mesh = build_hand_mesh(model, correctives=correctives)
    wrap = HandWrapper(model, cfg, mesh).eval()

    batch, _ = build_hand_batch(box, img_path=img_path)
    image, ray32, cond = make_onnx_inputs(model, cfg, batch)
    with torch.no_grad():
        wrap(image, ray32, cond)  # smoke run

    keepdir = out.replace(".onnx", "_fp32")
    tmpdir = keepdir if keep_fp32 else tempfile.mkdtemp(prefix="sam3dbody_hand_export_")
    os.makedirs(tmpdir, exist_ok=True)
    fp32 = os.path.join(tmpdir, "hand_fp32.onnx")
    print("[export-hand] tracing + ONNX export (fp32) ...")
    torch.onnx.export(
        wrap, (image, ray32, cond), fp32,
        input_names=["image", "ray_cond", "condition_info"],
        output_names=OUTPUT_NAMES, opset_version=17, dynamo=False,
        do_constant_folding=True)
    _convert_and_save_fp16(fp32, out)
    if not keep_fp32:
        shutil.rmtree(tmpdir, ignore_errors=True)
    write_sidecars(out, model)
    return out


def write_sidecars(out, model):
    """Emit the small side artifacts the C++ refiner needs:
      * mhr_wrist.bin -- joint_rotation[[77(L),41(R)]] rotmats (18 float32), the
        wrist-twist rest rotations for the wrist-angle criterion. Absent -> the
        pipeline skips CRITERIA 1 (box-size gate only).
      * <out>.json    -- the I/O contract sidecar (mirrors the body export).
    """
    import json
    outdir = os.path.dirname(out)
    jr = model.head_pose.joint_rotation.detach().float().cpu().numpy()  # (127,3,3)
    wrist = np.concatenate([jr[77].reshape(-1), jr[41].reshape(-1)]).astype("<f4")
    wpath = os.path.join(outdir, "mhr_wrist.bin")
    wrist.tofile(wpath)
    print(f"[export-hand] wrote {wpath} ({wrist.nbytes} bytes: joint_rotation[77,41])")

    meta = {
        "name": "sam3dbody_hand",
        "kind": "hand-decoder 2nd pass (M7)",
        "precision": "fp16 (fp16 I/O)",
        "inputs": {
            "image": [1, 3, 512, 512],
            "ray_cond": [1, 2, 32, 32],
            "condition_info": [1, 3],
        },
        "outputs": {"pred": [1, 519], "wrist_global": [1, 2, 3, 3]},
        "crop": {"padding": 0.9, "imagenet_norm": True, "left_hand": "H-flip"},
        "wrist_joint_idxs": {"left": 78, "right": 42},
        "merge": "replace_hands_in_pose; wrist-angle (1.4 rad) + box-size (64px) gated",
        "sidecar": "mhr_wrist.bin (joint_rotation[77,41], 18xf32)",
        "size_bytes": os.path.getsize(out),
    }
    jpath = out.rsplit(".", 1)[0] + ".json"
    with open(jpath, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[export-hand] wrote {jpath}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--keep-fp32", action="store_true")
    ap.add_argument("--correctives", action="store_true")
    ap.add_argument("--img", default=None, help="image to derive the hand crop from")
    ap.add_argument("--parity", action="store_true",
                    help="run torch-wrapper vs reference-hand-pass parity only")
    args = ap.parse_args()
    if args.parity:
        from validate_hand import run_parity
        run_parity(args.img)
        return
    export(args.out, args.keep_fp32, args.correctives, args.img)


if __name__ == "__main__":
    main()
