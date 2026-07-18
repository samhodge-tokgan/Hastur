# Copyright the Hastur authors.
# SPDX-License-Identifier: Apache-2.0
#
# mhr_common.py -- shared helpers for the MHR mesh oracle / asset extraction.
#
# Loads the reference SAM-3D-Body checkpoints and reproduces MHRHead.forward
# starting from a raw pred[519] vector (i.e. skipping the FFN `proj`), using the
# EXACT reference modules (roma, mhr_utils, and the TorchScript MHR). This is the
# numerical ground truth that both the ONNX oracle and the C++ MhrModel validate
# against.
#
# Coordinate/unit conventions (from sam_3d_body/models/heads/mhr_head.py):
#   * skel_state is [tx,ty,tz, qx,qy,qz,qw, s]  (quaternion xyzw, scalar scale).
#   * verts and joints leave the TorchScript in centimetres -> divided by 100.
#   * verts[...,[1,2]] *= -1 and joints[...,[1,2]] *= -1 (camera-frame flip).
#   * expression params are zeroed at inference; jaw + hand euler params zeroed.

import os
import sys

import numpy as np
import torch

# --- reference repo wiring ---------------------------------------------------
REF_ROOT = os.environ.get(
    "SAM3D_ROOT", "/Users/sam/Documents/github/sam-3d-body"
)
CKPT = os.path.join(REF_ROOT, "checkpoints/sam-3d-body-dinov3/model.ckpt")
MHR_PT = os.path.join(REF_ROOT, "checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt")

sys.path.insert(0, REF_ROOT)

# Disable the optional momentum python package so the head path falls back to the
# TorchScript (which is what we ship-replace); mhr_head.py keys off this env var.
os.environ["MOMENTUM_ENABLED"] = "0"

# Importing sam_3d_body triggers a torch.hub.list('pytorch/vision', force_reload=True)
# side-effect in its package __init__; neutralize it so the offline extraction/oracle
# tools don't depend on GitHub reachability (it is unused on the mesh path).
import torch.hub as _hub  # noqa: E402
_hub.list = lambda *a, **k: []  # type: ignore

import roma  # noqa: E402
from sam_3d_body.models.modules import rot6d_to_rotmat  # noqa: E402
from sam_3d_body.models.modules.mhr_utils import (  # noqa: E402
    compact_cont_to_model_params_body,
    compact_cont_to_model_params_hand,
    mhr_param_hand_mask,
)

# 519-d layout (mirrors src/MeshTypes.h).
G6D, BODY_CONT, SHAPE, SCALE, HANDPCA, EXPR = 6, 260, 45, 28, 54, 72
LN2 = 0.6931471805599453


def load_mhr_ts(device="cpu"):
    m = torch.jit.load(MHR_PT, map_location=device).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def load_head_buffers(device="cpu"):
    """Pull the MHRHead buffers for the *body* head (head_pose.*) from the ckpt."""
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    pref = "head_pose."
    names = [
        "scale_mean", "scale_comps", "keypoint_mapping",
        "hand_pose_mean", "hand_pose_comps",
        "hand_joint_idxs_left", "hand_joint_idxs_right",
    ]
    out = {}
    for n in names:
        out[n] = sd[pref + n].to(device)
    return out


class MhrReference:
    """Reproduces MHRHead.forward (body head, enable_hand_model=False) from pred[519]."""

    def __init__(self, device="cpu"):
        self.device = device
        self.mhr = load_mhr_ts(device)
        b = load_head_buffers(device)
        self.scale_mean = b["scale_mean"]
        self.scale_comps = b["scale_comps"]
        self.keypoint_mapping = b["keypoint_mapping"]
        self.hand_pose_mean = b["hand_pose_mean"]
        self.hand_pose_comps = b["hand_pose_comps"]
        self.hand_joint_idxs_left = b["hand_joint_idxs_left"].long()
        self.hand_joint_idxs_right = b["hand_joint_idxs_right"].long()

    def decode_params(self, pred):
        """pred[B,519] -> (model_params[B,204], shape[B,45], expr[B,72])."""
        c = 0
        global_rot_6d = pred[:, c:c + G6D]; c += G6D
        global_rot_rotmat = rot6d_to_rotmat(global_rot_6d)
        global_rot_euler = roma.rotmat_to_euler("ZYX", global_rot_rotmat)
        global_trans = torch.zeros_like(global_rot_euler)

        pred_pose_cont = pred[:, c:c + BODY_CONT]; c += BODY_CONT
        pred_pose_euler = compact_cont_to_model_params_body(pred_pose_cont)
        pred_pose_euler[:, mhr_param_hand_mask] = 0
        pred_pose_euler[:, -3:] = 0  # jaw

        pred_shape = pred[:, c:c + SHAPE]; c += SHAPE
        pred_scale = pred[:, c:c + SCALE]; c += SCALE
        pred_hand = pred[:, c:c + 2 * HANDPCA]; c += 2 * HANDPCA
        pred_face = pred[:, c:c + EXPR] * 0; c += EXPR

        body_pose_params = pred_pose_euler[..., :130]
        scales = self.scale_mean[None, :] + pred_scale @ self.scale_comps
        full_pose = torch.cat([global_trans * 10, global_rot_euler, body_pose_params], dim=1)

        # replace_hands_in_pose
        lh, rh = torch.split(pred_hand, [HANDPCA, HANDPCA], dim=1)
        lh_mp = compact_cont_to_model_params_hand(
            self.hand_pose_mean + torch.einsum("da,ab->db", lh, self.hand_pose_comps))
        rh_mp = compact_cont_to_model_params_hand(
            self.hand_pose_mean + torch.einsum("da,ab->db", rh, self.hand_pose_comps))
        full_pose[:, self.hand_joint_idxs_left] = lh_mp
        full_pose[:, self.hand_joint_idxs_right] = rh_mp

        model_params = torch.cat([full_pose, scales], dim=1)  # B,204
        return model_params, pred_shape, pred_face

    @torch.no_grad()
    def forward(self, pred):
        """pred[B,519] -> dict(verts[B,V,3], joints[B,127,3], keypoints[B,70,3]) in metres, flipped."""
        model_params, shape, expr = self.decode_params(pred)
        verts, skel_state = self.mhr(shape, model_params, expr)
        jcoords = skel_state[:, :, :3].clone()
        verts = verts / 100.0
        jcoords = jcoords / 100.0

        model_vert_joints = torch.cat([verts, jcoords], dim=1)  # B,18566,3
        B = verts.shape[0]
        kpts = (self.keypoint_mapping @ model_vert_joints.permute(1, 0, 2).flatten(1, 2)
                ).reshape(-1, B, 3).permute(1, 0, 2)  # B,308,3
        j3d = kpts[:, :70].clone()

        verts[..., [1, 2]] *= -1
        j3d[..., [1, 2]] *= -1
        jcoords[..., [1, 2]] *= -1
        return {"verts": verts, "joints": jcoords, "keypoints": j3d,
                "model_params": model_params, "shape": shape}
