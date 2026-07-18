#!/usr/bin/env python
# Copyright the Hastur authors.
# SPDX-License-Identifier: LicenseRef-SAM-License
#
# validate_wrist.py -- ground-truth the M7 wrist-angle criterion (CRITERIA 1) for
# a given image, replicating sam3d_body.run_inference with the reference decoders +
# real TorchScript MHR head. Prints the per-hand angle_difference (radians) the
# reference would gate on (threshold 1.4), so the C++ wrist gate can be checked.
import os
import sys

import numpy as np
import roma
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import export_sam3dbody as B
import export_hand as H
from mhr_common import compact_cont_to_model_params_body, mhr_param_hand_mask


def rot_angle_diff(A, Bm):
    R = A @ Bm.transpose(-2, -1)
    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    return torch.acos(torch.clamp((tr - 1) / 2, -1, 1))


@torch.no_grad()
def body_joint_global_rots(model, cfg, pred):
    """Reference joint_global_rots from the real head_pose.mhr_forward, decoding
    pred[519] the way head_pose.forward does."""
    head = model.head_pose
    g6 = pred[:, :6]
    from sam_3d_body.models.modules import rot6d_to_rotmat
    grm = rot6d_to_rotmat(g6)
    grot = roma.rotmat_to_euler("ZYX", grm)
    gtrans = torch.zeros_like(grot)
    cont = pred[:, 6:266]
    pe = compact_cont_to_model_params_body(cont)
    pe[:, mhr_param_hand_mask] = 0
    pe[:, -3:] = 0
    shape = pred[:, 266:311]
    scale = pred[:, 311:339]
    hand = pred[:, 339:447]
    face = pred[:, 447:519] * 0
    out = head.mhr_forward(
        global_trans=gtrans, global_rot=grot, body_pose_params=pe,
        hand_pose_params=hand, scale_params=scale, shape_params=shape,
        expr_params=face, return_keypoints=True, return_joint_coords=True,
        return_model_params=True, return_joint_rotations=True)
    return out[-1]  # joint_global_rots (1,127,3,3)


@torch.no_grad()
def main(img_path):
    import cv2
    model, cfg = B.load_model()
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    Himg, Wimg = img.shape[:2]
    S = cfg.MODEL.IMAGE_SIZE[0]

    # --- body decoder on the person box (detector box via HASTUR_BOX env) ---
    if os.environ.get("HASTUR_BOX"):
        box = np.array([[float(v) for v in os.environ["HASTUR_BOX"].split(",")]],
                       np.float32)
    else:
        box = np.array([[0, 0, Wimg, Himg]], np.float32)
    print("body box:", box.tolist())
    bbatch = B.build_batch(box=box, img_path=img_path)
    ref = B.run_reference(model, cfg, bbatch)
    pred = ref["pred"]

    # ori_local_wrist (XZY of body pose params [41,43,42]=L, [31,33,32]=R)
    body133 = compact_cont_to_model_params_body(pred[:, 6:266])
    wr = body133[:, [41, 43, 42, 31, 33, 32]].reshape(1, 2, 3)
    ori_local = roma.euler_to_rotmat("XZY", wr)  # (1,2,3,3)

    # body FK lowarm [76(L),40(R)] via the REAL head
    jgr = body_joint_global_rots(model, cfg, pred)  # (1,127,3,3)
    lowarm = jgr[:, [76, 40]]  # (1,2,3,3)
    joint_rotation = model.head_pose.joint_rotation[[77, 41]]  # (2,3,3)

    # hand boxes (crop -> full frame)
    hb = ref["hand_box"][0]  # (2,4) cx,cy,w,h in [0,1] of the 512 body crop
    aff = bbatch["affine_trans"][0, 0]  # (2,3) frame->crop
    a00 = aff[0, 0].item()

    def to_full(side):
        cx, cy, w, h = (hb[side] * S).tolist()
        s = max(w, h)
        fx = (cx - aff[0, 2].item()) / a00
        fy = (cy - aff[1, 2].item()) / a00
        fs = s / a00
        return np.array([[fx - fs/2, fy - fs/2, fx + fs/2, fy + fs/2]], np.float32), fs

    pred_wrist = []
    for side in range(2):  # 0=L,1=R
        fbox, fs = to_full(side)
        if side == 0:
            flip = img[:, ::-1].copy()
            tmp = fbox.copy()
            fbox[:, 0] = Wimg - tmp[:, 2] - 1
            fbox[:, 2] = Wimg - tmp[:, 0] - 1
            hbatch = _batch_from_frame(cfg, flip, fbox)
            hp = H.run_reference_hand(model, cfg, hbatch)
            w42 = hp["wrist_global"][:, 1].clone()  # joint 42
            w42[:, [1, 2], :] *= -1  # unflip -> left wrist
            pred_wrist.append(w42)
        else:
            hbatch = _batch_from_frame(cfg, img, fbox)
            hp = H.run_reference_hand(model, cfg, hbatch)
            pred_wrist.append(hp["wrist_global"][:, 1].clone())  # joint 42
        print(f"side {side} box_full_px_size={fs:.1f}")

    pred_global_wrist = torch.stack(pred_wrist, dim=1)  # (1,2,3,3)
    wrist_zero = lowarm @ joint_rotation[None]
    fused = torch.einsum("kabc,kabd->kadc", pred_global_wrist, wrist_zero)
    ang = rot_angle_diff(ori_local, fused)[0]  # (2,)
    print(f"REFERENCE wrist angle_difference: L={ang[0]:.3f} R={ang[1]:.3f} rad "
          f"(threshold 1.4; <1.4 = refine)")


def _batch_from_frame(cfg, frame, box):
    from sam_3d_body.data.transforms import (Compose, GetBBoxCenterScale,
                                             TopdownAffine, VisionTransformWrapper)
    from torchvision.transforms import ToTensor
    from sam_3d_body.data.utils.prepare_batch import prepare_batch
    tf = Compose([GetBBoxCenterScale(padding=0.9),
                  TopdownAffine(input_size=[512, 512], use_udp=False),
                  VisionTransformWrapper(ToTensor())])
    batch = prepare_batch(frame, tf, box)
    for k in list(batch.keys()):
        if torch.is_tensor(batch[k]):
            batch[k] = batch[k].float().cpu()
    return batch


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else
         os.path.join(B.REF_ROOT, "notebook/images/dancing.jpg"))
