# Copyright the Hastur authors.
# SPDX-License-Identifier: LicenseRef-SAM-License
#
# mhr_meshmodule.py -- a PURE-TORCH reimplementation of the MHR mesh path
# (blendshape -> parameter-transform -> local skel state -> FK -> pose-corrective
# -> LBS -> keypoints -> /100 + [1,2] flip), built only from standard ops so it
# (a) exports cleanly to ONNX (the numerical oracle), and (b) is the exact spec
# the C++ MhrModel ports. It is validated bit-for-bit against the shipped
# TorchScript in validate against MhrReference.
import torch
import torch.nn as nn
import torch.nn.functional as F

from mhr_common import (SCALE, SHAPE, MhrReference, compact_cont_to_model_params_body,
                        compact_cont_to_model_params_hand, rot6d_to_rotmat, roma,
                        mhr_param_hand_mask, G6D, BODY_CONT, HANDPCA, EXPR, LN2)


# ---- quaternion / skel-state helpers (xyzw), matching pymomentum backend ----
def quat_mul(q1, q2):
    x1, y1, z1, w1 = q1.unbind(-1)
    x2, y2, z2, w2 = q2.unbind(-1)
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    return torch.stack([x, y, z, w], -1)


def quat_rotate(q, v):
    # assumes q normalized; split -> r=w (scalar), axis=xyz
    axis = q[..., :3]
    r = q[..., 3:4]
    av = torch.cross(axis, v, dim=-1)
    aav = torch.cross(axis, av, dim=-1)
    return v + 2.0 * (av * r + aav)


def euler_xyz_to_quat(e):
    roll, pitch, yaw = e.unbind(-1)
    cy = torch.cos(yaw * 0.5); sy = torch.sin(yaw * 0.5)
    cp = torch.cos(pitch * 0.5); sp = torch.sin(pitch * 0.5)
    cr = torch.cos(roll * 0.5); sr = torch.sin(roll * 0.5)
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    w = cr * cp * cy + sr * sp * sy
    return torch.stack([x, y, z, w], -1)


def skel_normalize_q(ss):
    t, q, s = ss[..., :3], ss[..., 3:7], ss[..., 7:8]
    q = F.normalize(q, dim=-1)
    return t, q, s


def skel_multiply(s1, s2):
    # s_res = s1 . s2 (parent . child), quaternions normalized first
    t1, q1, sc1 = skel_normalize_q(s1)
    t2, q2, sc2 = skel_normalize_q(s2)
    t_res = t1 + sc1 * quat_rotate(q1, t2)
    s_res = sc1 * sc2
    q_res = quat_mul(q1, q2)
    return torch.cat([t_res, q_res, s_res], -1)


def rotmat_to_unitquat(M):
    # roma/scipy decision-matrix method. M: (B,3,3) -> quat xyzw (B,4).
    B = M.shape[0]
    tr = M[:, 0, 0] + M[:, 1, 1] + M[:, 2, 2]
    dec = torch.stack([M[:, 0, 0], M[:, 1, 1], M[:, 2, 2], tr], dim=1)  # (B,4)
    cand = []
    for i in range(3):
        j = (i + 1) % 3
        k = (j + 1) % 3
        q = torch.zeros(B, 4, dtype=M.dtype, device=M.device)
        q[:, i] = 1.0 - tr + 2.0 * M[:, i, i]
        q[:, j] = M[:, j, i] + M[:, i, j]
        q[:, k] = M[:, k, i] + M[:, i, k]
        q[:, 3] = M[:, k, j] - M[:, j, k]
        cand.append(q)
    q3 = torch.stack([M[:, 2, 1] - M[:, 1, 2], M[:, 0, 2] - M[:, 2, 0],
                      M[:, 1, 0] - M[:, 0, 1], 1.0 + tr], dim=1)
    cand.append(q3)
    stacked = torch.stack(cand, dim=1)  # (B,4,4) [choice, comp]
    choice = dec.argmax(dim=1)          # (B,)
    # one-hot select (ONNX-friendly; avoids advanced-index gather)
    onehot = F.one_hot(choice, num_classes=4).to(M.dtype)  # (B,4)
    quat = (onehot.unsqueeze(-1) * stacked).sum(dim=1)      # (B,4)
    return F.normalize(quat, dim=-1)


def quat_to_euler_zyx(quat, eps=1e-7):
    # roma unitquat_to_euler, intrinsic "ZYX" specialized (sqrt, not hypot).
    qx, qy, qz, qw = quat.unbind(-1)
    a = qw - qy
    b = qx + qz
    c = qy + qw
    d = qz - qx
    hcd = torch.sqrt(c * c + d * d)
    hab = torch.sqrt(a * a + b * b)
    a1 = 2.0 * torch.atan2(hcd, hab)
    half_sum = torch.atan2(b, a)
    half_diff = torch.atan2(d, c)
    a2 = half_sum - half_diff          # angle_first (intrinsic)
    a0 = half_sum + half_diff          # angle_third
    case1 = torch.abs(a1) <= eps
    case2 = torch.abs(a1 - torch.pi) <= eps
    deg = case1 | case2
    a2 = torch.where(deg, torch.zeros_like(a2), a2)
    a0 = torch.where(case1, 2.0 * half_sum, a0)
    a0 = torch.where(case2, 2.0 * half_diff, a0)
    a1 = a1 - torch.pi / 2.0
    out = torch.stack([a0, a1, a2], dim=-1)
    out = torch.where(out < -torch.pi, out + 2 * torch.pi, out)
    out = torch.where(out > torch.pi, out - 2 * torch.pi, out)
    return out


def batch6DFromXYZ(r):
    # r: (...,3) -> (...,6) = [col0(3), col1(3)] of the XYZ rotation matrix.
    rc = torch.cos(r); rs = torch.sin(r)
    cx, cy, cz = rc.unbind(-1)
    sx, sy, sz = rs.unbind(-1)
    m00 = cy * cz
    m10 = cy * sz
    m20 = -sy
    m01 = -cx * sz + sx * sy * cz
    m11 = cx * cz + sx * sy * sz
    m21 = sx * cy
    col0 = torch.stack([m00, m10, m20], -1)
    col1 = torch.stack([m01, m11, m21], -1)
    return torch.cat([col0, col1], -1)


def xyz_from_6d(poses):
    # inverse of batch6DFromXYZ: (...,6) -> XYZ euler (...,3). sqrt, not hypot.
    x_raw = poses[..., :3]
    y_raw = poses[..., 3:]
    x = F.normalize(x_raw, dim=-1)
    z = F.normalize(torch.cross(x, y_raw, dim=-1), dim=-1)
    y = torch.cross(z, x, dim=-1)
    m = torch.stack([x, y, z], dim=-1)  # columns x,y,z
    sy = torch.sqrt(m[..., 0, 0] ** 2 + m[..., 1, 0] ** 2)
    singular = sy < 1e-6
    ex = torch.where(singular, torch.atan2(-m[..., 1, 2], m[..., 1, 1]),
                     torch.atan2(m[..., 2, 1], m[..., 2, 2]))
    ey = torch.atan2(-m[..., 2, 0], sy)
    ez = torch.where(singular, torch.zeros_like(sy), torch.atan2(m[..., 1, 0], m[..., 0, 0]))
    return torch.stack([ex, ey, ez], dim=-1)


# Fixed index tables (mhr_utils.py) as flat long tensors.
_B3 = torch.tensor([0, 2, 4, 6, 8, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                    24, 25, 26, 27, 28, 29, 34, 35, 36, 37, 38, 39, 44, 45, 46, 53, 54,
                    55, 64, 65, 66, 85, 69, 73, 86, 70, 79, 87, 71, 82, 88, 72, 76, 91,
                    92, 93, 112, 96, 100, 113, 97, 106, 114, 98, 109, 115, 99, 103, 130,
                    131, 132], dtype=torch.long)
_B1 = torch.tensor([1, 3, 5, 7, 9, 11, 30, 31, 32, 33, 40, 41, 42, 43, 47, 48, 49, 50, 51,
                    52, 56, 57, 58, 59, 60, 61, 62, 63, 67, 68, 74, 75, 77, 78, 80, 81, 83,
                    84, 89, 90, 94, 95, 101, 102, 104, 105, 107, 108, 110, 111, 116, 117,
                    118, 119, 120, 121, 122, 123], dtype=torch.long)
_BT = torch.tensor([124, 125, 126, 127, 128, 129], dtype=torch.long)
_HAND_DOFS = [3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 2, 3, 1, 1]


def compact_cont_body_exp(cont):
    # (B,260) -> body model params (B,133), ONNX-friendly (index_copy scatters).
    B = cont.shape[0]
    e3 = xyz_from_6d(cont[:, :138].reshape(B, 23, 6)).reshape(B, 69)
    o1 = cont[:, 138:254].reshape(B, 58, 2)
    e1 = torch.atan2(o1[..., 0], o1[..., 1])            # (B,58)
    et = cont[:, 254:260]                                # (B,6)
    body = torch.zeros(B, 133, dtype=cont.dtype, device=cont.device)
    body = body.index_copy(1, _B3.to(cont.device), e3)
    body = body.index_copy(1, _B1.to(cont.device), e1)
    body = body.index_copy(1, _BT.to(cont.device), et)
    return body


def compact_cont_hand_exp(cont54):
    # (B,54) -> hand model params (B,27), ONNX-friendly.
    B = cont54.shape[0]
    cpos = 0
    three_cont, three_mp = [], []
    one_cont, one_mp = [], []
    mp = 0
    for k in _HAND_DOFS:
        if k == 3:
            three_cont += list(range(cpos, cpos + 6))
            three_mp += list(range(mp, mp + 3))
            cpos += 6; mp += 3
        else:
            for a in range(k):
                one_cont += [cpos + a * 2, cpos + a * 2 + 1]
                one_mp += [mp + a]
            cpos += 2 * k; mp += k
    dev = cont54.device
    tc = cont54[:, torch.tensor(three_cont, device=dev)].reshape(B, -1, 6)
    te = xyz_from_6d(tc).reshape(B, -1)                       # (B,15)
    oc = cont54[:, torch.tensor(one_cont, device=dev)].reshape(B, -1, 2)
    oe = torch.atan2(oc[..., 0], oc[..., 1])                  # (B,12)
    out = torch.zeros(B, 27, dtype=cont54.dtype, device=dev)
    out = out.index_copy(1, torch.tensor(three_mp, device=dev), te)
    out = out.index_copy(1, torch.tensor(one_mp, device=dev), oe)
    return out


class MhrMeshModule(nn.Module):
    def __init__(self, ref: MhrReference):
        super().__init__()
        m = ref.mhr
        B = dict(m.named_buffers())

        def reg(name, t):
            self.register_buffer(name, t.detach().clone(), persistent=False)

        reg("base_shape", B["character_torch.blend_shape.base_shape"])
        reg("shape_vectors", B["character_torch.blend_shape.shape_vectors"])
        reg("parameter_transform", B["character_torch.parameter_transform.parameter_transform"])
        reg("joint_translation_offsets", B["character_torch.skeleton.joint_translation_offsets"])
        reg("joint_prerotations", B["character_torch.skeleton.joint_prerotations"])
        reg("inverse_bind_pose", B["character_torch.linear_blend_skinning.inverse_bind_pose"])
        reg("skin_vert_idx", B["character_torch.linear_blend_skinning.vert_indices_flattened"].long())
        reg("skin_joint_idx", B["character_torch.linear_blend_skinning.skin_indices_flattened"].long())
        reg("skin_weights", B["character_torch.linear_blend_skinning.skin_weights_flattened"])
        pmi = B["character_torch.skeleton.pmi"]
        sizes = list(m.character_torch.skeleton._pmi_buffer_sizes)
        self.pmi_levels = [t.long() for t in torch.split(pmi, sizes, dim=1)]

        # pose corrective dense weights (densify the sparse layer 0)
        p0 = m.pose_correctives_model.pose_dirs_predictor.__getattr__("0")
        si = p0.sparse_indices; sw = p0.sparse_weight; ss = p0.sparse_shape
        W0 = torch.sparse_coo_tensor(si, sw, list(ss)).to_dense()   # (3000,750)
        W2 = m.pose_correctives_model.pose_dirs_predictor.__getattr__("2").weight  # (55317,3000)
        reg("pc_W0", W0)
        reg("pc_W2", W2.detach().clone())

        # head decode buffers
        reg("scale_mean", ref.scale_mean)
        reg("scale_comps", ref.scale_comps)
        reg("hand_pose_mean", ref.hand_pose_mean)
        reg("hand_pose_comps", ref.hand_pose_comps)
        reg("keypoint_mapping", ref.keypoint_mapping)
        self.register_buffer("hand_idx_left", ref.hand_joint_idxs_left.long(), persistent=False)
        self.register_buffer("hand_idx_right", ref.hand_joint_idxs_right.long(), persistent=False)
        # float mask zeroing hand euler params + jaw (last 3) in the body-133 vector
        bmask = torch.ones(133)
        bmask[mhr_param_hand_mask] = 0.0
        bmask[-3:] = 0.0
        self.register_buffer("body_zero_mask", bmask, persistent=False)
        self.V = self.base_shape.shape[0]
        self.J = self.joint_translation_offsets.shape[0]

    # ---- param decode (pred[B,519] -> model_params[B,204], shape[B,45]) ----
    # Fully ONNX-exportable: no boolean index_put or advanced-index assignment.
    def decode(self, pred):
        c = 0
        g6 = pred[:, c:c + G6D]; c += G6D
        grm = rot6d_to_rotmat(g6)
        # ONNX-exportable equivalent of roma.rotmat_to_euler("ZYX", grm)
        grot = quat_to_euler_zyx(rotmat_to_unitquat(grm))
        gtrans = torch.zeros_like(grot)
        cont = pred[:, c:c + BODY_CONT]; c += BODY_CONT
        pe = compact_cont_body_exp(cont) * self.body_zero_mask[None]
        shape = pred[:, c:c + SHAPE]; c += SHAPE
        scale = pred[:, c:c + SCALE]; c += SCALE
        hand = pred[:, c:c + 2 * HANDPCA]; c += 2 * HANDPCA
        body = pe[:, :130]
        scales = self.scale_mean[None] + scale @ self.scale_comps
        full = torch.cat([gtrans * 10, grot, body], dim=1)  # (B,136)
        lh, rh = torch.split(hand, [HANDPCA, HANDPCA], dim=1)
        lh_mp = compact_cont_hand_exp(self.hand_pose_mean + torch.einsum("da,ab->db", lh, self.hand_pose_comps))
        rh_mp = compact_cont_hand_exp(self.hand_pose_mean + torch.einsum("da,ab->db", rh, self.hand_pose_comps))
        full = full.index_copy(1, self.hand_idx_left, lh_mp)
        full = full.index_copy(1, self.hand_idx_right, rh_mp)
        return torch.cat([full, scales], dim=1), shape

    def local_skel(self, jp):
        B = jp.shape[0]
        j = jp.reshape(B, self.J, 7)
        t = j[..., :3] + self.joint_translation_offsets[None]
        q = euler_xyz_to_quat(j[..., 3:6])
        q = quat_mul(self.joint_prerotations[None].expand(B, -1, -1), q)
        s = torch.exp(j[..., 6:7] * LN2)
        return torch.cat([t, q, s], -1)   # (B,J,8)

    def fk(self, local, use_double=True):
        g = local.double() if use_double else local.clone()
        for lvl in self.pmi_levels:
            source = lvl[0]; target = lvl[1]
            s2 = g.index_select(-2, source)   # child current
            s1 = g.index_select(-2, target)   # ancestor current
            g = g.index_copy(-2, source, skel_multiply(s1, s2))
        return g.to(local.dtype)

    def pose_corrective(self, jp):
        B = jp.shape[0]
        j = jp.reshape(B, self.J, 7)
        eul = j[:, 2:, 3:6]                      # (B,125,3)
        feat = batch6DFromXYZ(eul).clone()       # (B,125,6)
        feat[:, :, 0] = feat[:, :, 0] - 1.0
        feat[:, :, 4] = feat[:, :, 4] - 1.0
        x = feat.flatten(1, 2)                   # (B,750)
        h = F.relu(x @ self.pc_W0.t())           # (B,3000)
        o = h @ self.pc_W2.t()                   # (B,55317)
        return o.reshape(B, self.V, 3)

    def lbs(self, global_skel, unposed):
        # joint_state = multiply(global_skel, inverse_bind_pose)
        jstate = skel_multiply(global_skel, self.inverse_bind_pose[None])  # (B,J,8)
        B = unposed.shape[0]
        js = jstate.index_select(-2, self.skin_joint_idx)      # (B,nnz,8)
        vp = unposed.index_select(-2, self.skin_vert_idx)      # (B,nnz,3)
        t, q, s = skel_normalize_q(js)
        transformed = t + quat_rotate(q, s * vp)               # (B,nnz,3)
        w = self.skin_weights[None, :, None]
        skinned = torch.zeros(B, self.V, 3, dtype=unposed.dtype, device=unposed.device)
        skinned = skinned.index_add(-2, self.skin_vert_idx, transformed * w)
        return skinned

    def forward(self, pred):
        model_params, shape = self.decode(pred)
        B = pred.shape[0]
        identity_rest = self.base_shape[None] + torch.einsum("nvd,bn->bvd", self.shape_vectors, shape)
        full249 = torch.cat([model_params, torch.zeros(B, SHAPE, dtype=pred.dtype, device=pred.device)], dim=1)
        jp = torch.einsum("dn,bn->bd", self.parameter_transform, full249)
        local = self.local_skel(jp)
        glob = self.fk(local)
        corr = self.pose_corrective(jp)
        unposed = identity_rest + corr
        verts_cm = self.lbs(glob, unposed)
        jcoords_cm = glob[..., :3]
        verts = verts_cm / 100.0
        jcoords = jcoords_cm / 100.0
        mvj = torch.cat([verts, jcoords], dim=1)                       # (B,18566,3)
        kpts = torch.einsum("kn,bnd->bkd", self.keypoint_mapping, mvj)  # (B,308,3)
        j3d = kpts[:, :70].clone()
        flip = torch.tensor([1.0, -1.0, -1.0], dtype=verts.dtype, device=verts.device)
        verts = verts * flip
        j3d = j3d * flip
        jcoords = jcoords * flip
        return verts, jcoords, j3d
