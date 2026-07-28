"""Sam3TrackerEngine — Python reference (numpy + onnxruntime).

Single-object, forward-only recurrent tracker driving the 5 exported SAM 3 graphs
with host-owned recurrent memory. This mirrors, 1:1, the C++ Sam3TrackerEngine so the
C++ port can be validated bit-close against it. All host-side logic (memory-frame
selection, temporal-pos assembly, obj-ptr token split + sinusoidal temporal pos,
no_mem_embed seed path, det seed) is reproduced here from sam3_tracker_base.py.

Config (probed from the built model):
  num_maskmem=7, mem_dim=64, hidden_dim=256, image_size=1008, backbone_stride=14,
  max_cond_frames_in_attn=4, keep_first_cond_frame=False,
  use_memory_selection=True, memory_temporal_stride_for_eval r=1, mf_threshold=0.01,
  max_obj_ptrs_in_encoder=16, sigmoid scale=20 bias=-10 (baked into G5),
  non_overlap=False, multimask=True, cond_frame_spatial/obj_ptr_embedding=None.
"""
import os, numpy as np, onnxruntime as ort

EXPORT = "/root/sam3_export"
DATA = os.path.join(EXPORT, "engine", "data")

NUM_MASKMEM = 7
MEM_DIM = 64
HID = 256
MAX_OBJ_PTRS = 16
MF_THRESH = 0.01
HW = 72 * 72       # 5184 top-level tokens
GRID = 72

def sess(name, ep="cpu"):
    so = ort.SessionOptions(); so.log_severity_level = 3
    providers = ["CPUExecutionProvider"] if ep == "cpu" else \
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ort.InferenceSession(os.path.join(EXPORT, name), sess_options=so, providers=providers)

def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))

def get_1d_sine_pe(pos_inds, dim=HID, temperature=10000.0):
    """pos_inds: [N] float -> [N,dim].  Matches sam3_tracker_utils.get_1d_sine_pe."""
    pe_dim = dim // 2
    dim_t = np.arange(pe_dim, dtype=np.float32)
    dim_t = temperature ** (2 * (dim_t // 2) / pe_dim)
    pe = pos_inds[:, None].astype(np.float32) / dim_t[None, :]
    return np.concatenate([np.sin(pe), np.cos(pe)], axis=-1).astype(np.float32)  # [N,dim]


class Sam3TrackerEngineRef:
    def __init__(self, ep="cpu"):
        self.g1 = sess("G1.onnx", ep)
        self.g3 = sess("G3.onnx", ep)
        self.g4 = sess("G4.onnx", ep)
        self.g5 = sess("G5.onnx", ep)
        c = lambda n: np.load(os.path.join(EXPORT, n)).astype(np.float32)
        self.tpos = c("const_maskmem_tpos_enc.npy")       # [7,1,1,64]
        self.no_mem = c("const_no_mem_embed.npy")          # [1,1,256]
        self.proj_w = c("const_objptr_tpos_proj_w.npy")    # [64,256]
        self.proj_b = c("const_objptr_tpos_proj_b.npy")    # [64]
        self.reset()

    def reset(self, total_frames=0):
        self.frames = {}       # idx -> record dict
        self.cond_idx = []     # conditioning frame indices (=[0])
        self.noncond_idx = []  # non-cond processed frame indices, ascending
        self.num_frames = 0    # frames processed so far
        self.total_frames = total_frames   # full sequence length (fixed up front)

    # ---- graph wrappers ----
    def _G1(self, image):
        o = self.g1.run(None, {"image": image.astype(np.float32)})
        return {"f0": o[0], "f1": o[1], "f2": o[2], "pos2": o[5]}   # tracker branch

    def _G3(self, src, src_pos, prompt, prompt_pos, n_optr):
        return self.g3.run(None, {"src": src, "src_pos": src_pos, "prompt": prompt,
                                  "prompt_pos": prompt_pos,
                                  "num_obj_ptr_tokens": np.array(n_optr, dtype=np.int64)})[0]

    def _G4(self, bf, hr0, hr1):
        o = self.g4.run(None, {"backbone_features": bf, "high_res_0": hr0, "high_res_1": hr1})
        return {"low": o[0], "high": o[1], "ious": o[2], "obj_ptr": o[3], "osl": o[4]}

    def _G5(self, pix_feat, high_res, osl):
        o = self.g5.run(None, {"pix_feat": pix_feat, "pred_masks_high_res": high_res,
                               "object_score_logits": osl})
        return o[0], o[1]     # maskmem_features [1,64,72,72], maskmem_pos_enc [1,64,72,72]

    # ---- helpers ----
    @staticmethod
    def _flatten_hwbc(x):        # [1,C,72,72] -> [5184,1,C]
        C = x.shape[1]
        return np.transpose(x.reshape(1, C, HW), (2, 0, 1)).copy()

    @staticmethod
    def _seq_to_bchw(seq):       # [5184,1,C] -> [1,C,72,72]
        C = seq.shape[2]
        return np.transpose(seq, (1, 2, 0)).reshape(1, C, GRID, GRID).copy()

    def _eff_iou(self, osl, ious):
        obj_norm = (sigmoid(osl) * 2 - 1) if osl.item() > 0 else 0.0
        iou_score = float(np.max(ious))
        return float(np.mean(obj_norm * iou_score)), iou_score

    # ---- seed / propagate ----
    def seed(self, frame0, seed_mask_1008, total_frames):
        """frame0 [1,3,1008,1008], seed_mask_1008 [1008,1008] soft. Returns high_res [1,1,1008,1008].
        total_frames = full sequence length (drives temporal-pos normalization, fixed up front)."""
        self.reset(total_frames)
        g = self._G1(frame0)
        pix = g["f2"]                                   # [1,256,72,72] raw tracker top feat
        src = self._flatten_hwbc(pix)                   # [5184,1,256]
        # cond frame: memory-less features = pix_feat + no_mem_embed
        bf_seq = src + self.no_mem                       # broadcast [1,1,256]
        bf = self._seq_to_bchw(bf_seq)                   # [1,256,72,72]
        g4 = self._G4(bf, g["f0"], g["f1"])              # obj_ptr approximates mask-prompt decoder
        obj_ptr = g4["obj_ptr"]                          # [1,256]
        # seed output + memory come from the DETECTION mask (not G4's mask):
        high = (seed_mask_1008[None, None] * 20.0 - 10.0).astype(np.float32)   # [1,1,1008,1008]
        osl = np.array([[10.0]], dtype=np.float32)       # obj present (20*1-10)
        mem, mempos = self._G5(pix, high, osl)
        eff, iou = self._eff_iou(osl, np.ones((1, 3), np.float32))
        self.frames[0] = dict(cond=True, mem=mem, mempos=mempos, obj_ptr=obj_ptr,
                              osl=osl, low=None, high=high, eff=eff, iou=iou)
        self.cond_idx = [0]; self.num_frames = 1
        return high

    def _frame_filter(self, frame_idx):
        """use_memory_selection valid_indices: ascending non-cond frames i<frame_idx with eff>thr."""
        max_num = min(self.total_frames, MAX_OBJ_PTRS)
        valid = []
        for i in range(frame_idx - 1, 0, -1):            # step -r, r=1; end exclusive 0
            rec = self.frames.get(i)
            if rec is None or rec["cond"]:
                continue
            if rec["eff"] > MF_THRESH:
                valid.insert(0, i)
            if len(valid) >= max_num - 1:
                break
        must = frame_idx - 1
        if must not in valid:
            valid.append(must)
        return valid

    def propagate(self, frame):
        """frame [1,3,1008,1008]. Returns (low_res [1,1,288,288], high_res [1,1,1008,1008])."""
        frame_idx = self.num_frames
        g = self._G1(frame)
        pix = g["f2"]
        src = self._flatten_hwbc(pix)                    # [5184,1,256]
        src_pos = self._flatten_hwbc(g["pos2"])          # [5184,1,256]

        valid = self._frame_filter(frame_idx)

        # ---- assemble spatial maskmem prompt tokens ----
        prompt_tok = []      # each [tokens,1,64]
        prompt_pos = []
        # cond frames first (t_pos=0). single cond frame 0.
        for t, rec in [(ci, self.frames[ci]) for ci in self.cond_idx]:
            feats = rec["mem"]                            # [1,64,72,72]
            prompt_tok.append(self._flatten_hwbc(feats))  # [5184,1,64]
            pos = self._flatten_hwbc(rec["mempos"]) + self.tpos[NUM_MASKMEM - 0 - 1]  # t=0
            prompt_pos.append(pos)
        # non-cond recent frames, t_pos=1..num_maskmem-1
        for t_pos in range(1, NUM_MASKMEM):
            t_rel = NUM_MASKMEM - t_pos
            if t_rel > len(valid):
                continue
            prev = valid[-t_rel]
            rec = self.frames.get(prev)
            if rec is None:
                continue
            feats = rec["mem"]
            prompt_tok.append(self._flatten_hwbc(feats))
            pos = self._flatten_hwbc(rec["mempos"]) + self.tpos[NUM_MASKMEM - t_pos - 1]
            prompt_pos.append(pos)

        # ---- object-pointer tokens ----
        pos_list, ptrs = [], []
        for ci in self.cond_idx:                          # cond obj_ptr, t<=frame_idx
            pos_list.append(float(frame_idx - ci))
            ptrs.append(self.frames[ci]["obj_ptr"])
        for t_diff in range(1, MAX_OBJ_PTRS):
            if t_diff >= len(valid):
                break
            t = valid[-t_diff]
            rec = self.frames.get(t)
            if rec is not None:
                pos_list.append(float(t_diff))
                ptrs.append(rec["obj_ptr"])
        n_optr = 0
        if len(ptrs) > 0:
            obj_ptrs = np.stack(ptrs, axis=0)             # [P,1,256]
            obj_pos = get_1d_sine_pe(np.array(pos_list, np.float32) / (min(self.total_frames, MAX_OBJ_PTRS) - 1))
            obj_pos = obj_pos @ self.proj_w.T + self.proj_b   # [P,64]
            obj_pos = obj_pos[:, None, :]                 # [P,1,64]
            # split each 256-ptr into 4 tokens of 64
            P = obj_ptrs.shape[0]
            obj_ptrs = obj_ptrs.reshape(P, 1, HID // MEM_DIM, MEM_DIM)
            obj_ptrs = np.transpose(obj_ptrs, (0, 2, 1, 3)).reshape(P * (HID // MEM_DIM), 1, MEM_DIM)
            obj_pos = np.repeat(obj_pos, HID // MEM_DIM, axis=0)   # [P*4,1,64]
            prompt_tok.append(obj_ptrs.astype(np.float32))
            prompt_pos.append(obj_pos.astype(np.float32))
            n_optr = obj_ptrs.shape[0]

        prompt = np.concatenate(prompt_tok, axis=0).astype(np.float32)
        prompt_pos = np.concatenate(prompt_pos, axis=0).astype(np.float32)

        memory = self._G3(src, src_pos, prompt, prompt_pos, n_optr)   # [5184,1,256]
        bf = self._seq_to_bchw(memory)                     # [1,256,72,72]
        g4 = self._G4(bf, g["f0"], g["f1"])
        low, high, ious, obj_ptr, osl = g4["low"], g4["high"], g4["ious"], g4["obj_ptr"], g4["osl"]
        mem, mempos = self._G5(pix, high, osl)             # memory from raw pix + this frame's mask
        eff, iou = self._eff_iou(osl, ious)
        self.frames[frame_idx] = dict(cond=False, mem=mem, mempos=mempos, obj_ptr=obj_ptr,
                                      osl=osl, low=low, high=high, eff=eff, iou=iou)
        self.noncond_idx.append(frame_idx); self.num_frames += 1
        return low, high


def iou(a, b):
    i = np.logical_and(a, b).sum(); u = np.logical_or(a, b).sum()
    return float(i) / float(u) if u > 0 else 1.0

def resize_to(mask_logits, H, W):
    """bilinear resize [1,1,h,w] logits -> [H,W] bool, matches _get_orig_video_res_output."""
    import torch
    t = torch.from_numpy(mask_logits)
    r = torch.nn.functional.interpolate(t, size=(H, W), mode="bilinear", align_corners=False)
    return (r.squeeze().numpy() > 0)

def main():
    import sys
    ep = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    H, W, WINDOW = open(os.path.join(DATA, "meta.txt")).read().split()
    H, W, WINDOW = int(H), int(W), int(WINDOW)
    frames = np.load(os.path.join(DATA, "frames.npy"))
    seed_mask = np.load(os.path.join(DATA, "seed_mask.npy"))
    golden = np.load(os.path.join(DATA, "golden_masks.npy"))

    eng = Sam3TrackerEngineRef(ep=ep)
    print(f"[engine_ref ep={ep}] providers G1:", eng.g1.get_providers())
    out_masks = {}
    ref_low = np.zeros((WINDOW, 288, 288), np.float32)     # low-res logits for bit-close check
    high0 = eng.seed(frames[0:1], seed_mask, total_frames=WINDOW)
    out_masks[0] = resize_to(high0, H, W)
    for f in range(1, WINDOW):
        low, high = eng.propagate(frames[f:f + 1])
        ref_low[f] = low[0, 0]
        out_masks[f] = resize_to(low, H, W)   # golden uses low_res(288)->video res
    np.save(os.path.join(DATA, f"ref_low_{ep}.npy"), ref_low)

    print("\nframe | IoU(ref,golden) | MSE | area_ref | area_gold")
    ious = []
    dump = np.zeros((WINDOW, H, W), np.uint8)
    for f in range(WINDOW):
        a = out_masks[f].astype(np.uint8); b = golden[f].astype(np.uint8)
        dump[f] = a
        j = iou(a, b); mse = float(((a.astype(np.float32) - b) ** 2).mean())
        ious.append(j)
        print(f"  {f:3d} |  {j:.4f} | {mse:.3e} | {int(a.sum()):7d} | {int(b.sum()):7d}")
    print(f"\nref-vs-golden mean IoU (tracked 1+) = {np.mean(ious[1:]):.4f}  min = {np.min(ious[1:]):.4f}")
    np.save(os.path.join(DATA, f"masks_ref_{ep}.npy"), dump)

if __name__ == "__main__":
    main()
