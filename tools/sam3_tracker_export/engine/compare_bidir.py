"""Compare a C++ tracker run against the Python BIDIRECTIONAL ground truth (ref_bidir.npz).
Reports: seed-frame id mapping, per-frame per-object IoU, overall mean/min IoU, track
lifetimes, and an ID-STABILITY / swap count (# consecutive-frame identity switches vs the
Python reference). Run it on the bidirectional bin and on the forward-only bin to contrast.
  python compare_bidir.py <cpp_bin> [label]
"""
import os, sys, numpy as np, torch
D = "/root/sam3_export/engine/data"
LR = 288
CPP = sys.argv[1] if len(sys.argv) > 1 else os.path.join(D, "cpp_bidir.bin")
LABEL = sys.argv[2] if len(sys.argv) > 2 else os.path.basename(CPP)
IOU_MATCH = 0.3   # min IoU for a cpp track to "be" a ref person on a frame

def upscale(low, H, W):
    t = torch.from_numpy(low.astype(np.float32))[None, None]
    r = torch.nn.functional.interpolate(t, size=(H, W), mode="bilinear", align_corners=False)
    return (r.squeeze().numpy() > 0)
def iou(a, b):
    i = np.logical_and(a, b).sum(); u = np.logical_or(a, b).sum(); return float(i)/u if u > 0 else 1.0

ref = np.load(os.path.join(D, "ref_bidir.npz"))
WIN = int(ref["window"][0]); H = int(ref["H"][0]); W = int(ref["W"][0]); SEED = int(ref["seed"][0])
ref_frames = {}
for f in range(WIN):
    if f"ids_{f}" in ref.files:
        ref_frames[f] = (ref[f"ids_{f}"], ref[f"masks_{f}"])

cpp_frames = {}
with open(CPP, "rb") as fh:
    nf = int(np.frombuffer(fh.read(4), np.int32)[0])
    cpp_seed = int(np.frombuffer(fh.read(4), np.int32)[0])
    for f in range(nf):
        nt = int(np.frombuffer(fh.read(4), np.int32)[0])
        items = []
        for _ in range(nt):
            tid = int(np.frombuffer(fh.read(4), np.int32)[0])
            low = np.frombuffer(fh.read(LR*LR*4), np.float32).reshape(LR, LR).copy()
            items.append((tid, low))
        cpp_frames[f] = items

print(f"=== {LABEL}  (WIN={WIN} seed={cpp_seed}, ref bidir seed={SEED}) ===")

# map cpp track id -> ref obj id via IoU at the cpp seed frame
r_ids, r_masks = ref_frames[cpp_seed]
c0 = cpp_frames[cpp_seed]
c_up = {tid: upscale(low, H, W) for tid, low in c0}
mapping = {}; used = set()
for tid in sorted(c_up):
    best = -1; bj = 0
    for k, rid in enumerate(r_ids.tolist()):
        if rid in used: continue
        j = iou(c_up[tid], r_masks[k] > 0)
        if j > bj: bj = j; best = rid
    if bj >= IOU_MATCH: mapping[tid] = best; used.add(best)
print(f"seed-frame {cpp_seed}: mapped {len(mapping)}/{len(c0)} cpp tracks to ref ids")

# per-frame per-object IoU under fixed mapping
allj = []
for f in range(WIN):
    if f not in ref_frames: continue
    r_ids, r_masks = ref_frames[f]; r_map = {rid: i for i, rid in enumerate(r_ids.tolist())}
    for tid, low in cpp_frames[f]:
        rid = mapping.get(tid, -1)
        if rid not in r_map: continue
        j = iou(upscale(low, H, W), r_masks[r_map[rid]] > 0)
        allj.append(j)
print(f"per-object IoU vs Python-bidir: mean={np.mean(allj):.4f} min={np.min(allj):.4f} "
      f"median={np.median(allj):.4f} (n={len(allj)})")

# ID stability / swap count: for each cpp track, per present frame assign the ref id of
# max IoU; a swap = a consecutive-frame change of that assigned ref id.
swaps = 0; drifted_tracks = 0; total_present = 0
purity_sum = 0.0; ntracks = 0
cpp_life = {}
for f in cpp_frames:
    for tid, _ in cpp_frames[f]: cpp_life.setdefault(tid, []).append(f)
for tid in sorted(cpp_life):
    seq = []
    for f in sorted(cpp_life[tid]):
        low = dict(cpp_frames[f])[tid]
        up = upscale(low, H, W)
        if not up.any(): continue
        r_ids, r_masks = ref_frames[f]
        best = -1; bj = IOU_MATCH
        for k, rid in enumerate(r_ids.tolist()):
            j = iou(up, r_masks[k] > 0)
            if j > bj: bj = j; best = rid
        if best >= 0: seq.append(best)
    if not seq: continue
    ntracks += 1; total_present += len(seq)
    tsw = sum(1 for i in range(1, len(seq)) if seq[i] != seq[i-1])
    swaps += tsw
    if tsw > 0: drifted_tracks += 1
    vals, cnts = np.unique(seq, return_counts=True)
    purity_sum += cnts.max() / len(seq)
print(f"ID stability: total swaps={swaps}  drifted tracks={drifted_tracks}/{ntracks}  "
      f"mean track purity={purity_sum/max(ntracks,1):.3f}  (present frames={total_present})")

# lifetimes
cl = {t: (cpp_life[t][0], cpp_life[t][-1], len(cpp_life[t])) for t in sorted(cpp_life)}
rl = {}
for f in ref_frames:
    for k, i in enumerate(ref_frames[f][0].tolist()):
        if ref_frames[f][1][k].sum() > 0: rl.setdefault(i, []).append(f)
rl = {t: (rl[t][0], rl[t][-1], len(rl[t])) for t in sorted(rl)}
print(f"cpp tracks={len(cl)}  ref objs={len(rl)}")
print("cpp lifetimes:", cl)
print("ref lifetimes:", rl)
