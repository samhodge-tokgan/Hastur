"""CUDA vs CPU parity for the bidirectional C++ run: compares two cpp_bidir bins
per-(frame,track) at the 288 low-res logit-binary level (the association-relevant field)."""
import sys, numpy as np
LR = 288
def load(path):
    fr = {}
    with open(path, "rb") as fh:
        nf = int(np.frombuffer(fh.read(4), np.int32)[0]); _seed = int(np.frombuffer(fh.read(4), np.int32)[0])
        for f in range(nf):
            nt = int(np.frombuffer(fh.read(4), np.int32)[0]); d = {}
            for _ in range(nt):
                tid = int(np.frombuffer(fh.read(4), np.int32)[0])
                d[tid] = np.frombuffer(fh.read(LR*LR*4), np.float32).reshape(LR, LR).copy()
            fr[f] = d
    return fr
a = load(sys.argv[1]); b = load(sys.argv[2])
ious = []; maxad = 0.0; npair = 0; missing = 0
for f in a:
    for tid, la in a[f].items():
        lb = b.get(f, {}).get(tid)
        if lb is None: missing += 1; continue
        ba, bb = la > 0, lb > 0
        i = np.logical_and(ba, bb).sum(); u = np.logical_or(ba, bb).sum()
        ious.append(float(i)/u if u > 0 else 1.0)
        maxad = max(maxad, float(np.abs(la - lb).max())); npair += 1
print(f"CPU/CUDA parity: matched (frame,track) pairs={npair} missing={missing}")
print(f"  mask IoU: mean={np.mean(ious):.5f} min={np.min(ious):.5f}   max|logit diff|={maxad:.4f}")
