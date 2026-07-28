import numpy as np, torch, os
D="/root/sam3_export/engine/data"; LR=288
def up(low,H,W):
    t=torch.from_numpy(low.astype(np.float32))[None,None]
    return (torch.nn.functional.interpolate(t,size=(H,W),mode="bilinear",align_corners=False).squeeze().numpy()>0)
def iou(a,b):
    i=np.logical_and(a,b).sum();u=np.logical_or(a,b).sum();return float(i)/u if u>0 else 1.0
ref=np.load(D+"/ref_bidir.npz");WIN=int(ref["window"][0]);H=int(ref["H"][0]);W=int(ref["W"][0]);SEED=int(ref["seed"][0])
rf={f:(ref[f"ids_{f}"],ref[f"masks_{f}"]) for f in range(WIN) if f"ids_{f}" in ref.files}
cf={}
with open(D+"/cpp_bidir_cuda.bin","rb") as fh:
    nf=int(np.frombuffer(fh.read(4),np.int32)[0]);sd=int(np.frombuffer(fh.read(4),np.int32)[0])
    for f in range(nf):
        nt=int(np.frombuffer(fh.read(4),np.int32)[0]);it=[]
        for _ in range(nt):
            tid=int(np.frombuffer(fh.read(4),np.int32)[0]);low=np.frombuffer(fh.read(LR*LR*4),np.float32).reshape(LR,LR).copy();it.append((tid,low))
        cf[f]=it
rid,rm=rf[SEED];c0={t:up(l,H,W) for t,l in cf[SEED]};mp={};us=set()
for t in sorted(c0):
    b=-1;bj=0.3
    for k,r in enumerate(rid.tolist()):
        if r in us:continue
        j=iou(c0[t],rm[k]>0)
        if j>bj:bj=j;b=r
    if b>=0:mp[t]=b;us.add(b)
def band(lo,hi):
    js=[]
    for f in range(lo,hi+1):
        if f not in rf:continue
        rid,rm=rf[f];rmap={r:i for i,r in enumerate(rid.tolist())}
        for t,l in cf[f]:
            r=mp.get(t,-1)
            if r not in rmap:continue
            js.append(iou(up(l,H,W),rm[rmap[r]]>0))
    return js
bwd=band(0,SEED-1);sd_=band(SEED,SEED);fwd=band(SEED+1,WIN-1)
print(f"seed frame {SEED}: coverage frames 0..{WIN-1}")
print(f"BACKWARD half (frames 0..{SEED-1}, {SEED} frames): mean IoU={np.mean(bwd):.4f} min={np.min(bwd):.4f} n={len(bwd)}")
print(f"FORWARD  half (frames {SEED+1}..{WIN-1}, {WIN-1-SEED} frames): mean IoU={np.mean(fwd):.4f} min={np.min(fwd):.4f} n={len(fwd)}")
print(f"forward-only-from-seed would cover ONLY {WIN-SEED}/{WIN} frames; bidirectional RECOVERS the {SEED} earlier frames at mean IoU {np.mean(bwd):.4f}")
