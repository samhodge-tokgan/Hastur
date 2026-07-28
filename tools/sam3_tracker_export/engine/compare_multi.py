import os, sys, numpy as np, torch
D="/root/sam3_export/engine/data"
LR=288
def upscale(low, H, W):  # [288,288] logits -> [H,W] bool, bilinear align_corners=False
    t=torch.from_numpy(low.astype(np.float32))[None,None]
    r=torch.nn.functional.interpolate(t,size=(H,W),mode="bilinear",align_corners=False)
    return (r.squeeze().numpy()>0)
def iou(a,b):
    i=np.logical_and(a,b).sum(); u=np.logical_or(a,b).sum(); return float(i)/u if u>0 else 1.0
# load ref
ref=np.load(os.path.join(D,"ref_multi.npz"))
WIN=int(ref["window"][0]); H=int(ref["H"][0]); W=int(ref["W"][0])
ref_frames={}
for f in range(WIN):
    ref_frames[f]=(ref[f"ids_{f}"], ref[f"masks_{f}"])  # ids, masks[N,H,W] uint8
# load cpp
cpp_frames={}
with open(os.path.join(D,"cpp_multi.bin"),"rb") as fh:
    nf=int(np.frombuffer(fh.read(4),np.int32)[0])
    for f in range(nf):
        nt=int(np.frombuffer(fh.read(4),np.int32)[0])
        items=[]
        for _ in range(nt):
            tid=int(np.frombuffer(fh.read(4),np.int32)[0])
            low=np.frombuffer(fh.read(LR*LR*4),np.float32).reshape(LR,LR).copy()
            items.append((tid,low))
        cpp_frames[f]=items
# fix mapping cpp track id -> ref obj id via IoU on frame 0
r_ids,r_masks=ref_frames[0]
c0=cpp_frames[0]
c_up={tid:upscale(low,H,W) for tid,low in c0}
mapping={}
used=set()
for tid in sorted(c_up):
    best=-1; bj=0
    for k,rid in enumerate(r_ids.tolist()):
        if rid in used: continue
        j=iou(c_up[tid], r_masks[k]>0)
        if j>bj: bj=j; best=rid
    mapping[tid]=best; used.add(best)
print("frame0 cpp_track_id -> ref_obj_id (IoU):")
for tid in sorted(mapping):
    k=r_ids.tolist().index(mapping[tid]); print(f"   cpp {tid} -> ref {mapping[tid]}  IoU={iou(c_up[tid], r_masks[k]>0):.4f}")

# per-frame per-object IoU under fixed mapping
print("\nframe | matched | mean IoU | min IoU | per-obj IoU (cpp_id:iou)")
allj=[]
for f in range(WIN):
    r_ids,r_masks=ref_frames[f]; r_map={rid:i for i,rid in enumerate(r_ids.tolist())}
    row=[]; js=[]
    for tid,low in cpp_frames[f]:
        rid=mapping.get(tid,-1)
        if rid not in r_map: continue
        up=upscale(low,H,W)
        j=iou(up, r_masks[r_map[rid]]>0); js.append(j); row.append(f"{tid}:{j:.3f}")
    if js:
        allj+=js
        print(f"  {f:3d} | {len(js):2d}/{len(r_ids)} | {np.mean(js):.4f} | {np.min(js):.4f} | "+" ".join(row))
print(f"\nOVERALL mean per-object IoU = {np.mean(allj):.4f}  min = {np.min(allj):.4f}  (n={len(allj)})")
# lifetimes
def life(frames):
    L={}
    for f in frames:
        ids = frames[f][0].tolist() if isinstance(frames[f],tuple) else [t for t,_ in frames[f]]
        for i in ids: L.setdefault(i,[]).append(f)
    return L
cl={}
for f in cpp_frames:
    for t,_ in cpp_frames[f]: cl.setdefault(t,[]).append(f)
rl={}
for f in ref_frames:
    for i in ref_frames[f][0].tolist(): rl.setdefault(i,[]).append(f)
print(f"\ncpp tracks={len(cl)} ids={sorted(cl)}  | ref objs={len(rl)} ids={sorted(rl)}")
print("cpp lifetimes:", {t:(cl[t][0],cl[t][-1],len(cl[t])) for t in sorted(cl)})
print("ref lifetimes:", {t:(rl[t][0],rl[t][-1],len(rl[t])) for t in sorted(rl)})
