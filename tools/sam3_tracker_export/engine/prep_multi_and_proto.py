import os, sys, numpy as np, torch
sys.path.insert(0,"/root/sam3_export")
import onnxruntime as ort
from common import build, load_frames, EXPORT

OUT="/root/sam3_export/engine/data"
WIN=20
# preprocess 20 frames (same TF as single-obj golden)
model=None
imgs,(H,W),_=load_frames(WIN,start=0)
frames=np.concatenate([im.detach().float().cpu().numpy() for im in imgs],axis=0).astype(np.float32)
np.save(os.path.join(OUT,"frames_multi.npy"),frames)
print("frames_multi",frames.shape,H,W)

def sess(n):
    so=ort.SessionOptions(); so.log_severity_level=3
    return ort.InferenceSession(os.path.join(EXPORT,n),sess_options=so,providers=["CPUExecutionProvider"])
g1=sess("G1.onnx"); g2=sess("G2.onnx")
lf=np.load(os.path.join(EXPORT,"const_lang_feats_person.npy"))
lm=np.load(os.path.join(EXPORT,"const_lang_mask_person.npy")).astype(bool)

def sig(x): return 1/(1+np.exp(-x))
def mask_iou(a,b):
    # a,b bool [h,w]
    i=np.logical_and(a,b).sum(); u=np.logical_or(a,b).sum(); return i/u if u>0 else 0.0

def detect(frame):
    o=g1.run(None,{"image":frame[None]})
    det_fpn0,det_fpn1,det_fpn2,det_pos72=o[6],o[7],o[8],o[9]
    pl,pb,pm,pres=g2.run(None,{"fpn0":det_fpn0,"fpn1":det_fpn1,"fpn2":det_fpn2,
        "pos72":det_pos72,"lang_feats":lf,"lang_mask":lm})
    probs=(sig(pl[...,0])*sig(pres)).squeeze(0)  # [200]
    keep=np.where(probs>0.5)[0]
    order=keep[np.argsort(-probs[keep])]
    binm={i:(pm[0,i]>0) for i in order}
    # greedy NMS iou>0.1
    sel=[]
    for i in order:
        ok=True
        for j in sel:
            if mask_iou(binm[i],binm[j])>0.1: ok=False; break
        if ok: sel.append(i)
    return sel, probs, pm

dc=np.load(os.path.join(OUT,"det_cap.npz"))
for f in [0,1,10,19]:
    sel,probs,pm=detect(frames[f])
    capmask=dc[f"mask_{f}"]; capsc=dc[f"scores_{f}"]
    # match each cap det to my sel by mask iou
    mysel_masks=[(pm[0,i]>0) for i in sel]
    ious=[]
    for k in range(capmask.shape[0]):
        cm=capmask[k]>0
        best=max((mask_iou(cm,mm) for mm in mysel_masks), default=0)
        ious.append(best)
    print(f"frame {f}: my_ndet={len(sel)} cap_ndet={capmask.shape[0]} myscores={np.round(np.sort(probs[sel])[::-1],3).tolist()} match_iou_to_cap={np.round(ious,3).tolist()}")
