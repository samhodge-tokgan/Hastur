import sys, numpy as np
LR=288
def load(path):
    fr={}
    with open(path,"rb") as fh:
        nf=int(np.frombuffer(fh.read(4),np.int32)[0])
        for f in range(nf):
            nt=int(np.frombuffer(fh.read(4),np.int32)[0])
            d={}
            for _ in range(nt):
                tid=int(np.frombuffer(fh.read(4),np.int32)[0])
                low=np.frombuffer(fh.read(LR*LR*4),np.float32).reshape(LR,LR).copy()
                d[tid]=low
            fr[f]=d
    return fr
a=load(sys.argv[1]); b=load(sys.argv[2])
def iou(x,y):
    x=x>0;y=y>0;i=np.logical_and(x,y).sum();u=np.logical_or(x,y).sum();return float(i)/u if u>0 else 1.0
maxdiff=0; ious=[]
nf=min(len(a),len(b))
for f in range(nf):
    for tid in a[f]:
        if tid in b[f]:
            md=np.max(np.abs(a[f][tid]-b[f][tid])); maxdiff=max(maxdiff,md)
            ious.append(iou(a[f][tid],b[f][tid]))
print(f"frames compared={nf}  tracks/frame matched by id")
print(f"CPU-vs-CUDA  max|logit diff|={maxdiff:.4e}  mean mask IoU={np.mean(ious):.6f}  min IoU={np.min(ious):.6f}  n={len(ious)}")
