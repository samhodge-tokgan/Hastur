import numpy as np, torch
from common import build, load_frames, log, DEV
from validate import sess, npf

model = build(fp32=True)
imgs, hw, _ = load_frames(1, start=0); H,W = hw
g1=sess("G1.onnx"); g2=sess("G2.onnx")
o=g1.run(None,{"image":npf(imgs[0])})
det_vis72, det_pos72 = o[6], o[7]
txt=model.detector.backbone.forward_text(["person"],device=DEV)
lf=npf(txt["language_features"]); lm=txt["language_mask"].cpu().numpy()
logits,boxes,masks,presence=g2.run(None,{"vis72":det_vis72,"pos72":det_pos72,"lang_feats":lf,"lang_mask":lm})
probs=(1/(1+np.exp(-logits))*(1/(1+np.exp(-presence)))[:,None]).squeeze()
masks=masks[0]
def area72(i): return int((masks[i]>0).sum())  # at 72x72
order=np.argsort(-probs)
log("ORT-G2 top-8 dets (idx, prob, area@72x72, full-frame-frac):")
for i in order[:8]:
    a=area72(i); log(f"  #{i} prob={probs[i]:.3f} area72={a} frac={a/(72*72):.2f}")
log(f"n dets>0.5 = {(probs>0.5).sum()}")
# selection by max prob:
bi=int(order[0]); log(f"max-prob pick #{bi} prob={probs[bi]:.3f} area72={area72(bi)} frac={area72(bi)/5184:.2f}")
