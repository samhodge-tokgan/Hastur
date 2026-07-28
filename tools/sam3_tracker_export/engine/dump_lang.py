import os, numpy as np, torch, sys
sys.path.insert(0,"/root/sam3_export")
from common import build, DEV
m=build(fp32=True)
with torch.inference_mode():
    txt=m.detector.backbone.forward_text(["person"],device=DEV)
lf=txt["language_features"].float().cpu().numpy()
lm=txt["language_mask"].cpu().numpy()
print("lang_feats",lf.shape,lf.dtype,"lang_mask",lm.shape,lm.dtype, lm.sum())
np.save("/root/sam3_export/const_lang_feats_person.npy", lf.astype(np.float32))
np.save("/root/sam3_export/const_lang_mask_person.npy", lm.astype(np.float32))
print("saved")
