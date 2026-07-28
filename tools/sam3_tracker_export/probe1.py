import os, glob, sys, time
import torch, numpy as np
from PIL import Image
from torchvision.transforms import v2
torch.manual_seed(0)
dev = "cuda"
FRAMES = "/root/hastur_matany/clip6077718/frames/"
RES = 1008
def log(*a): print(*a, flush=True)

t0=time.time()
from sam3.model_builder import build_sam3_video_model
model = build_sam3_video_model(device=dev)
model.eval()
log("built model in %.1fs" % (time.time()-t0))
det = model.detector; trk = model.tracker
log("detector", type(det).__name__, "tracker", type(trk).__name__,
    "trk.backbone", type(trk.backbone).__name__ if trk.backbone is not None else None)

transform = v2.Compose([
    v2.ToDtype(torch.uint8, scale=True),
    v2.Resize(size=(RES, RES)),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
])
allp = sorted(glob.glob(os.path.join(FRAMES, "*.png")))
img0 = Image.open(allp[0]).convert("RGB")
W0,H0 = img0.size
log("orig size", W0, H0, "nframes", len(allp))
imt = transform(v2.functional.to_image(img0).to(dev)).unsqueeze(0)
log("input image tensor", imt.shape, imt.dtype)

def dump(name, v, ind="  "):
    if torch.is_tensor(v): log(ind, name, tuple(v.shape), v.dtype)
    elif isinstance(v, dict):
        log(ind, name, "(dict)", list(v.keys()))
        for kk,vv in v.items(): dump(kk, vv, ind+"  ")
    elif isinstance(v,(list,tuple)):
        log(ind, name, "(list len %d)"%len(v))
        for i,e in enumerate(v): dump("[%d]"%i, e, ind+"  ")
    else:
        log(ind, name, type(v).__name__)

with torch.inference_mode():
    bb = det.backbone.forward_image(imt)
    dump("backbone_out", bb)
    txt = det.backbone.forward_text(["person"], device=dev)
    dump("text_out", txt)
