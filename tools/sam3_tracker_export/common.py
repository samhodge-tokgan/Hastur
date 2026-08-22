"""Shared helpers: build model (fp32), load frames, run G1 backbone, detect person seed."""
import os, glob, time, torch
import numpy as np
from PIL import Image
from torchvision.transforms import v2

DEV = "cuda"
FRAMES = "/root/hastur_matany/clip6077718/frames/"
RES = 1008
EXPORT = "/root/sam3_export"

def log(*a): print(*a, flush=True)

_TF = v2.Compose([
    v2.ToDtype(torch.uint8, scale=True),
    v2.Resize(size=(RES, RES)),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
])

def build(fp32=True):
    from sam3.model_builder import build_sam3_video_model
    model = build_sam3_video_model(device=DEV)
    model.eval()
    if fp32:
        model.float()
    return model

def load_frames(n, start=0):
    allp = sorted(glob.glob(os.path.join(FRAMES, "*.png")))
    sel = allp[start:start+n]
    pil = [Image.open(p).convert("RGB") for p in sel]
    W,H = pil[0].size
    imgs = [_TF(v2.functional.to_image(p).to(DEV)).unsqueeze(0).float() for p in pil]
    return imgs, (H, W), sel

@torch.inference_mode()
def g1_backbone(model, img):
    """Run shared ViT-L backbone; return (raw_backbone_out_for_detector, tracker_backbone_out)."""
    det = model.detector; trk = model.tracker
    bb = det.backbone.forward_image(img)          # dict with sam2_backbone_out
    sam2 = bb["sam2_backbone_out"]
    fpn = [f.float() for f in sam2["backbone_fpn"]]
    pos = [p.float() for p in sam2["vision_pos_enc"]]
    # tracker applies conv_s0/conv_s1 to fpn[0],[1]
    tfpn0 = trk.sam_mask_decoder.conv_s0(fpn[0])
    tfpn1 = trk.sam_mask_decoder.conv_s1(fpn[1])
    tracker_bb = {"backbone_fpn": [tfpn0, tfpn1, fpn[2]],
                  "vision_pos_enc": pos}
    return bb, tracker_bb

@torch.inference_mode()
def detect_person(model, img, video_hw, conf=0.5):
    """Return best person binary mask at video resolution (H,W) float tensor, plus stats."""
    from sam3.model.data_misc import FindStage
    from sam3.model import box_ops
    det = model.detector
    bb = det.backbone.forward_image(img)
    txt = det.backbone.forward_text(["person"], device=DEV)
    bb.update(txt)
    find_stage = FindStage(
        img_ids=torch.tensor([0], device=DEV, dtype=torch.long),
        text_ids=torch.tensor([0], device=DEV, dtype=torch.long),
        input_boxes=None, input_boxes_mask=None, input_boxes_label=None,
        input_points=None, input_points_mask=None,
    )
    dummy = det._get_dummy_prompt()
    out = det.forward_grounding(backbone_out=bb, find_input=find_stage,
                               geometric_prompt=dummy, find_target=None)
    logits = out["pred_logits"].float()      # [B,Q,1]
    boxes = out["pred_boxes"].float()        # [B,Q,4] cxcywh norm
    masks = out["pred_masks"].float()        # [B,Q,h,w] logits
    presence = out["presence_logit_dec"].sigmoid().float().unsqueeze(1)
    probs = (logits.sigmoid() * presence).squeeze(-1)  # [B,Q]
    probs = probs.squeeze(0)                  # [Q]
    masks = masks.squeeze(0)                  # [Q,h,w]
    keep = probs > conf
    idxs = torch.nonzero(keep).squeeze(-1)
    log(f"[detect] {len(idxs)} person dets > {conf}; masks {tuple(masks.shape)}")
    H, W = video_hw
    best = None; best_area = -1; best_i = -1
    for i in idxs.tolist():
        m = masks[i][None,None]
        mm = torch.nn.functional.interpolate(m, size=(H,W), mode="bilinear", align_corners=False)
        binm = (mm.squeeze() > 0).float()
        area = binm.sum().item()
        if area > best_area:
            best_area = area; best = binm; best_i = i
    log(f"[detect] chose det #{best_i} prob={probs[best_i].item():.3f} area={best_area:.0f}")
    return best, {"prob": probs[best_i].item(), "n_dets": len(idxs)}
