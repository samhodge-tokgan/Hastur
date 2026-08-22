"""Prep fixtures for the Sam3TrackerEngine validation (single object, soccer window).

Produces, under /root/sam3_export/engine/data/ :
  frames.npy        [WINDOW,3,1008,1008] float32  (preprocessed G1 inputs)
  seed_mask.npy     [1008,1008]          float32  (soft mask_inputs the seed frame feeds
                                                    _use_mask_as_output; = add_new_mask resize)
  golden_masks.npy  [WINDOW,H,W]         uint8    (PyTorch reference per-frame masks)
  meta.txt          "H W WINDOW"

The model is only used here to (a) reproduce the exact detector person seed and its
add_new_mask 1008-resize, and (b) pull golden_masks that capture.py already saved.
"""
import os, numpy as np, torch
from common import build, load_frames, detect_person, log, EXPORT

WINDOW = 9
OUT = os.path.join(EXPORT, "engine", "data")
os.makedirs(OUT, exist_ok=True)

def main():
    model = build(fp32=True)
    imgs, (H, W), _ = load_frames(WINDOW, start=0)
    log(f"video {H}x{W}, {len(imgs)} frames")

    frames = np.concatenate([im.detach().float().cpu().numpy() for im in imgs], axis=0)
    np.save(os.path.join(OUT, "frames.npy"), frames.astype(np.float32))
    log("frames", frames.shape)

    # detector person seed (video-res binary), then replicate add_new_mask's 1008 resize
    seed_video, st = detect_person(model, imgs[0], (H, W))   # [H,W] float {0,1}
    log("seed prob", round(st["prob"], 3), "area", int(seed_video.sum().item()))
    mi = seed_video[None, None].float()
    mask_1008 = torch.nn.functional.interpolate(
        mi, size=(1008, 1008), mode="bilinear", align_corners=False, antialias=True
    ).squeeze().cpu().numpy().astype(np.float32)      # soft [0,1]
    np.save(os.path.join(OUT, "seed_mask.npy"), mask_1008)
    log("seed_mask_1008", mask_1008.shape, "sum>0.5", int((mask_1008 > 0.5).sum()))

    g = torch.load(os.path.join(EXPORT, "golden.pt"), map_location="cpu", weights_only=False)
    gm = g["golden_masks"]
    golden = np.stack([gm[f] for f in range(WINDOW)], axis=0).astype(np.uint8)
    np.save(os.path.join(OUT, "golden_masks.npy"), golden)
    log("golden_masks", golden.shape, "areas", [int(golden[f].sum()) for f in range(WINDOW)])

    with open(os.path.join(OUT, "meta.txt"), "w") as f:
        f.write(f"{H} {W} {WINDOW}\n")
    log("wrote", OUT)

if __name__ == "__main__":
    main()
