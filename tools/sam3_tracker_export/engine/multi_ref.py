"""Multi-object Python reference: full SAM3 video model (text='person', forward-only).
Dumps per-frame obj_ids + video-res binary masks (ground truth) AND per-frame detector
outputs (masks/scores/boxes) so the C++ association can be validated apples-to-apples.
Runs in-process (rank 0, world 1)."""
import os, glob, sys, numpy as np, torch
sys.path.insert(0, "/root/sam3")
from common import FRAMES  # /root/hastur_matany/clip6077718/frames/
from sam3.model_builder import build_sam3_video_model

WINDOW = int(sys.argv[1]) if len(sys.argv) > 1 else 20
OUT = "/root/sam3_export/engine/data"
os.makedirs(OUT, exist_ok=True)

def log(*a): print(*a, flush=True)

def main():
    # temp dir of the first WINDOW frames named 00000.png.. (integer sort -> same order as load_frames)
    allp = sorted(glob.glob(os.path.join(FRAMES, "*.png")))[:WINDOW]
    tmp = "/root/sam3_export/engine/data/refframes"
    os.makedirs(tmp, exist_ok=True)
    for i, p in enumerate(allp):
        dst = os.path.join(tmp, f"{i:05d}.png")
        if os.path.islink(dst) or os.path.exists(dst): os.remove(dst)
        os.symlink(p, dst)

    model = build_sam3_video_model(device="cuda")
    model.eval(); model.rank = 0; model.world_size = 1
    if hasattr(model, "detector"): model.detector.rank = 0; model.detector.world_size = 1

    # capture per-frame detector outputs
    det_cap = {}
    orig_rbd = model.run_backbone_and_detection
    def rbd(*a, **k):
        det_out = orig_rbd(*a, **k)
        fi = k.get("frame_idx")
        det_cap[fi] = {
            "mask": det_out["mask"].detach().float().cpu().numpy(),   # [N,h,w] logits
            "scores": det_out["scores"].detach().float().cpu().numpy(),
            "bbox": det_out["bbox"].detach().float().cpu().numpy(),   # xyxy norm
        }
        return det_out
    model.run_backbone_and_detection = rbd

    st = model.init_state(resource_path=tmp)
    H, W = st["orig_height"], st["orig_width"]
    log(f"video {H}x{W}, {st['num_frames']} frames")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        model.add_prompt(st, frame_idx=0, text_str="person")
        per_frame = {}
        for fidx, out in model.propagate_in_video(st, start_frame_idx=0,
                                                  max_frame_num_to_track=WINDOW, reverse=False):
            ids = np.asarray(out["out_obj_ids"]).astype(np.int64)
            masks = np.asarray(out["out_binary_masks"]).astype(np.uint8)  # [N,H,W]
            per_frame[fidx] = (ids, masks)
            log(f"  frame {fidx}: {len(ids)} objs ids={ids.tolist()} areas={[int(m.sum()) for m in masks]}")

    # save ground truth: pack as object dict per frame
    np.savez_compressed(os.path.join(OUT, "ref_multi.npz"),
        **{f"ids_{f}": per_frame[f][0] for f in per_frame},
        **{f"masks_{f}": per_frame[f][1] for f in per_frame},
        window=np.array([WINDOW]), H=np.array([H]), W=np.array([W]))
    # save detector captures
    np.savez_compressed(os.path.join(OUT, "det_cap.npz"),
        **{f"mask_{f}": det_cap[f]["mask"] for f in det_cap},
        **{f"scores_{f}": det_cap[f]["scores"] for f in det_cap},
        **{f"bbox_{f}": det_cap[f]["bbox"] for f in det_cap})
    log("saved ref_multi.npz + det_cap.npz")
    # lifetimes
    life = {}
    for f in sorted(per_frame):
        for oid in per_frame[f][0].tolist():
            life.setdefault(oid, []).append(f)
    log("track lifetimes (obj_id: [first..last] nframes):")
    for oid in sorted(life): log(f"  id {oid}: {life[oid][0]}..{life[oid][-1]} ({len(life[oid])} frames)")

if __name__ == "__main__":
    main()
