"""Python BIDIRECTIONAL reference (5d-iii ground truth) on the soccer window.
Seeds text='person' at the mid-window max-visibility frame, then runs the model's
propagate_in_video generator TWICE from that seed: reverse=False (seed->end) then
reverse=True (seed->start) -- the run_soccer propagation_direction='both' recipe.
Dumps per-frame obj_ids + video-res binary masks (ground truth) over the whole window,
plus the per-frame detector captures (for apples-to-apples C++ association checks)."""
import os, glob, sys, numpy as np, torch
sys.path.insert(0, "/root/sam3")
sys.path.insert(0, "/root/sam3_export")
from common import FRAMES
from sam3.model_builder import build_sam3_video_model

START = int(sys.argv[1]) if len(sys.argv) > 1 else 180
WIN   = int(sys.argv[2]) if len(sys.argv) > 2 else 61
SEED_GLOBAL = int(sys.argv[3]) if len(sys.argv) > 3 else 209
SEED = SEED_GLOBAL - START
OUT = "/root/sam3_export/engine/data"

def log(*a): print(*a, flush=True)

def main():
    allp = sorted(glob.glob(os.path.join(FRAMES, "*.png")))[START:START + WIN]
    assert len(allp) == WIN
    tmp = os.path.join(OUT, "refframes_bidir")
    os.makedirs(tmp, exist_ok=True)
    for i, p in enumerate(allp):
        dst = os.path.join(tmp, f"{i:05d}.png")
        if os.path.islink(dst) or os.path.exists(dst): os.remove(dst)
        os.symlink(p, dst)

    model = build_sam3_video_model(device="cuda")
    model.eval(); model.rank = 0; model.world_size = 1
    if hasattr(model, "detector"): model.detector.rank = 0; model.detector.world_size = 1

    det_cap = {}
    orig_rbd = model.run_backbone_and_detection
    def rbd(*a, **k):
        det_out = orig_rbd(*a, **k)
        fi = k.get("frame_idx")
        det_cap[fi] = {
            "mask": det_out["mask"].detach().float().cpu().numpy(),
            "scores": det_out["scores"].detach().float().cpu().numpy(),
        }
        return det_out
    model.run_backbone_and_detection = rbd

    st = model.init_state(resource_path=tmp)
    H, W = st["orig_height"], st["orig_width"]
    log(f"video {H}x{W}, {st['num_frames']} frames, seed local {SEED} (global {SEED_GLOBAL})")
    per_frame = {}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        r = model.add_prompt(st, frame_idx=SEED, text_str="person")
        # forward: seed -> end
        for fidx, out in model.propagate_in_video(st, start_frame_idx=SEED,
                max_frame_num_to_track=WIN - 1 - SEED, reverse=False):
            ids = np.asarray(out["out_obj_ids"]).astype(np.int64)
            masks = np.asarray(out["out_binary_masks"]).astype(np.uint8)
            per_frame[fidx] = (ids, masks)
        # backward: seed -> start (re-run from the same seed state)
        for fidx, out in model.propagate_in_video(st, start_frame_idx=SEED,
                max_frame_num_to_track=SEED, reverse=True):
            if fidx == SEED and fidx in per_frame:
                continue  # keep the forward pass's seed-frame output
            ids = np.asarray(out["out_obj_ids"]).astype(np.int64)
            masks = np.asarray(out["out_binary_masks"]).astype(np.uint8)
            per_frame[fidx] = (ids, masks)

    for f in sorted(per_frame):
        ids, masks = per_frame[f]
        vis = [int(i) for i, m in zip(ids.tolist(), masks) if m.sum() > 0]
        log(f"  frame {f:3d}: {len(vis)} vis objs ids={vis}")

    np.savez_compressed(os.path.join(OUT, "ref_bidir.npz"),
        **{f"ids_{f}": per_frame[f][0] for f in per_frame},
        **{f"masks_{f}": per_frame[f][1] for f in per_frame},
        window=np.array([WIN]), H=np.array([H]), W=np.array([W]), seed=np.array([SEED]))
    np.savez_compressed(os.path.join(OUT, "det_cap_bidir.npz"),
        **{f"mask_{f}": det_cap[f]["mask"] for f in det_cap},
        **{f"scores_{f}": det_cap[f]["scores"] for f in det_cap})
    log("saved ref_bidir.npz + det_cap_bidir.npz")

    life = {}
    for f in sorted(per_frame):
        for k, oid in enumerate(per_frame[f][0].tolist()):
            if per_frame[f][1][k].sum() > 0: life.setdefault(oid, []).append(f)
    log("track lifetimes (obj_id: first..last nframes):")
    for oid in sorted(life):
        log(f"  id {oid}: {life[oid][0]}..{life[oid][-1]} ({len(life[oid])} frames)")

if __name__ == "__main__":
    main()
