"""Bidirectional feasibility: seed mid-clip, propagate BACKWARD (reverse=True -> negated
temporal-pos sign in _prepare_memory_conditioned_features). Pure PyTorch, quick sanity."""
import numpy as np, torch, time
from common import build, load_frames, g1_backbone, detect_person, log

WINDOW = 9
SEED = 8  # seed at last frame, track backward to 0

def main():
    model = build(fp32=True); trk = model.tracker
    imgs, (H,W), _ = load_frames(WINDOW, start=0)
    cached = {i: (img, g1_backbone(model, img)[1]) for i, img in enumerate(imgs)}
    seed_mask, st = detect_person(model, imgs[SEED], (H,W))
    log(f"seed at frame {SEED}, prob {st['prob']:.3f}, area {int(seed_mask.sum())}")
    state = trk.init_state(video_height=H, video_width=W, num_frames=WINDOW)
    state["cached_features"] = dict(cached)
    trk.add_new_mask(state, frame_idx=SEED, obj_id=1, mask=seed_mask)
    masks = {}
    t0 = time.time()
    for fidx,_,_,vrm,osc in trk.propagate_in_video(state, start_frame_idx=SEED,
            max_frame_num_to_track=WINDOW-1, reverse=True, propagate_preflight=True, tqdm_disable=True):
        masks[fidx] = (vrm.squeeze() > 0).detach().cpu().numpy().astype(np.uint8)
    log("reverse propagate %.1fs; frames tracked: %s" % (time.time()-t0, sorted(masks)))
    def iou(a,b):
        i=np.logical_and(a,b).sum(); u=np.logical_or(a,b).sum(); return i/u if u else 1.0
    log("frame | area | IoU vs next-inner")
    order = sorted(masks, reverse=True)
    prev = None
    for f in order:
        j = iou(masks[f], prev) if prev is not None else 1.0
        log(f"  {f:3d} | {masks[f].sum():7d} | {j:.4f}")
        prev = masks[f]

if __name__ == "__main__":
    main()
