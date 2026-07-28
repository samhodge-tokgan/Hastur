"""Prep a mid-clip soccer window for the bidirectional (5d-iii) validation.
Preprocesses WIN frames starting at global START (same TF as the single-obj golden),
saves frames_bidir.npy [WIN,3,1008,1008] + meta_bidir.txt (H W WIN SEED_LOCAL).
The window spans the max-visibility seed (global 209 == run_soccer SEED_IDX) so the
forward pass covers SEED..end and the backward pass covers SEED..0."""
import os, sys, numpy as np
sys.path.insert(0, "/root/sam3_export")
from common import load_frames

OUT = "/root/sam3_export/engine/data"
START = int(sys.argv[1]) if len(sys.argv) > 1 else 180   # global idx of first window frame
WIN   = int(sys.argv[2]) if len(sys.argv) > 2 else 61    # window length
SEED_GLOBAL = int(sys.argv[3]) if len(sys.argv) > 3 else 209  # run_soccer max-visibility seed
SEED_LOCAL = SEED_GLOBAL - START
assert 0 <= SEED_LOCAL < WIN, f"seed {SEED_GLOBAL} not in window [{START},{START+WIN})"

imgs, (H, W), sel = load_frames(WIN, start=START)
frames = np.concatenate([im.detach().float().cpu().numpy() for im in imgs], 0).astype(np.float32)
np.save(os.path.join(OUT, "frames_bidir.npy"), frames)
with open(os.path.join(OUT, "meta_bidir.txt"), "w") as f:
    f.write(f"{H} {W} {WIN} {SEED_LOCAL}\n")
print("frames_bidir", frames.shape, "H,W", H, W, "WIN", WIN,
      "seed_local", SEED_LOCAL, "global", SEED_GLOBAL, "->", os.path.basename(sel[SEED_LOCAL]))
