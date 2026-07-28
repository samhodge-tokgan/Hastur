"""FULL 5-graph end-to-end single-object validation.
A (reference): PyTorch detector person seed + PyTorch tracker (Sam3 single-object recurrence), GPU.
B (all ONNX): G1-ORT backbone -> G2-ORT detect person seed -> G1/G3/G4/G5-ORT tracker recurrence.
Compare per-frame masks A-vs-B (IoU/MSE)."""
import os, time, numpy as np, torch
import onnxruntime as ort
from common import build, load_frames, detect_person, log, EXPORT, DEV
from validate import sess, npf, build_cached_pytorch, build_cached_onnx, patch_onnx, restore, run_prop, iou, WINDOW

def ort_detect_person(g1, g2, model, img, hw, conf=0.5):
    """Seed mask from the REAL exported G1->G2 ONNX pipeline (text='person')."""
    o = g1.run(None, {"image": npf(img)})
    det_fpn0, det_fpn1, det_fpn2, det_pos72 = o[6], o[7], o[8], o[9]
    txt = model.detector.backbone.forward_text(["person"], device=DEV)  # text encoder is host-side
    lf = npf(txt["language_features"]); lm = txt["language_mask"].cpu().numpy()
    p = g2.run(None, {"fpn0": det_fpn0, "fpn1": det_fpn1, "fpn2": det_fpn2,
                      "pos72": det_pos72, "lang_feats": lf, "lang_mask": lm})
    logits, boxes, masks, presence = p
    probs = (1/(1+np.exp(-logits)) * (1/(1+np.exp(-presence)))[:,None]).squeeze()  # [200]
    masks = masks[0]  # [200,72,72]
    H, W = hw
    keep = np.nonzero(probs > conf)[0]
    # select the highest-confidence detection (robust to CPU/GPU mask-boundary drift)
    bi = int(keep[np.argmax(probs[keep])])
    mt = torch.from_numpy(masks[bi])[None,None]
    mm = torch.nn.functional.interpolate(mt, size=(H,W), mode="bilinear", align_corners=False)
    best = (mm.squeeze() > 0).float()
    log(f"[ORT-G2 detect] {len(keep)} dets>{conf}; chose #{bi} prob={probs[bi]:.3f} area={best.sum().item():.0f}")
    return best.to(DEV)

def run_prop_full(model, imgs, hw, sessions):
    """B: seed from ORT G1->G2, propagate with ORT G1/G3/G4/G5."""
    trk = model.tracker; H,W = hw
    seed = ort_detect_person(sessions["G1"], sessions["G2"], model, imgs[0], hw)
    cached = build_cached_onnx(model, imgs, sessions["G1"])
    saved = patch_onnx(trk, sessions["G3"], sessions["G4"], sessions["G5"])
    state = trk.init_state(video_height=H, video_width=W, num_frames=WINDOW)
    state["cached_features"] = dict(cached)
    trk.add_new_mask(state, frame_idx=0, obj_id=1, mask=seed)
    masks={}
    for fidx,_,_,vrm,_ in trk.propagate_in_video(state, start_frame_idx=0,
            max_frame_num_to_track=WINDOW-1, reverse=False, propagate_preflight=True, tqdm_disable=True):
        masks[fidx] = (vrm.squeeze()>0).detach().cpu().numpy().astype(np.uint8)
    restore(saved)
    return masks

def main():
    model = build(fp32=True)
    imgs, hw, _ = load_frames(WINDOW, start=0)
    seed_pt, st = detect_person(model, imgs[0], hw)
    log("PyTorch seed prob", round(st["prob"],3))
    sessions = {n: sess(f"{n}.onnx") for n in ["G1","G2","G3","G4","G5"]}
    log("loaded 5 ORT sessions")

    t0=time.time(); A = run_prop(model, imgs, hw, seed_pt, use_onnx=False); log("A(pytorch, all-torch) %.1fs"%(time.time()-t0))
    t0=time.time(); B = run_prop_full(model, imgs, hw, sessions); log("B(full 5-graph ONNX) %.1fs"%(time.time()-t0))

    log("\nframe | IoU(A,B) | MSE | areaA | areaB")
    ious=[]
    for f in sorted(A):
        a=A[f]; b=B[f]; j=iou(a,b); mse=float(((a.astype(np.float32)-b.astype(np.float32))**2).mean())
        ious.append(j); log(f"  {f:3d} |  {j:.4f} | {mse:.3e} | {a.sum():7d} | {b.sum():7d}")
    log(f"\nFULL 5-graph: mean IoU (frames 1+) = {np.mean(ious[1:]):.4f}  min = {np.min(ious[1:]):.4f}")

if __name__ == "__main__":
    main()
