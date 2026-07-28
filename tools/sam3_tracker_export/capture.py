"""Run A: fp32 golden tracker propagation on the soccer clip, seeded by detector person mask.
Captures per-graph example IO (G1/G3/G4/G5) and golden per-frame masks."""
import os, time, pickle, torch
import numpy as np
from common import build, load_frames, g1_backbone, detect_person, log, EXPORT, DEV

WINDOW = 9   # seed frame 0 + track 1..8
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

def tocpu(x):
    if torch.is_tensor(x): return x.detach().float().cpu()
    if isinstance(x,(list,tuple)): return [tocpu(e) for e in x]
    if isinstance(x,dict): return {k:tocpu(v) for k,v in x.items()}
    return x

def main():
    model = build(fp32=True)
    trk = model.tracker
    imgs, (H,W), paths = load_frames(WINDOW, start=0)
    log(f"loaded {len(imgs)} frames, video {H}x{W}")

    # ---- G1 backbone per frame + cache tracker features ----
    cached = {}
    g1_examples = {}
    for i, img in enumerate(imgs):
        raw, tbb = g1_backbone(model, img)
        cached[i] = (img, tbb)
        if i in (0,1):
            g1_examples[i] = {"image": tocpu(img),
                              "fpn": tocpu(tbb["backbone_fpn"]),
                              "pos": tocpu(tbb["vision_pos_enc"])}
    log("G1 done; fpn shapes", [tuple(t.shape) for t in cached[0][1]["backbone_fpn"]])

    # ---- detector G2 person seed on frame 0 ----
    seed_mask, stats = detect_person(model, imgs[0], (H,W))
    assert seed_mask is not None, "no person detected"

    # ---- hooks to capture golden IO of G3/G4/G5 ----
    cap = {"G3": [], "G4": [], "G5": []}

    enc = trk.transformer.encoder
    orig_enc = enc.forward
    def enc_fwd(src, prompt, **kw):
        out = orig_enc(src, prompt, **kw)
        s = src[0] if isinstance(src, list) else src
        sp = kw.get("src_pos"); sp = sp[0] if isinstance(sp, list) else sp
        cap["G3"].append({
            "src": tocpu(s), "prompt": tocpu(prompt),
            "src_pos": tocpu(sp), "prompt_pos": tocpu(kw.get("prompt_pos")),
            "num_obj_ptr_tokens": kw.get("num_obj_ptr_tokens", 0),
            "memory": tocpu(out["memory"]),
        })
        return out
    enc.forward = enc_fwd

    orig_heads = trk._forward_sam_heads
    def heads_fwd(backbone_features, point_inputs=None, mask_inputs=None,
                  high_res_features=None, multimask_output=False, gt_masks=None):
        out = orig_heads(backbone_features=backbone_features, point_inputs=point_inputs,
                         mask_inputs=mask_inputs, high_res_features=high_res_features,
                         multimask_output=multimask_output, gt_masks=gt_masks)
        (lrm_multi, hrm_multi, ious, low_res_masks, high_res_masks, obj_ptr, osl) = out
        cap["G4"].append({
            "backbone_features": tocpu(backbone_features),
            "high_res_0": tocpu(high_res_features[0]) if high_res_features else None,
            "high_res_1": tocpu(high_res_features[1]) if high_res_features else None,
            "point_coords": tocpu(point_inputs["point_coords"]) if point_inputs else None,
            "point_labels": tocpu(point_inputs["point_labels"]) if point_inputs else None,
            "has_mask_input": mask_inputs is not None,
            "multimask_output": bool(multimask_output),
            "low_res_masks": tocpu(low_res_masks), "high_res_masks": tocpu(high_res_masks),
            "ious": tocpu(ious), "obj_ptr": tocpu(obj_ptr),
            "object_score_logits": tocpu(osl),
            "low_res_multimasks": tocpu(lrm_multi),
        })
        return out
    trk._forward_sam_heads = heads_fwd

    orig_mem = trk._encode_new_memory
    def mem_fwd(image, current_vision_feats, feat_sizes, pred_masks_high_res,
                object_score_logits, is_mask_from_pts, output_dict=None, is_init_cond_frame=False):
        mf, mpe = orig_mem(image=image, current_vision_feats=current_vision_feats,
                           feat_sizes=feat_sizes, pred_masks_high_res=pred_masks_high_res,
                           object_score_logits=object_score_logits, is_mask_from_pts=is_mask_from_pts,
                           output_dict=output_dict, is_init_cond_frame=is_init_cond_frame)
        pix = current_vision_feats[-1]  # (HW,B,C)
        cap["G5"].append({
            "pix_feat_hwbc": tocpu(pix),
            "pred_masks_high_res": tocpu(pred_masks_high_res),
            "object_score_logits": tocpu(object_score_logits),
            "is_mask_from_pts": bool(is_mask_from_pts),
            "maskmem_features": tocpu(mf),
            "maskmem_pos_enc": tocpu(mpe),
        })
        return mf, mpe
    trk._encode_new_memory = mem_fwd

    # ---- seed + propagate ----
    state = trk.init_state(video_height=H, video_width=W, num_frames=WINDOW)
    state["cached_features"] = dict(cached)
    trk.add_new_mask(state, frame_idx=0, obj_id=1, mask=seed_mask)

    golden_masks = {}
    t0=time.time()
    for fidx, obj_ids, lrm, vrm, oscore in trk.propagate_in_video(
            state, start_frame_idx=0, max_frame_num_to_track=WINDOW-1,
            reverse=False, propagate_preflight=True, tqdm_disable=True):
        # vrm: [num_obj,1,H,W] logits
        golden_masks[fidx] = (vrm.squeeze() > 0).detach().cpu().numpy().astype(np.uint8)
        log(f"  frame {fidx}: mask area={golden_masks[fidx].sum()} objscore={oscore.flatten().tolist()}")
    log("propagate %.1fs" % (time.time()-t0))
    log("captured G3=%d G4=%d G5=%d frames" % (len(cap["G3"]), len(cap["G4"]), len(cap["G5"])))

    torch.save({"g1": g1_examples, "cap": cap,
                "golden_masks": golden_masks, "video_hw": (H,W),
                "seed_area": int(seed_mask.sum().item())},
               os.path.join(EXPORT, "golden.pt"))
    log("saved golden.pt")

if __name__ == "__main__":
    main()
