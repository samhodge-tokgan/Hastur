"""End-to-end single-object recurrent validation.
Run A: pure PyTorch fp32 tracker (GPU).  Run B: identical driver but the 4 neural graphs
(G1,G3,G4,G5) routed through ONNXRuntime-CPU, memory bank held by the tracker host code.
Compare per-frame masks A-vs-B (IoU/MSE)."""
import os, time, numpy as np, torch
import onnxruntime as ort
from common import build, load_frames, g1_backbone, detect_person, log, EXPORT, DEV

WINDOW = 9
def sess(name):
    so = ort.SessionOptions(); so.log_severity_level = 3
    return ort.InferenceSession(os.path.join(EXPORT, name), sess_options=so,
                                providers=["CPUExecutionProvider"])
def npf(x): return x.detach().float().cpu().numpy()

def build_cached_pytorch(model, imgs):
    c = {}
    for i, img in enumerate(imgs):
        _, tbb = g1_backbone(model, img)
        c[i] = (img, tbb)
    return c

def build_cached_onnx(model, imgs, g1):
    c = {}
    for i, img in enumerate(imgs):
        o = g1.run(None, {"image": npf(img)})
        tbb = {"backbone_fpn": [torch.from_numpy(o[0]).to(DEV), torch.from_numpy(o[1]).to(DEV),
                               torch.from_numpy(o[2]).to(DEV)],
               "vision_pos_enc": [torch.from_numpy(o[3]).to(DEV), torch.from_numpy(o[4]).to(DEV),
                                 torch.from_numpy(o[5]).to(DEV)]}
        c[i] = (img, tbb)
    return c

def patch_onnx(trk, g3, g4, g5):
    """Monkeypatch the 3 recurrent neural methods to use ORT (tracking path only)."""
    enc = trk.transformer.encoder; orig_enc = enc.forward
    def enc_fwd(src, prompt, **kw):
        s = src[0] if isinstance(src, list) else src
        sp = kw["src_pos"]; sp = sp[0] if isinstance(sp, list) else sp
        nopt = int(kw.get("num_obj_ptr_tokens", 0))
        mem = g3.run(None, {"src": npf(s), "src_pos": npf(sp), "prompt": npf(prompt),
                            "prompt_pos": npf(kw["prompt_pos"]),
                            "num_obj_ptr_tokens": np.array(nopt, dtype=np.int64)})[0]
        return {"memory": torch.from_numpy(mem).to(s.device), "pos_embed": sp, "padding_mask": None}
    enc.forward = enc_fwd

    orig_heads = trk._forward_sam_heads
    def heads_fwd(backbone_features, point_inputs=None, mask_inputs=None,
                  high_res_features=None, multimask_output=False, gt_masks=None):
        if point_inputs is None and mask_inputs is None and high_res_features is not None:
            o = g4.run(None, {"backbone_features": npf(backbone_features),
                              "high_res_0": npf(high_res_features[0]),
                              "high_res_1": npf(high_res_features[1])})
            dev = backbone_features.device
            lrm = torch.from_numpy(o[0]).to(dev); hrm = torch.from_numpy(o[1]).to(dev)
            ious = torch.from_numpy(o[2]).to(dev); optr = torch.from_numpy(o[3]).to(dev)
            osl = torch.from_numpy(o[4]).to(dev)
            return (None, None, ious, lrm, hrm, optr, osl)
        return orig_heads(backbone_features=backbone_features, point_inputs=point_inputs,
                          mask_inputs=mask_inputs, high_res_features=high_res_features,
                          multimask_output=multimask_output, gt_masks=gt_masks)
    trk._forward_sam_heads = heads_fwd

    orig_mem = trk._encode_new_memory
    def mem_fwd(image, current_vision_feats, feat_sizes, pred_masks_high_res,
                object_score_logits, is_mask_from_pts, output_dict=None, is_init_cond_frame=False):
        if is_mask_from_pts:  # seed frame: keep PyTorch (G5 exported = tracking sigmoid path)
            return orig_mem(image=image, current_vision_feats=current_vision_feats,
                            feat_sizes=feat_sizes, pred_masks_high_res=pred_masks_high_res,
                            object_score_logits=object_score_logits, is_mask_from_pts=is_mask_from_pts,
                            output_dict=output_dict, is_init_cond_frame=is_init_cond_frame)
        pix = current_vision_feats[-1].permute(1,2,0).reshape(1,256,72,72).contiguous()
        o = g5.run(None, {"pix_feat": npf(pix), "pred_masks_high_res": npf(pred_masks_high_res),
                          "object_score_logits": npf(object_score_logits)})
        dev = pix.device
        return torch.from_numpy(o[0]).to(dev), [torch.from_numpy(o[1]).to(dev)]
    trk._encode_new_memory = mem_fwd
    return (enc, orig_enc), (trk, orig_heads, orig_mem)

def restore(saved):
    (enc, oe), (trk, oh, om) = saved
    enc.forward = oe; trk._forward_sam_heads = oh; trk._encode_new_memory = om

def run_prop(model, imgs, hw, seed_mask, use_onnx, sessions=None):
    trk = model.tracker
    H, W = hw
    if use_onnx:
        cached = build_cached_onnx(model, imgs, sessions["G1"])
        saved = patch_onnx(trk, sessions["G3"], sessions["G4"], sessions["G5"])
    else:
        cached = build_cached_pytorch(model, imgs)
        saved = None
    state = trk.init_state(video_height=H, video_width=W, num_frames=WINDOW)
    state["cached_features"] = dict(cached)
    trk.add_new_mask(state, frame_idx=0, obj_id=1, mask=seed_mask)
    masks = {}
    for fidx, obj_ids, lrm, vrm, osc in trk.propagate_in_video(
            state, start_frame_idx=0, max_frame_num_to_track=WINDOW-1,
            reverse=False, propagate_preflight=True, tqdm_disable=True):
        masks[fidx] = (vrm.squeeze() > 0).detach().cpu().numpy().astype(np.uint8)
    if saved: restore(saved)
    return masks

def iou(a, b):
    i = np.logical_and(a, b).sum(); u = np.logical_or(a, b).sum()
    return float(i)/float(u) if u > 0 else 1.0

def main():
    model = build(fp32=True)
    imgs, hw, _ = load_frames(WINDOW, start=0)
    seed_mask, st = detect_person(model, imgs[0], hw)
    log("seed prob", st["prob"])
    sessions = {n: sess(f"{n}.onnx") for n in ["G1","G3","G4","G5"]}
    log("loaded ORT sessions")

    t0=time.time(); A = run_prop(model, imgs, hw, seed_mask, use_onnx=False); log("A(pytorch) %.1fs"%(time.time()-t0))
    t0=time.time(); B = run_prop(model, imgs, hw, seed_mask, use_onnx=True, sessions=sessions); log("B(onnx) %.1fs"%(time.time()-t0))

    log("\nframe | IoU(A,B) | MSE(prob-mask) | areaA | areaB")
    ious=[]
    for f in sorted(A):
        a=A[f]; b=B[f]; j=iou(a,b); mse=float(((a.astype(np.float32)-b.astype(np.float32))**2).mean())
        ious.append(j)
        log(f"  {f:3d} |  {j:.4f} |   {mse:.3e}   | {a.sum():7d} | {b.sum():7d}")
    log(f"\nmean IoU (tracked frames 1+) = {np.mean(ious[1:]):.4f}  min = {np.min(ious[1:]):.4f}")

if __name__ == "__main__":
    main()
