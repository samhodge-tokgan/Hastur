import os, time, types, torch, numpy as np, torch.nn as nn, torch.nn.functional as F
from common import build, load_frames, log, EXPORT, DEV
from sam3.model.data_misc import FindStage
from sam3.model.box_ops import box_cxcywh_to_xyxy

# ---- minimal export shim: skip the box/point pool-project branch when there are 0
# geometric prompts (text-only seed). That branch runs grid_sample / roi_align / pin_memory
# unconditionally (gated on module presence, not prompt count); with a static-0 count the
# guard drops those ONNX-hostile nodes from the trace. Numerically identical (empty prompt).
def _encode_points_shim(self, points, points_mask, points_labels, img_feats):
    points_embed = None
    n_points, bs = points.shape[:2]
    if n_points == 0:  # text-only seed: return clean empty (0,bs,C), no 0-dim Adds
        return self.label_embed(points_labels.long()), points_mask
    if self.points_direct_project is not None:
        points_embed = self.points_direct_project(points)
    if self.points_pool_project is not None and n_points > 0:
        grid = points.transpose(0, 1).unsqueeze(2); grid = (grid * 2) - 1
        sampled = F.grid_sample(img_feats, grid, align_corners=False)
        sampled = sampled.squeeze(-1).permute(2, 0, 1)
        proj = self.points_pool_project(sampled)
        points_embed = proj if points_embed is None else points_embed + proj
    if self.points_pos_enc_project is not None:
        x, y = points.unbind(-1)
        enc_x, enc_y = self.pos_enc._encode_xy(x.flatten(), y.flatten())
        enc_x = enc_x.view(n_points, bs, enc_x.shape[-1]); enc_y = enc_y.view(n_points, bs, enc_y.shape[-1])
        enc = torch.cat([enc_x, enc_y], -1); proj = self.points_pos_enc_project(enc)
        points_embed = proj if points_embed is None else points_embed + proj
    type_embed = self.label_embed(points_labels.long())
    if points_embed is None: points_embed = 0.0
    return type_embed + points_embed, points_mask

def _encode_boxes_shim(self, boxes, boxes_mask, boxes_labels, img_feats):
    boxes_embed = None
    n_boxes, bs = boxes.shape[:2]
    if n_boxes == 0:  # text-only seed: return clean empty (0,bs,C), no 0-dim Adds
        return self.label_embed(boxes_labels.long()), boxes_mask
    if self.boxes_direct_project is not None:
        boxes_embed = self.boxes_direct_project(boxes)
    if self.boxes_pool_project is not None and n_boxes > 0:
        import torchvision
        H, W = img_feats.shape[-2:]
        boxes_xyxy = box_cxcywh_to_xyxy(boxes)
        scale = torch.tensor([W, H, W, H], dtype=boxes_xyxy.dtype, device=boxes_xyxy.device).view(1, 1, 4)
        boxes_xyxy = boxes_xyxy * scale
        sampled = torchvision.ops.roi_align(img_feats, boxes_xyxy.float().transpose(0, 1).unbind(0), self.roi_size)
        proj = self.boxes_pool_project(sampled).view(bs, n_boxes, self.d_model).transpose(0, 1)
        boxes_embed = proj if boxes_embed is None else boxes_embed + proj
    if self.boxes_pos_enc_project is not None:
        cx, cy, w, h = boxes.unbind(-1)
        enc = self.pos_enc.encode_boxes(cx.flatten(), cy.flatten(), w.flatten(), h.flatten())
        enc = enc.view(boxes.shape[0], boxes.shape[1], enc.shape[-1])
        proj = self.boxes_pos_enc_project(enc)
        boxes_embed = proj if boxes_embed is None else boxes_embed + proj
    type_embed = self.label_embed(boxes_labels.long())
    if boxes_embed is None: boxes_embed = 0.0
    return type_embed + boxes_embed, boxes_mask


import numpy as _np
def _rpb_shim(self, reference_boxes, feat_size):
    # decoder box-relative-position-bias: drop the tensor-valued size-cache '==' check
    # (decoder.py:341) that forces an aten.item() data-dependent guard. Recompute coords.
    H, W = feat_size
    boxes_xyxy = box_cxcywh_to_xyxy(reference_boxes).transpose(0, 1)
    bs, num_queries, _ = boxes_xyxy.shape
    # reuse the concrete coord cache (populated by the eager warm-up pass) so we never run
    # arange() on the tensor-valued H/W (which would mint unbacked symints); skip the '==' guard
    if self.compilable_cord_cache is None:
        self.compilable_cord_cache = self._get_coords(H, W, reference_boxes.device)
    coords_h, coords_w = self.compilable_cord_cache
    deltas_y = coords_h.view(1, -1, 1) - boxes_xyxy.reshape(-1, 1, 4)[:, :, 1:4:2]
    deltas_y = deltas_y.view(bs, num_queries, -1, 2)
    deltas_x = coords_w.view(1, -1, 1) - boxes_xyxy.reshape(-1, 1, 4)[:, :, 0:3:2]
    deltas_x = deltas_x.view(bs, num_queries, -1, 2)
    if self.boxRPB in ["log", "both"]:
        dxl = deltas_x * 8; dxl = torch.sign(dxl) * torch.log2(torch.abs(dxl) + 1.0) / _np.log2(8)
        dyl = deltas_y * 8; dyl = torch.sign(dyl) * torch.log2(torch.abs(dyl) + 1.0) / _np.log2(8)
        if self.boxRPB == "log": deltas_x, deltas_y = dxl, dyl
        else: deltas_x = torch.cat([deltas_x, dxl], -1); deltas_y = torch.cat([deltas_y, dyl], -1)
    deltas_x = self.boxRPB_embed_x(x=deltas_x)
    deltas_y = self.boxRPB_embed_y(x=deltas_y)
    B = deltas_y.unsqueeze(3) + deltas_x.unsqueeze(2)
    B = B.flatten(2, 3).permute(0, 3, 1, 2).contiguous()
    return B


class G2Detector(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.det = model.detector
        ge = self.det.geometry_encoder
        ge._encode_points = types.MethodType(_encode_points_shim, ge)
        ge._encode_boxes = types.MethodType(_encode_boxes_shim, ge)
        dec = self.det.transformer.decoder
        dec._get_rpb_matrix = types.MethodType(_rpb_shim, dec)
    def forward(self, fpn0, fpn1, fpn2, pos72, lang_feats, lang_mask):
        # encoder uses last level (72x72); seg head uses full pyramid [288,144,72]
        bb = {"backbone_fpn":[fpn0, fpn1, fpn2], "vision_pos_enc":[pos72],
              "vision_features":fpn2, "language_features":lang_feats, "language_mask":lang_mask}
        fs = FindStage(img_ids=torch.zeros(1,dtype=torch.long,device=fpn2.device),
                       text_ids=torch.zeros(1,dtype=torch.long,device=fpn2.device),
                       input_boxes=None, input_boxes_mask=None, input_boxes_label=None,
                       input_points=None, input_points_mask=None)
        dummy = self.det._get_dummy_prompt()
        out = self.det.forward_grounding(backbone_out=bb, find_input=fs,
                                         geometric_prompt=dummy, find_target=None)
        return out["pred_logits"], out["pred_boxes"], out["pred_masks"], out["presence_logit_dec"]

def main():
    model = build(fp32=True)
    imgs, hw, _ = load_frames(1, start=0)
    det = model.detector
    with torch.inference_mode():
        raw = det.backbone.forward_image(imgs[0])
        txt = det.backbone.forward_text(["person"], device=DEV)
    fpn0 = raw["backbone_fpn"][0].float().clone()
    fpn1 = raw["backbone_fpn"][1].float().clone()
    fpn2 = raw["backbone_fpn"][2].float().clone()
    pos72 = raw["vision_pos_enc"][-1].float().clone()
    lf = txt["language_features"].float().clone(); lm = txt["language_mask"].clone()
    args = (fpn0, fpn1, fpn2, pos72, lf, lm)
    log("G2 inputs", [tuple(a.shape) for a in args])
    mod = G2Detector(model).cuda().eval().float()
    for p in mod.parameters(): p.requires_grad_(False)
    with torch.no_grad(), torch.autocast("cuda", enabled=False):
        ref = mod(*args)
    log("G2 fwd OK; out shapes", [tuple(r.shape) for r in ref])
    path=os.path.join(EXPORT,"G2.onnx"); method=None; err=""
    names=["fpn0","fpn1","fpn2","pos72","lang_feats","lang_mask"]
    onames=["pred_logits","pred_boxes","pred_masks","presence_logit"]
    dyn=None
    for dynn,op in [(True,18)]:
        try:
            t0=time.time()
            with torch.no_grad(), torch.autocast("cuda", enabled=False):
                torch.onnx.export(mod,args,path,input_names=names,
                    output_names=onames,dynamic_axes=dyn,opset_version=op,dynamo=dynn)
            method="dynamo" if dynn else "legacy-ts"; log(f"G2 exported via {method} in {time.time()-t0:.0f}s"); break
        except Exception as e:
            import traceback; err+="\n"+traceback.format_exc()[-3500:]
    if method is None:
        log("G2 EXPORT FAILED:",err); return
    mb=os.path.getsize(path)/1e6
    if os.path.exists(path+".data"): mb+=os.path.getsize(path+".data")/1e6
    import onnxruntime as ort
    so=ort.SessionOptions(); so.log_severity_level=3
    s=ort.InferenceSession(path,sess_options=so,providers=["CPUExecutionProvider"])
    o=s.run(None,{"fpn0":fpn0.cpu().numpy(),"fpn1":fpn1.cpu().numpy(),"fpn2":fpn2.cpu().numpy(),
                  "pos72":pos72.cpu().numpy(),"lang_feats":lf.cpu().numpy(),"lang_mask":lm.cpu().numpy()})
    log(f"### G2 OK ({method}, {mb:.1f} MB) [ref=GPU fp32, ort=CPU]")
    for i,(r,x) in enumerate(zip(ref,o)):
        r=r.detach().cpu().numpy();x=np.asarray(x)
        if r.shape!=x.shape: log(f"  {onames[i]} SHAPE ref{r.shape} ort{x.shape}"); continue
        log(f"  {onames[i]} {r.shape}: max_abs={np.abs(r-x).max():.3e} mse={((r-x)**2).mean():.3e}")

if __name__=="__main__":
    main()
