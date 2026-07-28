"""nn.Module wrappers for the 5 per-frame graphs. Each takes/returns plain tensors."""
import torch, torch.nn as nn, torch.nn.functional as F


class G1Backbone(nn.Module):
    """Shared ViT-L/14 image backbone -> tracker FPN feats + pos embeds."""
    def __init__(self, model):
        super().__init__()
        self.backbone = model.detector.backbone
        self.conv_s0 = model.tracker.sam_mask_decoder.conv_s0
        self.conv_s1 = model.tracker.sam_mask_decoder.conv_s1
        # swap ViT attention complex-RoPE -> real-valued RoPE (ONNX-clean, numerically identical)
        n = 0
        for m in self.backbone.modules():
            fc = getattr(m, "freqs_cis", None)
            if hasattr(m, "use_rope_real") and fc is not None and fc.is_complex() \
               and not getattr(m, "use_ve_rope", False):
                m.use_rope_real = True
                m.register_buffer("freqs_cis_real", fc.real.contiguous(), persistent=False)
                m.register_buffer("freqs_cis_imag", fc.imag.contiguous(), persistent=False)
                n += 1
        print(f"[G1] switched {n} ViT attn blocks to real RoPE", flush=True)
        # replace fused perflib addmm_act (aten::_addmm_activation, ONNX-unsupported +
        # forces bf16) with a plain Linear+activation path (numerically identical)
        import types
        from sam3.model.vitdet import Mlp
        def _mlp_fwd(self, x):
            x = self.act(self.fc1(x)); x = self.drop1(x); x = self.norm(x)
            x = self.fc2(x); x = self.drop2(x); return x
        k = 0
        for m in self.backbone.modules():
            if isinstance(m, Mlp):
                m.forward = types.MethodType(_mlp_fwd, m); k += 1
        print(f"[G1] replaced fused MLP in {k} blocks", flush=True)
    def forward(self, image):
        out = self.backbone.forward_image(image)
        bb = out["sam2_backbone_out"]                  # tracker (SAM2) branch
        fpn = bb["backbone_fpn"]; pos = bb["vision_pos_enc"]
        f0 = self.conv_s0(fpn[0]); f1 = self.conv_s1(fpn[1]); f2 = fpn[2]
        # detector (SAM3) branch: full FPN pyramid (seg head needs all 3 levels) + last-level pos
        dfpn = out["backbone_fpn"]           # [288,144,72] all 256ch
        det_pos72 = out["vision_pos_enc"][-1]
        return f0, f1, f2, pos[0], pos[1], pos[2], dfpn[0], dfpn[1], dfpn[2], det_pos72


class G3MemAttn(nn.Module):
    """Memory-attention encoder (4-layer RoPE) with REAL-valued RoPE.
    Inputs (seq-first): src[HW,1,256], src_pos[HW,1,256], prompt[L,1,64], prompt_pos[L,1,64],
    num_obj_ptr_tokens (0-d int64: trailing prompt tokens excluded from RoPE)."""
    def __init__(self, model):
        super().__init__()
        self.encoder = model.tracker.transformer.encoder
        # switch every RoPEAttention to real-valued RoPE (numerically identical, ONNX-clean)
        from sam3.sam.transformer import RoPEAttention
        dev = next(self.encoder.parameters()).device
        for m in self.encoder.modules():
            if isinstance(m, RoPEAttention):
                m.use_rope_real = True
                m.freqs_cis = m.freqs_cis.to(dev)
                m.freqs_cis_real = m.freqs_cis.real.contiguous()
                m.freqs_cis_imag = m.freqs_cis.imag.contiguous()
    def forward(self, src, src_pos, prompt, prompt_pos, num_obj_ptr_tokens):
        out = self.encoder(
            src=[src], prompt=prompt, src_key_padding_mask=[None],
            src_pos=[src_pos], prompt_pos=prompt_pos, prompt_key_padding_mask=None,
            feat_sizes=[(72, 72)], num_obj_ptr_tokens=num_obj_ptr_tokens,
        )
        return out["memory"]


class G4SamHeads(nn.Module):
    """SAM prompt-encoder + mask-decoder (tracking path: no point prompt, multimask)."""
    def __init__(self, model):
        super().__init__()
        self.trk = model.tracker
    def forward(self, backbone_features, high_res_0, high_res_1):
        (lrm_multi, hrm_multi, ious, low_res_masks, high_res_masks,
         obj_ptr, osl) = self.trk._forward_sam_heads(
            backbone_features=backbone_features,
            point_inputs=None, mask_inputs=None,
            high_res_features=[high_res_0, high_res_1],
            multimask_output=True,
        )
        return low_res_masks, high_res_masks, ious, obj_ptr, osl


class G5MemEncoder(nn.Module):
    """SimpleMaskEncoder memory encoder (tracking path: sigmoid on mask logits)."""
    def __init__(self, model, antialias=True):
        super().__init__()
        self.trk = model.tracker
        self.mm = model.tracker.maskmem_backbone
        self.antialias = antialias
        self.scale = self.trk.sigmoid_scale_for_mem_enc
        self.bias = self.trk.sigmoid_bias_for_mem_enc
    def forward(self, pix_feat, pred_masks_high_res, object_score_logits):
        # pix_feat: (1,256,72,72)
        mask_for_mem = torch.sigmoid(pred_masks_high_res) * self.scale + self.bias
        # replicate SimpleMaskEncoder.forward with explicit interpolate (mask_downsampler)
        md = self.mm.mask_downsampler
        x = mask_for_mem
        if md.interpol_size is not None:
            x = F.interpolate(x, size=md.interpol_size, mode="bilinear",
                              align_corners=False, antialias=self.antialias)
        x = md.encoder(x)
        pf = self.mm.pix_feat_proj(pix_feat)
        x = pf + x
        x = self.mm.fuser(x)
        x = self.mm.out_proj(x)
        pos = self.mm.position_encoding(x).to(x.dtype)
        is_obj = (object_score_logits > 0).float()
        maskmem = x + (1 - is_obj[..., None, None]) * \
            self.trk.no_obj_embed_spatial[..., None, None].expand_as(x)
        return maskmem, pos
