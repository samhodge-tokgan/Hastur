import os, sys, time, traceback, torch, numpy as np
from common import build, log, EXPORT
import graphs

torch.backends.cuda.matmul.allow_tf32 = False
G = torch.load(os.path.join(EXPORT, "golden.pt"), weights_only=False)
CPU = torch.device("cpu")

def t(x): return torch.as_tensor(x, dtype=torch.float32)

def size_mb(p): return os.path.getsize(p)/1e6

def try_export(name, module, args, input_names, output_names, dynamic_axes, opset=17):
    """Export on CPU fp32. Try legacy TorchScript then dynamo. Return (path, method) or raises."""
    path = os.path.join(EXPORT, f"{name}.onnx")
    module = module.eval()
    errs = []
    # 1) legacy torchscript exporter
    try:
        torch.onnx.export(module, args, path, input_names=input_names,
                          output_names=output_names, dynamic_axes=dynamic_axes,
                          opset_version=opset, dynamo=False)
        return path, "legacy-ts"
    except Exception as e:
        errs.append("legacy: " + repr(e)[:400])
    # 2) dynamo exporter
    try:
        torch.onnx.export(module, args, path, input_names=input_names,
                          output_names=output_names, dynamic_axes=dynamic_axes,
                          opset_version=18, dynamo=True)
        return path, "dynamo"
    except Exception as e:
        errs.append("dynamo: " + repr(e)[:400])
    raise RuntimeError("BOTH exporters failed:\n  " + "\n  ".join(errs))

def ort_run(path, feeds):
    import onnxruntime as ort
    so = ort.SessionOptions(); so.log_severity_level = 3
    sess = ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])
    outs = sess.run(None, feeds)
    return outs, [o.name for o in sess.get_outputs()]

def report(name, ref_list, ort_list):
    lines = []
    for i,(r,o) in enumerate(zip(ref_list, ort_list)):
        r = np.asarray(r); o = np.asarray(o)
        if r.shape != o.shape:
            lines.append(f"  out{i}: SHAPE MISMATCH ref{r.shape} ort{o.shape}"); continue
        mad = float(np.abs(r-o).max()); mse = float(((r-o)**2).mean())
        lines.append(f"  out{i} {r.shape}: max_abs={mad:.3e} mse={mse:.3e}")
    return "\n".join(lines)

def main():
    which = sys.argv[1] if len(sys.argv)>1 else "all"
    model = build(fp32=True)
    model = model.to(CPU)  # export on CPU fp32 to match ORT-CPU
    results = {}

    # ---------------- G4 SAM heads ----------------
    if which in ("all","G4"):
        try:
            rec = G["cap"]["G4"][1]  # a tracking-path record (multimask True, no points)
            bf = t(rec["backbone_features"]); h0 = t(rec["high_res_0"]); h1 = t(rec["high_res_1"])
            mod = graphs.G4SamHeads(model).to(CPU).eval()
            with torch.no_grad():
                ref = mod(bf, h0, h1)
            path, method = try_export("G4", mod, (bf,h0,h1),
                ["backbone_features","high_res_0","high_res_1"],
                ["low_res_masks","high_res_masks","ious","obj_ptr","object_score_logits"],
                dynamic_axes=None)
            outs,_ = ort_run(path, {"backbone_features":bf.numpy(),"high_res_0":h0.numpy(),"high_res_1":h1.numpy()})
            rep = report("G4", [x.detach().cpu().numpy() for x in ref], outs)
            results["G4"] = f"OK ({method}, {size_mb(path):.1f} MB)\n"+rep
        except Exception as e:
            results["G4"] = "FAIL\n"+traceback.format_exc()[-1500:]
        log("### G4\n"+results["G4"])

    # ---------------- G5 memory encoder ----------------
    if which in ("all","G5"):
        for aa in (True, False):
            tag = "G5" if aa else "G5_noaa"
            try:
                rec = G["cap"]["G5"][1]
                pix = t(rec["pix_feat_hwbc"]).permute(1,2,0).reshape(1,256,72,72).contiguous()
                pm = t(rec["pred_masks_high_res"]); osl = t(rec["object_score_logits"])
                mod = graphs.G5MemEncoder(model, antialias=aa).to(CPU).eval()
                with torch.no_grad():
                    ref = mod(pix, pm, osl)
                path, method = try_export(tag, mod, (pix,pm,osl),
                    ["pix_feat","pred_masks_high_res","object_score_logits"],
                    ["maskmem_features","maskmem_pos_enc"], dynamic_axes=None)
                outs,_ = ort_run(path, {"pix_feat":pix.numpy(),"pred_masks_high_res":pm.numpy(),"object_score_logits":osl.numpy()})
                rep = report(tag, [x.detach().cpu().numpy() for x in ref], outs)
                results[tag] = f"OK ({method}, {size_mb(path):.1f} MB)\n"+rep
            except Exception as e:
                results[tag] = "FAIL\n"+traceback.format_exc()[-1200:]
            log(f"### {tag}\n"+results[tag])

    # ---------------- G3 memory attention (real RoPE, dynamic L) ----------------
    if which in ("all","G3"):
        try:
            rec = G["cap"]["G3"][2]  # L=15560, obj_ptr=8
            src = t(rec["src"]); sp = t(rec["src_pos"]); pr = t(rec["prompt"]); pp = t(rec["prompt_pos"])
            nopt = torch.tensor(rec["num_obj_ptr_tokens"], dtype=torch.int64)
            mod = graphs.G3MemAttn(model).to(CPU).eval()
            with torch.no_grad():
                ref = mod(src, sp, pr, pp, nopt)
            # sanity vs captured (complex-rope) golden memory
            gmem = t(rec["memory"])
            realvscomplex = float((ref-gmem).abs().max())
            log(f"  [G3] real-vs-complex RoPE max_abs on captured golden = {realvscomplex:.3e}")
            dyn = {"prompt":{0:"L"}, "prompt_pos":{0:"L"}}
            path, method = try_export("G3", mod, (src,sp,pr,pp,nopt),
                ["src","src_pos","prompt","prompt_pos","num_obj_ptr_tokens"],
                ["memory"], dynamic_axes=dyn)
            feeds = {"src":src.numpy(),"src_pos":sp.numpy(),"prompt":pr.numpy(),
                     "prompt_pos":pp.numpy(),"num_obj_ptr_tokens":nopt.numpy()}
            outs,_ = ort_run(path, feeds)
            rep = report("G3", [ref.detach().cpu().numpy()], outs)
            # also test a DIFFERENT L to prove dynamic axis
            rec2 = G["cap"]["G3"][0]  # L=5188 obj_ptr=4
            src2=t(rec2["src"]);sp2=t(rec2["src_pos"]);pr2=t(rec2["prompt"]);pp2=t(rec2["prompt_pos"])
            nopt2=torch.tensor(rec2["num_obj_ptr_tokens"],dtype=torch.int64)
            with torch.no_grad(): ref2 = mod(src2,sp2,pr2,pp2,nopt2)
            outs2,_ = ort_run(path, {"src":src2.numpy(),"src_pos":sp2.numpy(),"prompt":pr2.numpy(),
                                     "prompt_pos":pp2.numpy(),"num_obj_ptr_tokens":nopt2.numpy()})
            rep2 = report("G3@L=5188", [ref2.detach().cpu().numpy()], outs2)
            results["G3"] = f"OK ({method}, {size_mb(path):.1f} MB) realRoPEvsComplex={realvscomplex:.2e}\n"+rep+"\n dynamic-L recheck:\n"+rep2
        except Exception as e:
            results["G3"] = "FAIL\n"+traceback.format_exc()[-2000:]
        log("### G3\n"+results["G3"])

    # ---------------- G1 image backbone ----------------
    if which in ("all","G1"):
        try:
            img = t(G["g1"][0]["image"])
            mod = graphs.G1Backbone(model).to(CPU).eval()
            with torch.no_grad():
                ref = mod(img)
            path, method = try_export("G1", mod, (img,),
                ["image"], ["fpn0","fpn1","fpn2","pos0","pos1","pos2"], dynamic_axes=None)
            outs,_ = ort_run(path, {"image":img.numpy()})
            rep = report("G1", [x.detach().cpu().numpy() for x in ref], outs)
            results["G1"] = f"OK ({method}, {size_mb(path):.1f} MB)\n"+rep
        except Exception as e:
            results["G1"] = "FAIL\n"+traceback.format_exc()[-1800:]
        log("### G1\n"+results["G1"])

    log("\n\n========== SUMMARY ==========")
    for k,v in results.items():
        log(f"[{k}] {v.splitlines()[0]}")

if __name__ == "__main__":
    main()
