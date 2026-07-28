import os, time, traceback, torch, numpy as np
from common import build, log, EXPORT
import graphs
G = torch.load(os.path.join(EXPORT,"golden.pt"), weights_only=False)

def main():
    model = build(fp32=True)  # stays on cuda
    img = torch.as_tensor(G["g1"][0]["image"], dtype=torch.float32).cuda()
    mod = graphs.G1Backbone(model).cuda().eval().float()
    for p in mod.parameters(): p.requires_grad_(False)
    from collections import Counter
    c = Counter(str(p.dtype) for p in mod.parameters() if p.is_floating_point())
    cb = Counter(str(b.dtype) for b in mod.buffers() if b.is_floating_point())
    log("param dtypes", dict(c), "buffer dtypes", dict(cb))
    with torch.no_grad(), torch.autocast("cuda", enabled=False):
        ref = [x.float().cpu().numpy() for x in mod(img)]
    log("G1 gpu fp32 forward OK; out shapes", [r.shape for r in ref])
    path = os.path.join(EXPORT,"G1.onnx")
    method=None; err=None
    for dyn,op in [(False,17),(True,18)]:
        try:
            t0=time.time()
            with torch.no_grad(), torch.autocast("cuda", enabled=False):
                torch.onnx.export(mod,(img,),path,input_names=["image"],
                    output_names=["fpn0","fpn1","fpn2","pos0","pos1","pos2","det_fpn0","det_fpn1","det_fpn2","det_pos72"],
                    opset_version=op, dynamo=dyn)
            method = "dynamo" if dyn else "legacy-ts"
            log(f"exported G1 via {method} in {time.time()-t0:.0f}s")
            break
        except Exception as e:
            err=(err or "")+f"\n[{'dynamo' if dyn else 'legacy'}] "+repr(e)[:1600]
    if method is None:
        log("G1 EXPORT FAILED:", err); return
    mb=os.path.getsize(path)/1e6
    data=path+".data"
    if os.path.exists(data): mb+=os.path.getsize(data)/1e6
    import onnxruntime as ort
    so=ort.SessionOptions(); so.log_severity_level=3
    sess=ort.InferenceSession(path,sess_options=so,providers=["CPUExecutionProvider"])
    outs=sess.run(None,{"image":img.float().cpu().numpy()})
    log(f"### G1 OK ({method}, {mb:.1f} MB)  [ref=GPU fp32, ort=CPU fp32]")
    for i,(r,o) in enumerate(zip(ref,outs)):
        r=np.asarray(r);o=np.asarray(o)
        log(f"  out{i} {r.shape}: max_abs={np.abs(r-o).max():.3e} mse={((r-o)**2).mean():.3e}")

if __name__=="__main__":
    main()
