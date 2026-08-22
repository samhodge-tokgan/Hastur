import torch, numpy as np
from common import build, log
import graphs
model = build(fp32=True).to("cpu")
# check for any non-fp32 floating params/buffers
bad=[]
for n,p in model.detector.backbone.named_parameters():
    if p.is_floating_point() and p.dtype!=torch.float32: bad.append(("param",n,str(p.dtype)))
for n,b in model.detector.backbone.named_buffers():
    if b.is_floating_point() and b.dtype!=torch.float32: bad.append(("buf",n,str(b.dtype)))
log("non-fp32 floating tensors in backbone:", bad[:20], "total", len(bad))

mod = graphs.G1Backbone(model).to("cpu").eval()
seen=[]
def hook(m, inp, out):
    def dt(x):
        if torch.is_tensor(x) and x.is_floating_point(): return str(x.dtype)
        return None
    od = dt(out) if torch.is_tensor(out) else None
    if od == "torch.bfloat16":
        seen.append(type(m).__name__)
for m in mod.modules():
    m.register_forward_hook(hook)
img = torch.randn(1,3,1008,1008)
try:
    with torch.no_grad(): mod(img)
    log("forward OK")
except Exception as e:
    log("forward err:", repr(e)[:200])
log("first bf16-producing modules:", seen[:8])
