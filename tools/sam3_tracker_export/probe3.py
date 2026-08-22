import torch, math
from common import build, log
import graphs
model = build(fp32=True).to("cpu")
mod = graphs.G1Backbone(model).to("cpu").eval()
names={m:n for n,m in mod.named_modules()}
log_rows=[]
def hook(m, inp, out):
    def dt(x):
        return str(x.dtype) if torch.is_tensor(x) and x.is_floating_point() else None
    ins=[dt(x) for x in inp if torch.is_tensor(x)]
    od = dt(out) if torch.is_tensor(out) else (dt(out[0]) if isinstance(out,(list,tuple)) and out and torch.is_tensor(out[0]) else None)
    if od=="torch.bfloat16":
        log_rows.append((names.get(m,'?'), type(m).__name__, ins, od))
for m in mod.modules(): m.register_forward_hook(hook)
img=torch.randn(1,3,1008,1008)
try:
    with torch.no_grad(): mod(img)
except Exception as e: log("err",repr(e)[:120])
for r in log_rows[:6]: log(r)
