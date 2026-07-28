import torch, numpy as np
from common import EXPORT, log
import os
d = torch.load(os.path.join(EXPORT,"golden.pt"), weights_only=False)
def sh(x):
    if torch.is_tensor(x): return f"{tuple(x.shape)}:{str(x.dtype).replace('torch.','')}"
    if isinstance(x,(list,tuple)): return [sh(e) for e in x]
    return x
for g in ["G3","G4","G5"]:
    log("="*30, g)
    for j,rec in enumerate(d["cap"][g][:3]):
        log(f" record {j}:")
        for k,v in rec.items(): log("   ", k, "=", sh(v))
log("="*30, "G1 examples")
for i,ex in d["g1"].items():
    log(" frame",i, {k:sh(v) for k,v in ex.items()})
