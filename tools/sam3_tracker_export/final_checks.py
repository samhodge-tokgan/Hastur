import os, torch, numpy as np
from common import build, log, EXPORT
import graphs
G = torch.load(os.path.join(EXPORT,"golden.pt"), weights_only=False)
CPU=torch.device("cpu")
model = build(fp32=True).to(CPU)

# --- G3 real-vs-complex RoPE, SAME device (CPU), same inputs ---
rec = G["cap"]["G3"][2]
def t(x): return torch.as_tensor(x, dtype=torch.float32)
src=t(rec["src"]);sp=t(rec["src_pos"]);pr=t(rec["prompt"]);pp=t(rec["prompt_pos"])
nopt=torch.tensor(rec["num_obj_ptr_tokens"],dtype=torch.int64)
# complex-rope reference: fresh wrapper WITHOUT the real swap
from sam3.sam.transformer import RoPEAttention
enc = model.tracker.transformer.encoder
for m in enc.modules():
    if isinstance(m, RoPEAttention): m.freqs_cis = m.freqs_cis.cpu()
with torch.no_grad():
    out_complex = enc(src=[src],prompt=pr,src_key_padding_mask=[None],src_pos=[sp],
                      prompt_pos=pp,prompt_key_padding_mask=None,feat_sizes=[(72,72)],
                      num_obj_ptr_tokens=int(nopt))["memory"]
mod = graphs.G3MemAttn(model).to(CPU).eval()  # applies real swap in-place
with torch.no_grad():
    out_real = mod(src,sp,pr,pp,nopt)
log(f"G3 real-vs-complex RoPE (SAME CPU device): max_abs={float((out_real-out_complex).abs().max()):.3e} "
    f"mse={float(((out_real-out_complex)**2).mean()):.3e}")

# --- ONNX sizes ---
log("\nONNX artifact sizes:")
for f in sorted(os.listdir(EXPORT)):
    if f.endswith(".onnx") or f.endswith(".onnx.data"):
        log(f"  {f:16s} {os.path.getsize(os.path.join(EXPORT,f))/1e6:9.1f} MB")
