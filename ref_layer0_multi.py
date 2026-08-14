# Multi-token layer-0 reference (RoPE + causal attention) for Qwen2.5-0.5B.
# Compare layer-0 last-token hidden against the engine's dump.
import json, struct, numpy as np

def load(f, hdr, key, n):
    info = hdr[key]; s,e = info['data_offsets']
    f.seek(n+8+s); data = f.read(e-s)
    a = (np.frombuffer(data, dtype='<u2').astype(np.uint32) << 16).view(np.float32) \
        if info['dtype']=='BF16' else np.frombuffer(data, dtype='<f4')
    return a.reshape(info['shape'])

MODEL='models/qwen2.5-0.5b/model.safetensors'
IDS=[785,6722,315,9625,374]
EPS=1e-6; THETA=1000000.0; D=896; Hq=14; Hkv=2; hd=64

with open(MODEL,'rb') as f:
    n=struct.unpack('<Q',f.read(8))[0]; hdr=json.loads(f.read(n))
    T=lambda k: load(f,hdr,k,n)
    embed=T('model.embed_tokens.weight'); a_n=T('model.layers.0.input_layernorm.weight')
    q_w=T('model.layers.0.self_attn.q_proj.weight'); k_w=T('model.layers.0.self_attn.k_proj.weight')
    v_w=T('model.layers.0.self_attn.v_proj.weight'); o_w=T('model.layers.0.self_attn.o_proj.weight')

T_tok=len(IDS)
h=embed[IDS].astype(np.float64)          # [T, D]
rms = lambda x,w: x/np.sqrt((x**2).mean(axis=1,keepdims=True)+EPS)*w
normed=rms(h,a_n)
q=(normed@q_w.T).reshape(T_tok,Hq,hd)
k=(normed@k_w.T).reshape(T_tok,Hkv,hd)
v=(normed@v_w.T).reshape(T_tok,Hkv,hd)

# RoPE (rotate_half) at positions 0..T-1
half=hd//2
freq = 1.0/(THETA**(np.arange(half)*2.0/hd))       # [half]
def apply_rope(x):
    x2=x.copy()
    for t in range(T_tok):
        angle=freq*t
        c=np.cos(angle); s=np.sin(angle)           # [half]
        xa=x2[t,...,:half].copy(); xb=x2[t,...,half:].copy()
        x2[t,...,:half]=xa*c - xb*s
        x2[t,...,half:]=xb*c + xa*s
    return x2
q=apply_rope(q); k=apply_rope(k)

# causal attention: GQA, q head i -> kv head i*Hkv//Hq
attn=np.zeros((T_tok,Hq,hd))
for t in range(T_tok):
    for i in range(Hq):
        kvi=i*Hkv//Hq
        qh=q[t,i]; # scores vs all k up to t
        s = (qh[None,:]@k[:t+1,kvi,:].T)/np.sqrt(hd)   # [t+1]
        sm = np.exp(s-s.max()); sm/=sm.sum()
        attn[t,i]=sm@v[:t+1,kvi,:]
h = h + attn.reshape(T_tok,Hq*hd)@o_w.T

print("REF multi layer0 last-tok h[0:8]:", " ".join(f"{x:.6f}" for x in h[-1,:8]))
