# Single-token (position 0) full 24-layer reference forward for Qwen2.5-0.5B.
# Position 0 => RoPE identity; single-token attention => =V (softmax over 1).
import json, struct, numpy as np

def load_tensor(f, header, key, n):
    info = header[key]; start, end = info['data_offsets']
    f.seek(n + 8 + start); data = f.read(end - start)
    if info['dtype'] == 'BF16':
        a = (np.frombuffer(data, dtype='<u2').astype(np.uint32) << 16).view(np.float32)
    else:
        a = np.frombuffer(data, dtype='<f4')
    return a.reshape(info['shape'])

MODEL = 'models/qwen2.5-0.5b/model.safetensors'
TOKEN = 1001
EPS = 1e-6

with open(MODEL, 'rb') as f:
    n = struct.unpack('<Q', f.read(8))[0]
    header = json.loads(f.read(n))
    T = lambda k: load_tensor(f, header, k, n)

    embed = T('model.embed_tokens.weight')
    fin_n = T('model.norm.weight')
    NL = 24
    layers = []
    for i in range(NL):
        p = f'model.layers.{i}.'
        layers.append({
            'a_n':   T(p+'input_layernorm.weight'),
            'q':     T(p+'self_attn.q_proj.weight'),
            'k':     T(p+'self_attn.k_proj.weight'),
            'v':     T(p+'self_attn.v_proj.weight'),
            'o':     T(p+'self_attn.o_proj.weight'),
            'm_n':   T(p+'post_attention_layernorm.weight'),
            'gate':  T(p+'mlp.gate_proj.weight'),
            'up':    T(p+'mlp.up_proj.weight'),
            'down':  T(p+'mlp.down_proj.weight'),
        })

D = 896; Hq = 14; Hkv = 2; hd = D // Hq

def rms(x, w, eps):
    return x / np.sqrt((x**2).mean() + eps) * w

h = embed[TOKEN].astype(np.float64)

for i, L in enumerate(layers):
    normed = rms(h, L['a_n'], EPS)
    q = (normed @ L['q'].T).reshape(Hq, hd)
    k = (normed @ L['k'].T).reshape(Hkv, hd)
    v = (normed @ L['v'].T).reshape(Hkv, hd)
    # RoPE pos 0: identity.  attention single token => value (GQA)
    attn = np.zeros((Hq, hd))
    for j in range(Hq):
        attn[j] = v[j * Hkv // Hq]
    h = h + attn.reshape(-1) @ L['o'].T
    normed2 = rms(h, L['m_n'], EPS)
    g = normed2 @ L['gate'].T
    u = normed2 @ L['up'].T
    g = g / (1 + np.exp(-g))
    h = h + (g * u) @ L['down'].T
    if i in (0, 1, 23):
        print(f"REF layer{i} final h[0:8]:", " ".join(f"{x:.6f}" for x in h[:8]))

hn = rms(h, fin_n, EPS)
logits = hn @ embed.T
top = np.argsort(-logits)[:5]
print("REF final h[0:8]:", " ".join(f"{x:.6f}" for x in h[:8]))
print("REF top5 logits tokens:", top.tolist())
