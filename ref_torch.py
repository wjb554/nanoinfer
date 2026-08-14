# Ground-truth reference: run Qwen2.5-0.5B with HuggingFace transformers (CPU).
# Compares the engine's (and numpy's) logits against the real model.
import os
os.environ['HF_HUB_OFFLINE'] = '1'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

mdir = 'models/qwen2.5-0.5b'
tok = AutoTokenizer.from_pretrained(mdir, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(mdir, torch_dtype=torch.float32)
model.eval()

def report(label, ids, hidden=False):
    with torch.no_grad():
        out = model(input_ids=ids, output_hidden_states=hidden)
    logits = out.logits[0, -1]
    top = torch.topk(logits, 5)
    print(f'--- {label} ---')
    print('input tokens:', ids.tolist(), [repr(tok.decode([i])) for i in ids[0]])
    print('top5 ids:', top.indices.tolist())
    print('top5 text:', [repr(tok.decode([i])) for i in top.indices.tolist()])
    if hidden:
        h = out.hidden_states[-1][0, -1].detach().numpy()
        print('last-layer last-tok h[0:8]:', ' '.join(f'{x:.6f}' for x in h[:8]))

report('single token 1001', torch.tensor([[1001]]))
report('prompt "The capital of France is"', tok('The capital of France is', return_tensors='pt').input_ids, hidden=True)
