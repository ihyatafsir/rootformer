#!/usr/bin/env python3
"""probe_env.py -- read-only reconnaissance for the Ghazali-forgetting (EWC) study.

Answers, on the pod CPU with CUDA hidden:
  1. does the v12 tokenizer round-trip the shipped heritage token stream? (vocab alignment)
  2. does the released checkpoint load into UnifiedRootformerV12, and is lm_head tied?
  3. does base.forward(input_ids, labels) give a finite linguistic LM loss?
  4. what exactly are the trunk param names for layers 20-23?
  5. can the synthesis corpus be tokenized with the same tokenizer?

Nothing is written; no GPU is touched.
"""
import os, sys, json, time
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
import torch

REL = '/workspace/hf_v19_2_release'
sys.path.insert(0, REL)
sys.path.insert(0, os.path.join(REL, 'models'))

from pathlib import Path
BP = Path(REL) / 'data/rootformer_v12_arabic_blueprint.json'
CKPT = Path(REL) / 'checkpoints/rootformer_v19_2_synthesis_ar_backbone.awzan142.roots9490.tok10052.safetensors'

print('== 1. tokenizer / heritage alignment ==', flush=True)
from models.unified_rootformer_v12 import UnifiedRootformerV12
from safetensors.torch import load_file

base = UnifiedRootformerV12(str(BP), 'Qwen/Qwen2.5-0.5B', 'cpu', torch.float32)
tok = base.tokenizer
print('  vocab_size', base.vocab_size, 'config.vocab_size', base.config.vocab_size)
print('  tie_word_embeddings', base.config.tie_word_embeddings)

her = torch.load('/workspace/data/heritage_tokens_v12.pt', map_location='cpu')
print('  heritage', tuple(her.shape), her.dtype, 'min', int(her.min()), 'max', int(her.max()))
samp = her[1000:1060].tolist()
print('  decode[1000:1060]:', repr(tok.decode(samp))[:200])
samp2 = her[5_000_000:5_000_050].tolist()
print('  decode[5e6:5e6+50]:', repr(tok.decode(samp2))[:200])

# where did the embed rows get inserted? check id_to_token stability around 353
print('  id 350..360 ->', [tok.id_to_token.get(i, '?') for i in range(350, 361)])

print('== 2. checkpoint load ==', flush=True)
sd = load_file(str(CKPT))
missing, unexpected = base.load_state_dict(sd, strict=False)
bb_missing = [k for k in missing if k.startswith('backbone.')]
print('  total keys', len(sd), 'missing', len(missing), 'unexpected', len(unexpected))
print('  backbone-missing', len(bb_missing), bb_missing[:6])
print('  unexpected prefixes', sorted({k.split(".")[0] for k in unexpected}))
lm = base.backbone.lm_head.weight
emb = base.backbone.model.embed_tokens.weight
print('  lm_head is embed_tokens (tied):', lm.data_ptr() == emb.data_ptr(), tuple(lm.shape))

print('== 3. linguistic LM forward ==', flush=True)
ids = her[200000:200064].reshape(2, 32).long()
out = base.forward(input_ids=ids, labels=ids)
print('  loss_lm', float(out['loss_lm']), 'finite', bool(torch.isfinite(out['loss_lm'])))
print('  logits', tuple(out['logits'].shape))
print('  ppl', float(torch.exp(out['loss_lm'])))

print('== 4. trunk params layers 20-23 ==', flush=True)
names = [(n, tuple(p.shape)) for n, p in base.named_parameters() if n.startswith('backbone.layers.2')]
n_tr = sum(p.numel() for n, p in base.named_parameters()
           if n.startswith(tuple(f'backbone.layers.{i}.' for i in (20, 21, 22, 23))))
print('  n tensors', len(names), 'n params %.2fM' % (n_tr / 1e6))
for n, s in names[:8]:
    print('   ', n, s)

print('== 5. synthesis corpus tokenization ==', flush=True)
syn = '/workspace/rootformer_v12/v18_next_root_morph/data/unified_basran_andalusian_train.jsonl'
t0 = time.time()
n_ok = 0
ids_syn = []
with open(syn, encoding='utf-8') as f:
    for line in f:
        if n_ok >= 3:
            break
        try:
            it = json.loads(line)
        except Exception:
            continue
        ar = it.get('arabic', '').strip()
        if len(ar.split()) < 2:
            continue
        e = tok.encode(ar)
        ids_syn.append(e)
        print('   ar:', ar[:70], '-> ids', len(e), e[:12], 'decode:', repr(tok.decode(e))[:60])
        n_ok += 1
print('  tokenize time %.1fs for 3' % (time.time() - t0))

print('== 6. free-text encode smoke ==', flush=True)
for t in ['بسم الله الرحمن الرحيم', 'العلم نور', 'قال سيبويه']:
    e = tok.encode(t)
    print('  ', t, '->', len(e), 'ids', e[:10], 'rt=', repr(tok.decode(e))[:50])
print('PROBE_OK')
