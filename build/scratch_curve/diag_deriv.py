#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""diag_deriv.py -- why did curve_lm.held_out_metrics() print deriv real-seen 0.00%?

Reproduces the metric on the 6k checkpoint with BOTH the curve_lm code path and a direct
reimplementation, so any disagreement is visible.  Uses only test_deriv.
"""
import json
import random
import sys

import torch

sys.path.insert(0, '/workspace/hf_v19_2_release')
sys.path.insert(0, '/workspace/hf_v19_2_release/models')

D = '/workspace/sf_data'
meta = json.load(open(f'{D}/meta.json'))
NR = meta['n_roots']
SPECIAL = set(meta['special_root_ids'])
HOLD = set(meta['hold_roots'])
import scratch_lm as S      # noqa: E402
print('[*] loading', flush=True)
RAW = torch.load(f'{D}/streams.pt', weights_only=False)
dev = torch.device('cpu')
model = S.MorphemicLM((NR, meta['n_awzan'], meta['n_prefixes'], meta['n_suffixes'])).to(dev)
model.load_state_dict(torch.load('/workspace/scratch_lm_run/sf_morph.pt',
                                 map_location=dev, weights_only=True))
model.eval()

sp = torch.zeros(NR, dtype=torch.bool); sp[list(SPECIAL)] = True
hd = torch.zeros(NR, dtype=torch.bool); hd[list(HOLD)] = True

for nsent in (400, 2000):
    rng = random.Random(11)
    Dd = RAW['test_deriv']
    order = rng.sample(range(len(Dd['L'])), min(nsent, len(Dd['L'])))
    ctx = {h: [Dd[k][j][:S.CTX_WORDS][:-1] for j in order]
           for h, k in (('p', 'P'), ('r', 'R'), ('w', 'W'), ('s', 'S'))}
    tgt = {h: [Dd[k][j][:S.CTX_WORDS][1:] for j in order]
           for h, k in (('p', 'P'), ('r', 'R'), ('w', 'W'), ('s', 'S'))}
    T = max(len(x) for x in ctx['r'])
    m = torch.zeros((len(order), T), dtype=torch.bool)
    for b, x in enumerate(ctx['r']):
        m[b, :len(x)] = True
    with torch.no_grad():
        lr, lw, lp, ls = model(S.pad(ctx['p'], 0), S.pad(ctx['r'], 0),
                               S.pad(ctx['w'], 0), S.pad(ctx['s'], 0))
    r_tg = S.pad(tgt['r'], 0)
    p1 = lr.argmax(-1); p5 = lr.topk(5, -1).indices
    cats = {
        'heldout': hd[r_tg] & m,
        'real': (~sp[r_tg]) & (~hd[r_tg]) & m,
        'special': sp[r_tg] & m,
    }
    print(f'\n=== test_deriv, {len(order)} sentences, T={T}, positions={int(m.sum())} ===')
    for k, mm in cats.items():
        n = int(mm.sum())
        h1 = int(((p1 == r_tg) & mm).sum())
        h5 = int(((p5 == r_tg.unsqueeze(-1)).any(-1) & mm).sum())
        print(f'  {k:9s} n={n:6d}  top1 {h1/max(n,1)*100:6.3f}%  top5 {h5/max(n,1)*100:6.3f}%')
    print(f'  mask covers {int(m.sum())} of {m.numel()} cells; sum of cats = '
          f'{sum(int(x.sum()) for x in cats.values())}')
    # sanity: what does r_tg look like vs the mask?
    print(f'  r_tg shape {tuple(r_tg.shape)}  m shape {tuple(m.shape)}')
    print(f'  positions where m is True but r_tg==0 (would be counted as <PAD> targets): '
          f'{int(((r_tg == 0) & m).sum())}')
