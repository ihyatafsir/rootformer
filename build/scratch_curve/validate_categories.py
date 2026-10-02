#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""validate_categories.py -- does the category metric reproduce the PUBLISHED numbers?

Loads sf_morph.pt and reports test_gen / test_deriv heldout / real_seen / special exactly as
heldout_root_eval.py defines them.  If this lands on 1.490/5.228 (test_gen) and 1.468/4.916
(test_deriv) with n=19,126 / 19,753, then the metric implementation is correct and any
curve_lm number produced with the same function can be trusted.

This is the CHECK that curve_lm.held_out_metrics() needed: that function had the context
sliced with [1:] instead of [:-1], which mis-aligned the category masks.  This version uses
the correct x[:-1] / x[1:] convention and is self-contained.
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
CKPT = sys.argv[1] if len(sys.argv) > 1 else '/workspace/scratch_lm_run/sf_morph.pt'
MAXSENT = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
dev = torch.device('cpu')
print(f'[*] ckpt={CKPT} max_sent={MAXSENT}', flush=True)
DATA = torch.load(f'{D}/streams.pt', weights_only=False)
model = S.MorphemicLM((NR, meta['n_awzan'], meta['n_prefixes'], meta['n_suffixes'])).to(dev)
model.load_state_dict(torch.load(CKPT, map_location=dev, weights_only=True))
model.eval()

sp = torch.zeros(NR, dtype=torch.bool); sp[list(SPECIAL)] = True
hd = torch.zeros(NR, dtype=torch.bool); hd[list(HOLD)] = True


@torch.no_grad()
def cats(split, bs=16):
    rng = random.Random(11)                    # exactly heldout_root_eval.py's seed and bs
    Dd = DATA[split]
    order = rng.sample(range(len(Dd['L'])), min(MAXSENT, len(Dd['L'])))
    a = {k: {'n': 0, 'h1': 0, 'h5': 0} for k in ('heldout_root', 'real_seen_root', 'special')}
    for i in range(0, len(order), bs):
        ch = order[i:i + bs]
        # CORRECT convention: inputs x[:-1], targets x[1:]
        ctx = {h: [Dd[k][j][:S.CTX_WORDS][:-1] for j in ch]
               for h, k in (('p', 'P'), ('r', 'R'), ('w', 'W'), ('s', 'S'))}
        tgt = {h: [Dd[k][j][:S.CTX_WORDS][1:] for j in ch]
               for h, k in (('p', 'P'), ('r', 'R'), ('w', 'W'), ('s', 'S'))}
        if min(len(x) for x in ctx['r']) < 1:
            continue
        T = max(len(x) for x in ctx['r'])
        m = torch.zeros((len(ch), T), dtype=torch.bool)
        for b, x in enumerate(ctx['r']):
            m[b, :len(x)] = True
        lr, lw, lp, ls = model(S.pad(ctx['p'], 0), S.pad(ctx['r'], 0),
                               S.pad(ctx['w'], 0), S.pad(ctx['s'], 0))
        r_tg = S.pad(tgt['r'], 0)
        p1 = lr.argmax(-1)
        p5 = lr.topk(5, -1).indices
        for key, mm in (('heldout_root', hd[r_tg] & m),
                        ('real_seen_root', (~sp[r_tg]) & (~hd[r_tg]) & m),
                        ('special', sp[r_tg] & m)):
            a[key]['n'] += int(mm.sum())
            a[key]['h1'] += int(((p1 == r_tg) & mm).sum())
            a[key]['h5'] += int(((p5 == r_tg.unsqueeze(-1)).any(-1) & mm).sum())
    return {k: {'top1': v['h1'] / max(v['n'], 1), 'top5': v['h5'] / max(v['n'], 1), 'n': v['n']}
            for k, v in a.items()}


print(json.dumps({s: cats(s) for s in ('test_gen', 'test_deriv')}, indent=2))
