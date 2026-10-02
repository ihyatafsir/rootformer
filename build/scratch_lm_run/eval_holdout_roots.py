#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Held-out-root-only evaluation of a scratch_lm checkpoint.

The script's own root_accuracy() reports test_deriv 'real' roots aggregated over ALL real
roots (held-out and seen mixed).  This evaluates the sharper thesis question: accuracy on the
NEXT-ROOT targets that are literally IN the held-out root set -- roots for which the model
received zero training sentences.
"""
import json
import random
import sys

import torch

sys.path.insert(0, '/workspace/hf_v19_2_release')
import scratch_lm as S  # noqa: E402

D = '/workspace/sf_data'
CKPT = sys.argv[1] if len(sys.argv) > 1 else '/workspace/scratch_lm_run/sf_morph.pt'
meta = json.load(open(f'{D}/meta.json'))
data = torch.load(f'{D}/streams.pt', weights_only=False)
hold = set(meta['hold_roots'])

dev = torch.device(sys.argv[2] if len(sys.argv) > 2 else 'cuda')
model = S.MorphemicLM((meta['n_roots'], meta['n_awzan'],
                       meta['n_prefixes'], meta['n_suffixes'])).to(dev)
model.load_state_dict(torch.load(CKPT, map_location=dev, weights_only=True))
model.eval()

spec = torch.zeros(meta['n_roots'], dtype=torch.bool)
for i in meta['special_root_ids']:
    spec[i] = True
spec = spec.to(dev)
hold_t = torch.zeros(meta['n_roots'], dtype=torch.bool)
for i in hold:
    hold_t[i] = True
hold_t = hold_t.to(dev)


@torch.no_grad()
def acc(model, split, bs=16, max_sent=8000):
    rng = random.Random(11)
    Dd = data[split]
    order = rng.sample(range(len(Dd['L'])), min(max_sent, len(Dd['L'])))
    cats = ('heldout_root', 'real_seen_root', 'special')
    a = {k: {'n': 0, 'h1': 0, 'h5': 0} for k in cats}
    for i in range(0, len(order), bs):
        ch = order[i:i + bs]
        P = [Dd['P'][j][:S.CTX_WORDS] for j in ch]
        R = [Dd['R'][j][:S.CTX_WORDS] for j in ch]
        W = [Dd['W'][j][:S.CTX_WORDS] for j in ch]
        Ss = [Dd['S'][j][:S.CTX_WORDS] for j in ch]
        if min(len(x) for x in R) < 2:
            continue
        r_tg = S.pad([x[1:] for x in R], 0).to(dev)
        p_in = S.pad([x[:-1] for x in P], 0).to(dev)
        r_in = S.pad([x[:-1] for x in R], 0).to(dev)
        w_in = S.pad([x[:-1] for x in W], 0).to(dev)
        s_in = S.pad([x[:-1] for x in Ss], 0).to(dev)
        m = torch.zeros_like(r_tg, dtype=torch.bool)
        for bi, x in enumerate(R):
            m[bi, :min(len(x) - 1, r_tg.shape[1])] = True
        lr, _, _, _ = model(p_in, r_in, w_in, s_in)
        pred1 = lr.argmax(-1)
        pred5 = lr.topk(5, -1).indices
        hit1 = (pred1 == r_tg) & m
        hit5 = (pred5 == r_tg.unsqueeze(-1)).any(-1) & m
        for key, mm in (('heldout_root', hold_t[r_tg] & m),
                        ('real_seen_root', (~spec[r_tg]) & (~hold_t[r_tg]) & m),
                        ('special', spec[r_tg] & m)):
            a[key]['n'] += int(mm.sum())
            a[key]['h1'] += int((hit1 & mm).sum())
            a[key]['h5'] += int((hit5 & mm).sum())
    return {k: {'top1': v['h1'] / max(v['n'], 1), 'top5': v['h5'] / max(v['n'], 1),
                'n': v['n']} for k, v in a.items()}


out = {'ckpt': CKPT, 'n_hold_roots': len(hold)}
for sp in ('test_gen', 'test_deriv'):
    out[sp] = acc(model, sp)
print(json.dumps(out, indent=2))
json.dump(out, open('/workspace/scratch_lm_run/heldout_root_eval.json', 'w'), indent=2)

# unigram reference on exactly the same held-out-root positions == 0 by construction
tr = data['train']['R']
present = sum(1 for seq in tr for x in seq if x in hold)
print(f'\n[check] held-out root tokens in TRAIN stream = {present} '
      f'-> unigram held-out-root top1/top5 identically 0')
