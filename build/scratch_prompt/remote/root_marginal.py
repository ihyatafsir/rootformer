#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""root_marginal.py -- exact root-stream marginals and the degenerate-baseline comparison.

For each split: how many target positions, the frequency of each special root, and what a
CONSTANT predictor ("always <PARTICLE>") and the unigram top-5 would score.  This is the
number that decides whether the model's 41% on the `special` bucket is skill or frequency.
"""
import json
import sys

import torch

sys.path.insert(0, '/workspace/hf_v19_2_release')

D = '/workspace/sf_data'
OUT = '/workspace/scratch_prompt'
meta = json.load(open(f'{D}/meta.json'))
NR = meta['n_roots']
SP = set(meta['special_root_ids'])
HOLD = set(meta['hold_roots'])
PARTICLE = 4
UNK = 3
print('[*] loading ...', flush=True)
RAW = torch.load(f'{D}/streams.pt', weights_only=False)

res = {}
for split in ('train', 'test_gen', 'test_deriv'):
    R = RAW[split]['R']
    cnt = torch.zeros(NR, dtype=torch.long)
    n = 0
    for seq in R:
        if len(seq) < 2:
            continue
        for x in seq[1:]:
            cnt[x] += 1
            n += 1
    sp_mass = int(cnt[list(SP)].sum())
    hold_mass = int(cnt[list(HOLD)].sum())
    real_mass = n - sp_mass - hold_mass
    top = torch.topk(cnt, 12)
    top5 = top.indices[:5].tolist()
    res[split] = {
        'positions': n,
        'special_positions': sp_mass, 'special_share': sp_mass / max(n, 1),
        'heldout_root_positions': hold_mass, 'heldout_share': hold_mass / max(n, 1),
        'real_seen_positions': real_mass, 'real_seen_share': real_mass / max(n, 1),
        'top12': [{'id': int(i), 'name': None, 'n': int(c), 'share': float(c) / max(n, 1),
                   'kind': 'SPECIAL' if int(i) in SP else 'HELD-OUT' if int(i) in HOLD
                           else 'real'} for c, i in zip(top.values, top.indices)],
        'always_particle_acc': float(cnt[PARTICLE]) / max(n, 1),
        'always_particle_acc_real_only': 0.0,
        'unigram_top5_acc': float(cnt[top5].sum()) / max(n, 1),
        'unigram_top5_ids': top5,
        'unigram_top1_id': int(top5[0]),
    }
    print(f'\n=== {split}: {n} root target positions ===')
    print(f'  special        : {sp_mass:8d}  {sp_mass/max(n,1)*100:5.2f}%')
    print(f'  held-out roots : {hold_mass:8d}  {hold_mass/max(n,1)*100:5.2f}%')
    print(f'  real seen roots: {real_mass:8d}  {real_mass/max(n,1)*100:5.2f}%')
    print(f'  always-<PARTICLE> top1 = {cnt[PARTICLE]/max(n,1)*100:5.2f}%   '
          f'unigram top5 = {cnt[top5].sum()/max(n,1)*100:5.2f}%   '
          f'unigram top1 = id {top5[0]}')
    print('  top roots:')
    for c, i in zip(top.values, top.indices):
        i = int(i)
        kind = 'SPECIAL' if i in SP else 'HELD-OUT' if i in HOLD else 'real'
        print(f'    id={i:5d} n={int(c):8d} {float(c)/max(n,1)*100:6.2f}%  [{kind}]')

# also: within the SPECIAL bucket, the per-token breakdown on test splits
sys.path.insert(0, '/workspace/hf_v19_2_release/models')
import nrmp_vocab as nv  # noqa: E402
_V = next(v for k, v in vars(nv).items() if isinstance(v, type) and 'MorphemicVocab' in k)
VOCAB = _V('/workspace/hf_v19_2_release/data/rootformer_v12_arabic_blueprint.json')
for split in ('test_gen', 'test_deriv'):
    R = RAW[split]['R']
    cnt = {}
    for seq in R:
        for x in seq[1:]:
            cnt[x] = cnt.get(x, 0) + 1
    spc = sorted(((v, k) for k, v in cnt.items() if k in SP), reverse=True)
    tot = sum(cnt.values())
    spmass = sum(v for v, _ in spc)
    print(f'\n--- {split}: SPECIAL bucket composition (of {spmass} special positions) ---')
    for v, k in spc[:12]:
        print(f'    {v:8d}  {v/spmass*100:6.2f}% of specials  '
              f'{v/tot*100:5.2f}% of all  id={k} {VOCAB.id2root.get(k,"?")!r}')
    json.dump([{'id': k, 'name': VOCAB.id2root.get(k, '?'), 'n': v,
                'share_of_special': v / max(spmass, 1), 'share_of_all': v / max(tot, 1)}
               for v, k in spc[:20]],
              open(f'{OUT}/special_bucket_{split}.json', 'w'), ensure_ascii=False, indent=2)

for split in res:
    for t in res[split]['top12']:
        t['name'] = VOCAB.id2root.get(t['id'], '?')
json.dump(res, open(f'{OUT}/root_marginal.json', 'w'), ensure_ascii=False, indent=2)
print(f'\nwrote {OUT}/root_marginal.json')
