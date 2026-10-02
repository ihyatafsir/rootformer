#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Independent audit of the scratch_lm prepared data:
   (1) leak check: does any training/val sentence contain a held-out root?
   (2) independent re-derivation of the unigram baselines (bits/word, root acc).
Reads ONLY: /workspace/sf_data/{meta.json,streams.pt}.  CPU-only.
"""
import json
import math
import sys
from collections import Counter

import torch

D = '/workspace/sf_data'
meta = json.load(open(f'{D}/meta.json'))
hold = set(meta['hold_roots'])
n_roots = meta['n_roots']
id2root = {i: r for i, r in enumerate(meta.get('roots_list', []))}

print('== meta ==')
for k in ('n_roots', 'n_awzan', 'n_prefixes', 'n_suffixes', 'train_words', 'max_words'):
    print(f'  {k} = {meta[k]}')
print(f'  n_hold_roots = {len(hold)}')
print(f'  n_special_root_ids = {len(meta["special_root_ids"])}')
print(f'  counts = {meta["counts"]}')
print(f'  hold_roots ids = {sorted(hold)[:10]} ...')

data = torch.load(f'{D}/streams.pt', weights_only=False)
print('\n== streams loaded ==')
for sp, v in data.items():
    print(f'  {sp}: sentences={len(v["L"])} words={sum(v["L"])} '
          f'max_len={max(v["L"]) if v["L"] else 0}')

# ---------------- (1) LEAK CHECK ------------------------------------------------
print('\n== leak check: held-out roots in each split ==')
leak = {}
for sp in ('train', 'val', 'test_gen', 'test_deriv'):
    R = data[sp]['R']
    sent_with_hold = 0
    occ_hold = 0
    occ_total = 0
    distinct_hold_seen = set()
    for seq in R:
        occ_total += len(seq)
        hit = hold.intersection(seq)
        if hit:
            sent_with_hold += 1
            distinct_hold_seen |= hit
            occ_hold += sum(1 for x in seq if x in hold)
    leak[sp] = {'sentences': len(R), 'sentences_containing_heldout_root': sent_with_hold,
                'heldout_root_occurrences': occ_hold, 'total_word_positions': occ_total,
                'distinct_heldout_roots_present': len(distinct_hold_seen)}
    print(f'  {sp}: sentences={len(R)} sent_with_heldout_root={sent_with_hold} '
          f'heldout_occ={occ_hold}/{occ_total} distinct_heldout_roots={len(distinct_hold_seen)}')

# any held-out root id that is also a special token?  would make the holdout meaningless
print(f'  held-out ∩ special_ids = {sorted(hold & set(meta["special_root_ids"]))}')

# ---------------- (2) INDEPENDENT UNIGRAM BASELINE ------------------------------
C = [Counter(), Counter(), Counter(), Counter()]   # R, W, P, S
d = data['train']
for arr, c in zip((d['R'], d['W'], d['P'], d['S']), C):
    for seq in arr:
        c.update(seq)
TOT = [sum(c.values()) for c in C]
print('\n== train-stream unigram counts ==')
for name, c, t in zip('R W P S'.split(), C, TOT):
    print(f'  {name}: vocab_seen={len(c)} tokens={t}')
print(f'  held-out root ids with nonzero train count: '
      f'{sum(1 for r in hold if C[0].get(r, 0) > 0)}')
print(f'  <UNK> count in train roots = {C[0].get(0, 0)}  '
      f'(id0 is the first special)')

splits = ['val', 'test_gen', 'test_deriv']
ubits, npos_out = {}, {}
for sp in splits:
    bits, npos = 0.0, 0
    dd = data[sp]
    for si, (arr, c, T) in enumerate(zip((dd['R'], dd['W'], dd['P'], dd['S']), C, TOT)):
        V = max(c) + 2
        for seq in arr[:-1]:
            for x in seq[1:]:
                p = (c.get(x, 0) + 0.5) / (T + 0.5 * V)
                bits += -math.log2(max(p, 1e-12))
            if si == 0:
                npos += max(len(seq) - 1, 0)
    ubits[sp] = bits / max(npos, 1)
    npos_out[sp] = npos
print('\n== independent unigram bits/word (should match reported) ==')
for sp in splits:
    print(f'  {sp}: {ubits[sp]:.6f}  (npos={npos_out[sp]})')

# unigram root acc
top1 = C[0].most_common(1)[0][0]
top5 = {x for x, _ in C[0].most_common(5)}
sp_set = set(meta['special_root_ids'])
print(f'  unigram top1 root id = {top1}  top5 = {sorted(top5)}')
uacc = {}
for sp in splits:
    acc = {k: {'n': 0, 'h1': 0, 'h5': 0} for k in ('all', 'real', 'special')}
    for seq in data[sp]['R'][:-1]:
        for x in seq[1:]:
            keys = ['all'] + (['special'] if x in sp_set else ['real'])
            for k in keys:
                acc[k]['n'] += 1
                acc[k]['h1'] += (x == top1)
                acc[k]['h5'] += (x in top5)
    uacc[sp] = {k: {'top1': v['h1'] / max(v['n'], 1), 'top5': v['h5'] / max(v['n'], 1),
                    'n': v['n']} for k, v in acc.items()}
print('\n== independent unigram root acc (should match reported) ==')
for sp in splits:
    for k in ('all', 'real', 'special'):
        v = uacc[sp][k]
        print(f'  {sp:11s} {k:8s} top1={v["top1"]:.6f} top5={v["top5"]:.6f} n={v["n"]}')

json.dump({'leak': leak, 'unigram_bits': ubits, 'unigram_root_acc': uacc},
          open('/workspace/scratch_lm_run/audit_data.json', 'w'), indent=2)
print('\nwrote /workspace/scratch_lm_run/audit_data.json')
