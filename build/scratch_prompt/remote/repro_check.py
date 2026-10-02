#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""repro_check.py -- validate the harness against the project's OWN published numbers.

heldout_root_eval.py reported, for sf_morph.pt:
    test_gen  : real_seen_root top1 1.490% top5 5.228% n=19,126 ; special 41.471/68.348 n=16,631
    test_deriv: real_seen_root top1 1.468% top5 4.916% n=19,753 ; special 39.202/66.607 n=17,818
This reimplements that identity split (heldout / real_seen / special) on top of the harness
primitives, with the same seed and batch size.  If the numbers match, the harness's forward
pass, masking and id handling match the project's.
"""
import random
import sys
from collections import defaultdict

import torch

sys.path.insert(0, '/workspace/scratch_prompt')
import probe_scratch_lm as P  # noqa: E402

SPEC = torch.zeros(P.NR, dtype=torch.bool)
for i in P.SPECIAL_ROOT_IDS:
    SPEC[i] = True
HOLDT = torch.zeros(P.NR, dtype=torch.bool)
for i in P.HOLD:
    HOLDT[i] = True

print('CKPT under test:', P.CKPT, flush=True)
print('project-published reference (heldout_root_eval.log, same checkpoint):')
print('  test_gen  real_seen_root top1 1.490 top5 5.228 n=19,126 | special 41.471 68.348 n=16,631')
print('  test_deriv real_seen_root top1 1.468 top5 4.916 n=19,753 | special 39.202 66.607 n=17,818')
print(flush=True)

for split in ('test_gen', 'test_deriv'):
    a = defaultdict(lambda: {'n': 0, 'h1': 0, 'h5': 0})
    rng = random.Random(11)                      # same seed as heldout_root_eval.py
    D = P.DATA[split]
    order = rng.sample(range(len(D['L'])), min(8000, len(D['L'])))
    for i in range(0, len(order), 16):           # same bs as heldout_root_eval.py
        ch = order[i:i + 16]
        ctx = {h: [D[key][j][:P.CTX][:-1] for j in ch]
               for h, key in (('p', 'P'), ('r', 'R'), ('w', 'W'), ('s', 'S'))}
        tgt = {h: [D[key][j][:P.CTX][1:] for j in ch]
               for h, key in (('p', 'P'), ('r', 'R'), ('w', 'W'), ('s', 'S'))}
        if min(len(x) for x in ctx['r']) < 1:
            continue
        T = max(len(x) for x in ctx['r'])
        m = torch.zeros((len(ch), T), dtype=torch.bool)
        for b, x in enumerate(ctx['r']):
            m[b, :len(x)] = True
        lg = P.fwd(ctx)
        r_tg = P.S.pad(tgt['r'], 0).to(P.device)
        pred1 = lg['r'].argmax(-1)
        pred5 = lg['r'].topk(5, -1).indices
        hit1 = (pred1 == r_tg) & m
        hit5 = (pred5 == r_tg.unsqueeze(-1)).any(-1) & m
        for key, mm in (('heldout_root', HOLDT[r_tg] & m),
                        ('real_seen_root', (~SPEC[r_tg]) & (~HOLDT[r_tg]) & m),
                        ('special', SPEC[r_tg] & m)):
            a[key]['n'] += int(mm.sum())
            a[key]['h1'] += int((hit1 & mm).sum())
            a[key]['h5'] += int((hit5 & mm).sum())
    print(f'{split}:', flush=True)
    for k, v in a.items():
        print(f"   {k:16s} top1 {v['h1']/max(v['n'],1)*100:6.3f}%  "
              f"top5 {v['h5']/max(v['n'],1)*100:6.3f}%  n={v['n']}", flush=True)
