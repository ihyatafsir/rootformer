#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Instrumented reproduction of scratch_lm.py train() -- SAME code path (imports scratch_lm
and reuses MorphemicLM / batches_morph / pad / eval_bits), same seeds, same data order.

Adds only observation:
  * running-mean TRAIN loss per 500-step block (the script prints one noisy batch instead)
  * val bits/word every 500 steps (the script only does this every 10,000 steps, so at
    --steps 6000 it never fires during training)
  * gradient probe: are the held-out roots' embedding/head rows receiving gradient?
  * exact peak VRAM via torch.cuda.max_memory_allocated()

Writes ONLY /workspace/scratch_lm_run/loss_curve.json.  Does not save weights.
Note: the periodic eval() calls suppress dropout, which shifts the torch RNG stream relative
to the untouched run, so this is an instrumented reproduction, not a bit-exact replay.
"""
import json
import random
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, '/workspace/hf_v19_2_release')
import scratch_lm as S  # noqa: E402

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
BS = 16
LR = 6e-4
OUTJ = '/workspace/scratch_lm_run/loss_curve.json'

meta = json.load(open('/workspace/sf_data/meta.json'))
data = torch.load('/workspace/sf_data/streams.pt', weights_only=False)
hold = torch.tensor(sorted(meta['hold_roots']))
n_roots = meta['n_roots']
seen_mask = torch.ones(n_roots, dtype=torch.bool)
seen_mask[hold] = False
for i in meta['special_root_ids']:
    seen_mask[i] = False
seen = torch.nonzero(seen_mask).flatten()[:512]

device = torch.device('cuda')
rng = random.Random(0)
torch.manual_seed(0)
model = S.MorphemicLM((meta['n_roots'], meta['n_awzan'],
                       meta['n_prefixes'], meta['n_suffixes'])).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01, betas=(0.9, 0.95))
sched = torch.optim.lr_scheduler.OneCycleLR(
    opt, max_lr=LR, total_steps=STEPS,
    pct_start=max(0.02, min(0.3, 200.0 / max(STEPS, 1))))
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
curve, block_loss, block_n = [], 0.0, 0
grad_probe = {}
step = 0
while step < STEPS:
    for b in S.batches_morph(data, 'train', BS, rng):
        p_in = S.pad([x[:-1] for x in b['p']], 0).to(device)
        r_in = S.pad([x[:-1] for x in b['r']], 0).to(device)
        w_in = S.pad([x[:-1] for x in b['w']], 0).to(device)
        s_in = S.pad([x[:-1] for x in b['s']], 0).to(device)
        r_tg = S.pad([x[1:] for x in b['r']], 0).to(device)
        w_tg = S.pad([x[1:] for x in b['w']], 0).to(device)
        p_tg = S.pad([x[1:] for x in b['p']], 0).to(device)
        s_tg = S.pad([x[1:] for x in b['s']], 0).to(device)
        lr_, lw, lp, ls = model(p_in, r_in, w_in, s_in)
        loss = (F.cross_entropy(lr_.reshape(-1, lr_.shape[-1]), r_tg.reshape(-1), ignore_index=0)
                + 0.5 * F.cross_entropy(lw.reshape(-1, lw.shape[-1]), w_tg.reshape(-1), ignore_index=0)
                + 0.25 * F.cross_entropy(lp.reshape(-1, lp.shape[-1]), p_tg.reshape(-1), ignore_index=0)
                + 0.25 * F.cross_entropy(ls.reshape(-1, ls.shape[-1]), s_tg.reshape(-1), ignore_index=0))
        loss.backward()
        step += 1
        if step in (1, 100, 1000, 3000):
            with torch.no_grad():
                ge = model.e_r.weight.grad
                gh = model.h_r.weight.grad
                grad_probe[step] = {
                    'e_r_holdout_meanabsgrad': float(ge[hold].abs().mean()) if ge is not None else None,
                    'e_r_seen_meanabsgrad': float(ge[seen].abs().mean()) if ge is not None else None,
                    'h_r_holdout_meanabsgrad': float(gh[hold].abs().mean()) if gh is not None else None,
                    'h_r_seen_meanabsgrad': float(gh[seen].abs().mean()) if gh is not None else None,
                }
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        block_loss += float(loss.detach()); block_n += 1
        if step % 500 == 0:
            vb = S.eval_bits(model, data, ['val'], device)['val']
            curve.append({'step': step, 'train_loss_mean500': block_loss / max(block_n, 1),
                          'val_bits_per_word': vb, 'lr': sched.get_last_lr()[0],
                          'peak_vram_MiB': torch.cuda.max_memory_allocated() / 2**20})
            print(f'  step {step}: train_loss(mean500)={block_loss/max(block_n,1):.4f} '
                  f'val_bits/word={vb:.4f} peakVRAM={torch.cuda.max_memory_allocated()/2**20:.1f}MiB',
                  flush=True)
            block_loss, block_n = 0.0, 0
            model.train()
        if step >= STEPS:
            break

res = {'arm': 'instrumented_reproduction', 'steps': STEPS, 'batch': BS, 'lr': LR,
       'wall_seconds': time.time() - t0,
       'peak_vram_MiB': torch.cuda.max_memory_allocated() / 2**20,
       'curve': curve, 'grad_probe': grad_probe}
json.dump(res, open(OUTJ, 'w'), indent=2)
print(f'\nwall={res["wall_seconds"]:.1f}s peak_vram={res["peak_vram_MiB"]:.1f}MiB')
print('grad probe (mean |grad| on root rows):')
print(json.dumps(grad_probe, indent=2))
print(f'wrote {OUTJ}')
