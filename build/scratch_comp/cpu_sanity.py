#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CPU-only sanity run on the REAL re-prepared dataset: one training step per arm, then
eval_bits and eval_heads on a few sentences.  Catches shape/table bugs without using the GPU."""
import json
import random
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, '/workspace/scratch_comp')
import scratch_lm_comp as S  # noqa: E402

D = '/workspace/scratch_comp/sf_data_9490'
dev = torch.device('cpu')
meta = json.load(open(f'{D}/meta.json'))
tables = S.Tables(meta)
print('n_roots=%d n_gate=%d n_sym=%d n_alphabet=%d hold=%d'
      % (meta['n_roots'], tables.n_gate, meta['n_sym'], tables.n_alphabet, len(tables.hold)))
print('alphabet=%r' % ''.join(tables.alphabet))
print('gate_tab[0]=%d  gate_tab[special4]=%d  gate_tab[content]=%d'
      % (tables.gate_tab[0], tables.gate_tab[tables.special_ids[4]],
         tables.gate_tab[meta['content_root_ids'][0]]))
data = torch.load(f'{D}/streams.pt', weights_only=False)
sizes = (meta['n_roots'], meta['n_awzan'], meta['n_prefixes'], meta['n_suffixes'])
t0 = time.time()

for arm in ('comp', 'type'):
    torch.manual_seed(0)
    if arm == 'comp':
        model = S.LMComp(sizes, tables.n_gate, meta['n_sym'])
    else:
        model = S.LMType(sizes)
    nparam = sum(p.numel() for p in model.parameters())
    b = next(S.batches_morph(data, 'train', 4, random.Random(0)))
    p_in = S.pad([x[:-1] for x in b['p']], 0); r_in = S.pad([x[:-1] for x in b['r']], 0)
    w_in = S.pad([x[:-1] for x in b['w']], 0); s_in = S.pad([x[:-1] for x in b['s']], 0)
    r_tg = S.pad([x[1:] for x in b['r']], 0); w_tg = S.pad([x[1:] for x in b['w']], 0)
    p_tg = S.pad([x[1:] for x in b['p']], 0); s_tg = S.pad([x[1:] for x in b['s']], 0)
    h, root_logits, lw, lp, ls = model(p_in, r_in, w_in, s_in)
    print(f'[{arm}] params={nparam/1e6:.3f}M  h={tuple(h.shape)} root_logits={tuple(root_logits.shape)}')
    assert torch.isfinite(h).all(), 'non-finite backbone'
    loss_nonroot = (0.5 * F.cross_entropy(lw.reshape(-1, lw.shape[-1]), w_tg.reshape(-1), ignore_index=0)
                    + 0.25 * F.cross_entropy(lp.reshape(-1, lp.shape[-1]), p_tg.reshape(-1), ignore_index=0)
                    + 0.25 * F.cross_entropy(ls.reshape(-1, ls.shape[-1]), s_tg.reshape(-1), ignore_index=0))
    if arm == 'comp':
        gate_tg = tables.gate_tab[r_tg]
        loss_gate = F.cross_entropy(root_logits.reshape(-1, root_logits.shape[-1]),
                                    gate_tg.reshape(-1), ignore_index=-100)
        slot_logp, slot_valid, _ = model.root.teacher_forced(
            h, tables.letters[r_tg], tables.lens[r_tg])
        loss_letters = -(slot_logp * slot_valid).sum() / slot_valid.sum().clamp(min=1)
        loss = loss_gate + loss_letters + loss_nonroot
        print(f'[{arm}] loss={float(loss):.4f} gate={float(loss_gate):.4f} '
              f'letters={float(loss_letters):.4f} nonroot={float(loss_nonroot):.4f} '
              f'valid_slots={int(slot_valid.sum())}')
    else:
        loss = F.cross_entropy(root_logits.reshape(-1, root_logits.shape[-1]),
                               r_tg.reshape(-1), ignore_index=0) + loss_nonroot
        print(f'[{arm}] loss={float(loss):.4f}')
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    assert torch.isfinite(loss) and torch.isfinite(gn), 'NaN/Inf'
    print(f'[{arm}] gradnorm={float(gn):.3f}  step OK  ({time.time()-t0:.0f}s)')

    for sp in ('test_deriv',):
        bb = S.eval_bits(model, data, sp, dev, tables, arm, max_sent=8)
        print(f'[{arm}] bits {sp}: {bb}')
        hr, recs = S.eval_heads(model, data, sp, dev, tables, arm, max_sent=8,
                                max_rounds=100, max_rounds_free=100, free_max_pos=10)
        print(f'[{arm}] heads {sp}: joint={hr["acc"]["real_hold"]} '
              f'content={hr["content_only_acc"]} inexact={hr["n_bfs_inexact_joint"]}/{hr["n_bfs_inexact_free"]}')
        if recs:
            pc = S.partial_credit(recs, tables)
            print(f'[{arm}] partial credit n={pc["n"]} content_top1={pc["content_top1_rate"]:.3f} '
                  f'slot={pc["slot_acc_top1"]:.3f}')
print('\nSANITY OK (%.0fs)' % (time.time() - t0))
