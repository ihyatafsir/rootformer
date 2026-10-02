#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Per-target root NLL (bits) on test_deriv, split by target category.

This is the sharpest single measurement of the head fix: a categorical softmax over root TYPES
can only ever push an unseen root's row DOWN, so its NLL on held-out-root targets is bounded
below by roughly log2(n_roots).  A compositional decoder can put real mass on the unseen
string, so its NLL is strictly smaller.  Run for both heads on the same data/checkpoints."""
import json
import math
import random
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, '/workspace/scratch_comp')
import scratch_lm_comp as S  # noqa: E402

D = '/workspace/scratch_comp/sf_data_9490'
meta = json.load(open(f'{D}/meta.json'))
data = torch.load(f'{D}/streams.pt', weights_only=False)
tables = S.Tables(meta)
dev = torch.device(sys.argv[1] if len(sys.argv) > 1 else 'cuda')
sizes = (meta['n_roots'], meta['n_awzan'], meta['n_prefixes'], meta['n_suffixes'])
CATS = ('special', 'real_seen', 'real_hold')
out = {}


@torch.no_grad()
def run(tag, root_head, bs=24):
    if root_head == 'comp':
        model = S.LMComp(sizes, tables.n_gate, meta['n_sym']).to(dev)
    else:
        model = S.LMType(sizes).to(dev)
    model.load_state_dict(torch.load(f'{D}/{tag}.pt', map_location=dev, weights_only=True))
    model.eval()
    acc = {c: {'bits': 0.0, 'n': 0, 'vals': []} for c in CATS}
    rng = random.Random(11)
    Dd = data['test_deriv']
    order = rng.sample(range(len(Dd['L'])), len(Dd['L']))
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
        m = m.to(dev)
        h, root_logits, _, _, _ = model(p_in, r_in, w_in, s_in)
        if root_head == 'comp':
            gate_tg = tables.gate_tab.to(dev)[r_tg]
            nll = F.cross_entropy(root_logits.reshape(-1, root_logits.shape[-1]),
                                  gate_tg.reshape(-1), reduction='none',
                                  ignore_index=-100).reshape(r_tg.shape)
            slot_logp, slot_valid, _ = model.root.teacher_forced(
                h, tables.letters.to(dev)[r_tg], tables.lens.to(dev)[r_tg])
            nll = nll - (slot_logp * slot_valid).sum(-1)
        else:
            nll = F.cross_entropy(root_logits.reshape(-1, root_logits.shape[-1]),
                                  r_tg.reshape(-1), reduction='none').reshape(r_tg.shape)
        bits = nll / math.log(2)
        hold_t = torch.tensor([r in tables.hold for r in range(meta['n_roots'])],
                              dtype=torch.bool, device=dev)
        spec_t = torch.tensor([r in tables.special_set for r in range(meta['n_roots'])],
                              dtype=torch.bool, device=dev)
        masks = {'special': spec_t[r_tg] & m, 'real_seen': (~spec_t[r_tg]) & (~hold_t[r_tg]) & m,
                 'real_hold': hold_t[r_tg] & m}
        for c, mm in masks.items():
            v = bits[mm]
            acc[c]['bits'] += float(v.sum())
            acc[c]['n'] += int(mm.sum())
            acc[c]['vals'].extend(v.tolist())
    res = {}
    for c in CATS:
        vals = sorted(acc[c]['vals'])
        n = max(len(vals), 1)
        res[c] = {'n': acc[c]['n'], 'mean_bits': acc[c]['bits'] / n,
                  'median_bits': vals[len(vals) // 2] if vals else None,
                  'p10_bits': vals[int(0.1 * len(vals))] if vals else None,
                  'frac_below_10bit': sum(1 for v in vals if v < 10) / n}
        print(f'  {tag}: {c:10s} n={res[c]["n"]:6d} mean={res[c]["mean_bits"]:.3f} '
              f'median={res[c]["median_bits"]:.3f} p10={res[c]["p10_bits"]:.3f} '
              f'frac<10bit={res[c]["frac_below_10bit"]:.4f}', flush=True)
    return res


for tag, rh in (('comp6k', 'comp'), ('comp24k', 'comp'), ('type6k', 'type')):
    print(f'== {tag} ({rh}) test_deriv root NLL in bits ==', flush=True)
    out[tag] = run(tag, rh)
    out[tag]['root_head'] = rh
print('log2(n_roots) = %.3f' % math.log2(meta['n_roots']))
json.dump(out, open('/workspace/scratch_comp/heldout_nll.json', 'w'), indent=2)
print('wrote /workspace/scratch_comp/heldout_nll.json')
