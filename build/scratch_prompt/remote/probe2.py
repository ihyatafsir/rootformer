#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""probe2.py -- fast, faithful per-head evaluation of /workspace/scratch_lm_run/sf_morph.pt.

Why a 48-word window is faithful: scratch_lm.MorphemicLM is CAUSALLY masked (upper-triangular
-inf attn_mask) with ABSOLUTE positional embeddings, so the logits at position t depend only on
input positions 0..t.  Right-padding cannot change any real position's prediction.  Only
~0.5% of sentences here exceed 20 words and ~0.03% exceed 48.

Sampling: sentences are bucketed by length and batches are drawn uniformly WITHIN each bucket,
padding only to that bucket's longest sentence.  Per-bucket rates are recombined with weights =
the bucket's true share of target positions, so aggregates are unbiased stratified estimates of
the full split.  We score ALL positions, which is scratch_lm.root_accuracy's convention -- the
function whose published numbers (test_deriv real root 1.468%) we reproduce in repro_check.py.
`train_seen` is not sampled at all: it replays scratch_lm.train's own sampler and scores the
exact sentences the optimiser saw.
"""
import json
import random
import sys
import time

sys.path.insert(0, '/workspace/hf_v19_2_release')
sys.path.insert(0, '/workspace/hf_v19_2_release/models')

import torch

D = '/workspace/sf_data'
OUT = '/workspace/scratch_prompt'
CKPT = '/workspace/scratch_lm_run/sf_morph.pt'
W = 48
NB = 120
BS = 64
BUCKETS = [(2, 8), (8, 16), (16, 24), (24, W + 1)]
IDX = {'p': 0, 'r': 1, 'w': 2, 's': 3}

meta = json.load(open(f'{D}/meta.json'))
NR, NW, NP, NS = meta['n_roots'], meta['n_awzan'], meta['n_prefixes'], meta['n_suffixes']
SPECIAL = set(meta['special_root_ids'])
HOLD = set(meta['hold_roots'])
HEADS = ('p', 'r', 'w', 's')
NAME = {'p': 'prefix', 'r': 'root', 'w': 'wazn', 's': 'suffix'}
LIM = {'p': NP, 'r': NR, 'w': NW, 's': NS}

print('[*] loading streams.pt ...', flush=True)
RAW = torch.load(f'{D}/streams.pt', weights_only=False)
import scratch_lm as S        # noqa: E402
import nrmp_vocab as nv       # noqa: E402

_V = next(v for k, v in vars(nv).items() if isinstance(v, type) and 'MorphemicVocab' in k)
VOCAB = _V('/workspace/hf_v19_2_release/data/rootformer_v12_arabic_blueprint.json')
device = torch.device('cpu')
model = S.MorphemicLM((NR, NW, NP, NS)).to(device)
model.load_state_dict(torch.load(CKPT, map_location=device, weights_only=True))
model.eval()
print('[*] checkpoint loaded', flush=True)


def pack(split):
    d = RAW[split]
    n = len(d['L'])
    ids = torch.zeros((n, 4, W), dtype=torch.int32)
    ln = torch.zeros(n, dtype=torch.int32)
    for i in range(n):
        L = min(int(d['L'][i]), W)
        ln[i] = L
        for k, key in enumerate(('P', 'R', 'W', 'S')):
            seq = d[key][i][:L]
            ids[i, k, :len(seq)] = torch.tensor(seq, dtype=torch.int32)
    return ids, ln


PACK = {}
for sp in ('train', 'val', 'test_gen', 'test_deriv'):
    PACK[sp] = pack(sp)
    ln = PACK[sp][1]
    print(f'  packed {sp:11s} n={ln.numel():8d} target_positions='
          f'{int((ln-1).clamp(min=0).sum()):9d} n>=9={int((ln>=9).sum()):8d} '
          f'n>=25={int((ln>=25).sum()):7d}', flush=True)


def logits_of(sub):
    """sub [B,4,T+1] -> dict head -> logits [B,T,*]"""
    lr, lw, lp, ls = model(sub[:, 0, :-1].long(), sub[:, 1, :-1].long(),
                           sub[:, 2, :-1].long(), sub[:, 3, :-1].long())
    return {'r': lr, 'w': lw, 'p': lp, 's': ls}


def valid_mask(lns, T, floor=0):
    """Target j (0-based) is valid iff sentence length > j+1+floor, so j <= L-2-floor."""
    B = len(lns)
    m = torch.zeros((B, T), dtype=torch.bool)
    for bi, L in enumerate(lns):
        keep = int(L) - 1 - floor
        if keep > 0:
            m[bi, :min(keep, T)] = True
    return m


def acc_from(lg, sub, lns, floor=0):
    T = lg['r'].shape[1]
    m = valid_mask(lns, T, floor)
    out = {}
    for h in HEADS:
        t = sub[:, IDX[h], 1:].long()
        top5 = lg[h].topk(5, -1).indices
        out[h] = {'n': int(m.sum()),
                  'h1': int(((top5[..., 0] == t) & m).sum()),
                  'h5': int(((top5 == t.unsqueeze(-1)).any(-1) & m).sum())}
    return out


@torch.no_grad()
def score_buckets(split, nb=NB, floor=0, seed=11):
    ids, ln = PACK[split]
    rng = random.Random(seed)
    buck = [torch.nonzero((ln >= lo) & (ln < hi)).flatten().tolist() for lo, hi in BUCKETS]
    true_pos = [sum(max(int(ln[i]) - 1 - floor, 0) for i in idx) for idx in buck]
    tot_pos = sum(true_pos)
    res = {h: {'n': 0.0, 'h1': 0.0, 'h5': 0.0} for h in HEADS}
    per_bucket = []
    for (lo, hi), idx, tp in zip(BUCKETS, buck, true_pos):
        if not idx or tp == 0:
            per_bucket.append({'range': [lo, hi], 'true_pos': 0, 'weight': 0,
                               'sampled_n': 0, 'root_top1': None, 'root_top5': None})
            continue
        acc = {h: {'n': 0, 'h1': 0, 'h5': 0} for h in HEADS}
        for _ in range(nb):
            ch = rng.sample(idx, min(BS, len(idx)))
            lns = [int(ln[j]) for j in ch]
            T = max(max(lns) - 1, 1)
            sub = ids[ch][:, :, :T + 1]
            lg = logits_of(sub)
            a = acc_from(lg, sub, lns, floor)
            for h in HEADS:
                for k in ('n', 'h1', 'h5'):
                    acc[h][k] += a[h][k]
        wt = tp / tot_pos
        for h in HEADS:
            if acc[h]['n']:
                res[h]['n'] += tp
                res[h]['h1'] += wt * (acc[h]['h1'] / acc[h]['n'])
                res[h]['h5'] += wt * (acc[h]['h5'] / acc[h]['n'])
        per_bucket.append({'range': [lo, hi], 'true_pos': tp, 'weight': wt,
                           'sampled_n': acc['r']['n'],
                           'root_top1': acc['r']['h1'] / max(acc['r']['n'], 1),
                           'root_top5': acc['r']['h5'] / max(acc['r']['n'], 1)})
    return res, per_bucket


@torch.no_grad()
def score_list(split, idx, floor=0, bs=BS):
    ids, ln = PACK[split]
    acc = {h: {'n': 0, 'h1': 0, 'h5': 0} for h in HEADS}
    for i in range(0, len(idx), bs):
        ch = idx[i:i + bs]
        lns = [min(int(ln[j]), W) for j in ch]
        T = max(max(lns) - 1, 1)
        sub = ids[ch][:, :, :T + 1]
        lg = logits_of(sub)
        a = acc_from(lg, sub, lns, floor)
        for h in HEADS:
            for k in ('n', 'h1', 'h5'):
                acc[h][k] += a[h][k]
    return acc


def seen_indices():
    ln = PACK['train'][1].tolist()
    n = len(ln)
    rng = random.Random(0)
    bs, max_sent = 16, 64
    seen, steps = set(), 0
    while steps < 6000:
        order = rng.sample(range(n), min(bs * max_sent, n))
        cur = []
        for i in order:
            if ln[i] < 2:
                continue
            cur.append(i)
            if len(cur) == bs:
                seen.update(cur)
                cur = []
                steps += 1
                if steps >= 6000:
                    break
    return sorted(seen), steps


def table(rows, title):
    print('\n' + '=' * 106)
    print(title)
    print('=' * 106)
    print(f"{'split':12s} {'ctx>=':>5s} {'positions':>10s} | " +
          ' | '.join(f'{NAME[h]:>13s}' for h in HEADS))
    for sp, fl, r in rows:
        print(f'{sp:12s} {fl:5d} {int(r["r"]["n"]):10d} | ' + ' | '.join(
            f'{r[h]["h1"]*100:5.2f}/{r[h]["h5"]*100:5.2f}' for h in HEADS))


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'acc'
    if mode != 'acc':
        sys.exit('unknown mode')
    rows, saved = [], {}
    for split, floor in (('train', 0), ('train', 8), ('val', 0),
                         ('test_gen', 0), ('test_gen', 8),
                         ('test_deriv', 0), ('test_deriv', 8)):
        t0 = time.time()
        r, pb = score_buckets(split, floor=floor)
        rows.append((split, floor, r))
        saved[f'{split}|{floor}'] = {h: {'top1': r[h]['h1'], 'top5': r[h]['h5'],
                                         'positions': r[h]['n']} for h in HEADS}
        print(f'  {split:11s} ctx>={floor}: {time.time()-t0:.0f}s', flush=True)
        for b in pb:
            if b['weight']:
                print(f'      bucket {str(b["range"]):9s} true_pos={b["true_pos"]:8d} '
                      f'w={b["weight"]*100:5.1f}% sampled={b["sampled_n"]:7d} '
                      f'root {b["root_top1"]*100:5.2f}/{b["root_top5"]*100:5.2f}')
    seen, steps = seen_indices()
    print(f'\n[*] replayed scratch_lm.train sampler: {len(seen)} distinct TRAIN sentences '
          f'optimised on in {steps} steps', flush=True)
    for floor in (0, 8):
        t0 = time.time()
        a = score_list('train', seen, floor=floor)
        rows.append(('train_seen', floor, a))
        saved[f'train_seen|{floor}'] = {h: {'top1': a[h]['h1'] / max(a[h]['n'], 1),
                                            'top5': a[h]['h5'] / max(a[h]['n'], 1),
                                            'positions': a[h]['n']} for h in HEADS}
        print(f'  train_seen ctx>={floor}: {time.time()-t0:.0f}s', flush=True)
    table(rows, 'PER-HEAD ACCURACY (top-1 / top-5 %).  Stratified over sentence length; '
                'train_seen = the exact sentences the optimiser saw.')

    # ---------------- unigram reference ----------------
    marg, ug = {}, {}
    for k, key in enumerate(('P', 'R', 'W', 'S')):
        flat = PACK['train'][0][:, k, :].flatten().long()
        c = torch.bincount(flat, minlength=(NR, NW, NP, NS)[k])
        c[0] = 0                                   # id 0 is padding, never a target
        marg[key] = c
        tot = int(c.sum())
        top5 = torch.topk(c, 5).indices.tolist()
        p = c.float() / tot
        ug[key] = {'top1_id': top5[0], 'top1_p': float(p[top5[0]]), 'top5_ids': top5,
                   'top5_mass': float(p[top5].sum()), 'n_distinct': int((c > 0).sum()),
                   'n_tokens': tot,
                   'H_bits': float(-(p[p > 0] * torch.log2(p[p > 0])).sum())}
    print('\n' + '=' * 106)
    print('UNIGRAM REFERENCE -- empirical TRAIN marginal per stream (the context-free baseline)')
    print('=' * 106)
    for k, h in zip(('P', 'R', 'W', 'S'), HEADS):
        u = ug[k]
        nm = (VOCAB.id2prefix.get(u['top1_id'], '?') if h == 'p' else
              VOCAB.id2root.get(u['top1_id'], '?') if h == 'r' else
              VOCAB.id2wazn.get(u['top1_id'], '?') if h == 'w' else
              VOCAB.id2suffix.get(u['top1_id'], '?'))
        print(f"  {NAME[h]:6s} top1={nm!r:16s}(id {u['top1_id']:5d}) "
              f"p={u['top1_p']*100:6.2f}%  top5mass={u['top5_mass']*100:6.2f}%  "
              f"distinct={u['n_distinct']:6d}  H={u['H_bits']:6.3f} bits")

    print('\n  unigram-model top-1/top-5 accuracy on the SAME positions as the model:')
    res_ug = {}
    for sp in ('train', 'test_gen', 'test_deriv'):
        ids, ln = PACK[sp]
        pm = torch.zeros(ids.shape[0], W, dtype=torch.bool)
        for i in range(ids.shape[0]):
            pm[i, :max(int(ln[i]) - 1, 0)] = True
        pm = pm.flatten()
        row = {}
        for k, h in zip(('P', 'R', 'W', 'S'), HEADS):
            flat = ids[:, k, :].flatten().long()[pm]
            t1 = ug[k]['top1_id']
            top5 = torch.tensor(ug[k]['top5_ids'])
            row[h] = {'n': int(flat.numel()),
                      'top1': float((flat == t1).float().mean()),
                      'top5': float(torch.isin(flat, top5).float().mean())}
        res_ug[sp] = row
        print(f"    {sp:11s} " + '  '.join(
            f'{NAME[h]}:{row[h]["top1"]*100:5.2f}/{row[h]["top5"]*100:5.2f}'
            for h in HEADS))
    sp_ids = torch.zeros(NR, dtype=torch.bool)
    for i in SPECIAL:
        sp_ids[i] = True
    print('\n  unigram on REAL (non-special) root targets only:')
    for sp in ('test_gen', 'test_deriv'):
        ids, ln = PACK[sp]
        pm = torch.zeros(ids.shape[0], W, dtype=torch.bool)
        for i in range(ids.shape[0]):
            pm[i, :max(int(ln[i]) - 1, 0)] = True
        flat = ids[:, 1, :].flatten().long()[pm.flatten()]
        real = ~sp_ids[flat]
        t1 = ug['R']['top1_id']
        top5 = torch.tensor(ug['R']['top5_ids'])
        print(f"    {sp:11s} top1={float((flat == t1).float()[real].mean())*100:6.3f}%  "
              f"top5={float(torch.isin(flat, top5).float()[real].mean())*100:6.3f}%  "
              f"n_real={int(real.sum())}")
    json.dump({'per_head': saved, 'unigram': ug, 'unigram_acc': res_ug,
               'train_seen_sentences': len(seen), 'train_seen_steps': steps},
              open(f'{OUT}/per_head_acc.json', 'w'), indent=2)
    print(f'\nwrote {OUT}/per_head_acc.json')
