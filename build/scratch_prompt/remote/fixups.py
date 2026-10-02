#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fixups.py -- the two things probe2.py got wrong, computed correctly.

(1) probe2.tbl() printed the WEIGHTED float in the position column and fed weighted floats to
    the percent formatter, so the train_seen row came out as ~6e7%.  probe2.run_indices()
    itself stored honest integer counts; only the summary table was wrong.  This script
    recomputes per-head top-1/top-5 for train / train_seen / val / test_gen / test_deriv with
    a single consistent convention and prints them cleanly.
(2) The unigram reference crashed on `torch.isin(flat, top5_ids)` because the list was turned
    into a string tensor.  Fixed here, and extended to REAL-only root targets, which is the
    comparison that decides whether the model learned anything at all.
"""
import json
import random
import sys

import torch

sys.path.insert(0, '/workspace/hf_v19_2_release')
sys.path.insert(0, '/workspace/hf_v19_2_release/models')

D = '/workspace/sf_data'
OUT = '/workspace/scratch_prompt'
CKPT = '/workspace/scratch_lm_run/sf_morph.pt'
W = 48
BS = 64
NB = 120
BUCKETS = [(2, 8), (8, 16), (16, 24), (24, W + 1)]
IDX = {'p': 0, 'r': 1, 'w': 2, 's': 3}
NAME = {'p': 'prefix', 'r': 'root', 'w': 'wazn', 's': 'suffix'}
HEADS = ('p', 'r', 'w', 's')

meta = json.load(open(f'{D}/meta.json'))
NR, NW, NP, NS = meta['n_roots'], meta['n_awzan'], meta['n_prefixes'], meta['n_suffixes']
SPECIAL = meta['special_root_ids']
HOLD = set(meta['hold_roots'])
print('[*] loading ...', flush=True)
RAW = torch.load(f'{D}/streams.pt', weights_only=False)
import scratch_lm as S      # noqa: E402
import nrmp_vocab as nv     # noqa: E402
_V = next(v for k, v in vars(nv).items() if isinstance(v, type) and 'MorphemicVocab' in k)
VOCAB = _V('/workspace/hf_v19_2_release/data/rootformer_v12_arabic_blueprint.json')
device = torch.device('cpu')
model = S.MorphemicLM((NR, NW, NP, NS)).to(device)
model.load_state_dict(torch.load(CKPT, map_location=device, weights_only=True))
model.eval()


def pack(split):
    d = RAW[split]
    n = len(d['L'])
    ids = torch.zeros((n, 4, W), dtype=torch.int32)
    ln = torch.zeros(n, dtype=torch.int32)
    for i in range(n):
        L = min(int(d['L'][i]), W)
        ln[i] = L
        for k, key in enumerate(('P', 'R', 'W', 'S')):
            ids[i, k, :L] = torch.tensor(d[key][i][:L], dtype=torch.int32)
    return ids, ln


print('[*] packing ...', flush=True)
PACK = {sp: pack(sp) for sp in ('train', 'val', 'test_gen', 'test_deriv')}
print('[*] packed', flush=True)


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
    return sorted(seen)


@torch.no_grad()
def score_ids(split, ch):
    ids, ln = PACK[split]
    lns = [min(int(ln[j]), W) for j in ch]
    T = max(max(lns) - 1, 1)
    sub = ids[ch][:, :, :T + 1].long()
    lr, lw, lp, ls = model(sub[:, 0, :-1], sub[:, 1, :-1], sub[:, 2, :-1], sub[:, 3, :-1])
    lg = {'r': lr, 'w': lw, 'p': lp, 's': ls}
    m = torch.zeros((len(ch), T), dtype=torch.bool)
    for b, L in enumerate(lns):
        m[b, :L - 1] = True
    out = {}
    for h in HEADS:
        t = sub[:, IDX[h], 1:]
        top5 = lg[h].topk(5, -1).indices
        out[h] = {'n': int(m.sum()),
                  'h1': int(((top5[..., 0] == t) & m).sum()),
                  'h5': int(((top5 == t.unsqueeze(-1)).any(-1) & m).sum())}
    return out


def add(a, b):
    for h in HEADS:
        for k in ('n', 'h1', 'h5'):
            a[h][k] += b[h][k]
    return a


def zero():
    return {h: {'n': 0, 'h1': 0, 'h5': 0} for h in HEADS}


def rate(a):
    return {h: (a[h]['h1'] / max(a[h]['n'], 1), a[h]['h5'] / max(a[h]['n'], 1), a[h]['n'])
            for h in HEADS}


@torch.no_grad()
def score_sampled(split, nb=NB, seed=11):
    ids, ln = PACK[split]
    rng = random.Random(seed)
    buck = [torch.nonzero((ln >= lo) & (ln < hi)).flatten().tolist() for lo, hi in BUCKETS]
    tp = [sum(max(int(ln[i]) - 1, 0) for i in idx) for idx in buck]
    tot = sum(tp)
    agg = {h: 0.0 for h in HEADS}
    agg5 = {h: 0.0 for h in HEADS}
    npos = 0
    for idx, w in zip(buck, tp):
        if not idx or w == 0:
            continue
        a = zero()
        for _ in range(nb):
            add(a, score_ids(split, rng.sample(idx, min(BS, len(idx)))))
        ww = w / tot
        npos += w
        for h in HEADS:
            agg[h] += ww * a[h]['h1'] / max(a[h]['n'], 1)
            agg5[h] += ww * a[h]['h5'] / max(a[h]['n'], 1)
    return {h: (agg[h], agg5[h], npos) for h in HEADS}


res = {}
for split in ('train', 'val', 'test_gen', 'test_deriv'):
    res[split] = score_sampled(split)
    print(f'  scored {split}', flush=True)

seen = seen_indices()
print(f'[*] train_seen: {len(seen)} sentences', flush=True)
a = zero()
for i in range(0, len(seen), BS):
    add(a, score_ids('train', seen[i:i + BS]))
res['train_seen'] = rate(a)
print('  scored train_seen', flush=True)

print('\n' + '=' * 96)
print('PER-HEAD ACCURACY (top-1 / top-5 %), consistent convention')
print('=' * 96)
print(f"{'split':12s} {'positions':>10s} | " + ' | '.join(f'{NAME[h]:>15s}' for h in HEADS))
for sp in ('train', 'train_seen', 'val', 'test_gen', 'test_deriv'):
    z = res[sp]
    print(f'{sp:12s} {int(z["r"][2]):10d} | ' + ' | '.join(
        f'{z[h][0]*100:6.2f}/{z[h][1]*100:5.2f}' for h in HEADS))

# ---------------- unigram ----------------
marg = {}
for i, key in enumerate(('P', 'R', 'W', 'S')):
    h = HEADS[i]
    flat = PACK['train'][0][:, i, :].flatten().long()
    c = torch.bincount(flat, minlength=(NP, NR, NW, NS)[i]).long()
    c[0] = 0
    marg[h] = c

print('\n' + '=' * 96)
print('UNIGRAM on the SAME target positions (train marginal, id 0 excluded as padding)')
print('=' * 96)
ug = {}
for sp in ('train', 'test_gen', 'test_deriv'):
    ids, ln = PACK[sp]
    pm = torch.zeros(ids.shape[0], W, dtype=torch.bool)
    for i in range(ids.shape[0]):
        pm[i, :max(int(ln[i]) - 1, 0)] = True
    pmf = pm.flatten()
    row = {}
    for i, h in enumerate(HEADS):
        flat = ids[:, i, :].flatten().long()[pmf]
        c = marg[h]
        top1 = int(torch.argmax(c))
        top5 = torch.topk(c, 5).indices
        row[h] = {'n': int(flat.numel()),
                  'top1': float((flat == top1).double().mean()),
                  'top5': float(torch.isin(flat, top5).double().mean())}
    ug[sp] = row
    print(f'{sp:12s} ' + '  '.join(
        f'{NAME[h]}: {row[h]["top1"]*100:5.2f}/{row[h]["top5"]*100:5.2f}' for h in HEADS))

print('\nROOT head, REAL (non-special, non-held-out) targets only:')
sp_ids = torch.zeros(NR, dtype=torch.bool)
for i in SPECIAL:
    sp_ids[i] = True
hold_ids = torch.zeros(NR, dtype=torch.bool)
for i in HOLD:
    hold_ids[i] = True
rh = {}
for sp in ('test_gen', 'test_deriv'):
    ids, ln = PACK[sp]
    pm = torch.zeros(ids.shape[0], W, dtype=torch.bool)
    for i in range(ids.shape[0]):
        pm[i, :max(int(ln[i]) - 1, 0)] = True
    flat = ids[:, 1, :].flatten().long()[pm.flatten()]
    real = ~sp_ids[flat] & ~hold_ids[flat]
    c = marg['r']
    top1 = int(torch.argmax(c))
    top5 = torch.topk(c, 5).indices
    rh[sp] = {'n_real': int(real.sum()),
              'unigram_top1_on_real': float((flat == top1).double()[real].mean()),
              'unigram_top5_on_real': float(torch.isin(flat, top5).double()[real].mean()),
              'unigram_top1_id': top1,
              'unigram_top1_name': VOCAB.id2root.get(top1, '?'),
              'unigram_top5_names': [VOCAB.id2root.get(int(x), '?') for x in top5]}
    print(f'  {sp}: n_real={rh[sp]["n_real"]:,}  unigram top1={rh[sp]["unigram_top1_on_real"]*100:.3f}% '
          f'top5={rh[sp]["unigram_top5_on_real"]*100:.3f}%   '
          f'(unigram top1 root = {rh[sp]["unigram_top1_name"]!r}; top5 = {rh[sp]["unigram_top5_names"]})')

print('\nMODEL root head on the SAME real-only targets (from collapse_analysis / heldout eval):')
print('  published: test_gen real_seen_root top1 1.490% top5 5.228% (n=19,126)')
print('             test_deriv real_seen_root top1 1.468% top5 4.916% (n=19,753)')

json.dump({'per_head': {sp: {h: list(z[h]) for h in HEADS} for sp, z in res.items()},
           'unigram_same_positions': ug, 'unigram_real_root_only': rh,
           'unigram_top5_root_ids': [int(x) for x in torch.topk(marg['r'], 5).indices],
           'unigram_root_top1_id': int(torch.argmax(marg['r']))},
          open(f'{OUT}/per_head_acc.json', 'w'), indent=2)
print(f'\nwrote {OUT}/per_head_acc.json')
