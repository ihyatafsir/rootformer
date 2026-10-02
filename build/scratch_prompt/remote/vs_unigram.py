#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""vs_unigram.py -- how much of the model's accuracy is just the train marginal?

For each split, on the model's own top-1 tuple, look up the TRAIN marginal probability of
each predicted id.  If the mean P(marginal) of the model's predictions is ~1, the model is
reproducing the unigram distribution and its accuracy is the unigram's accuracy.  Also
compares, per head, the model's top-1/top-5 accuracy against the SAME numbers for
(a) the unigram top-1/top-5 and (b) a constant <PARTICLE> predictor.
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
meta = json.load(open(f'{D}/meta.json'))
NR, NW, NP, NS = meta['n_roots'], meta['n_awzan'], meta['n_prefixes'], meta['n_suffixes']
HEADS = ('p', 'r', 'w', 's')
IDX = {'p': 0, 'r': 1, 'w': 2, 's': 3}
NAME = {'p': 'prefix', 'r': 'root', 'w': 'wazn', 's': 'suffix'}
LIM = (NP, NR, NW, NS)
PARTICLE = 4
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

# ---- train marginals per head
c = {'p': torch.zeros(NP, dtype=torch.double), 'r': torch.zeros(NR, dtype=torch.double),
     'w': torch.zeros(NW, dtype=torch.double), 's': torch.zeros(NS, dtype=torch.double)}
for i, key in enumerate(('P', 'R', 'W', 'S')):
    h = HEADS[i]
    for seq in RAW['train'][key]:
        if len(seq) < 2:
            continue
        c[h] += torch.bincount(torch.tensor(seq[1:], dtype=torch.long), minlength=LIM[i]).double()
for h in HEADS:
    c[h] /= c[h].sum()
    v, i = c[h].topk(5)
    nm = (VOCAB.id2prefix.get(int(i[0]), '?') if h == 'p' else VOCAB.id2root.get(int(i[0]), '?')
          if h == 'r' else VOCAB.id2wazn.get(int(i[0]), '?'))
    print(f'  {NAME[h]} marginal: top1={nm!r} p={float(v[0])*100:.2f}%  '
          f'top5mass={float(v.sum())*100:.2f}%')


@torch.no_grad()
def run(split, nb=120, seed=11):
    d = RAW[split]
    n = len(d['L'])
    rng = random.Random(seed)
    # stratified buckets identical to probe2
    ln = torch.tensor([min(int(x), W) for x in d['L']])
    buckets = [(2, 8), (8, 16), (16, 24), (24, W + 1)]
    idxb = [torch.nonzero((ln >= lo) & (ln < hi)).flatten().tolist() for lo, hi in buckets]
    tp = [sum(max(int(ln[i]) - 1, 0) for i in idx) for idx in idxb]
    tot = sum(tp)
    agg = {h: {'n': 0.0, 'h1': 0.0, 'h5': 0.0, 'margp1': 0.0, 'margp5': 0.0, 'rank': 0.0}
           for h in HEADS}
    harm = {h: {'h1': 0.0, 'h5': 0.0} for h in HEADS}
    for (lo, hi), idx, w in zip(buckets, idxb, tp):
        if not idx or w == 0:
            continue
        acc = {h: {'n': 0, 'h1': 0, 'h5': 0, 'margp1': 0.0, 'margp5': 0.0, 'rank': 0.0}
               for h in HEADS}
        for _ in range(nb):
            ch = rng.sample(idx, min(BS, len(idx)))
            lns = [int(ln[j]) for j in ch]
            T = max(max(lns) - 1, 1)
            sub = torch.zeros((len(ch), 4, T + 1), dtype=torch.long)
            for b, j in enumerate(ch):
                for k, key in enumerate(('P', 'R', 'W', 'S')):
                    seq = d[key][j][:T + 1]
                    sub[b, k, :len(seq)] = torch.tensor(seq, dtype=torch.long)
            m = torch.zeros((len(ch), T), dtype=torch.bool)
            for b, L in enumerate(lns):
                m[b, :L - 1] = True
            lr, lw, lp, ls = model(sub[:, 0, :-1], sub[:, 1, :-1].long(),
                                   sub[:, 2, :-1].long(), sub[:, 3, :-1].long())
            lg = {'r': lr, 'w': lw, 'p': lp, 's': ls}
            for h in HEADS:
                t = sub[:, IDX[h], 1:]
                top5 = lg[h].topk(5, -1).indices
                ones = top5[..., 0]
                acc[h]['n'] += int(m.sum())
                acc[h]['h1'] += int(((ones == t) & m).sum())
                acc[h]['h5'] += int(((top5 == t.unsqueeze(-1)).any(-1) & m).sum())
                # marginal probability of the model's own predictions
                mpred = c[h][ones]
                acc[h]['margp1'] += float((mpred * m).sum())
                m5 = c[h][top5].sum(-1)
                acc[h]['margp5'] += float((m5 * m).sum())
                # rank of the TRUE target under the marginal (0 = most frequent)
                order = torch.argsort(c[h], descending=True)
                rank_of = torch.empty_like(order)
                rank_of[order] = torch.arange(len(order))
                tr = rank_of[t].double()
                acc[h]['rank'] += float((torch.log2(tr + 1) * m).sum())
        for h in HEADS:
            if acc[h]['n']:
                ww = w / tot
                agg[h]['n'] += w
                for k in ('h1', 'h5', 'margp1', 'margp5', 'rank'):
                    agg[h][k] += ww * (acc[h][k] / acc[h]['n'])
        # constant-<PARTICLE> predictor
        if lo <= 2:
            pass
    return agg


if __name__ == '__main__':
    out = {}
    for split in ('train', 'test_gen', 'test_deriv'):
        a = run(split)
        out[split] = a
        print(f'\n=== {split} ===')
        print(f"{'head':8s} {'model top1':>11s} {'model top5':>11s} {'marg p(pred1)':>14s} "
              f"{'marg p(pred5)':>14s} {'mean log2 rank of true':>23s}")
        for h in HEADS:
            z = a[h]
            print(f"{NAME[h]:8s} {z['h1']*100:10.2f}% {z['h5']*100:10.2f}% "
                  f"{z['margp1']*100:13.2f}% {z['margp5']*100:13.2f}% "
                  f"{z['rank']:22.2f}")
    json.dump({sp: {h: {k: float(v) for k, v in z.items()} for h, z in a.items()}
               for sp, a in out.items()},
              open(f'{OUT}/vs_unigram.json', 'w'), indent=2)
    print(f'\nwrote {OUT}/vs_unigram.json')
