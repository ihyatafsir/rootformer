#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CPU-only pre-flight for echo_identity.py: validates every data/indexing path and
produces the arm-4 occurrence census (how much of the val stream is even affected by the
9015-row attention root_embed clamp).  No GPU, no model, a few seconds."""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

RELEASE = Path('/workspace/hf_v19_2_release')
sys.path.insert(0, str(RELEASE))
sys.path.insert(0, str(RELEASE / 'models'))
WIN, STRIDE, CUT = 128, 64, 9015
CACHE = Path('/workspace/nrmp_cache_9490')

import nrmp_vocab as nv                                                    # noqa: E402

V = next(v for k, v in vars(nv).items() if isinstance(v, type) and 'MorphemicVocab' in k)
vocab = V(str(RELEASE / 'data/rootformer_v12_arabic_blueprint.json'))
print(f'vocab num_roots={vocab.num_roots} awzan={vocab.num_awzan} '
      f'prefixes={vocab.num_prefixes} suffixes={vocab.num_suffixes}')

tr4 = [t.long() for t in torch.load(CACHE / 'train.pt', map_location='cpu')]
va4 = [t.long() for t in torch.load(CACHE / 'val.pt', map_location='cpu')]
print(f'train stream {tr4[1].numel()} root tokens | val stream {va4[1].numel()}')

specials = sorted({vocab.PAD_ROOT, vocab.BOS_ROOT, vocab.EOS_ROOT, vocab.UNK_ROOT,
                   vocab.root2id['<PARTICLE>']} |
                  {i for i, r in enumerate(vocab.roots_list) if r.startswith('<P:')})
sp = torch.tensor(specials)
print(f'specials n={len(specials)} ids={specials[:12]}...')


def census(y, tag):
    n = int(y.numel())
    gt = int((y >= CUT).sum())
    eq = int((y == CUT - 1).sum())
    d = {'n': n, 'gt_9014': gt, 'pct_gt': 100.0 * gt / n, 'eq_9014': eq,
         'distinct_ids_total': int(torch.unique(y).numel()),
         'distinct_ids_gt': int(torch.unique(y[y >= CUT]).numel())}
    print(f'{tag:34s} n={n:9d}  >9014: {gt:7d} ({d["pct_gt"]:.3f}%)  ==9014: {eq:6d}  '
          f'distinct>9014={d["distinct_ids_gt"]:4d}/{d["distinct_ids_total"]}')
    return d


out = {'CUT': CUT, 'num_roots': int(vocab.num_roots),
       'aliased_ids': int(vocab.num_roots - CUT)}
out['train_full'] = census(tr4[1], 'train stream (all root tokens)')
out['val_full'] = census(va4[1], 'val stream (all root tokens)')
out['train_full_nonspecial'] = census(tr4[1][~torch.isin(tr4[1], sp)], 'train non-special')
out['val_full_nonspecial'] = census(va4[1][~torch.isin(va4[1], sp)], 'val non-special')


def starts_for(t, n, seed):
    s = list(range(0, t.numel() - WIN - 1, STRIDE))
    np.random.default_rng(seed).shuffle(s)
    return sorted(s[:n])


va_st = starts_for(va4[1], 300, 1)
tr_st = starts_for(tr4[1], 3000, 0)
print(f'va_st n={len(va_st)} first={va_st[:4]} last={va_st[-2:]}')
print(f'tr_st n={len(tr_st)} first={tr_st[:4]} last={tr_st[-2:]}')


def wt(t, st):
    return torch.stack([t[j:j + WIN] for j in st])


Tv = wt(va4[1], va_st)
Tr = wt(tr4[1], tr_st)
for tag, T in (('val', Tv), ('train', Tr)):
    m = ~torch.isin(T[:, 1:].reshape(-1), sp)
    mid = ~torch.isin(T[:, :-1].reshape(-1), sp)
    out[f'{tag}_windows'] = {
        'identity_positions': int(mid.sum()), 'nextword_positions': int(m.sum()),
        'identity_gt_9014': int((T[:, :-1].reshape(-1)[mid] >= CUT).sum()),
        'nextword_gt_9014': int((T[:, 1:].reshape(-1)[m] >= CUT).sum()),
    }
    for name, ids in (('identity', T[:, :-1].reshape(-1)[mid]),
                      ('nextword', T[:, 1:].reshape(-1)[m])):
        c = Counter(ids.tolist())
        top = c.most_common(5)
        maj = top[0][1] / ids.numel()
        maj5 = sum(v for _, v in top) / ids.numel()
        sub_le, sub_gt = ids[ids < CUT], ids[ids >= CUT]
        out[f'{tag}_windows'][f'{name}_marginal'] = {
            'majority_acc@1': maj, 'majority_acc@5': maj5,
            'majority_ids': [int(k) for k, _ in top], 'n': int(ids.numel())}
        print(f'{tag}/{name}: n={ids.numel()} marginal@1={100*maj:.3f}% '
              f'marginal@5={100*maj5:.3f}% | <=9014 n={sub_le.numel()} '
              f'gt n={sub_gt.numel()}')
        if sub_gt.numel():
            cg = Counter(sub_gt.tolist())
            print(f'    within-group marginal (gt9014): '
                  f'{100*cg.most_common(1)[0][1]/sub_gt.numel():.3f}% '
                  f'({cg.most_common(3)})')


def unique_words(streams):
    L = min(int(t.numel()) for t in streams)
    P, R, W, S = [t.long()[:L] for t in streams]
    key = (((P * vocab.num_roots + R) * vocab.num_awzan + W) * vocab.num_suffixes + S)
    u = torch.unique(key)
    s = u % vocab.num_suffixes
    u = u // vocab.num_suffixes
    w = u % vocab.num_awzan
    u = u // vocab.num_awzan
    r = u % vocab.num_roots
    p = u // vocab.num_roots
    return p, r, w, s


for tag, st in (('val', va4), ('train', tr4)):
    p, r, w, s = unique_words(st)
    k = ~torch.isin(r, sp)
    p, r, w, s = p[k], r[k], w[k], s[k]
    out[f'{tag}_unique_words'] = {
        'n': int(p.numel()), 'gt_9014': int((r >= CUT).sum()),
        'distinct_gt_9014': int(torch.unique(r[r >= CUT]).numel())}
    print(f'{tag} unique words: {p.numel()}  (>=9015 roots {int((r>=CUT).sum())}, '
          f'distinct {int(torch.unique(r[r>=CUT]).numel())})')
    # sanity: the packed key must round-trip for a sample of real events
    if tag == 'val':
        Pp, Rr, Ww, Ss = [t.long()[:5000] for t in st]
        key = (((Pp * vocab.num_roots + Rr) * vocab.num_awzan + Ww)
               * vocab.num_suffixes + Ss)
        ss = key % vocab.num_suffixes
        kk = key // vocab.num_suffixes
        ww = kk % vocab.num_awzan
        kk = kk // vocab.num_awzan
        rr = kk % vocab.num_roots
        pp = kk // vocab.num_roots
        assert torch.equal(ss, Ss) and torch.equal(ww, Ww)
        assert torch.equal(rr, Rr) and torch.equal(pp, Pp)
        print('packed-key round-trip OK')

print(json.dumps(out, indent=2))
Path('/workspace/echo_test/precheck_cpu.json').write_text(json.dumps(out, indent=2))
print('wrote /workspace/echo_test/precheck_cpu.json')
