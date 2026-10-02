#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
real_unigram.py -- a MEANINGFUL context-free baseline for real-root prediction.

The previous baseline scored 0.00% on real roots because a context-free unigram's five most
frequent "roots" are all special tokens (<PARTICLE>, <UNK>, <P:...>), so it never predicts a real
root at all. That makes the comparison look better than it is.

This restricts the unigram to real triconsonantal roots and renormalises, then scores top-1/top-5
on real-root positions. That is the bar a model must clear to claim it has learned root structure.

Usage: python real_unigram.py
"""
import json
import math
import sys
from collections import Counter

import torch

sys.path.insert(0, '/workspace/hf_v19_2_release')

SF = '/workspace/sf_data'


def main():
    meta = json.load(open(f'{SF}/meta.json'))
    data = torch.load(f'{SF}/streams.pt', weights_only=False)
    specials = set(meta['special_root_ids'])
    n_roots = meta['n_roots']

    # ---- unigram over REAL roots only ------------------------------------------------
    C = Counter()
    for seq in data['train']['R']:
        C.update(seq)
    real = Counter({r: c for r, c in C.items() if r not in specials})
    top5 = [r for r, _ in real.most_common(5)]
    top1 = top5[0]
    tot = sum(real.values())
    print(f'[*] real-root unigram built from {tot} training positions, '
          f'{len(real)} distinct real roots')
    print(f'[*] top-5 real roots: {top5}')

    out = {'top5_real_roots': top5, 'distinct_real_roots': len(real)}
    for sp in ('test_gen', 'test_deriv'):
        n = h1 = h5 = 0
        for seq in data[sp]['R'][:-1]:
            for x in seq[1:]:
                if x in specials:
                    continue
                n += 1
                h1 += (x == top1)
                h5 += (x in top5)
        # a proper unigram also has a probability mass; report its bits on real positions
        bits = 0.0
        for seq in data[sp]['R'][:-1]:
            for x in seq[1:]:
                if x in specials:
                    continue
                p = (real.get(x, 0) + 0.5) / (tot + 0.5 * (len(real) + 1))
                bits += -math.log2(max(p, 1e-12))
        out[sp] = {'n_real': n, 'top1': h1 / max(n, 1), 'top5': h5 / max(n, 1),
                   'bits_per_real_root': bits / max(n, 1)}
        print(f'  {sp:<12} real-root n={n:>6}  unigram top1 {100*h1/max(n,1):5.2f}%  '
              f'top5 {100*h5/max(n,1):5.2f}%  '
              f'root bits {bits/max(n,1):6.3f}')

    json.dump(out, open('/workspace/real_unigram.json', 'w'), indent=2)
    print('\nwrote /workspace/real_unigram.json')
    print('\nThis is the bar. Compare against the model\'s real-root top-5 '
          '(test_gen 7.94%, test_deriv 6.03%).')


if __name__ == '__main__':
    main()
