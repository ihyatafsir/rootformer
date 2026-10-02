#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
real_ngram.py -- is ~8% top-5 on real roots the TASK CEILING, or is the model undertrained?

Builds interpolated n-gram models (1..4) over the REAL-root stream only (special tokens collapsed
to a single OTHER symbol) and scores top-1/top-5 on the real-root positions of the held-out works.

If a 4-gram also lands near 8%, next-root prediction from context is close to its ceiling on this
corpus and no amount of training fixes it. If the 4-gram reaches 20%+, the neural model is badly
underfit and more compute should close the gap.

Usage: python real_ngram.py
"""
import json
import math
import sys
from collections import Counter, defaultdict

import torch

sys.path.insert(0, '/workspace/hf_v19_2_release')
SF = '/workspace/sf_data'
OTHER = 1


def main():
    meta = json.load(open(f'{SF}/meta.json'))
    data = torch.load(f'{SF}/streams.pt', weights_only=False)
    specials = set(meta['special_root_ids'])

    def norm(seq):
        return [OTHER if x in specials else x for x in seq]

    # a 4-gram over all 13M training positions needs ~10GB; 300k sentences is ample for the
    # estimate and keeps memory sane
    train = [norm(s) for s in data['train']['R'][:300000]]
    print(f'[*] {len(train)} training sentences, {sum(len(s) for s in train)} positions '
          f'(capped for memory)')

    # interpolated n-gram counts
    ctx_counts = {n: defaultdict(Counter) for n in range(1, 5)}
    tot = Counter()
    for seq in train:
        for t in range(1, len(seq)):
            tot[seq[t]] += 1
            for n in range(1, 5):
                key = tuple(seq[max(0, t - n + 1):t])
                ctx_counts[n][key][seq[t]] += 1
    V = meta['n_roots']
    uni_total = sum(tot.values())
    print(f'[*] unigram mass {uni_total}, {len(tot)} distinct symbols\n')

    TOP = [s for s, _ in tot.most_common(64)]

    def predict(ctx, topk=5):
        """Interpolated backoff. The unigram floor is applied only to observed candidates and
        the global top-64 -- applying it over all 9,114 symbols per position is O(V) and far too
        slow."""
        scores = defaultdict(float)
        for n in range(1, 5):
            key = tuple(ctx[-(n - 1):]) if n > 1 else ()
            c = ctx_counts[n].get(key)
            if not c:
                continue
            z = sum(c.values())
            w = 0.4 ** (n - 1)
            for sym, k in c.items():
                scores[sym] += w * (k / z)
        for sym in list(scores) + TOP:
            scores[sym] += 0.1 * (tot.get(sym, 0) / uni_total)
        # OTHER is not a legal answer for a real-root position. Leaving it in made the argmax
        # always OTHER (0.00% top-1) and wasted a top-5 slot -- an unfair bar.
        scores.pop(OTHER, None)
        return sorted(scores.items(), key=lambda x: -x[1])[:topk]

    out = {}
    for sp in ('test_gen', 'test_deriv'):
        seqs = [norm(s) for s in data[sp]['R'][:4000]]
        n = h1 = h5 = 0
        for seq in seqs:
            for t in range(1, len(seq)):
                tgt = seq[t]
                if tgt == OTHER:
                    continue
                n += 1
                pred = [s for s, _ in predict(seq[:t], 5)]
                if pred and pred[0] == tgt:
                    h1 += 1
                if tgt in pred:
                    h5 += 1
        out[sp] = {'n_real': n, 'top1': h1 / max(n, 1), 'top5': h5 / max(n, 1)}
        print(f'  {sp:<12} real n={n:>6}  4-gram top1 {100*h1/max(n,1):5.2f}%  '
              f'top5 {100*h5/max(n,1):5.2f}%')

    json.dump(out, open('/workspace/real_ngram.json', 'w'), indent=2)
    print('\nCompare: from-scratch neural model real-root top5 '
          'test_gen 7.94% / test_deriv 6.03%')
    print('         real-root unigram      top5 test_gen 7.65% / test_deriv 6.24%')


if __name__ == '__main__':
    main()
