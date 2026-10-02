#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ngram_baseline.py -- honest context-model reference for next-root prediction.

Counts are fit on the TRAIN root stream and evaluated on held-out sentences. Nothing is fit
on the evaluation set (the earlier ~39% "oracle bigram" was fit and scored on the same pairs,
which is an upper bound, not a baseline).

Models: unigram, bigram, trigram (with simple backoff), and a 4-gram.
Also reports the same metrics restricted to positions whose gold root is a radical (particles
and catch-alls excluded), matching nrmp_governed_eval.py.

Usage:
  python ngram_baseline.py --train-stream /workspace/nrmp_cache/train.pt --n 150
"""
import argparse
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / 'models'))

DIAC = re.compile(r'[\u064b-\u0652\u0670\u0640]')


def load_eval_sentences(n, seed=0, min_words=5, max_words=60):
    import glob
    sents = []
    for f in (sorted(glob.glob('/workspace/scholastic_sanitized/*.txt'))[:4]
              + sorted(glob.glob('/workspace/andalusian_canon_sanitized/*.txt'))[:3]):
        txt = Path(f).read_text(encoding='utf-8', errors='ignore')
        for s in re.split(r'[\n.!?؟]+', txt):
            s = s.strip()
            if min_words <= len(s.split()) <= max_words and re.search(r'[\u0600-\u06FF]', s):
                sents.append(s)
    random.Random(seed).shuffle(sents)
    return sents[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-stream', default='/workspace/nrmp_cache/train.pt')
    ap.add_argument('--n', type=int, default=150)
    ap.add_argument('--out', default='/workspace/ngram_baseline.json')
    args = ap.parse_args()

    import nrmp_vocab as nv
    cls = next(v for k, v in vars(nv).items() if isinstance(v, type) and 'MorphemicVocab' in k)
    vocab = cls(str(ROOT_DIR / 'data/rootformer_v12_arabic_blueprint.json'))

    specials = {vocab.PAD_ROOT, vocab.BOS_ROOT, vocab.EOS_ROOT, vocab.UNK_ROOT,
                vocab.root2id['<PARTICLE>']}
    specials |= {i for i, r in enumerate(vocab.roots_list) if r.startswith('<P:')}

    t = torch.load(args.train_stream, map_location='cpu')
    stream = t[1].tolist()
    print(f'[*] train root stream: {len(stream)} tokens')

    uni, bi, tri, quad = Counter(), defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
    prev1 = prev2 = prev3 = None
    for r in stream:
        uni[r] += 1
        if prev1 is not None:
            bi[prev1][r] += 1
        if prev2 is not None:
            tri[(prev2, prev1)][r] += 1
        if prev3 is not None:
            quad[(prev3, prev2, prev1)][r] += 1
        prev3, prev2, prev1 = prev2, prev1, r

    top_uni = uni.most_common(1)[0][0]
    print(f'[*] distinct roots in train: {len(uni)}   most common: {vocab.id2root.get(top_uni)}')
    print(f'[*] distinct bigram contexts: {len(bi)}   trigram: {len(tri)}   quad: {len(quad)}')

    def predict_bi(ctx):
        c = bi.get(ctx)
        return c.most_common(1)[0][0] if c else top_uni

    def predict_tri(ctx2, ctx1):
        c = tri.get((ctx2, ctx1))
        if c:
            return c.most_common(1)[0][0]
        return predict_bi(ctx1)

    def predict_quad(c3, c2, c1):
        c = quad.get((c3, c2, c1))
        if c:
            return c.most_common(1)[0][0]
        return predict_tri(c2, c1)

    def dist_bi(ctx):
        return bi.get(ctx)

    sents = load_eval_sentences(args.n)
    print(f'[*] held-out sentences: {len(sents)}\n')

    stats = {k: {'n': 0, 'top1': 0, 'top5': 0, 'ce': 0.0} for k in
             ('unigram', 'bigram', 'trigram', '4gram')}

    for sent in sents:
        enc = vocab.encode_sentence(sent)
        rs = [x[1] for x in enc]
        if len(rs) < 2:
            continue
        for i in range(1, len(rs)):
            tgt = rs[i]
            if tgt in specials:
                continue
            c1 = rs[i - 1]
            c2 = rs[i - 2] if i >= 2 else None
            c3 = rs[i - 3] if i >= 3 else None

            preds = {
                'unigram': top_uni,
                'bigram': predict_bi(c1),
                'trigram': predict_tri(c2, c1) if c2 is not None else predict_bi(c1),
                '4gram': predict_quad(c3, c2, c1) if (c3 is not None and c2 is not None)
                         else predict_tri(c2, c1) if c2 is not None else predict_bi(c1),
            }
            for name, p in preds.items():
                stats[name]['n'] += 1
                if p == tgt:
                    stats[name]['top1'] += 1
                # top-5 from the bigram distribution of the same order where available
                if name == 'bigram':
                    d = bi.get(c1)
                elif name == 'trigram':
                    d = tri.get((c2, c1)) or bi.get(c1)
                elif name == '4gram':
                    d = quad.get((c3, c2, c1)) or tri.get((c2, c1)) or bi.get(c1)
                else:
                    d = uni
                if d:
                    cand = [r for r, _ in d.most_common(5)]
                    if tgt in cand:
                        stats[name]['top5'] += 1
                    tot = sum(d.values())
                    p_t = d.get(tgt, 0) / tot if tot else 1e-9
                    stats[name]['ce'] += -math.log(max(p_t, 1e-9))
                # unigram smoothing for unseen

    print(f'{"model":<10}{"n":>7}{"acc@1":>9}{"acc@5":>9}{"ppl":>10}')
    print('-' * 45)
    out = {}
    for name, s in stats.items():
        n = max(s['n'], 1)
        ppl = math.exp(min(s['ce'] / n, 25.0))
        out[name] = {'n': s['n'], 'acc@1': s['top1'] / n, 'acc@5': s['top5'] / n, 'ppl': ppl}
        print(f'{name:<10}{s["n"]:>7}{100*s["top1"]/n:>9.2f}{100*s["top5"]/n:>9.2f}{ppl:>10.1f}')

    json.dump(out, open(args.out, 'w'), indent=2)
    print(f'\nwrote {args.out}')
    print('\nThese counts are fit on TRAIN and scored on held-out text -- no leakage.')


if __name__ == '__main__':
    main()
