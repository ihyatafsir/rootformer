#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_aligned_cache.py -- INDEPENDENT verification of the rebuilt, index-aligned NRMP cache.

Runs in a FRESH process and re-derives the split + the per-word prefix stream from the corpus
using the original writer's own `sentences_from` / `encode_sentence`, then checks:

  1. all four streams of the aligned cache have IDENTICAL length (= word events)
  2. streams 1/2/3 of the aligned cache are element-wise IDENTICAL to the shipped cache over
     their FULL length (the root/wazn/suffix data must be untouched by the fix)
  3. stream 0 of the aligned cache equals the re-derived true prefix for the first N words
  4. the marker-count delta on stream 0 equals exactly 2 * sentences (per marker value)

Read-only.
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch


def md5(path, chunk=1 << 20):
    import hashlib
    h = hashlib.md5()
    with open(path, 'rb') as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--release', default='/workspace/hf_v19_2_release')
    ap.add_argument('--aligned-dir', default='/workspace/head_fix/nrmp_cache_9490_aligned')
    ap.add_argument('--shipped', default='/workspace/nrmp_cache_9490')
    ap.add_argument('--verify-words', type=int, default=20000)
    ap.add_argument('--report', default='/workspace/head_fix/verify_aligned_cache.json')
    args = ap.parse_args()

    os.environ['ROOTFORMER_VALIDATED_SEG'] = '0'
    release = Path(args.release)
    sys.path.insert(0, str(release))
    sys.path.insert(0, str(release / 'models'))
    import nrmp_train as NP

    # ---- split, via the original writer's code path ---------------------------------------
    all_pairs = []
    for c in NP.CORPORA:
        if not Path(c).exists():
            continue
        all_pairs += NP.sentences_from(c, min_words=4, max_words=64, limit=120000)
    seen, uniq = set(), []
    for f, s in all_pairs:
        if s in seen:
            continue
        seen.add(s)
        uniq.append((f, s))
    files = sorted({f for f, _ in uniq})
    random.Random(NP.SEED).shuffle(files)
    val_files = set(files[:max(1, int(len(files) * 0.08))])
    vocab = NP.build_vocab()
    print(f'[*] split reproduced: {len(uniq)} unique sentences, {len(files)} files, '
          f'{len(val_files)} val files')

    # ---- accumulate the TRUE prefix stream (and r/w/s) ------------------------------------
    N = args.verify_words
    want = {sp: [[], [], [], []] for sp in ('train', 'val')}
    n_words = {'train': 0, 'val': 0}
    for f, s in uniq:
        split = 'val' if f in val_files else 'train'
        enc = vocab.encode_sentence(s)
        if len(enc) < 2:
            continue
        w = want[split]
        for j, (p, r, wz, sf) in enumerate(enc):
            if n_words[split] + j < N:
                w[0].append(p); w[1].append(r); w[2].append(wz); w[3].append(sf)
        n_words[split] += len(enc)
        if n_words['train'] >= N and n_words['val'] >= N:
            break

    REP = {'aligned_dir': args.aligned_dir, 'shipped': args.shipped,
           'md5': {}, 'checks': {}}
    for sp in ('train', 'val'):
        a_path = Path(args.aligned_dir) / f'{sp}.pt'
        s_path = Path(args.shipped) / f'{sp}.pt'
        a = [t.long() for t in torch.load(a_path, map_location='cpu')]
        s = [t.long() for t in torch.load(s_path, map_location='cpu')]
        REP['md5'][sp] = {'aligned': md5(a_path), 'shipped': md5(s_path),
                          'aligned_bytes': a_path.stat().st_size,
                          'shipped_bytes': s_path.stat().st_size}
        c = {}
        lens = [int(t.numel()) for t in a]
        c['aligned_lengths'] = lens
        c['all_four_streams_equal'] = len(set(lens)) == 1
        c['shipped_prefix_len'] = int(s[0].numel())
        c['stream0_shrank_by'] = int(s[0].numel() - a[0].numel())
        # 2. streams 1..3 must be bit-identical to the shipped cache over their FULL length
        for idx, name in ((1, 'root'), (2, 'wazn'), (3, 'suffix')):
            n = min(a[idx].numel(), s[idx].numel())
            eq = int((a[idx][:n] == s[idx][:n]).sum())
            c[f'{name}_identical_to_shipped_pct'] = 100.0 * eq / max(n, 1)
            c[f'{name}_len_aligned'] = int(a[idx].numel())
            c[f'{name}_len_shipped'] = int(s[idx].numel())
        # 3. stream 0 equals the re-derived true prefix
        b = torch.tensor(want[sp][0], dtype=torch.long)
        k = min(N, int(b.numel()), int(a[0].numel()))
        eq0 = int((a[0][:k] == b[:k]).sum())
        c['n_prefix_checked'] = k
        c['aligned_stream0_equals_true_prefix_pct'] = 100.0 * eq0 / max(k, 1)
        # 4. marker-count delta on stream 0
        for nm, val in (('BOS_ROOT', vocab.BOS_ROOT), ('EOS_ROOT', vocab.EOS_ROOT)):
            c[f'stream0_count_{nm}'] = {'shipped': int((s[0] == val).sum()),
                                        'aligned': int((a[0] == val).sum()),
                                        'delta': int((s[0] == val).sum() - (a[0] == val).sum())}
        REP['checks'][sp] = c
        print(f'[*] {sp}: ' + json.dumps(c, indent=1))

    # sentences per split, from the shipped manifest
    mf = Path(args.shipped) / 'manifest_9490.json'
    if mf.exists():
        m = json.loads(mf.read_text())['counts']
        for sp in ('train', 'val'):
            exp = m[sp]['sentences']
            for nm in ('BOS_ROOT', 'EOS_ROOT'):
                got = REP['checks'][sp][f'stream0_count_{nm}']['delta']
                REP['checks'][sp][f'{nm}_delta_equals_sentences'] = (got == exp)
                print(f'[*] {sp} {nm} marker delta {got} == sentences {exp} -> {got == exp}')

    Path(args.report).write_text(json.dumps(REP, indent=2))
    print(f'[*] report -> {args.report}')


if __name__ == '__main__':
    main()
