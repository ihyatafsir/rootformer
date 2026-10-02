#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nrmp_rebuild_cache.py -- rebuild /workspace/nrmp_cache/{train,val}.pt against the LIVE
9,490-root vocabulary, without touching the shipped originals.

!!! FIXED 2026-10-02: STREAM-0 INDEX MISALIGNMENT !!!
----------------------------------------------------
This script used to append BOS_ROOT/EOS_ROOT to stream 0 ONLY (the historical writer's bug), so
stream 0 held n_k+2 entries per n_k-word sentence while streams 1..3 held n_k.  Both consumers
pair the streams BY INDEX (nrmt_train.py:375, nrmp_train.py cmd_train), so the model was fed the
prefix of a DIFFERENT word.  MEASURED on the cache it produced: the prefix read for word i agrees
with the true prefix 58.8% of the time -- exactly the agreement of a random permutation of the
same stream (58.8%), i.e. ZERO information -- and on the 20.6% of positions whose prefix is
informative it matches only 4.5% of the time.  Sanity arithmetic straight out of the emitted
manifest: val 972383 = 837575 + 2*67404, train 8311213 = 7063767 + 2*623723.

`--prefix-layout aligned` (the DEFAULT now) writes exactly one prefix id per word event.
`legacy` restores the historical bytes exactly (md5-verified: an independent rebuild in legacy
mode reproduces the shipped train.pt/val.pt md5s bit-for-bit).  The verified, dependency-free
replacement with a full alignment report is `/workspace/head_fix/nrmp_cache_align.py`.

WHY
---
The shipped cache (2026-09-30 18:52) was built when the root inventory held 9,114 entries
(ids 0..9113).  It therefore contains ZERO occurrences of the 199 roots at 9114..9312 and of
the 177 roots at 9313..9489, so those rows can never receive gradient.  The vocabulary is now
append-only 9,490 and the checkpoint has been extended to match; only the cache is stale.

THE ORIGINAL WRITER (RECOVERED, NOT REINVENTED)
-----------------------------------------------
/workspace/hf_v19_2_release/nrmp_train.py  md5 c24675c52d13cd329ab27813b862fef5
  == rootformer/build/nrmp_train.py       md5 c24675c52d13cd329ab27813b862fef5   (bit-identical)

Its `--prepare` (cmd_prepare, lines 86..141) IS the writer.  There is no external split file:
the split is DERIVED deterministically from
    CORPORA (7 dirs, in order) -> sentences_from(min_words=4, max_words=64, limit=per_corpus)
    -> de-duplicate identical sentences (first occurrence wins, order preserved)
    -> files = sorted({source file})          # vocab-independent
    -> random.Random(1337).shuffle(files)
    -> n_val = max(1, int(len(files) * 0.08)); val_files = set(files[:n_val])
    -> per sentence: BOS_ROOT to stream 0; then (prefix, root, wazn, suffix) appended to
       streams 0,1,2,3; then EOS_ROOT to stream 0
    -> torch.save([torch.tensor(x, dtype=torch.int32) for x in streams[split]])

This script imports that module and calls its own sentences_from/build_vocab, so the split is
reproduced by the original code rather than re-implemented.

LAYOUT / DTYPE (verified against the consumer)
----------------------------------------------
Consumer: nrmt_train.py:312-313
    tr4 = [t.long() for t in torch.load(Path(args.cache) / 'train.pt', map_location='cpu')]
    ...
    P, R, W, S = streams           # index 0=prefix, 1=root, 2=wazn, 3=suffix
A plain list of exactly 4 int32 tensors, saved with plain torch.save.  NO window/packing step
is stored: the cache is a FLAT per-word stream.  Windowing happens in the reader
(WIN, STRIDE, CTX = 128, 64, 12 at nrmt_train.py:48; starts_for() picks window offsets and
extract_backbone() slices t[j:j+WIN]).  CTX=12 is a 12-root *novelty* context, not a packed
record, so nothing about 13-grams is persisted here.

TOKENIZER MODE
--------------
The shipped cache predates models/validated_segmentation.py (created 2026-10-01 17:14; the
tokenizer was patched 2026-10-01 14:36) and the module's own docstring states the cache was
built with the greedy analyzer.  We therefore default to ROOTFORMER_VALIDATED_SEG=0 so that the
ONLY difference from the shipped cache is vocabulary coverage, keeping the two comparable.
Pass --seg-mode 1 to build the alternative stream instead.

Outputs (never overwrites /workspace/nrmp_cache):
    <out-dir>/train.pt, <out-dir>/val.pt, <out-dir>/manifest_9490.json
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

RELEASE = Path(os.environ.get('NRMP_RELEASE', '/workspace/hf_v19_2_release'))
SHIPPED = Path('/workspace/nrmp_cache')


def md5(path, chunk=1 << 20):
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
    ap.add_argument('--out-dir', default='/workspace/nrmp_cache_9490')
    ap.add_argument('--per-corpus', type=int, default=120000,
                    help='must match the original run; original default is 120000')
    ap.add_argument('--seg-mode', default='0', choices=['0', '1', '2'],
                    help='ROOTFORMER_VALIDATED_SEG; 0 = greedy (matches the shipped cache)')
    ap.add_argument('--prefix-layout', default='aligned', choices=['legacy', 'aligned', 'all'],
                    help='stream-0 layout.  aligned (default, FIXED): exactly one prefix id per '
                         'word event, so P[i]/R[i]/W[i]/S[i] pair by index as both consumers '
                         'assume.  legacy: reproduce the historical BOS/EOS-into-stream-0-only '
                         'writer exactly (n_k+2 entries per n_k-word sentence).  all: BOS/EOS '
                         'into all four streams (PAD in wazn/suffix).')
    ap.add_argument('--min-words', type=int, default=4)
    ap.add_argument('--max-words', type=int, default=64)
    ap.add_argument('--split-only', action='store_true',
                    help='compute and print the split, write nothing')
    ap.add_argument('--compare-shipped', action='store_true',
                    help='also report element-wise agreement with the shipped cache')
    ap.add_argument('--limit-train', type=int, default=0,
                    help='smoke test: truncate the train stream to N word events')
    args = ap.parse_args()

    # MUST be set before the vocab/tokenizer is constructed.  mode() reads it lazily anyway.
    os.environ['ROOTFORMER_VALIDATED_SEG'] = args.seg_mode
    os.environ.setdefault('ROOTFORMER_VALIDATED_SEG', '0')

    sys.path.insert(0, str(RELEASE))
    sys.path.insert(0, str(RELEASE / 'models'))
    import torch
    import nrmp_train as NP          # the recovered original writer

    assert Path(NP.__file__).resolve().parent == RELEASE, NP.__file__

    print(f'[*] release          : {RELEASE}')
    print(f'[*] original writer  : {Path(NP.__file__)}  md5 {md5(Path(NP.__file__))}')
    print(f'[*] ROOTFORMER_VALIDATED_SEG={os.environ["ROOTFORMER_VALIDATED_SEG"]} '
          f'(0 = greedy, matches shipped cache)')

    vocab = NP.build_vocab()
    print(f'[*] live vocab       : num_roots={vocab.num_roots} '
          f'prefixes={vocab.num_prefixes} awzan={vocab.num_awzan} suffixes={vocab.num_suffixes}')
    print(f'[*] PAD/BOS/EOS/UNK  : {vocab.PAD_ROOT}/{vocab.BOS_ROOT}/{vocab.EOS_ROOT}/{vocab.UNK_ROOT} '
          f'| <PARTICLE>={vocab.root2id["<PARTICLE>"]}')

    # ---- the split, produced by the original code path -------------------------------------
    t0 = time.time()
    all_pairs = []
    for c in NP.CORPORA:
        if not Path(c).exists():
            print(f'  [skip missing] {c}')
            continue
        got = NP.sentences_from(c, min_words=args.min_words, max_words=args.max_words,
                                limit=args.per_corpus)
        print(f'  {c}: {len(got)} sentences')
        all_pairs += got
    seen, uniq = set(), []
    for f, s in all_pairs:
        if s in seen:
            continue
        seen.add(s)
        uniq.append((f, s))
    files = sorted({f for f, _ in uniq})
    random.Random(NP.SEED).shuffle(files)
    n_val = max(1, int(len(files) * 0.08))
    val_files = set(files[:n_val])
    print(f'[*] seed={NP.SEED} | unique sentences={len(uniq)} | files={len(files)} | '
          f'val files={len(val_files)}  ({100*len(val_files)/max(len(files),1):.2f}%)  '
          f'[{time.time()-t0:.0f}s]')

    split_rule = {
        'recovered_from': str(Path(NP.__file__)),
        'writer_md5': md5(Path(NP.__file__)),
        'corpora': NP.CORPORA,
        'seed': NP.SEED,
        'shuffle': 'random.Random(SEED).shuffle(sorted(files))',
        'val_fraction': 0.08,
        'n_val_files': n_val,
        'n_files': len(files),
        'sentence_filter': {'min_words': args.min_words, 'max_words': args.max_words,
                            'split_regex': r'[\n.!?؟]+', 'per_corpus_limit': args.per_corpus},
        'dedup': 'identical sentence strings, first occurrence wins',
        'level': 'FILE (no file straddles the split)',
        'seg_mode': args.seg_mode,
    }
    if args.split_only:
        split_rule['val_files'] = sorted(val_files)
        print(json.dumps(split_rule, ensure_ascii=False, indent=2))
        return

    # ---- encode -----------------------------------------------------------------------------
    streams = {'train': [[], [], [], []], 'val': [[], [], [], []]}
    special_root = {vocab.PAD_ROOT, vocab.BOS_ROOT, vocab.EOS_ROOT,
                    vocab.UNK_ROOT, vocab.root2id['<PARTICLE>']}
    n_words = {'train': 0, 'val': 0}
    n_junk = {'train': 0, 'val': 0}
    n_sent = {'train': 0, 'val': 0}
    n_skipped = {'train': 0, 'val': 0}
    t0 = time.time()
    for i, (f, s) in enumerate(uniq):
        split = 'val' if f in val_files else 'train'
        enc = vocab.encode_sentence(s)
        if len(enc) < 2:
            n_skipped[split] += 1
            continue
        st = streams[split]
        n_sent[split] += 1
        if args.prefix_layout == 'legacy':
            st[0].append(vocab.BOS_ROOT)          # historical: stream 0 ONLY (the bug)
        elif args.prefix_layout == 'all':
            st[0].append(vocab.BOS_ROOT)
            st[1].append(vocab.BOS_ROOT)
            st[2].append(vocab.PAD_WAZN)
            st[3].append(vocab.PAD_SUFFIX)
        for (p, r, w, sf) in enc:
            st[0].append(p)
            st[1].append(r)
            st[2].append(w)
            st[3].append(sf)
            n_words[split] += 1
            if r in special_root:
                n_junk[split] += 1
        if args.prefix_layout == 'legacy':
            st[0].append(vocab.EOS_ROOT)
        elif args.prefix_layout == 'all':
            st[0].append(vocab.EOS_ROOT)
            st[1].append(vocab.EOS_ROOT)
            st[2].append(vocab.PAD_WAZN)
            st[3].append(vocab.PAD_SUFFIX)
        if i and i % 50000 == 0:
            print(f'    {i}/{len(uniq)} sentences  train_words={n_words["train"]} '
                  f'val_words={n_words["val"]}  [{time.time()-t0:.0f}s]', flush=True)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = dict(split_rule)
    manifest['val_files'] = sorted(val_files)
    manifest['counts'] = {}
    manifest['outputs'] = {}

    for split in ('train', 'val'):
        if split == 'train' and args.limit_train:
            streams[split] = [x[:args.limit_train] for x in streams[split]]
        tens = [torch.tensor(x, dtype=torch.int32) for x in streams[split]]
        dst = out / f'{split}.pt'
        torch.save(tens, dst)
        root = tens[1]
        ids = torch.unique(root)
        n_new = int((root >= 9313).sum())
        n_mid = int(((root >= 9114) & (root < 9313)).sum())
        stats = {
            'word_events': n_words[split],
            'sentences': n_sent[split],
            'skipped_short_sentences': n_skipped[split],
            'junk_root_pct': round(100 * n_junk[split] / max(n_words[split], 1), 3),
            'shapes': [list(t.shape) for t in tens],
            'dtypes': [str(t.dtype) for t in tens],
            'root_max': int(root.max()),
            'root_min': int(root.min()),
            'distinct_root_ids': int(ids.numel()),
            'occurrences_ge_9114': n_mid,
            'occurrences_ge_9313': n_new,
            'distinct_ids_9114_9312': int(((ids >= 9114) & (ids < 9313)).sum()),
            'distinct_ids_ge_9313': int((ids >= 9313).sum()),
            'md5': md5(dst),
            'bytes': dst.stat().st_size,
        }
        manifest['counts'][split] = stats
        manifest['outputs'][split] = str(dst)
        print(f'[*] {split}: {tens[1].numel()} word events, prefix stream {tens[0].numel()} '
              f'({n_sent[split]} sentences) -> {dst}')
        print(f'    root max={stats["root_max"]} distinct={stats["distinct_root_ids"]} | '
              f'ids>=9114: {n_mid} occ / {stats["distinct_ids_9114_9312"]} ids | '
              f'ids>=9313: {n_new} occ / {stats["distinct_ids_ge_9313"]} ids')

    if args.compare_shipped:
        print('[*] comparing against the SHIPPED cache (must stay untouched)')
        manifest['comparison_vs_shipped'] = {}
        for split in ('train', 'val'):
            ship = SHIPPED / f'{split}.pt'
            if not ship.exists():
                continue
            s4 = torch.load(ship, map_location='cpu')
            n4 = torch.load(out / f'{split}.pt', map_location='cpu')
            cmp = {'shipped_md5': md5(ship)}
            # structural agreement, then element-wise agreement on the root stream
            cmp['shipped_shapes'] = [list(t.shape) for t in s4]
            for idx, name in enumerate(('prefix', 'root', 'wazn', 'suffix')):
                a, b = s4[idx].long(), n4[idx].long()
                n = min(a.numel(), b.numel())
                eq = int((a[:n] == b[:n]).sum())
                cmp[name] = {
                    'len_shipped': int(a.numel()), 'len_rebuilt': int(b.numel()),
                    'compared': n,
                    'identical': eq,
                    'agreement_pct': round(100.0 * eq / max(n, 1), 4),
                    'shipped_max': int(a.max()), 'rebuilt_max': int(b.max()),
                }
            manifest['comparison_vs_shipped'][split] = cmp
            r = cmp['root']
            print(f'    {split}: root agreement {r["agreement_pct"]}% on {r["compared"]} '
                  f'positions (shipped len {r["len_shipped"]} vs rebuilt {r["len_rebuilt"]})')
            for name in ('prefix', 'wazn', 'suffix'):
                print(f'      {name}: {cmp[name]["agreement_pct"]}%')

    mf = out / 'manifest_9490.json'
    mf.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f'[*] manifest -> {mf}')
    print(f'[*] total {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
