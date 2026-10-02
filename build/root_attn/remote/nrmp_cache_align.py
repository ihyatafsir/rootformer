#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nrmp_cache_align.py -- VERIFY and FIX the prefix-stream index misalignment of the NRMP cache.

THE DEFECT (writer: /workspace/hf_v19_2_release/nrmp_train.py cmd_prepare, lines 124/133;
reproduced verbatim in /workspace/nrmp_rebuild_cache.py lines 181/190):

    st[0].append(vocab.BOS_ROOT)          # <- stream 0 ONLY, once per sentence
    for (p, r, w, sf) in enc:
        st[0].append(p)                   # prefix
        st[1].append(r); st[2].append(w); st[3].append(sf)
    st[0].append(vocab.EOS_ROOT)          # <- stream 0 ONLY, once per sentence

Stream 0 gets n_k + 2 entries for a sentence of n_k words; streams 1..3 get n_k.  BOTH
consumers pair the streams BY INDEX:

    nrmt_train.py:375   Tr, Wr, Pr, Sr = (win_tensor(tr4[i], tr_st) for i in (1, 2, 0, 3))
    nrmp_train.py:262   tr = chunk(load_stream('train'), args.seq_len, device)   # all 4 sliced

so the model is fed (prefix of a DIFFERENT word, root_t, wazn_t, suffix_t).  After the first
sentence, stream 0 runs 2*sentence_index positions ahead of streams 1..3.

SELF-EVIDENCE (the writer's own manifest, /workspace/nrmp_cache_9490/manifest_9490.json):
    val  : 972383 = 837575 + 2*67404        train: 8311213 = 7063767 + 2*623723
exactly 2 extra stream-0 entries per sentence, the extra count equalling `sentences`.

MODES
  default (no --save-cache) : reproduce the first --verify-words word events of each split
                              independently from the corpus, check element-wise equality with
                              --compare-cache (proves the reconstruction), then quantify the
                              misalignment.
  --save-cache              : additionally write the full rebuilt cache for --prefix-layout.

--prefix-layout
  legacy    BOS/EOS into stream 0 only            (byte-identical reproduction of the bug)
  aligned   ONE prefix id per word, no markers    (THE FIX; default)
  all       BOS/EOS into all four streams; PAD in wazn/suffix at the marker rows

Read-only w.r.t. every existing module and w.r.t. /workspace/nrmp_cache_9490.
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

LAYOUTS = ('legacy', 'aligned', 'all')


def md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, 'rb') as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def build_split(NP, per_corpus, min_words, max_words):
    """Derive the train/val split with the ORIGINAL writer's own code path."""
    all_pairs = []
    for c in NP.CORPORA:
        if not Path(c).exists():
            print(f'  [skip missing] {c}')
            continue
        got = NP.sentences_from(c, min_words=min_words, max_words=max_words, limit=per_corpus)
        print(f'  {c}: {len(got)} sentences', flush=True)
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
    return uniq, val_files, len(files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--release', default='/workspace/hf_v19_2_release')
    ap.add_argument('--out-dir', default='/workspace/head_fix/nrmp_cache_9490_aligned')
    ap.add_argument('--prefix-layout', default='aligned', choices=list(LAYOUTS))
    ap.add_argument('--per-corpus', type=int, default=120000)
    ap.add_argument('--min-words', type=int, default=4)
    ap.add_argument('--max-words', type=int, default=64)
    ap.add_argument('--seg-mode', default='0', choices=['0', '1', '2'])
    ap.add_argument('--verify-words', type=int, default=20000,
                    help='independently reproduce and check this many word events per split '
                         '(0 = skip the reconstruction check)')
    ap.add_argument('--compare-cache', default='/workspace/nrmp_cache_9490')
    ap.add_argument('--save-cache', action='store_true',
                    help='write the full train.pt/val.pt for --prefix-layout to --out-dir')
    ap.add_argument('--report', default='/workspace/head_fix/cache_alignment_report.json')
    args = ap.parse_args()

    os.environ['ROOTFORMER_VALIDATED_SEG'] = args.seg_mode
    release = Path(args.release)
    sys.path.insert(0, str(release))
    sys.path.insert(0, str(release / 'models'))
    import torch

    t0 = time.time()
    import nrmp_train as NP
    assert Path(NP.__file__).resolve().parent == release, NP.__file__
    uniq, val_files, n_files = build_split(NP, args.per_corpus, args.min_words, args.max_words)
    vocab = NP.build_vocab()

    REP = {
        'release': str(release),
        'writer': str(Path(NP.__file__)),
        'writer_md5': md5(Path(NP.__file__)),
        'bug': {
            'file': 'nrmp_train.py (release) / nrmp_rebuild_cache.py',
            'line_bos': 124, 'line_eos': 133,
            'code': 'st[0].append(vocab.BOS_ROOT) / st[0].append(vocab.EOS_ROOT)',
            'mechanism': 'stream 0 only -> n_k+2 entries per n_k-word sentence; consumers pair '
                         'P[i] with R[i], W[i], S[i] by index',
        },
        'split': {'seed': NP.SEED, 'n_uniq_sentences': len(uniq), 'n_files': n_files,
                  'n_val_files': len(val_files), 'per_corpus': args.per_corpus,
                  'min_words': args.min_words, 'max_words': args.max_words,
                  'seg_mode': args.seg_mode},
        'vocab': {'num_roots': vocab.num_roots, 'num_awzan': vocab.num_awzan,
                  'num_prefixes': vocab.num_prefixes, 'num_suffixes': vocab.num_suffixes,
                  'PAD_ROOT': vocab.PAD_ROOT, 'BOS_ROOT': vocab.BOS_ROOT,
                  'EOS_ROOT': vocab.EOS_ROOT, 'UNK_ROOT': vocab.UNK_ROOT,
                  'NONE_PREFIX': vocab.NONE_PREFIX,
                  'PAD_WAZN': vocab.PAD_WAZN, 'PAD_SUFFIX': vocab.PAD_SUFFIX},
        'prefix_layout': args.prefix_layout,
    }
    print(f'[*] PAD/BOS/EOS/UNK = {vocab.PAD_ROOT}/{vocab.BOS_ROOT}/{vocab.EOS_ROOT}/'
          f'{vocab.UNK_ROOT}  NONE_PREFIX={vocab.NONE_PREFIX}  '
          f'num_prefixes={vocab.num_prefixes} num_roots={vocab.num_roots} '
          f'num_awzan={vocab.num_awzan} num_suffixes={vocab.num_suffixes}')

    # ---- self-evidence straight from the shipped manifest ---------------------------------
    ship_mf = Path(args.compare_cache) / 'manifest_9490.json'
    if ship_mf.exists():
        m = json.loads(ship_mf.read_text())['counts']
        ev = {}
        for sp in ('train', 'val'):
            L0, L1 = m[sp]['shapes'][0][0], m[sp]['shapes'][1][0]
            nw, ns = m[sp]['word_events'], m[sp]['sentences']
            ev[sp] = {'prefix_stream_len': L0, 'root_stream_len': L1, 'diff': L0 - L1,
                      'word_events': nw, 'sentences': ns,
                      'diff_equals_2_sentences': (L0 - L1) == 2 * ns,
                      'prefix_len_equals_words_plus_2sents': L0 == nw + 2 * ns}
            print(f'[*] manifest {sp}: prefix {L0} - root {L1} = {L0-L1}  '
                  f'2*sentences = {2*ns}  -> {ev[sp]["diff_equals_2_sentences"]}')
        REP['manifest_self_evidence'] = ev

    # ---- encode --------------------------------------------------------------------------
    NV = args.verify_words
    want = args.prefix_layout
    streams = {sp: [[], [], [], []] for sp in ('train', 'val')} if args.save_cache else None
    n_words = {'train': 0, 'val': 0}
    n_sent = {'train': 0, 'val': 0}

    ver = {sp: {'legacy0': [], 'src': [], 'true_p': [], 'true_t': [], 'surface': [],
                'n_sents_seen': 0} for sp in ('train', 'val')}
    stop_after = NV if (NV > 0 and not args.save_cache) else 0

    print(f'[*] encoding {len(uniq)} sentences (seg_mode={args.seg_mode}, '
          f'save_cache={args.save_cache}) ...', flush=True)
    for i, (f, s) in enumerate(uniq):
        split = 'val' if f in val_files else 'train'
        enc = vocab.encode_sentence(s)
        if len(enc) < 2:
            continue
        n_sent[split] += 1
        sent_ord = n_sent[split] - 1
        surfaces = s.split()
        v = ver[split]
        n = len(enc)
        # (a) LEGACY bookkeeping -- always the buggy layout, independent of `want`, because it
        #     is what gets compared against the shipped cache.
        if len(v['legacy0']) < NV:
            v['legacy0'].append(vocab.BOS_ROOT)
            v['src'].append(('marker', sent_ord))
        # (b) chosen layout: opening marker
        if streams is not None:
            if want in ('legacy', 'all'):
                streams[split][0].append(vocab.BOS_ROOT)
            if want == 'all':
                streams[split][1].append(vocab.BOS_ROOT)
                streams[split][2].append(vocab.PAD_WAZN)
                streams[split][3].append(vocab.PAD_SUFFIX)
        for j, (p, r, w, sf) in enumerate(enc):
            wi = n_words[split] + j
            if streams is not None:
                streams[split][0].append(p)
                streams[split][1].append(r)
                streams[split][2].append(w)
                streams[split][3].append(sf)
            if len(v['legacy0']) < NV:
                v['legacy0'].append(p)
                v['src'].append(('word', wi))
            if wi < NV:
                v['true_p'].append(p)
                v['true_t'].append((r, w, sf))
                v['surface'].append(surfaces[j] if j < len(surfaces) else '')
        # (a') LEGACY bookkeeping: closing marker
        if len(v['legacy0']) < NV:
            v['legacy0'].append(vocab.EOS_ROOT)
            v['src'].append(('marker', sent_ord))
        # (b') chosen layout: closing marker
        if streams is not None:
            if want in ('legacy', 'all'):
                streams[split][0].append(vocab.EOS_ROOT)
            if want == 'all':
                streams[split][1].append(vocab.EOS_ROOT)
                streams[split][2].append(vocab.PAD_WAZN)
                streams[split][3].append(vocab.PAD_SUFFIX)
        n_words[split] += n
        v['n_sents_seen'] = n_sent[split]
        if stop_after and n_words['train'] >= NV and n_words['val'] >= NV:
            print(f'    reached {NV} word events per split at sentence {i} '
                  f'[{time.time()-t0:.0f}s]', flush=True)
            break
        if i and i % 50000 == 0:
            print(f'    {i}/{len(uniq)} train_words={n_words["train"]} '
                  f'val_words={n_words["val"]} [{time.time()-t0:.0f}s]', flush=True)

    REP['full_counts'] = {sp: {'word_events': n_words[sp], 'sentences': n_sent[sp],
                               'complete': not stop_after} for sp in ('train', 'val')}
    print(f'[*] encoded word events: {REP["full_counts"]}')

    # ---- verification: reconstruction vs shipped cache ------------------------------------
    vrep = {}
    for sp in ('train', 'val'):
        v = ver[sp]
        n = min(NV, len(v['legacy0']), len(v['true_p']))
        entry = {'n_checked': n, 'sents_covered': v['n_sents_seen']}
        if n:
            legacy0 = v['legacy0'][:n]
            tp = v['true_p'][:n]
            # exact-prefix agreement (no decode dependence)
            eq = sum(1 for a, b in zip(legacy0, tp) if a == b)
            entry['consumer_prefix_equals_true_prefix_pct'] = 100.0 * eq / n
            # majority-prefix baseline: how much of that agreement is just NONE_PREFIX?
            from collections import Counter
            cnt = Counter(tp)
            maj, majk = cnt.most_common(1)[0]
            entry['majority_true_prefix'] = maj
            entry['majority_true_prefix_pct'] = 100.0 * majk / n
            # null: agreement of a random permutation of the same multiset
            import random as _r
            sh = list(legacy0)
            _r.Random(1234).shuffle(sh)
            entry['null_shuffled_agreement_pct'] = \
                100.0 * sum(1 for a, b in zip(sh, tp) if a == b) / n
            # conditional on the TRUE prefix being informative (!= NONE_PREFIX)
            idx = [i for i in range(n) if tp[i] != vocab.NONE_PREFIX]
            entry['n_positions_with_informative_true_prefix'] = len(idx)
            entry['informative_true_prefix_pct'] = 100.0 * len(idx) / n
            entry['consumer_matches_true_pct_WHEN_informative'] = (
                100.0 * sum(1 for i in idx if legacy0[i] == tp[i]) / len(idx)) if idx else None
            # how often is the prefix the model reads itself informative?
            entry['consumer_prefix_informative_pct'] = \
                100.0 * sum(1 for x in legacy0 if x != vocab.NONE_PREFIX) / n
            marker = lambda x: x in (vocab.BOS_ROOT, vocab.EOS_ROOT)
            entry['consumer_value_in_BOS_or_EOS_set_pct'] = \
                100.0 * sum(1 for x in legacy0 if marker(x)) / n
            d = [abs(i - src[1]) for i, src in enumerate(v['src'][:n]) if src[0] == 'word']
            entry['mean_source_word_distance'] = (sum(d) / len(d)) if d else None
            entry['max_source_word_distance'] = max(d) if d else None
            entry['source_is_marker_pct'] = \
                100.0 * sum(1 for x in v['src'][:n] if x[0] == 'marker') / n
            # decode round trip: does (prefix, r, w, s) reconstruct the surface word?
            ok_true = 0
            ok_legacy = 0
            for i in range(n):
                r, w, sf = v['true_t'][i]
                surf = v['surface'][i]
                if not surf:
                    continue
                try:
                    if vocab.decode_word(v['true_p'][i], r, w, sf) == surf:
                        ok_true += 1
                    if vocab.decode_word(legacy0[i], r, w, sf) == surf:
                        ok_legacy += 1
                except Exception:
                    pass
            entry['decode_roundtrip_aligned_pct'] = 100.0 * ok_true / n
            entry['decode_roundtrip_legacy_consumer_pct'] = 100.0 * ok_legacy / n
        vrep[sp] = entry
        print(f'[*] {sp}: ' + json.dumps(entry))

        # byte-level comparison with the shipped cache, if present
        ship = Path(args.compare_cache) / f'{sp}.pt'
        if ship.exists() and n:
            s4 = torch.load(ship, map_location='cpu')
            cmp = {}
            for idx, name in enumerate(('prefix', 'root', 'wazn', 'suffix')):
                a = s4[idx].long()
                k = min(n, int(a.numel()))
                if idx == 0:
                    b = torch.tensor(v['legacy0'][:k], dtype=torch.long)
                    same = int((a[:k] == b).sum())
                    cmp['shipped_prefix_vs_reproduced_legacy_pct'] = 100.0 * same / max(k, 1)
                    t = torch.tensor(v['true_p'][:k], dtype=torch.long)
                    cmp['shipped_prefix_vs_true_prefix_pct'] = \
                        100.0 * int((a[:k] == t).sum()) / max(k, 1)
                else:
                    col = {1: 'true_r', 2: 'true_w', 3: 'true_s'}[idx]
                    if col == 'true_r':
                        b = torch.tensor([t[0] for t in v['true_t'][:k]], dtype=torch.long)
                    elif col == 'true_w':
                        b = torch.tensor([t[1] for t in v['true_t'][:k]], dtype=torch.long)
                    else:
                        b = torch.tensor([t[2] for t in v['true_t'][:k]], dtype=torch.long)
                    cmp[f'shipped_{name}_vs_reproduced_pct'] = \
                        100.0 * int((a[:k] == b).sum()) / max(k, 1)
            vrep[sp]['shipped_cache_comparison'] = cmp
            print(f'    shipped-cache comparison: ' + json.dumps(cmp))
    REP['verification'] = vrep

    # ---- write the rebuilt cache ---------------------------------------------------------
    if args.save_cache:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        REP['outputs'] = {}
        for sp in ('train', 'val'):
            tens = [torch.tensor(x, dtype=torch.int32) for x in streams[sp]]
            dst = out / f'{sp}.pt'
            torch.save(tens, dst)
            lens = [int(t.numel()) for t in tens]
            REP['outputs'][sp] = {
                'path': str(dst), 'md5': md5(dst), 'bytes': dst.stat().st_size,
                'lengths': lens, 'all_four_streams_equal_length': len(set(lens)) == 1,
                'word_events': n_words[sp], 'sentences': n_sent[sp],
                'root_max': int(tens[1].max()), 'root_min': int(tens[1].min()),
                'distinct_root_ids': int(torch.unique(tens[1]).numel()),
            }
            print(f'[*] wrote {dst}: lengths={lens} '
                  f'aligned={len(set(lens)) == 1}')
        REP['manifest'] = {
            'split': REP['split'], 'vocab': REP['vocab'], 'prefix_layout': want,
            'counts': REP['outputs'], 'writer': REP['writer'], 'writer_md5': REP['writer_md5'],
            'fix': 'stream 0 = one prefix id per word event; no BOS/EOS markers in stream 0 '
                   '(they collided with genuine prefix ids %d/%d)' % (vocab.BOS_ROOT,
                                                                     vocab.EOS_ROOT),
        }
        (out / 'manifest_aligned.json').write_text(
            json.dumps(REP['manifest'], ensure_ascii=False, indent=2))
        print(f'[*] manifest -> {out / "manifest_aligned.json"}')

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(REP, ensure_ascii=False, indent=2))
    print(f'[*] report -> {args.report}')
    print(f'[*] total {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
