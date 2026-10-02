#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_parallel_split.py -- assemble a leak-free seen/unseen Arabic->English split.

Why this is needed
------------------
The project's own bilingual material is not a benchmark:
  * its biggest "parallel" sources are LEXICONS, not sentences
    (lisan_farahidian 57,886 / semantic_anchor 7,583 / Basran_Heritage_Matrix 137,961);
  * the same work appears in several files, so pairs must be de-duplicated ACROSS files;
  * the Grand-100 evaluation set is 100% contained in training data.

What this does
--------------
1. Collects sentence-level pairs from every parallel file.
2. Drops lexicon-style sources by name AND by a form heuristic (no-space Arabic side, or an
   English side that is a bare gloss).
3. De-duplicates on (arabic, english) so a pair cannot sit in two splits via two files.
4. Holds out WHOLE WORKS for test -- never sentences from a work that also contributes training
   pairs -- so the split is leak-free at source level by construction.
5. VERIFIES the split: counts 13-gram overlap between test and train on both the Arabic and the
   English side. Anything above ~0 means the split is not clean and must be reported.

Outputs: /workspace/mt_split/{train,dev,test}.jsonl + split_report.json
"""
import argparse
import collections
import json
import pathlib
import random
import re
import sys

D = pathlib.Path('/workspace/rootformer_v12/v18_next_root_morph/data')
FILES = [
    'pure_gold_bilingual_train.jsonl', 'pure_gold_bilingual_val.jsonl',
    'unified_basran_andalusian_train.jsonl', 'unified_basran_andalusian_val.jsonl',
    'sovereign_classical_transmute_corpus.jsonl',
    'grand_neural_transmute_corpus.jsonl', 'lisan_3pillar_master_pairs.jsonl',
]
# sources that are lexical resources, not sentence translation
# Only genuinely lexical resources: root->gloss tables. NOTE: 'Heritage_Matrix' is NOT here --
# it contains real sentence translations as well as lexeme rows, and excluding it by name threw
# away ~138k legitimate pairs. The form heuristic below is what separates a gloss from a sentence.
LEXICON_PAT = re.compile(r'lisan_farahidian|semantic_anchor|master_pairs|3pillar|'
                         r'root_invariant|wazn_derivation', re.I)
AR = re.compile(r'[\u0600-\u06FF]')
ARABIC_ONLY = re.compile(r'^[\u0600-\u06FF\s]+$')


def is_sentence_pair(a, e):
    """Reject lexicon entries masquerading as sentence pairs."""
    if not a or not e:
        return False
    aw, ew = a.split(), e.split()
    if not (4 <= len(aw) <= 60 and 3 <= len(ew) <= 70):
        return False
    # a lexical gloss is a single Arabic token (or bare root) with a short English gloss
    if len(aw) == 1 and len(ew) <= 8:
        return False
    if len(aw) == 1 and len(a.strip()) <= 4:
        return False
    if ARABIC_ONLY.match(e.strip()):        # untranslated Arabic on the English side
        return False
    return True


def ngrams(tokens, n):
    return {tuple(tokens[i:i + n]) for i in range(max(0, len(tokens) - n + 1))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--test-works', nargs='*', default=None,
                    help='work names to hold out entirely (default: auto-pick)')
    ap.add_argument('--overlap-ngram', type=int, default=13)
    ap.add_argument('--out-dir', default='/workspace/mt_split')
    args = ap.parse_args()

    pairs = {}          # (ar,en) -> work
    skipped_lex = 0
    per_work = collections.Counter()
    for fn in FILES:
        p = D / fn
        if not p.exists():
            print(f'  [skip missing] {fn}')
            continue
        n = 0
        for line in open(p, encoding='utf-8', errors='ignore'):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                # DELIBERATE FALLBACK, kept: external JSONL corpus, scanned line by line.  A
                # malformed record is skipped like a blank line rather than aborting the split.
                continue
            a = (d.get('arabic') or '').strip()
            e = (d.get('english') or '').strip()
            src = str(d.get('source') or d.get('book') or 'unknown')
            if LEXICON_PAT.search(src):
                skipped_lex += 1
                continue
            if not is_sentence_pair(a, e):
                continue
            pairs.setdefault((a, e), src)
            per_work[src] += 1
            n += 1
        print(f'  {fn:<46} kept {n}')

    print(f'\n[*] distinct sentence pairs: {len(pairs)}  (lexicon rows skipped: {skipped_lex})')
    print(f'[*] distinct works: {len(per_work)}')

    # ---- pick held-out works -------------------------------------------------------------
    if args.test_works:
        test_works = set(args.test_works)
    else:
        # choose works with >=300 pairs so the test set is meaningful
        # hold out whole works until ~15% of pairs are in test, preferring small/medium works
        # so the split stays usable for training
        total = sum(per_work.values())
        target = 0.15 * total
        cand = sorted([(w, c) for w, c in per_work.items() if 300 <= c <= 20000],
                      key=lambda kv: kv[1])
        test_works, acc = set(), 0
        for w, c in cand:
            if acc + c > target * 1.25:
                continue
            test_works.add(w); acc += c
            if acc >= target:
                break
        if not test_works:
            test_works = {max(per_work.items(), key=lambda kv: kv[1])[0]}
    print(f'[*] held-out works: {sorted(test_works)}')

    train, test = [], []
    for (a, e), w in pairs.items():
        rec = {'arabic': a, 'english': e, 'work': w}
        (test if w in test_works else train).append(rec)

    random.Random(0).shuffle(train)
    random.Random(1).shuffle(test)
    dev = train[:len(train) // 20]
    train = train[len(train) // 20:]
    print(f'[*] train {len(train)} | dev {len(dev)} | test {len(test)}')

    # ---- VERIFY: 13-gram overlap, both sides --------------------------------------------
    N = args.overlap_ngram
    report = {'n_train': len(train), 'n_dev': len(dev), 'n_test': len(test),
              'test_works': sorted(test_works), 'overlap_ngram': N}
    for side in ('arabic', 'english'):
        tr = set()
        for r in train:
            tr |= ngrams(r[side].split(), N)
        hit = 0
        tot = 0
        for r in test:
            g = ngrams(r[side].split(), N)
            if not g:
                continue
            tot += len(g)
            hit += len(g & tr)
        cov = sum(1 for r in test if ngrams(r[side].split(), N) & tr)
        report[f'{side}_test_ngrams'] = tot
        report[f'{side}_test_ngrams_seen_in_train'] = hit
        report[f'{side}_test_rows_with_any_overlap'] = cov
        report[f'{side}_overlap_pct'] = 100.0 * hit / max(tot, 1)
        print(f'  {side:<8} test {N}-grams={tot} seen-in-train={hit} '
              f'({100.0*hit/max(tot,1):.3f}%)  rows affected={cov}/{len(test)}')

    # ---- CLEAN test: drop any test row that shares a 13-gram with train -------------------
    tr_ar, tr_en = set(), set()
    for r in train:
        tr_ar |= ngrams(r['arabic'].split(), N)
        tr_en |= ngrams(r['english'].split(), N)
    clean = [r for r in test
             if not (ngrams(r['arabic'].split(), N) & tr_ar)
             and not (ngrams(r['english'].split(), N) & tr_en)]
    report['n_test_clean'] = len(clean)
    report['n_test_dropped_for_overlap'] = len(test) - len(clean)
    print(f'[*] CLEAN test (zero {N}-gram overlap with train): {len(clean)} rows '
          f'({len(test)-len(clean)} dropped)')

    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (('train', train), ('dev', dev), ('test', test), ('test_clean', clean)):
        with open(out / f'{name}.jsonl', 'w', encoding='utf-8') as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
    # also a plain text version for sacrebleu-style tooling
    with open(out / 'test_clean.ar', 'w', encoding='utf-8') as fa, \
         open(out / 'test_clean.en', 'w', encoding='utf-8') as fe:
        for r in clean:
            fa.write(r['arabic'].replace('\n', ' ') + '\n')
            fe.write(r['english'].replace('\n', ' ') + '\n')
    json.dump(report, open(out / 'split_report.json', 'w'), indent=2)
    print(f'\nwrote {out}/ (train/dev/test.jsonl, test.ar, test.en, split_report.json)')
    print('A test-ngram overlap of 0.000% means the held-out works genuinely never occur in train.')


if __name__ == '__main__':
    main()
