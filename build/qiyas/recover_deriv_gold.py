#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
recover_deriv_gold.py -- recover the EXACT gold surface words of the 3,588 held-out-root
target positions scored in scratch_comp/RESULTS_COMP.md.

WHY THIS IS POSSIBLE WITHOUT THE STREAM
---------------------------------------
`streams.pt` stores only the tuple ids, not the surface words, so the words behind the
3,588 positions are not in any artifact.  But the split is deterministic and reconstructible
from `meta.json`:

  * `meta.json['test_files']` names the 16 held-out works (the split was made on basenames);
  * test_deriv = every sentence of those works that contains a held-out root, min 12 chars,
    >= 4 Arabic words, SPLIT on [\\n.!?؟؛:], deduplicated by diacritic-stripped text
    (exactly `sentences_by_file()` in scratch_lm_comp.py);
  * test_deriv had 3,627 sentences and the cap was 4,000, so NOTHING was truncated --
    sentence ORDER is irrelevant, only the set matters.

Then every word of those sentences whose encoded root is one of the 37 held-out roots is a
gold position.  The independent check is the MULTISET OF ROOTS: the per-root counts must
reproduce the histogram of `holdout_records` in comp24k_eval.json (كره 243, صبح 178, ...).
If they match, the recovered words ARE the gold surfaces.

Writes /workspace/qiyas/deriv_gold.json.  Reads only; writes only inside /workspace/qiyas/.
"""

from __future__ import annotations

import collections
import glob
import hashlib
import json
import os
import re
import sys
import time

sys.path.insert(0, '/workspace/hf_v19_2_release')
sys.path.insert(0, '/workspace/hf_v19_2_release/models')
sys.path.insert(0, '/workspace/qiyas')

OUT = '/workspace/qiyas'
AR = re.compile(r'[\u0600-\u06FF]')
DIAC = re.compile(r'[\u064b-\u0652\u0670\u0640]')
SPLIT = re.compile(r'[\n.!?؟؛:]+')

SOURCES = [
    '/workspace/scholastic_sanitized',
    '/workspace/andalusian_canon_sanitized',
    '/workspace/heritage_foundations',
    '/workspace/rootformer_v12/v18_next_root_morph/data',
    '/workspace/rootformer_v12/raw_translations',
    '/workspace/rootformer_v12/v17_deepseek_flash/data',
]


def log(*a):
    print(*a, flush=True)


def norm(s):
    return DIAC.sub('', s).strip()


def find_source(basename):
    hits = []
    for root in SOURCES:
        if not os.path.isdir(root):
            continue
        for ext in ('*.txt', '*.jsonl', '*.json'):
            for p in glob.glob(os.path.join(root, '**', ext), recursive=True):
                if os.path.basename(p) == basename:
                    hits.append(p)
    return hits


def chunks_of(path):
    if path.endswith('.jsonl'):
        out = []
        for line in open(path, encoding='utf-8', errors='ignore'):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            for k in ('arabic', 'text', 'ar'):
                if isinstance(d.get(k), str):
                    out.append(d[k])
                    break
        return out
    if path.endswith('.json'):
        d = json.load(open(path, encoding='utf-8', errors='ignore'))
        out = []

        def rec(x):
            if isinstance(x, str):
                if AR.search(x):
                    out.append(x)
            elif isinstance(x, dict):
                hit = False
                for k in ('arabic', 'text', 'ar', 'content'):
                    if isinstance(x.get(k), str):
                        out.append(x[k])
                        hit = True
                        break
                if not hit:
                    for v in x.values():
                        rec(v)
            elif isinstance(x, list):
                for v in x[:5000]:
                    rec(v)
        rec(d)
        return out
    return [open(path, encoding='utf-8', errors='ignore').read()]


def iter_all_sentences():
    """VERBATIM replication of scratch_lm_comp.sentences_by_file(): same SOURCES order, same
    per-root extension order, same sorted glob, same 400 MB skip, same filters.  The
    consumer deduplicates GLOBALLY in this order, so a sentence that also occurs in an
    earlier work is dropped from a later one -- which is why restricting the walk to the 16
    test works alone over-selects (6,651 instead of 3,627 sentences)."""
    for root in SOURCES:
        if not os.path.isdir(root):
            continue
        for ext in ('*.txt', '*.jsonl', '*.json'):
            for p in sorted(glob.glob(os.path.join(root, '**', ext), recursive=True)):
                if os.path.getsize(p) > 400 * 1024 * 1024:
                    continue
                key = os.path.basename(p)
                for ch in chunks_of(p):
                    for s in SPLIT.split(ch):
                        s = s.strip()
                        if len(s) < 12 or not AR.search(s):
                            continue
                        ws = [w for w in s.split() if AR.search(w)]
                        if len(ws) < 4:
                            continue
                        yield key, ' '.join(ws)


def main():
    os.makedirs(OUT, exist_ok=True)
    meta = json.load(open('/workspace/scratch_comp/sf_data_9490/meta.json', encoding='utf-8'))
    hold = set(meta['hold_roots_str'] if 'hold_roots_str' in meta else
               [meta['roots_list'][i] for i in meta['hold_roots']])
    test_files = set(meta['test_files'])
    log('[*] %d held-out roots, %d test files' % (len(hold), len(test_files)))

    from nrmp_vocab import FarāhīdianMorphemicVocab as V   # noqa: N806
    vocab = V('/workspace/hf_v19_2_release/data/rootformer_v12_arabic_blueprint.json')
    log('[*] vocab roots=%d awzan=%d' % (vocab.num_roots, vocab.num_awzan))

    # global dedup in walk order, keeping only the test works' sentences
    seen = set()
    keep = []
    t0 = time.time()
    n = 0
    for key, s in iter_all_sentences():
        n += 1
        h = hashlib.blake2b(norm(s).encode('utf-8'), digest_size=16).digest()
        if h in seen:
            continue
        seen.add(h)
        if key in test_files:
            keep.append(s)
        if n % 500000 == 0:
            log('    walk %d sentences (%.0fs) kept=%d' % (n, time.time() - t0, len(keep)))
    log('[*] walked %d sentences, %d unique, %d kept from the 16 test works (%.0fs)'
        % (n, len(seen), len(keep), time.time() - t0))

    hold_ids = {vocab.root2id[r] for r in hold if r in vocab.root2id}
    deriv = []
    for s in keep:
        try:
            e = vocab.encode_sentence(s)
        except Exception:
            continue
        if not e:
            continue
        if {t[1] for t in e} & hold_ids:
            deriv.append(s)
    log('[*] sentences containing a held-out root: %d  (RESULTS_COMP.md says 3,627)'
        % len(deriv))

    # TARGET POSITIONS.  eval_heads builds r_tg = R[1:] with r_in = R[:-1], so the model is
    # asked about every word EXCEPT THE FIRST of each sentence.  35,820 - 3,627 = 32,193
    # target positions, and 3,987 held-out occurrences minus the sentence-initial ones must
    # give the 3,588 the report scored.
    words = []
    hist = collections.Counter()
    for s in deriv:
        toks = [w for w in s.split() if AR.search(w)]
        for ti, w in enumerate(toks):
            try:
                p, r, wz, sfx = vocab.encode_word(w)
            except Exception:
                continue
            if r not in hold_ids:
                continue
            rec = {'word': w, 'root': vocab.id2root[r], 'wazn': vocab.id2wazn[wz],
                   'prefix': vocab.id2prefix[p], 'suffix': vocab.id2suffix[sfx],
                   'sentence_initial': (ti == 0), 'tok_index': ti}
            words.append(rec)
            if ti != 0:
                hist[vocab.id2root[r]] += 1
    n_all = len(words)
    n_initial = sum(1 for x in words if x['sentence_initial'])
    log('[*] held-out-root OCCURRENCES: %d (audit_comp.json says 3,987); '
        'sentence-initial: %d; TARGET positions: %d'
        % (n_all, n_initial, n_all - n_initial))
    words = [x for x in words if not x['sentence_initial']]

    # the independent check: the recorded histogram
    rec = json.load(open('/workspace/scratch_comp/sf_data_9490/comp24k_eval.json',
                         encoding='utf-8'))
    rec_hist = collections.Counter(x['true'] for x in rec['holdout_records'])
    ok = (len(words) == 3588) and (dict(hist) == dict(rec_hist))
    diff = {k: (hist.get(k, 0), rec_hist.get(k, 0)) for k in set(hist) | set(rec_hist)
            if hist.get(k, 0) != rec_hist.get(k, 0)}
    log('[*] recovered %d gold positions; RESULTS_COMP.md scored 3,588' % len(words))
    log('[*] histogram matches the recorded holdout_records: %s' % ok)
    if diff:
        log('    differing roots (recovered, recorded): %s' % diff)

    json.dump({
        'n': len(words),
        'n_expected': 3588,
        'n_sentences': len(deriv),
        'n_sentences_expected': 3627,
        'histogram_matches_recorded': bool(ok),
        'histogram_diff': diff,
        'recorded_histogram': dict(rec_hist),
        'recovered_histogram': dict(hist),
        'positions': words,
    }, open(os.path.join(OUT, 'deriv_gold.json'), 'w', encoding='utf-8'),
        ensure_ascii=False, indent=1)
    log('[*] wrote %s/deriv_gold.json' % OUT)


if __name__ == '__main__':
    main()
