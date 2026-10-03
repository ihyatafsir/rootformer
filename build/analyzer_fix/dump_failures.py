#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dump_failures.py -- reproduce the published mode-1 baseline on the SAME 188,722-type sample
and dump every failing type with its full (prefix, root, wazn, suffix) tuple + decode.

This is the A/B harness.  It is deliberately a re-implementation of eval_heldout.py's
scoring core (identical DIAC_RE, identical exact/skel definitions, identical type dedup),
so that a number printed here is comparable with the published 57.5884 % / 82.1065 %.

Usage:
    python dump_failures.py --dump /workspace/analyzer_fix/failures.jsonl
    python dump_failures.py --scores /workspace/analyzer_fix/scores.json

Nothing is written outside the --dump / --scores paths.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import sys

RELEASE = os.environ.get('AF_RELEASE', '/workspace/hf_v19_2_release')
BLUEPRINT = RELEASE + '/data/rootformer_v12_arabic_blueprint.json'
SAMPLE_CACHE = '/workspace/eval_heldout_sample.json'

DIAC_RE = re.compile('[\u064b-\u0652\u0670\u0640\u06d6-\u06ed]')


def md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as fh:
        for blk in iter(lambda: fh.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()


def vocab_class():
    import nrmp_vocab as nv
    if hasattr(nv, 'FarahidianMorphemicVocab'):
        return nv.FarahidianMorphemicVocab
    return next(v for k, v in vars(nv).items()
                if isinstance(v, type) and 'MorphemicVocab' in k)


def load_types():
    """Union of the per-file count dicts == the 188,722 distinct held-out forms.

    Occurrence weight = max count over files (the published occurrence figures weight a form
    by its total sample count; the union dedups by surface).  We keep both the sum (the
    'forms' metric of eval_heldout is a per-file sum, so a form repeated across files is
    counted once per file) -- see --scores for the reconciliation.
    """
    d = json.load(open(SAMPLE_CACHE, encoding='utf-8'))
    forms = collections.Counter()
    for f in d['files']:
        for w, n in f['counts'].items():
            forms[w] += n
    return forms, d['meta']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump', default=None)
    ap.add_argument('--scores', default=None)
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()

    sys.path.insert(0, RELEASE)
    sys.path.insert(0, RELEASE + '/models')
    V = vocab_class()
    v = V(BLUEPRINT)

    forms, meta = load_types()
    types = list(forms.keys())
    if a.limit:
        types = types[:a.limit]
    print('[*] types=%d  sampled_forms=%s' % (len(types), meta.get('sampled_forms')))

    import time
    t0 = time.time()
    n_rt = n_rt_skel = 0
    occ_rt = occ_tot = 0
    fails = []
    for i, w in enumerate(types):
        cnt = forms[w]
        pid, rid, wid, sid = v.encode_word(w)
        dec = v.decode_word(pid, rid, wid, sid)
        clean = DIAC_RE.sub('', w)
        exact = 1 if dec == w else 0
        skel = 1 if dec == clean else 0
        n_rt += exact
        n_rt_skel += skel
        occ_tot += cnt
        occ_rt += exact * cnt
        if not exact:
            rstr = v.id2root.get(rid, '<UNK>')
            fails.append({
                'w': w, 'n': cnt, 'p': v.id2prefix.get(pid, '<NONE>'),
                'r': rstr, 'wz': v.id2wazn.get(wid, '<NONE>'),
                's': v.id2suffix.get(sid, '<NONE>'),
                'dec': dec, 'skel': skel,
            })
        if (i + 1) % 50000 == 0:
            print('    %d/%d  %.0fs' % (i + 1, len(types), time.time() - t0), flush=True)

    rt_type = 100.0 * n_rt / len(types)
    rt_skel_type = 100.0 * n_rt_skel / len(types)
    rt_occ = 100.0 * occ_rt / max(occ_tot, 1)
    print('[=] type  round-trip %.4f %%  (skel %.4f %%)' % (rt_type, rt_skel_type))
    print('[=] occurrence round-trip %.4f %%  (%d/%d)' % (rt_occ, occ_rt, occ_tot))
    print('[=] failing types %d' % len(fails))

    scores = {
        'release': RELEASE,
        'md5': {'nrmp_vocab.py': md5(RELEASE + '/nrmp_vocab.py'),
                'models/validated_segmentation.py':
                    md5(RELEASE + '/models/validated_segmentation.py'),
                'models/morphemic_tokenizer_v12_arabic.py':
                    md5(RELEASE + '/models/morphemic_tokenizer_v12_arabic.py')},
        'types': len(types),
        'type_rt': round(rt_type, 4),
        'type_rt_skel': round(rt_skel_type, 4),
        'occ_rt': round(rt_occ, 4),
        'failing_types': len(fails),
        'nrmp_vocab_roots': int(v.num_roots),
        'nrmp_vocab_awzan': int(v.num_awzan),
    }
    if a.scores:
        with open(a.scores, 'w', encoding='utf-8') as fh:
            json.dump(scores, fh, ensure_ascii=False, indent=2)
        print('[*] wrote %s' % a.scores)
    if a.dump:
        with open(a.dump, 'w', encoding='utf-8') as fh:
            for x in fails:
                fh.write(json.dumps(x, ensure_ascii=False) + '\n')
        print('[*] wrote %s (%d lines)' % (a.dump, len(fails)))


if __name__ == '__main__':
    main()
