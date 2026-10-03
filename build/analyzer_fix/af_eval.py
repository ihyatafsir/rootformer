#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
af_eval.py -- A/B harness for the AF_ANALYZER_FIX patch.

Scores ALL 188,722 published held-out types with a vocab module loaded from an arbitrary path
(the shipped module, or the patched copy), reports the published metrics, and can dump the full
per-type tuple so two runs can be compared byte-for-byte.

    python af_eval.py --vocab /workspace/hf_v19_2_release/nrmp_vocab.py --dump base.jsonl
    AF_ANALYZER_FIX=1 python af_eval.py --vocab .../patched/nrmp_vocab.py --dump fix.jsonl
    python af_eval.py --diff base.jsonl fix.jsonl
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
import os
import re
import sys
import time

RELEASE = '/workspace/hf_v19_2_release'
BLUEPRINT = RELEASE + '/data/rootformer_v12_arabic_blueprint.json'
SAMPLE = '/workspace/eval_heldout_sample.json'
DIAC_RE = re.compile('[\u064b-\u0652\u0670\u0640\u06d6-\u06ed]')


def load_module(path):
    sys.path.insert(0, RELEASE)
    sys.path.insert(0, RELEASE + '/models')
    name = 'af_vocab_%s' % hashlib.md5(path.encode()).hexdigest()[:8]
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def load_types():
    d = json.load(open(SAMPLE, encoding='utf-8'))
    forms = collections.Counter()
    for f in d['files']:
        for w, n in f['counts'].items():
            forms[w] += n
    return forms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--vocab', default=RELEASE + '/nrmp_vocab.py')
    ap.add_argument('--dump', default=None)
    ap.add_argument('--diff', nargs=2, default=None)
    ap.add_argument('--tag', default='')
    a = ap.parse_args()

    if a.diff:
        A = [json.loads(l) for l in open(a.diff[0], encoding='utf-8')]
        B = [json.loads(l) for l in open(a.diff[1], encoding='utf-8')]
        assert len(A) == len(B), (len(A), len(B))
        changed = [i for i in range(len(A)) if A[i] != B[i]]
        broke = [i for i in changed
                 if (A[i].get('ok') and not B[i].get('ok'))]
        fixed = [i for i in changed
                 if (not A[i].get('ok') and B[i].get('ok'))]
        neutral = len(changed) - len(broke) - len(fixed)
        nA = sum(1 for x in A if x.get('ok'))
        nB = sum(1 for x in B if x.get('ok'))
        print('[diff] %s -> %s' % (a.diff[0], a.diff[1]))
        print('  types                 %d' % len(A))
        print('  round-trip A          %d (%.4f %%)' % (nA, 100.0 * nA / len(A)))
        print('  round-trip B          %d (%.4f %%)' % (nB, 100.0 * nB / len(A)))
        print('  changed tuples        %d' % len(changed))
        print('  FIXED  (fail->pass)   %d' % len(fixed))
        print('  BROKEN (pass->fail)   %d' % len(broke))
        print('  neutral (same ok)     %d' % neutral)
        for nm, idx in (('BROKEN', broke), ('FIXED', fixed)):
            print('  --- %s, top 15 by occurrences ---' % nm)
            for i in sorted(idx, key=lambda j: -B[j]['n'])[:15]:
                print('     %-16s n=%-6d A=(%s,%s,%s,%s)->%r  B=(%s,%s,%s,%s)->%r'
                      % (A[i]['w'], A[i]['n'], A[i]['p'], A[i]['r'], A[i]['wz'], A[i]['s'],
                         A[i]['d'], B[i]['p'], B[i]['r'], B[i]['wz'], B[i]['s'], B[i]['d']))
        return

    m = load_module(a.vocab)
    V = next(v for k, v in vars(m).items() if isinstance(v, type) and 'MorphemicVocab' in k)
    v = V(BLUEPRINT)
    forms = load_types()
    types = list(forms.keys())
    t0 = time.time()
    n_rt = n_skel = occ_rt = occ_tot = 0
    out = []
    for i, w in enumerate(types):
        n = forms[w]
        pid, rid, wid, sid = v.encode_word(w)
        dec = v.decode_word(pid, rid, wid, sid)
        clean = DIAC_RE.sub('', w)
        ok = 1 if dec == w else 0
        skel = 1 if dec == clean else 0
        n_rt += ok
        n_skel += skel
        occ_tot += n
        occ_rt += ok * n
        if a.dump:
            out.append({'w': w, 'n': n, 'p': v.id2prefix.get(pid, '<NONE>'),
                        'r': v.id2root.get(rid, '<UNK>'), 'wz': v.id2wazn.get(wid, '<NONE>'),
                        's': v.id2suffix.get(sid, '<NONE>'), 'd': dec, 'ok': ok})
        if (i + 1) % 50000 == 0:
            print('    %d/%d %.0fs' % (i + 1, len(types), time.time() - t0), flush=True)

    print('[%s] vocab=%s' % (a.tag or 'eval', a.vocab))
    print('  num_roots=%d num_awzan=%d' % (v.num_roots, v.num_awzan))
    print('  TYPE  round-trip %.4f %% (%d/%d)' % (100.0 * n_rt / len(types), n_rt, len(types)))
    print('  TYPE  skel        %.4f %%' % (100.0 * n_skel / len(types)))
    print('  OCC   round-trip %.4f %% (%d/%d)' % (100.0 * occ_rt / occ_tot, occ_rt, occ_tot))
    print('  failing types %d' % (len(types) - n_rt))
    if a.dump:
        with open(a.dump, 'w', encoding='utf-8') as fh:
            for x in out:
                fh.write(json.dumps(x, ensure_ascii=False) + '\n')
        print('  wrote %s' % a.dump)
        json.dump({'vocab': a.vocab, 'type_rt': round(100.0 * n_rt / len(types), 4),
                   'occ_rt': round(100.0 * occ_rt / occ_tot, 4),
                   'failing': len(types) - n_rt,
                   'roots': int(v.num_roots), 'awzan': int(v.num_awzan),
                   'af_analyzer_fix': os.environ.get('AF_ANALYZER_FIX', '0')},
                  open(a.dump + '.summary.json', 'w', encoding='utf-8'), indent=2)


if __name__ == '__main__':
    main()
