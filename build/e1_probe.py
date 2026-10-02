#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""e1_probe.py -- how much of the E1 (<PARTICLE>) residual is reachable by the tokenizer's
OWN root calculator (validated_segmentation._candidate_roots + tasrif engine _generates)?

Bounded random sample (seeded), CPU-only, single process, nice'd.
"""
import json, os, random, sys, time

os.environ.setdefault('ROOTFORMER_VALIDATED_SEG', '1')
R = '/workspace/taxpin'
for p in (R + '/models', R):
    if p not in sys.path:
        sys.path.insert(0, p)

import nrmp_vocab as nv
import validated_segmentation as vs
cls = next(c for k, c in vars(nv).items() if isinstance(c, type) and 'MorphemicVocab' in k)
v = cls(R + '/data/rootformer_v12_arabic_blueprint.json')
tok = v.base_tok
val = vs.validator(tok)
import morphemic_tokenizer_v12_arabic as mt

PROC = list(mt.PROCLITICS) + ['س', 'است', 'مست', 'يست', 'تست', 'ان', 'مت', 'ت', 'ي', 'ن', 'أ', 'م']
ENC = list(mt.ENCLITICS) + ['ة', 'ات', 'ون', 'ين', 'ان', 'كما', 'تما', 'تن', 'ني', 'ت']
PATTERNS = ['فَعَلَ', 'فَعِلَ', 'فَعُلَ', 'فَعِيل', 'فِعَال', 'فَاعِل', 'مَفْعُول', 'مُفْتَعِل',
            'يَفْعَلُ', 'يَفْعِلُ', 'تَفَاعَلَ', 'اِفْتِعَال', 'اِسْتِفْعَال', 'فُعُول', 'أَفْعَال',
            'مَفَاعِل', 'فَعْلَلَ', 'اِفْتَعَلَ', 'اِسْتَفْعَلَ', 'تَفَعَّلَ']

rows = [json.loads(l) for l in open('/workspace/taxonomy_out/failures_m1.jsonl', encoding='utf-8')]
E1 = [r for r in rows if r['cause'] == 'E1_particle_echo_no_analysis']
random.seed(20261001)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
sample = random.sample(E1, min(N, len(E1)))
print('[*] E1 types=%d, sampling %d' % (len(E1), len(sample)))

t0 = time.time()
res = {'recoverable_attested_root': 0, 'recoverable_unattested': 0, 'not_recoverable': 0}
ex = {'recoverable_attested_root': [], 'recoverable_unattested': []}
for i, r in enumerate(sample):
    C = r['w']
    hit = None
    for pre in [''] + PROC:
        if pre and not C.startswith(pre):
            continue
        rest = C[len(pre):]
        if len(rest) < 2:
            continue
        for suf in [''] + ENC:
            if suf and not rest.endswith(suf):
                continue
            stem = rest[:len(rest) - len(suf)] if suf else rest
            if len(stem) < 2:
                continue
            for pat in PATTERNS:
                try:
                    cands = val._candidate_roots(stem, pat)
                except Exception:
                    continue
                for cr in cands:
                    root0 = cr[0] if isinstance(cr, (tuple, list)) else cr
                    try:
                        rr = val._real_root(root0)
                    except Exception:
                        continue
                    if rr == val.ATTESTED:
                        hit = (pre, root0, pat, suf)
                        break
                    if hit is None:
                        hit = ('unatt', pre, root0, pat, suf)
                if hit and len(hit) == 4:
                    break
            if hit and len(hit) == 4:
                break
        if hit and len(hit) == 4:
            break
    if hit and len(hit) == 4:
        res['recoverable_attested_root'] += 1
        if len(ex['recoverable_attested_root']) < 12:
            ex['recoverable_attested_root'].append((C, r['count'], hit))
    elif hit:
        res['recoverable_unattested'] += 1
        if len(ex['recoverable_unattested']) < 12:
            ex['recoverable_unattested'].append((C, r['count'], hit))
    else:
        res['not_recoverable'] += 1
    if (i + 1) % 250 == 0:
        print('    %d/%d %.0fs %s' % (i + 1, len(sample), time.time() - t0, res))

out = {'sample_n': len(sample), 'e1_types': len(E1), 'seed': 20261001, 'result': res,
       'examples': {k: [(x[0], x[1], list(x[2]) if not isinstance(x[2], str) else x[2]) if not isinstance(x[2], tuple) else (x[0], x[1], list(x[2])) for x in v2] for k, v2 in ex.items()}}
json.dump(out, open('/workspace/taxonomy_out/e1_probe.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('[*] done', res)
