#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
probe5.py -- the clean, single criterion.

For EVERY failing type that carries a real root:
    reading = (root, wazn)
    ask the project's OWN grammar engine -- tasrif_engine.TasrifEngine.generate(root, wazn)
    -- whether THAT reading, realised by the grammar, yields the surface stem.

  * yes -> the analysis is right and the shipped hand-written realiser cannot emit it
           => REALISER GAP (the fix is in the realiser, and the grammar engine already has it)
  * no  -> the reading itself is wrong
           => ANALYSER (then: can ANY (root', wazn') generate the stem? that is al-Khalil's
              calculation -- tasrif_engine._align / candidate_pairs)
"""
import collections
import json
import re
import sys

RELEASE = '/workspace/hf_v19_2_release'
BLUEPRINT = RELEASE + '/data/rootformer_v12_arabic_blueprint.json'
sys.path.insert(0, RELEASE)
sys.path.insert(0, RELEASE + '/models')

import nrmp_vocab as nv
from tasrif_engine import TasrifEngine, Pattern
import morphemic_tokenizer_v12_arabic as MT

DIAC_RE = re.compile('[\u064b-\u0652\u0670\u0640\u06d6-\u06ed]')
sd = lambda s: DIAC_RE.sub('', s or '')
NONE = ('<NONE>', '<PAD>', '<UNK>', '', None)

V = next(v for k, v in vars(nv).items() if isinstance(v, type) and 'MorphemicVocab' in k)
v = V(BLUEPRINT)
eng = TasrifEngine(vocab=None)
print('[*] roots=%d awzan=%d' % (v.num_roots, v.num_awzan))

AWZAN = [w for w in v.awzan_list if not w.startswith('<')]

F = [json.loads(l) for l in open('/workspace/analyzer_fix/failures.jsonl', encoding='utf-8')]


def stems(w, pt, st):
    """All surface stems obtainable by stripping the claimed prefix/suffix (defensively:
    also the whole surface, and the surface minus the suffix only / prefix only)."""
    cw = sd(w)
    out = []
    cands = [(sd(pt) if pt else '', sd(st) if st else ''),
             (sd(pt) if pt else '', ''), ('', sd(st) if st else '')]
    for a, b in cands:
        s = cw
        if a and s.startswith(a):
            s = s[len(a):]
        if b and s.endswith(b):
            s = s[:len(s) - len(b)]
        if s and s not in out:
            out.append(s)
    if cw not in out:
        out.append(cw)
    return out


buckets = collections.Counter()
ex = collections.defaultdict(list)
for x in F:
    r, wz, p, s, w = x['r'], x['wz'], x['p'], x['s'], x['w']
    if r.startswith('<'):
        continue
    pt = '' if p in NONE else p
    st = '' if s in NONE else s
    S = stems(w, pt, st)
    g = None
    if wz not in NONE:
        try:
            res = eng.generate(r, wz)
            g = sd(res[0]) if res else None
        except Exception:
            g = None
    if g is not None and g in S:
        buckets['REALISER_GAP_same_reading_engine_ok'] += 1
        ex['REALISER_GAP_same_reading_engine_ok'].append((x, g))
    elif wz in NONE:
        buckets['NO_WAZN_ASSIGNED'] += 1
        ex['NO_WAZN_ASSIGNED'].append((x, g))
    else:
        buckets['ANALYSIS_WRONG_or_engine_gap'] += 1
        ex['ANALYSIS_WRONG_or_engine_gap'].append((x, g))

print('\n--- real-root failures: %d ---' % sum(buckets.values()))
for k, c in buckets.most_common():
    print('  %-38s %7d  %6.2f%%' % (k, c, 100.0 * c / sum(buckets.values())))

for k in ('REALISER_GAP_same_reading_engine_ok', 'NO_WAZN_ASSIGNED',
          'ANALYSIS_WRONG_or_engine_gap'):
    print('\n--- %s top 25 ---' % k)
    for xx, g in sorted(ex[k], key=lambda z: -z[0]['n'])[:25]:
        print('  %-14s n=%-6d p=%-6s r=%-8s wz=%-14s s=%-5s dec=%-12s engine=%r'
              % (xx['w'], xx['n'], xx['p'], xx['r'], xx['wz'], xx['s'], xx['dec'], g))
