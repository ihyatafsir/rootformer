#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
probe4.py -- split the REAL-root failures by WHO is at fault: the analyzer (wrong root /
wrong wazn) or the realiser (wazn not implemented).

Method (all mechanical, all measured):
  * `fell_through` = the shipped hand-written realiser returned the bare root, i.e. it has no
    branch for this wazn.  Tested by EXECUTION (call it), not by string presence.
  * for a fell-through reading, ask the project's OWN grammar engine,
    tasrif_engine.TasrifEngine.generate(root, wazn), whether the reading generates the surface.
      - yes -> the ANALYSIS IS RIGHT and the REALISER cannot emit it  -> REALISER GAP
      - no  -> the root or the wazn is wrong                          -> ANALYSER (wrong root/wazn)
  * for a non-fell-through reading whose decode is still wrong, the realiser produced something
    else; classify by whether the root's radicals are consistent with the surface.
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
from tasrif_engine import TasrifEngine
import morphemic_tokenizer_v12_arabic as MT

DIAC_RE = re.compile('[\u064b-\u0652\u0670\u0640\u06d6-\u06ed]')


def sd(s):
    return DIAC_RE.sub('', s or '')


V = next(v for k, v in vars(nv).items() if isinstance(v, type) and 'MorphemicVocab' in k)
v = V(BLUEPRINT)
eng = TasrifEngine(vocab=None)
print('[*] engine ok; roots=%d awzan=%d' % (v.num_roots, v.num_awzan))

F = [json.loads(l) for l in open('/workspace/analyzer_fix/failures.jsonl', encoding='utf-8')]
REAL = [x for x in F if not x['r'].startswith('<')]

buckets = collections.Counter()
ex = collections.defaultdict(list)
gen_err = 0
for x in REAL:
    r, wz, p, s, w = x['r'], x['wz'], x['p'], x['s'], x['w']
    pt = '' if p in ('<NONE>', '<PAD>', '<UNK>') else p
    st = '' if s in ('<NONE>', '<PAD>', '<UNK>') else s
    # the surface stem the realiser had to produce
    cw = sd(w)
    stem = cw
    if pt and stem.startswith(sd(pt)):
        stem = stem[len(sd(pt)):]
    if st and stem.endswith(sd(st)):
        stem = stem[:len(stem) - len(sd(st))]
    if not stem:
        stem = cw
    # EXECUTE the shipped realiser
    try:
        got = v.base_tok.realize_root_and_wazn(r, wz)
    except Exception:
        got = None
    bare = MT.ROOT_CANONICAL_MAP_REV.get(r, r)
    fell = (sd(got) == sd(bare)) and sd(bare) != stem
    if fell:
        # does the grammar engine generate the stem from this reading?
        g = None
        try:
            res = eng.generate(r, wz)
            g = sd(res[0]) if res else None
        except Exception:
            gen_err += 1
            g = None
        if g is not None and (g == stem or stem in (g,)):
            buckets['REALISER_GAP_engine_can_generate'] += 1
            ex['REALISER_GAP_engine_can_generate'].append(x)
        elif g is not None and len(g) == len(stem) and g == stem:
            buckets['REALISER_GAP_engine_can_generate'] += 1
            ex['REALISER_GAP_engine_can_generate'].append(x)
        else:
            buckets['ENGINE_ALSO_CANNOT_generate'] += 1
            ex['ENGINE_ALSO_CANNOT_generate'].append((x, g))
    else:
        buckets['realiser_ran_but_wrong'] += 1
        ex['realiser_ran_but_wrong'].append(x)

print('[*] generate() raised on %d readings' % gen_err)
print('\n--- REAL-root failures (%d types) ---' % len(REAL))
for k, c in buckets.most_common():
    print('  %-34s %7d  %6.2f%%' % (k, c, 100.0 * c / len(REAL)))

print('\n--- REALISER_GAP_engine_can_generate: top 15 ---')
for x in sorted(ex['REALISER_GAP_engine_can_generate'], key=lambda z: -z['n'])[:15]:
    print('  %-16s n=%-6d p=%-6s r=%-8s wz=%-14s s=%-6s dec=%-14s'
          % (x['w'], x['n'], x['p'], x['r'], x['wz'], x['s'], x['dec']))

print('\n--- ENGINE_ALSO_CANNOT_generate: top 20 (with what the engine returned) ---')
for x, g in sorted(ex['ENGINE_ALSO_CANNOT_generate'], key=lambda z: -z[0]['n'])[:20]:
    print('  %-16s n=%-6d r=%-8s wz=%-14s dec=%-12s engine=%r'
          % (x['w'], x['n'], x['r'], x['wz'], x['dec'], g))

print('\n--- realiser_ran_but_wrong: top 15 ---')
for x in sorted(ex['realiser_ran_but_wrong'], key=lambda z: -z['n'])[:15]:
    print('  %-16s n=%-6d p=%-6s r=%-8s wz=%-14s s=%-6s -> %-14s'
          % (x['w'], x['n'], x['p'], x['r'], x['wz'], x['s'], x['dec']))
