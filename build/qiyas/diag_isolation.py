#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diag_isolation.py -- what actually carries the ruling when the root's own cell is withheld?

The first isolation attempt emptied `self.wazn_cells`, which is the UNREACHABLE last resort at
qiyas_engine.py:397: `wazn_only` is always appended to the chain (lines 322-323) and is therefore
consulted IN the loop at line 390.  So the correct isolation removes 'wazn_only' from the CHAIN.

Prints, for held-out (root, wazn) pairs, the exact firing level of rule(), then the corrected
isolation.  CPU only.
"""
from __future__ import annotations

import collections
import json
import random
import sys

sys.path.insert(0, '/workspace/hf_v19_2_release')
sys.path.insert(0, '/workspace/hf_v19_2_release/models')
sys.path.insert(0, '/workspace/qiyas')

from qiyas_engine import Qiyas          # noqa: E402
import nrmp_vocab as nv                 # noqa: E402

V = nv.FarāhīdianMorphemicVocab
vocab = V('/workspace/hf_v19_2_release/data/rootformer_v12_arabic_blueprint.json')
assert (int(vocab.num_roots), int(vocab.num_awzan)) == (9490, 142), 'VOCAB SHRINK TRAP'

af = lambda x: '' if (x is None or x in ('<NONE>', '<PAD>', '<UNK>')) else x
raw = json.load(open('/workspace/lisan_root_fix/du_after.json', encoding='utf-8'))['words']
types = list(raw.keys())
bok = {w: bool(v[0]) for w, v in raw.items()}

obs = []
for w in types:
    p_id, r_id, wz_id, s_id = vocab.encode_word(w)
    r, wz = vocab.id2root[r_id], vocab.id2wazn[wz_id]
    p, s = af(vocab.id2prefix[p_id]), af(vocab.id2suffix[s_id])
    if r.startswith('<') or not r or not wz or wz in ('<NONE>', '<PAD>', '<UNK>'):
        continue
    if not bok.get(w) or len(p) + len(s) > len(w):
        continue
    stem = w[len(p):len(w) - len(s)] if s else w[len(p):]
    if not stem or p + stem + s != w:
        continue
    obs.append((r, wz, stem))
print('asl =', len(obs))

# reproduce the Tier-1 pair holdout exactly (same seed/frac as two_condition_rerun.py)
rng = random.Random(5)
pairs = sorted({(r, wz) for r, wz, _ in obs})
rng.shuffle(pairs)
k = int(0.10 * len(pairs))
test_pairs = set(pairs[:k])
train_obs = [o for o in obs if (o[0], o[1]) not in test_pairs]
test_obs = [o for o in obs if (o[0], o[1]) in test_pairs]
print('held-out pairs =', len(test_pairs), ' held-out obs =', len(test_obs))

q = Qiyas('identity', backoff=True)
for r, w, s in train_obs:
    q.observe(r, w, s)
q.induce()
print('chain =', q.chain)
print('tables present =', sorted(q.tables))
print("identity table size =", len(q.tables['identity']),
      " wazn_only table size =", len(q.tables['wazn_only']))

# ---- (A) WHERE does the ruling come from?
fired = collections.Counter()
for r, wz, s in test_obs:
    rule = q.rule(r, wz)
    if rule is None:
        fired['NONE'] += 1
    else:
        fired[rule[1].split(':')[0]] += 1
print('(A) firing level on held-out pairs:', dict(fired))

# is the identity cell for a held-out pair genuinely absent?
sample = sorted(test_pairs)[:5]
for r, wz in sample:
    print('    held-out pair %s/%s in identity table: %s' % (r, wz, (r, wz) in q.tables['identity']))

# ---- (B) CORRECTED isolation: remove 'wazn_only' FROM THE CHAIN
q2 = Qiyas('identity', backoff=False)
for r, w, s in train_obs:
    q2.observe(r, w, s)
q2.induce()
q2.chain = [n for n in q2.chain if n != 'wazn_only']
print('(B) chain after removing wazn_only =', q2.chain)
none_n = 0
for r, wz, s in test_obs:
    if q2.rule(r, wz) is None:
        none_n += 1
print('(B) rule() returns None for %d / %d held-out far\' observations' % (none_n, len(test_obs)))

# ---- (C) sanity: on ATTESTED pairs the identity cell must be present and correct
att = sorted({(r, wz) for r, wz, _ in train_obs})[:2000]
hit = 0
for r, wz in att:
    if q2.rule(r, wz) is not None:
        hit += 1
print('(C) identity cell present on %d / %d ATTESTED pairs (control)' % (hit, len(att)))
