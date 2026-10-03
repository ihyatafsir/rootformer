"""probe_rule.py -- which branch of rule() serves a held-out (root, wazn) with the chain cut to
['identity'] alone?  Instrument, do not speculate."""
import sys
import collections
import json
import random

sys.path.insert(0, '/workspace/hf_v19_2_release')
sys.path.insert(0, '/workspace/hf_v19_2_release/models')
sys.path.insert(0, '/workspace/qiyas')

from qiyas_engine import Qiyas          # noqa: E402
import nrmp_vocab as nv                 # noqa: E402

vocab = nv.FarāhīdianMorphemicVocab(
    '/workspace/hf_v19_2_release/data/rootformer_v12_arabic_blueprint.json')
af = lambda x: '' if (x is None or x in ('<NONE>', '<PAD>', '<UNK>')) else x
raw = json.load(open('/workspace/lisan_root_fix/du_after.json', encoding='utf-8'))['words']
bok = {w: bool(v[0]) for w, v in raw.items()}
obs = []
for w in raw:
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

rng = random.Random(5)
pairs = sorted({(r, wz) for r, wz, _ in obs})
rng.shuffle(pairs)
test_pairs = set(pairs[:int(0.10 * len(pairs))])
train_obs = [o for o in obs if (o[0], o[1]) not in test_pairs]
test_obs = [o for o in obs if (o[0], o[1]) in test_pairs]

q = Qiyas('identity', backoff=False)
for r, w, s in train_obs:
    q.observe(r, w, s)
q.induce()
q.chain = [n for n in q.chain if n != 'wazn_only']
print("chain =", q.chain)
print("type(q.tables['identity']) =", type(q.tables['identity']))

r0, wz0, s0 = test_obs[0]
print("first held-out (root, wazn) =", r0, wz0, "surface =", s0)
print("label =", q._label('identity', r0))
print("key in identity table:", (r0, wz0) in q.tables['identity'])
print("table.get(key) =", q.tables['identity'].get((r0, wz0)))
print("_modal(table, key) =", q._modal(q.tables['identity'], (r0, wz0)))
print("rule() =", q.rule(r0, wz0))
print("wazn_cells.get(wazn) =", q.wazn_cells.get(wz0))
print("type(q.wazn_cells) =", type(q.wazn_cells))

# how many test obs have their identity cell present?
present = sum(1 for r, wz, s in test_obs if (r, wz) in q.tables['identity'])
print("held-out obs whose identity cell IS present: %d / %d" % (present, len(test_obs)))

# now cut the tables to identity-only and re-count
q2 = Qiyas('identity', backoff=False)
for r, w, s in train_obs:
    q2.observe(r, w, s)
q2.induce()
q2.tables = {'identity': q2.tables['identity']}
q2.wazn_cells = collections.defaultdict(collections.Counter)
none_n = sum(1 for r, wz, s in test_obs if q2.rule(r, wz) is None)
print("with tables={'identity'} and wazn_cells emptied: None for %d / %d" % (none_n, len(test_obs)))
