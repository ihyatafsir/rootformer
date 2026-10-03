"""probe_final.py -- settle the isolation question with the engine's own source printed.

Question: with the root's own (root, wazn) cell withheld, does the ROOT carry the ruling, or does
the context-free wazn->alignment table carry it?  Print the relevant source so the answer is
checkable, then run both cuts explicitly.
"""
import sys
import collections
import inspect
import json
import random

sys.path.insert(0, '/workspace/hf_v19_2_release')
sys.path.insert(0, '/workspace/hf_v19_2_release/models')
sys.path.insert(0, '/workspace/qiyas')

from qiyas_engine import Qiyas          # noqa: E402
import nrmp_vocab as nv                 # noqa: E402

print("=" * 90)
print("ENGINE SOURCE -- __init__ chain construction, and rule()")
print("=" * 90)
src = inspect.getsource(Qiyas.__init__)
for line in src.splitlines():
    if 'chain' in line or 'BACKOFF' in line or 'wazn_only' in line:
        print("  init | " + line.rstrip())
print()
for line in inspect.getsource(Qiyas.rule).splitlines():
    print("  rule | " + line.rstrip())

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
print("\nasl=%d  held-out pairs=%d  held-out obs=%d" % (len(obs), len(test_pairs), len(test_obs)))

q = Qiyas('identity', backoff=False)
for r, w, s in train_obs:
    q.observe(r, w, s)
q.induce()
print("\nq.chain AS CONSTRUCTED           =", q.chain)
print("q.tables keys                    =", sorted(q.tables))

print("\n--- CUT 1: chain cut to ['identity'] only (wazn_cells LEFT INTACT) ---")
q1 = Qiyas('identity', backoff=False)
for r, w, s in train_obs:
    q1.observe(r, w, s)
q1.induce()
q1.chain = [n for n in q1.chain if n != 'wazn_only']
print("   chain =", q1.chain, " tables =", sorted(q1.tables))
fired = collections.Counter()
for r, wz, s in test_obs:
    rule = q1.rule(r, wz)
    fired['NONE' if rule is None else rule[1].split(':')[0]] += 1
print("   firing:", dict(fired))
print("   >>> the tail lookup `self.wazn_cells` at the end of rule() is OUTSIDE the loop and")
print("       still serves the ruling, which is why this cut does not isolate anything.")

print("\n--- CUT 2: identity table ONLY (tables={'identity'}, wazn_cells emptied) ---")
q2 = Qiyas('identity', backoff=False)
for r, w, s in train_obs:
    q2.observe(r, w, s)
q2.induce()
q2.tables = {'identity': q2.tables['identity']}
q2.wazn_cells = collections.defaultdict(collections.Counter)
print("   chain =", q2.chain, " tables =", sorted(q2.tables))
none_n = 0
for r, wz, s in test_obs:
    try:
        if q2.rule(r, wz) is None:
            none_n += 1
    except KeyError as e:
        print("   KeyError:", e)
        break
print("   rule() returns None for %d / %d held-out far' observations" % (none_n, len(test_obs)))

print("\n--- CUT 3: the direct attribution, measured on the real deployed chain ---")
q3 = Qiyas('identity', backoff=True)
for r, w, s in train_obs:
    q3.observe(r, w, s)
q3.induce()
own_cell = sum(1 for r, wz, s in test_obs if (r, wz) in q3.tables['identity'])
print("   identity's OWN cell present for : %d / %d held-out obs" % (own_cell, len(test_obs)))
att = collections.Counter()
for r, wz, s in test_obs:
    rule = q3.rule(r, wz)
    att['NONE' if rule is None else rule[1].split(':')[0]] += 1
print("   who actually carries the ruling :", dict(att))
