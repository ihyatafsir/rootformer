"""probe_isolate_clean.py -- the isolation test done correctly.

Both earlier attempts were wrong, for a reason the engine's source makes plain:
  * `rule()` consults `self.chain` first, then falls through to `self.wazn_cells` OUTSIDE the loop.
  * `__init__` appends 'wazn_only' to the chain even when backoff=False (line 322-323), so
    "identity without back-off" was never actually run anywhere in this project.

Correct cut: give the engine the identity table and NOTHING else -- chain=['identity'] and
wazn_cells emptied -- in that order, so the tail lookup finds nothing either.

Question answered: with the root's own (root, wazn) cell withheld, does the ROOT carry a ruling?
"""
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
assert (int(vocab.num_roots), int(vocab.num_awzan)) == (9490, 142), 'VOCAB SHRINK TRAP'
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
print("asl=%d  held-out pairs=%d  held-out obs=%d" % (len(obs), len(test_pairs), len(test_obs)))
# the roots in held-out pairs are SEEN roots: the passage's own setting
print("held-out pairs involve %d distinct roots, every one of them SEEN in the asl set: %s"
      % (len({r for r, _ in test_pairs}),
         {r for r, _ in test_pairs} <= {r for r, _, _ in train_obs}))

q = Qiyas('identity', backoff=False)
for r, w, s in train_obs:
    q.observe(r, w, s)
q.induce()

# ---- THE CORRECT CUT -------------------------------------------------------------
keep = q.tables['identity']
q.chain = ['identity']
q.tables = {'identity': keep}
q.wazn_cells = collections.defaultdict(collections.Counter)
print("\nCUT: chain=%s  tables=%s  wazn_cells emptied" % (q.chain, sorted(q.tables)))

none_n = 0
carried = 0
for r, wz, s in test_obs:
    if q.rule(r, wz) is None:
        none_n += 1
    else:
        carried += 1
print("  held-out far' observations            : %d" % len(test_obs))
print("  root-as-'illa carries a ruling for    : %d" % carried)
print("  rule() returns None for              : %d" % none_n)
print("  the root's own cell is present for   : %d"
      % sum(1 for r, wz, _ in test_obs if (r, wz) in keep))

# ---- CONTROL: on ATTESTED pairs the same cut must still work ---------------------
att = sorted({(r, wz) for r, wz, _ in train_obs})[:3000]
hit = sum(1 for r, wz in att if q.rule(r, wz) is not None)
print("\nCONTROL (head-independent style): on %d ATTESTED pairs the same cut returns a rule %d times"
      % (len(att), hit))
print("  -> a non-zero control shows the cut removed only the withheld cells, not the mechanism.")
