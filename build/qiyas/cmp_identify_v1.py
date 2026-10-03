"""cmp_identify_v1.py -- why does the inverted-index harness give 31.22 % where the original
gives 84.48 % on the SAME 37-root candidate set?  Do not guess; diff the two implementations
position by position on a sample."""
import sys
import collections
import json

sys.path.insert(0, '/workspace/hf_v19_2_release')
sys.path.insert(0, '/workspace/hf_v19_2_release/models')
sys.path.insert(0, '/workspace/qiyas')

from qiyas_engine import Qiyas          # noqa: E402
import nrmp_vocab as nv                 # noqa: E402

AF = ('<NONE>', '<PAD>', '<UNK>')
affix = lambda x: '' if (x is None or x in AF) else x
vocab = nv.FarāhīdianMorphemicVocab(
    '/workspace/hf_v19_2_release/data/rootformer_v12_arabic_blueprint.json')
gold = json.load(open('/workspace/qiyas/deriv_gold.json', encoding='utf-8'))
pos = [{'word': x['word'], 'root': x['root'], 'wazn': x['wazn'],
        'prefix': affix(x['prefix']), 'suffix': affix(x['suffix'])}
       for x in gold['positions']]
hold = sorted({x['root'] for x in pos})
du = json.load(open('/workspace/lisan_root_fix/du_after.json', encoding='utf-8'))['words']
hset = set(hold)

obs = []
for w in du:
    p_id, r_id, wz_id, s_id = vocab.encode_word(w)
    r, wz = vocab.id2root[r_id], vocab.id2wazn[wz_id]
    p, s = affix(vocab.id2prefix[p_id]), affix(vocab.id2suffix[s_id])
    if not r or r.startswith('<') or r in hset:
        continue
    if not wz or wz in AF or len(p) + len(s) > len(w):
        continue
    stem = w[len(p):len(w) - len(s)] if s else w[len(p):]
    if not stem or p + stem + s != w:
        continue
    obs.append((r, wz, stem))
print("asl =", len(obs), " roots =", len({o[0] for o in obs}))
q = Qiyas('strict7', backoff=True)
for r, w, s in obs:
    q.observe(r, w, s)
q.induce()

wazn_all = [w for w in vocab.awzan_list if isinstance(w, str) and not w.startswith('<')]
print("awzan tried =", len(wazn_all), " hold roots =", len(hold))

# ---- the ORIGINAL approach, verbatim from qiyas_deriv3588.build_grid/identify
grid = {}
for r in hold:
    row = {}
    for wz in wazn_all:
        res = q.realize(r, wz)
        if res is not None:
            row[wz] = res[:5]
    grid[r] = row

SAMPLE = pos[:400]
orig_cands = []
for x in SAMPLE:
    w, goldr, p, s = x['word'], x['root'], x['prefix'], x['suffix']
    c = []
    for r in hold:
        for wz, res in grid[r].items():
            qy = res[0]
            if (p + qy + s) == w or qy == w:
                c.append((r, wz, qy, res[1], res[2], res[3]))
    orig_cands.append(c)

# ---- the INVERTED-INDEX approach
rows = []
for r in hold:
    for wz, res in grid[r].items():
        rows.append((r, wz, res[0], res[1], res[2], res[3]))
index = collections.defaultdict(list)
for e in rows:
    index[e[2]].append(e)
print("index surfaces =", len(index), " rows =", len(rows))

inv_cands = []
for x in SAMPLE:
    w, p, s = x['word'], x['prefix'], x['suffix']
    raw = list(index.get(w, ()))
    c = [e for e in raw if (p + e[2] + s) == w or e[2] == w]
    inv_cands.append(c)

# ---- compare
n_same = 0
shown = 0
for i, x in enumerate(SAMPLE):
    A = {c[:2] for c in orig_cands[i]}
    B = {c[:2] for c in inv_cands[i]}
    if A == B:
        n_same += 1
        continue
    if shown < 12:
        shown += 1
        print("\nMISMATCH  word=%r gold=%s p=%r s=%r" % (x['word'], x['root'], x['prefix'], x['suffix']))
        print("   orig  n=%d %s" % (len(A), sorted(A)[:6]))
        print("   inv   n=%d %s" % (len(B), sorted(B)[:6]))
        print("   only-orig %s" % sorted(A - B)[:6])
        print("   only-inv  %s" % sorted(B - A)[:6])
print("\nsample=%d  identical candidate sets=%d  differing=%d" % (len(SAMPLE), n_same, len(SAMPLE) - n_same))

# how many positions have NO candidate under each, and is gold present?
og = sum(1 for i, x in enumerate(SAMPLE) if any(c[0] == x['root'] for c in orig_cands[i]))
ig = sum(1 for i, x in enumerate(SAMPLE) if any(c[0] == x['root'] for c in inv_cands[i]))
print("gold root present: orig %d/%d   inv %d/%d" % (og, len(SAMPLE), ig, len(SAMPLE)))
