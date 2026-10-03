"""probe_gold.py -- for 3 concrete gold positions, print every quantity both harnesses use."""
import sys
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
print("gold position 0 raw record:", json.dumps(gold['positions'][0], ensure_ascii=False))
print("parsed pos[0]:", pos[0])

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
q = Qiyas('strict7', backoff=True)
for r, w, s in obs:
    q.observe(r, w, s)
q.induce()
wazn_all = [w for w in vocab.awzan_list if isinstance(w, str) and not w.startswith('<')]

targets = ['الابتدا', 'المبتدا', 'النفي', 'أصبحا', 'ابتدا', 'أبدا']
for x in pos:
    if x['word'] not in targets:
        continue
    w, goldr, p, s = x['word'], x['root'], x['prefix'], x['suffix']
    print("\n=== word=%r gold_root=%s p=%r s=%r  (len w=%d, len p=%d, len s=%d)"
          % (w, goldr, p, s, len(w), len(p), len(s)))
    stem = w[len(p):len(w) - len(s)] if s else w[len(p):]
    print("    causal stem        = %r" % stem)
    # gold root's own realization for the gold wazn, plus the best over all wazn
    hits = []
    for wz in wazn_all:
        res = q.realize(goldr, wz)
        if res is None:
            continue
        qy = res[0]
        if (p + qy + s) == w or qy == w:
            hits.append((wz, qy))
    print("    gold-root wazn that satisfy the test: %s" % hits[:6])
    # does the stem resolve in the index space at all?
    n_stem = 0
    for r2 in hold:
        for wz in wazn_all:
            res = q.realize(r2, wz)
            if res is not None and res[0] == stem:
                n_stem += 1
    print("    held-out realizations equal to the STEM %r : %d" % (stem, n_stem))
    targets.remove(w)
