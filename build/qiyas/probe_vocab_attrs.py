import sys
sys.path.insert(0, '/workspace/hf_v19_2_release')
sys.path.insert(0, '/workspace/hf_v19_2_release/models')
sys.path.insert(0, '/workspace/qiyas')
import nrmp_vocab as nv

V = nv.FarāhīdianMorphemicVocab
v = V('/workspace/hf_v19_2_release/data/rootformer_v12_arabic_blueprint.json')
print("roots=%s awzan=%s" % (v.num_roots, v.num_awzan))
for attr in ('roots_list', 'awzan_list', 'id2root', 'id2wazn', 'root2id'):
    print("  has %-12s %s" % (attr, hasattr(v, attr)))
print("roots_list len", len(v.roots_list))
print("awzan_list len", len(v.awzan_list))
print("sample roots", v.roots_list[:3])
print("sample awzan", v.awzan_list[:3])
