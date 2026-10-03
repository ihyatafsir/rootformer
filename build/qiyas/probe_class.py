import sys
sys.path.insert(0, '/workspace/hf_v19_2_release')
sys.path.insert(0, '/workspace/hf_v19_2_release/models')
sys.path.insert(0, '/workspace/qiyas')
import nrmp_vocab as nv

print("module:", nv.__file__)
print("md5:")
import hashlib
print(" ", hashlib.md5(open(nv.__file__, 'rb').read()).hexdigest())

classes = [(k, v) for k, v in vars(nv).items() if isinstance(v, type)]
print("all classes:", [k for k, _ in classes])
print("has MorphemicVocab attr:", hasattr(nv, 'MorphemicVocab'))
for k, v in classes:
    if 'Morphemic' in k:
        print("  target:", k, v)
        V = v
v = V('/workspace/hf_v19_2_release/data/rootformer_v12_arabic_blueprint.json')
print("roots=%s awzan=%s" % (v.num_roots, v.num_awzan))
print("encode_word sample:", v.encode_word('كاتب'))
