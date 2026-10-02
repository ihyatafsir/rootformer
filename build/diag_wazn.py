import sys
sys.path.insert(0, "/workspace/hf_v19_2_release")
sys.path.insert(0, "/workspace/hf_v19_2_release/models")
import nrmp_vocab as nv
V = next(v for k, v in vars(nv).items() if isinstance(v, type) and 'MorphemicVocab' in k)
for bp in ("/workspace/hf_v19_2_release/data/rootformer_v12_arabic_blueprint.json",
           "/workspace/hf_v19_2_release/data/rootformer_v12_blueprint_137awzan.json"):
    v = V(bp)
    print("blueprint:", bp.split("/")[-1], "| num_awzan =", v.num_awzan)
    for w in ("حقائق", "دقائق", "والحقائق", "مداد"):
        p, r, wz, s = v.encode_word(w)
        rn = v.id2root.get(r, "?")
        wn = v.id2wazn.get(wz, "?")
        print("   %-10s root=%-8s wazn=%s" % (w, rn, wn))
    print()
