import sys
sys.path.insert(0, '.')
import nrmp_vocab as nv
V = next(v for k, v in vars(nv).items() if isinstance(v, type) and 'MorphemicVocab' in k)(
    'data/rootformer_v12_arabic_blueprint.json')
from andalusian_realizer import AndalusianRealizer
r = AndalusianRealizer().install(V)
print('installed on vocab as:', type(V.andalusian_realizer).__name__)
for w, case in [('يذهب', 'jazm'), ('ضربه', None), ('مصانع', 'jarr'), ('الغلام', 'jarr')]:
    t = V.encode_word(w)
    o = r.analyze_tuple(V, *t, case=case)
    print('%-8s case=%-5s irab=%-12r host=%-5s slot=%-14s closed=%s (%s)' % (
        w, case, o['irab'], o['host_class'], o['saturated_slot'],
        o['slot_closed'], o['slot_reason']))
