#!/usr/bin/env python3
"""Build staged variants of the patched nrmp_vocab.py for per-mechanism attribution.

Stage B -- DECODE ONLY: al-qalb generation + the control-token invariant.
Stage C -- B + closed-class inventory expansion and the three encoder recovery paths
           (without the ya/alif-maqsura orthographic variant closure).
Stage D -- the full patch (== nrmp_vocab.py.new).
"""
import re

SRC = 'nrmp_vocab.py.new'
text = open(SRC, encoding='utf-8').read()

ORIG_ROOTS_BLOCK = (
    "        self.roots_list = self.special_roots + particle_roots + raw_roots\n"
    "        self.root2id = {r: i for i, r in enumerate(self.roots_list)}\n"
)

ORIG_MAP_ROOT = '''        # Map Root
        if r is None or r == '':
            if clean_w in self.COMMON_PARTICLES or f'<P:{clean_w}>' in self.root2id:
                r_id = self.root2id.get(f'<P:{clean_w}>', self.root2id['<PARTICLE>'])
            else:
                r_id = self.root2id['<PARTICLE>']
        else:
            particle_cand = f'<P:{r}>'
            if particle_cand in self.root2id:
                r_id = self.root2id[particle_cand]
            else:
                r_id = self.root2id.get(r, self.root2id['<UNK>'])
'''


def cut_between(s, start_marker, end_marker):
    i = s.index(start_marker)
    j = s.index(end_marker, i)
    return s[:i], s[i:j], s[j:]


# ---- stage C: full patch minus the orthographic variant closure -------------------------------
c = text.replace("            out.update(_ya_alif_variants(p))\n", "")
assert c != text, 'variant-closure line not found'
open('stage_C.py', 'w', encoding='utf-8').write(c)

# ---- stage B: decode-only ---------------------------------------------------------------------
b = text
# 1. drop the appended-particle block in __init__
pre, mid, post = cut_between(b, '        # ---- CLOSED-CLASS SURFACE RECOVERY: APPEND-ONLY',
                             "        self.root2id = {r: i for i, r in enumerate(self.roots_list)}")
b = pre + ORIG_ROOTS_BLOCK + post
# 2. drop the _closed_class_split method
pre, mid, post = cut_between(b, '    def _closed_class_split(self, clean):',
                             '    def encode_word(self, word: str)')
b = pre + post
# 3. restore the original root mapping in encode_word
pre, mid, post = cut_between(b, '        # Map Root\n', '        # Map Wazn\n')
b = pre + ORIG_MAP_ROOT + post
assert '_closed_class_split' not in b.split('class Far')[1], 'stage B still references the split'
assert 'self._closed_class_split(clean_w)' not in b, 'stage B still contains a recovery path'
assert 'self._closed_inventory' not in b, 'stage B still builds the inventory'
assert 'AL_QALB_PARTICLES' in b, 'stage B lost al-qalb'
assert 'A CONTROL TOKEN IS NEVER SURFACE TEXT' in b, 'stage B lost the invariant'
open('stage_B.py', 'w', encoding='utf-8').write(b)

for f in ('stage_B.py', 'stage_C.py'):
    compile(open(f, encoding='utf-8').read(), f, 'exec')
    print(f, 'OK', len(open(f, encoding='utf-8').read()))
