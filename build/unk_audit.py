#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
unk_audit.py -- how often does the morphemic analyzer actually fail?

The corpus inventory reported "100% analysable" because encode_sentence() returns non-empty for
every word. But a tuple can carry the <UNK> root, which is not an analysis. This measures the real
failure rate, per word and in sentence context, and prints the most frequent roots so that special
tokens showing up at the top of the frequency table are visible.
"""
import re
import sys
from collections import Counter

sys.path.insert(0, '/workspace/hf_v19_2_release')
sys.path.insert(0, '/workspace/hf_v19_2_release/models')
import glob

import nrmp_vocab as nv

V = next(v for k, v in vars(nv).items() if isinstance(v, type) and 'MorphemicVocab' in k)
v = V('/workspace/hf_v19_2_release/data/rootformer_v12_arabic_blueprint.json')

print(f'UNK_ROOT id      : {v.UNK_ROOT} -> {v.id2root.get(v.UNK_ROOT)!r}')
print(f'<PARTICLE> id    : {v.root2id.get("<PARTICLE>")}')
print(f'num_roots        : {v.num_roots}')
print()

f = sorted(glob.glob('/workspace/scholastic_sanitized/*.txt'))[0]
txt = open(f, encoding='utf-8', errors='ignore').read()
words = [w for w in re.split(r'\s+', txt) if re.search(r'[\u0600-\u06FF]', w)][:20000]

unk_w = sum(1 for w in words if any(t[1] == v.UNK_ROOT for t in v.encode_sentence(w)))
print(f'per-WORD   : {unk_w}/{len(words)} = {100*unk_w/len(words):.2f}% yield an UNK root')

sents = [s.strip() for s in re.split(r'[\n.!?؟]+', txt) if len(s.split()) >= 4][:400]
n = r = 0
for s in sents:
    for t in v.encode_sentence(s):
        n += 1
        r += (t[1] == v.UNK_ROOT)
print(f'in-SENTENCE: {r}/{n} = {100*r/max(n,1):.2f}% of positions carry an UNK root')
print()

c = Counter()
for w in words[:200000]:
    for t in v.encode_sentence(w):
        c[v.id2root.get(t[1], '?')] += 1
tot = sum(c.values())
print('top-14 most frequent "roots" (per-word analysis):')
for k, n2 in c.most_common(14):
    tag = '  <-- SPECIAL' if str(k).startswith('<') else ''
    print(f'   {str(k):<16} {n2:>8}  {100*n2/tot:5.2f}%{tag}')

real = sum(n2 for k, n2 in c.items() if not str(k).startswith('<') and len(str(k)) == 3)
print(f'\nreal triconsonantal roots: {100*real/tot:.2f}% of analysed positions')
print(f'distinct real roots seen : {len([k for k in c if not str(k).startswith("<") and len(str(k))==3])}')
