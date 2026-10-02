#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Which content roots contain non-28-letter-Arabic symbols? Are any of the OLD
held-out roots among them? (Feasibility check for a leak-free radical alphabet.)"""
import json
import sys

sys.path.insert(0, '/workspace/hf_v19_2_release')
sys.path.insert(0, '/workspace/hf_v19_2_release/models')
import nrmp_vocab as nv

V = next(v for k, v in vars(nv).items() if isinstance(v, type) and 'MorphemicVocab' in k)
vocab = V('/workspace/hf_v19_2_release/data/rootformer_v12_arabic_blueprint.json')

ARABIC28 = set('ابتثجحخدذرزسشصضطظعغفقكلمنهوي')
content = {i: r for i, r in enumerate(vocab.roots_list) if not str(r).startswith('<')}
odd = {i: r for i, r in content.items() if any(ch not in ARABIC28 for ch in r)}
print('content roots with any symbol outside the 28 Arabic letters: %d' % len(odd))
for i, r in sorted(odd.items()):
    print('   id=%d root=%r  nonstandard=%r' % (i, r, sorted(set(r) - ARABIC28)))

meta = json.load(open('/workspace/sf_data/meta.json'))
hold = meta['hold_roots']
print('\nOLD hold_roots (n=%d):' % len(hold))
for r in hold:
    print('   id=%d %r' % (r, vocab.id2root.get(r)))
overlap = sorted(set(hold) & set(odd))
print('old held-out roots with nonstandard symbols: %d %r' % (len(overlap), overlap))
