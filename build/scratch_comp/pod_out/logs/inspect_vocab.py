#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Inspect the CURRENT release vocab for the compositional-root-decoder design.

Read-only. Answers: how many roots/awzan, the root-length distribution over CONTENT
(non-special) roots, and the radical alphabet + its coverage.
"""
import json
import sys
from collections import Counter

sys.path.insert(0, '/workspace/hf_v19_2_release')
sys.path.insert(0, '/workspace/hf_v19_2_release/models')

import nrmp_vocab as nv

V = next(v for k, v in vars(nv).items() if isinstance(v, type) and 'MorphemicVocab' in k)
vocab = V('/workspace/hf_v19_2_release/data/rootformer_v12_arabic_blueprint.json')

print('roots=%d awzan=%d prefixes=%d suffixes=%d'
      % (vocab.num_roots, vocab.num_awzan, vocab.num_prefixes, vocab.num_suffixes))

specials = [r for r in vocab.roots_list if str(r).startswith('<')]
content = [r for r in vocab.roots_list if not str(r).startswith('<')]
print('special roots=%d  content roots=%d' % (len(specials), len(content)))
print('first 10 specials:', specials[:10])

len_c = Counter(len(r) for r in content)
print('content root length distribution:', dict(sorted(len_c.items())))
print('  -> fraction len<=5: %.6f' % (sum(v for k, v in len_c.items() if k <= 5) / len(content)))
print('  -> roots longer than 5:', [r for r in content if len(r) > 5][:40])

allchars = Counter()
for r in content:
    for ch in r:
        allchars[ch] += 1
print('distinct chars across content roots: %d' % len(allchars))
print('char frequency (sorted desc):')
for ch, n in allchars.most_common():
    print('   %r U+%04X %d' % (ch, ord(ch), n))

# alphabet that covers ~all roots
alpha = [ch for ch, _ in allchars.most_common()]
for k in (28, 29, 30, 32, 35, 40, len(alpha)):
    if k > len(alpha):
        continue
    a = set(alpha[:k])
    miss = [r for r in content if any(ch not in a for ch in r)]
    print('alphabet size %d -> roots not coverable: %d' % (k, len(miss)))

print('special root ids:', [i for i, r in enumerate(vocab.roots_list) if str(r).startswith('<')][:80])

meta = json.load(open('/workspace/sf_data/meta.json'))
print('EXISTING sf_data meta: n_roots=%s n_awzan=%s n_prefixes=%s n_suffixes=%s train_words=%s'
      % (meta['n_roots'], meta['n_awzan'], meta['n_prefixes'], meta['n_suffixes'],
         meta['train_words']))
print('  hold_roots n=%d' % len(meta['hold_roots']))
print('  counts=%s' % (meta['counts'],))
print('  test_files=%d val_files=%d' % (len(meta['test_files']), len(meta['val_files'])))
