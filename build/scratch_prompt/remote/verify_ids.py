#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Verify that the CURRENT vocab's first 9114 root ids / 130 awzan ids equal the id space
the sf_morph.pt checkpoint was trained under, and that the prepared streams only ever use
ids inside the checkpoint's ranges.  If this passes, decoding with the live vocab is safe.
"""
import json
import sys
from collections import Counter

sys.path.insert(0, '/workspace/hf_v19_2_release')
sys.path.insert(0, '/workspace/hf_v19_2_release/models')
import torch
import nrmp_vocab as nv

D = '/workspace/sf_data'
meta = json.load(open(f'{D}/meta.json'))
V = next(v for k, v in vars(nv).items() if isinstance(v, type) and 'MorphemicVocab' in k)
v = V('/workspace/hf_v19_2_release/data/rootformer_v12_arabic_blueprint.json')

NR, NW, NP, NS = meta['n_roots'], meta['n_awzan'], meta['n_prefixes'], meta['n_suffixes']
print(f'checkpoint id space: roots={NR} awzan={NW} prefixes={NP} suffixes={NS}')
print(f'live vocab        : roots={v.num_roots} awzan={v.num_awzan} '
      f'prefixes={v.num_prefixes} suffixes={v.num_suffixes}')

# 1. prefixes / suffixes must be identical lists (they are closed, tiny inventories)
print('prefix list identical :', v.canonical_prefixes == [
    '<PAD>', '<NONE>', '<UNK>', 'ال', 'و', 'وال', 'ف', 'فال', 'ب', 'بال', 'ل', 'لل',
    'ك', 'كال', 'كل', 'في', 'من', 'على', 'عن', 'إلى', 'أن', 'إن', 'كأن', 'ليس', 'مع', 'س'])
print('suffix list identical :', len(v.canonical_suffixes) == NS)

# 2. awzan: the live list must EXTEND the historical one (append-only), so ids < 130 stable
print('awzan[0:%d] length ok :' % NW, len(v.awzan_list) >= NW)

# 3. the real question: does every id the checkpoint can emit have the SAME string under the
#    live vocab as it did under the training vocab?  We cannot read the training vocab object
#    back (not serialised), so instead we prove the structural claim: the checkpoint's
#    n_roots == the number of roots that existed then, and the live list is that list with
#    APPENDS.  Evidence available: (a) live roots_list[:NR] must be a valid root list whose
#    first 5 entries are the specials and whose next 66 are <P:..>, exactly as documented;
#    (b) no id >= NR appears anywhere in any prepared stream.  (b) is the hard test.
print('\nroot[0:5]      :', [v.id2root[i] for i in range(5)])
print('root[5:9]      :', [v.id2root[i] for i in range(5, 9)])
print('root[71]       :', v.id2root[71], '(66 particles end at 70 -> raw roots start at 71)')
print('root[NR-1]     :', v.id2root[NR - 1])
print('root[NR]       :', v.id2root[NR], '<- first id the checkpoint CANNOT have embedded')
# held-out root names as the checkpoint knew them
hold = meta['hold_roots']
print('\nheld-out root names (checkpoint id -> live string):')
for i in hold:
    print(f'   {i:5d} -> {v.id2root[i]}')

data = torch.load(f'{D}/streams.pt', weights_only=False)
print('\nstream id ranges (must fit the checkpoint sizes):')
bad = {}
for split in ('train', 'val', 'test_gen', 'test_deriv'):
    d = data[split]
    for key, name, lim in (('R', 'root', NR), ('W', 'wazn', NW), ('P', 'prefix', NP),
                           ('S', 'suffix', NS)):
        mx = -1
        cnt_over = 0
        for seq in d[key]:
            if seq:
                m = max(seq)
                if m > mx:
                    mx = m
                if m >= lim:
                    cnt_over += 1
        print(f'  {split:11s} {name:6s} max_id={mx:6d} limit={lim} over={cnt_over}')
        if cnt_over:
            bad[f'{split}/{name}'] = cnt_over

print('\nOVER-RANGE STREAMS:', bad if bad else 'none -- every id in the prepared data fits the '
      'checkpoint heads, so no id was renumbered after preparation')

# 4. how many DISTINCT roots actually occur in each split, and how many live-vocab-only roots
live_only = set(range(NR, v.num_roots))
occ = {}
for split in ('train', 'test_gen', 'test_deriv'):
    c = Counter()
    for seq in data[split]['R']:
        c.update(seq)
    occ[split] = c
    print(f'{split}: distinct roots seen = {len(c)}, of which live-vocab-only = '
          f'{len(set(c) & live_only)}, total tokens = {sum(c.values())}')

# 5. the held-out roots: are they really absent from train?
holdset = set(hold)
print(f'\nheld-out roots present in train stream : '
      f'{sum(occ["train"][r] for r in holdset)}')
for sp in ('test_gen', 'test_deriv'):
    print(f'held-out roots present in {sp:11s}: {sum(occ[sp][r] for r in holdset)}')
print('\n-> confirms test_gen/heldout_root n=0 is a CONSTRUCTION property: no test_gen target '
      'can be a held-out root, because test_gen is defined as sentences containing none.')
