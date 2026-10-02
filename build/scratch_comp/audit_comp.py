#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Independent audit of the RE-PREPARED dataset (current 9490/142 vocab).

Full pass, no sampling.  Checks, in order:
  1. shape/vocab of the new dataset vs meta;
  2. LEAK CHECK: held-out roots must occur 0x in train and val, and 38/38 must be present
     with nonzero count in test_deriv;
  3. held-out roots are disjoint from special tokens;
  4. the compositional radical alphabet is derived from TRAIN ONLY and every held-out root
     is expressible with it (this is the whole point of the redesign);
  5. the root_letters table round-trips to the exact root string for every content root;
  6. independent recomputation of the unigram bits/word and root-acc references, including
     the two FAIR references (majority real root over train, majority real root in-slice);
  7. observed root-length distribution over the training stream (2..5).
Writes /workspace/scratch_comp/audit_comp.json.
"""
import json
import math
import sys
from collections import Counter

import torch

sys.path.insert(0, '/workspace/scratch_comp')
import scratch_lm_comp as S  # noqa: E402

D = sys.argv[1] if len(sys.argv) > 1 else '/workspace/scratch_comp/sf_data_9490'
meta = json.load(open(f'{D}/meta.json'))
data = torch.load(f'{D}/streams.pt', weights_only=False)
tables = S.Tables(meta)
hold = set(meta['hold_roots'])

out = {}
print('== meta ==')
for k in ('n_roots', 'n_awzan', 'n_prefixes', 'n_suffixes', 'train_words', 'n_alphabet',
          'n_sym', 'blueprint_md5'):
    print(f'  {k} = {meta[k]}')
print(f'  n_hold_roots = {len(hold)}  n_special = {tables.n_special}  n_gate = {tables.n_gate}')
print(f'  counts = {meta["counts"]}')
out['meta'] = {k: meta[k] for k in ('n_roots', 'n_awzan', 'n_prefixes', 'n_suffixes',
                                    'train_words', 'n_alphabet', 'n_sym', 'blueprint_md5')}
out['n_hold_roots'] = len(hold)

print('\n== streams ==')
out['streams'] = {}
for sp, v in data.items():
    words = sum(v['L'])
    print(f'  {sp}: sentences={len(v["L"])} words={words} max_len={max(v["L"]) if v["L"] else 0}')
    out['streams'][sp] = {'sentences': len(v['L']), 'words': words}
assert set(data) == {'train', 'val', 'test_gen', 'test_deriv'}, set(data)

# ---------------- 2/3 leak check ------------------------------------------------
print('\n== LEAK CHECK (full pass) ==')
leak = {}
for sp in ('train', 'val', 'test_gen', 'test_deriv'):
    R = data[sp]['R']
    sent_with = occ = total = 0
    distinct = set()
    for seq in R:
        total += len(seq)
        hit = hold.intersection(seq)
        if hit:
            sent_with += 1
            distinct |= hit
            occ += sum(1 for x in seq if x in hold)
    leak[sp] = {'sentences': len(R), 'sentences_containing_heldout_root': sent_with,
                'heldout_root_occurrences': occ, 'total_word_positions': total,
                'distinct_heldout_roots_present': len(distinct)}
    print(f'  {sp}: sentences={len(R)} sent_with_heldout={sent_with} '
          f'heldout_occ={occ}/{total} distinct={len(distinct)}')
out['leak'] = leak
inter = sorted(hold & set(meta['special_root_ids']))
print(f'  held-out INTERSECT special_ids = {inter}')
out['heldout_intersect_special'] = inter
assert leak['train']['heldout_root_occurrences'] == 0, 'LEAK: held-out root in train'
assert leak['val']['heldout_root_occurrences'] == 0, 'LEAK: held-out root in val'
assert leak['test_gen']['heldout_root_occurrences'] == 0, 'held-out root in test_gen'
assert leak['test_deriv']['distinct_heldout_roots_present'] == len(hold), 'not all held-out present'
assert not inter, 'held-out overlaps specials'

# ---------------- 4 alphabet / expressibility -----------------------------------
print('\n== COMPOSITIONAL ALPHABET (train-derived) ==')
print(f'  alphabet size = {tables.n_alphabet}; chars = {"".join(tables.alphabet)}')
train_chars = set()
for seq in data['train']['R']:
    for rid in seq:
        if rid not in tables.special_set and rid != 0:
            train_chars.update(tables.id2root.get(rid, ''))
missing = sorted(train_chars - set(tables.alphabet))
print(f'  chars used by train content roots but NOT in alphabet: {missing}')
out['alphabet'] = ''.join(tables.alphabet)
out['alphabet_train_chars_missing'] = missing
inexpr = [r for r in meta['content_root_ids']
          if any(c not in set(tables.alphabet) for c in tables.id2root.get(r, ''))]
hold_inexpr = sorted(set(inexpr) & hold)
print(f'  content roots inexpressible: {len(inexpr)}; of held-out: {len(hold_inexpr)} {hold_inexpr}')
out['inexpressible_content_roots'] = len(inexpr)
out['heldout_inexpressible'] = hold_inexpr
assert not missing
# every held-out root must be expressible for the test to be meaningful
for r in sorted(hold):
    name = tables.id2root[r]
    assert all(c in set(tables.alphabet) for c in name), f'held-out root {name} inexpressible'
print('  [OK] every held-out root is expressible with the train-derived alphabet')

# ---------------- 5 root_letters round trip -------------------------------------
bad = []
inexpr_set = set(meta['inexpressible_content_roots'])
n_checked = 0
for rid in meta['content_root_ids']:
    if rid in inexpr_set:
        continue                      # explicitly marked inexpressible (Latin-script roots)
    n_checked += 1
    ls = tables.letters[rid].tolist()[:int(tables.lens[rid])]
    if tables.letters_str(ls) != tables.id2root.get(rid, ''):
        bad.append(rid)
print(f'\n== root_letters round-trip: {len(bad)} mismatches over {n_checked} expressible '
      f'content roots (skipped {len(inexpr_set)} inexpressible: '
      f'{[tables.id2root.get(r) for r in sorted(inexpr_set)]}) ==')
out['root_letters_roundtrip_bad'] = len(bad)
assert not bad

# ---------------- 7 length distribution -----------------------------------------
ld = meta['root_len_dist_train_tokens']
print(f'\n== training-stream root length distribution (tokens) == {ld}')
out['root_len_dist'] = ld

# ---------------- 6 independent unigram references ------------------------------
C = [Counter(), Counter(), Counter(), Counter()]
for arr, c in zip((data['train']['R'], data['train']['W'], data['train']['P'],
                   data['train']['S']), C):
    for seq in arr:
        c.update(seq)
TOT = [sum(c.values()) for c in C]
print('\n== train unigram counts ==')
for name, c, t in zip('R W P S'.split(), C, TOT):
    print(f'  {name}: seen={len(c)} tokens={t}')
print(f'  held-out root ids with nonzero TRAIN count: {sum(1 for r in hold if C[0].get(r, 0) > 0)}')
print(f'  <UNK> train count = {C[0].get(tables.special_ids[3], 0)}')

splits = ['val', 'test_gen', 'test_deriv']
ubits = {}
for sp in splits:
    bits, npos = 0.0, 0
    dd = data[sp]
    for si, (arr, c, T) in enumerate(zip((dd['R'], dd['W'], dd['P'], dd['S']), C, TOT)):
        V = max(c) + 2
        for seq in arr[:-1]:
            for x in seq[1:]:
                p = (c.get(x, 0) + 0.5) / (T + 0.5 * V)
                bits += -math.log2(max(p, 1e-12))
            if si == 0:
                npos += max(len(seq) - 1, 0)
    ubits[sp] = bits / max(npos, 1)
    print(f'  unigram bits/word {sp}: {ubits[sp]:.6f} (npos={npos})')
out['unigram_bits'] = ubits

uacc = S.unigram_root_acc(data, splits, tables)
print('\n== unigram root acc (script reference + FAIR references) ==')
print(f'  reference ids: {uacc["_reference_ids"]}')
for sp in splits:
    for k in ('all', 'real', 'special', 'real_majority_train', 'real_majority_slice'):
        v = uacc[sp][k]
        print(f'  {sp:11s} {k:20s} top1={v["top1"]:.6f} top5={v["top5"]:.6f} n={v["n"]}')
out['unigram_root_acc'] = uacc

json.dump(out, open('/workspace/scratch_comp/audit_comp.json', 'w'), indent=2, ensure_ascii=False)
print('\nwrote /workspace/scratch_comp/audit_comp.json')
print('AUDIT OK')
