#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extend_awzan.py -- add the 12 missing wazn patterns to the blueprint, with id remapping.

The 125-pattern inventory lacks 7 of the 10 sighat muntaha al-jumu' (including FA-CAYN-ALIF-LAM,
which is what 'haqa'iq' needs) and 5 broken plurals. That absence forces the analyser to mislabel
such words, which then blocks generation and segmentation validation.

Inserting at the END of the awzan partition shifts every id after it, so this rebuilds the vocab
and VERIFIES that:
  * every control / alphabet / proclitic / particle token keeps its id
  * every existing wazn keeps its id (they are all before the insertion point)
  * every root shifts consistently, and root2id/id2root stay inverse
  * the partition table is contiguous and covers the whole vocab

Writes a NEW file; the original is untouched.

Usage: python extend_awzan.py
"""
import json
import sys

SRC = ('/home/grem3/Documents/deepseek-harness/default-workspace/rootformer/'
       'local_prep/code/data/rootformer_v12_arabic_blueprint.json')
DST = ('/home/grem3/Documents/deepseek-harness/default-workspace/rootformer/'
       'results/rootformer_v12_blueprint_137awzan.json')

NEW_PATTERNS = ['فَعَائِل', 'فَعَالِيل', 'فَعَالِل', 'فَوَاعِل', 'فَعَائِيل',
                'أَفَاعِيل', 'فَعَاوِيل', 'فُعُل', 'فِعْلَى', 'أَفْعِل', 'فُعَّل', 'فِعَّال']


def main():
    d = json.load(open(SRC))
    i2t = d['vocab_id_to_token']
    t2i = d['vocab_token_to_id']
    parts = d['partitions']
    N = int(d['total_vocab_size'])

    aw_lo, aw_hi = parts['classical_awzan']
    ru_lo, ru_hi = parts['classical_roots']
    pd_lo, pd_hi = parts['simd_padding']

    # sanity: the partitions must be contiguous and cover 0..N
    assert aw_hi == ru_lo, (aw_hi, ru_lo)
    assert ru_hi == pd_lo, (ru_hi, pd_lo)
    assert pd_hi == N, (pd_hi, N)

    # existing wazn tokens, and their display form
    aw_tokens = [i2t[str(i)] for i in range(aw_lo, aw_hi)]
    assert len(aw_tokens) == 125, len(aw_tokens)
    existing = {t.split('wazn_')[-1].rstrip('>') for t in aw_tokens}
    added = [p for p in NEW_PATTERNS if p not in existing]
    print(f'[*] existing awzan {len(aw_tokens)}; adding {len(added)} -> '
          f'{len(aw_tokens)+len(added)}')

    # ---- rebuild the id space -------------------------------------------------------
    new_i2t, new_t2i = {}, {}
    nid = 0
    parts_new = {}

    def emit(tok):
        nonlocal nid
        new_i2t[str(nid)] = tok
        new_t2i[tok] = nid
        nid += 1

    # 1. everything before the awzan partition, unchanged
    for i in range(0, aw_lo):
        emit(i2t[str(i)])
    assert nid == aw_lo
    # 2. awzan, with the additions appended
    nm = 0
    for tok in aw_tokens:
        emit(tok); nm += 1
    for p in added:
        emit(f'<wazn_{p}>'); nm += 1
    aw_hi_new = nid
    parts_new['classical_awzan'] = [aw_lo, aw_hi_new]
    # 3. roots, shifted
    for i in range(ru_lo, ru_hi):
        emit(i2t[str(i)])
    ru_hi_new = nid
    parts_new['classical_roots'] = [aw_hi_new, ru_hi_new]
    # 4. padding, shifted
    for i in range(pd_lo, pd_hi):
        emit(i2t[str(i)])
    pd_hi_new = nid
    parts_new['simd_padding'] = [ru_hi_new, pd_hi_new]

    # the earlier partitions are unchanged
    for k in ('control_and_punctuation', 'arabic_alphabet_and_diacritics',
              'proclitics_and_enclitics', 'sibawayh_particles'):
        parts_new[k] = parts[k]

    out = dict(d)
    out['vocab_id_to_token'] = new_i2t
    out['vocab_token_to_id'] = new_t2i
    out['partitions'] = parts_new
    out['total_vocab_size'] = pd_hi_new
    out['wazn_extension'] = {'added': added, 'from': 125, 'to': nm}

    # ---- VERIFY ------------------------------------------------------------------------
    errs = []
    # a) pre-awzan tokens keep their ids
    for i in range(0, aw_lo):
        if new_i2t[str(i)] != i2t[str(i)]:
            errs.append(f'pre-awzan id {i} changed')
    # b) existing awzan keep their ids
    for k, tok in enumerate(aw_tokens):
        if new_i2t[str(aw_lo + k)] != tok:
            errs.append(f'awzan id {aw_lo+k} changed')
    # c) the new patterns are present and reachable
    for p in added:
        tok = f'<wazn_{p}>'
        if tok not in new_t2i:
            errs.append(f'{tok} missing')
        elif new_i2t[str(new_t2i[tok])] != tok:
            errs.append(f'{tok} map not inverse')
    # d) every root shifted by exactly len(added) and stays inverse
    shift = len(added)
    for i in range(ru_lo, ru_hi):
        tok = i2t[str(i)]
        if new_i2t[str(i + shift)] != tok:
            errs.append(f'root {tok} not shifted by {shift}')
            break
    # e) partitions contiguous and complete
    order = ['control_and_punctuation', 'arabic_alphabet_and_diacritics',
             'proclitics_and_enclitics', 'sibawayh_particles', 'classical_awzan',
             'classical_roots', 'simd_padding']
    cur = 0
    for k in order:
        lo, hi = parts_new[k]
        if lo != cur:
            errs.append(f'partition {k} starts at {lo}, expected {cur}')
        cur = hi
    if cur != pd_hi_new:
        errs.append(f'partitions end at {cur}, vocab size {pd_hi_new}')
    # f) bijection
    if len(new_i2t) != pd_hi_new or len(new_t2i) != pd_hi_new:
        errs.append('id/token maps are not a bijection')
    if set(new_i2t.values()) != set(new_t2i.keys()):
        errs.append('id->token and token->id disagree')

    print(f'\n[*] new vocab size: {pd_hi_new}  (was {N})')
    for k in order:
        lo, hi = parts_new[k]
        print(f'    {k:<34} {lo:>5}..{hi-1:<5} ({hi-lo})')
    print()
    if errs:
        print(f'VERIFICATION FAILED ({len(errs)}):')
        for e in errs[:10]:
            print('   ', e)
        sys.exit(1)
    print('VERIFICATION PASSED:')
    print('   pre-awzan tokens unchanged; existing 125 wazn ids unchanged')
    print(f'   {len(added)} new wazn tokens added at {aw_hi}..{aw_hi+len(added)-1}')
    print(f'   every root shifted by exactly {shift}; maps are a bijection')
    print('   partitions contiguous and cover the whole vocab')
    json.dump(out, open(DST, 'w'), ensure_ascii=False)
    print(f'\nwrote {DST}')


if __name__ == '__main__':
    main()
