#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
khalil_fidelity.py -- is the Al-Khalil implementation faithful to Kitab al-'Ayn?

The question for a 1200-year-old validated system is not whether it 'works' but whether the
implementation matches what was specified. al-'Ayn states FOUR letter-incompatibility rules:

  1. الهمزة والغين لا تجتمعان في بناء كلمة واحدة            (hamza + ghayn)
  2. القاف والكاف لا يجتمعان في كلمة واحدة                  (qaf + kaf), except Arabized loans
  3. الجيم مع القاف لا يأتلف إلا بفصل لازم                   (jim + qaf), needs a separator
  4. الضاد والصاد لا يأتلفان في كلمة واحدة أصليّة الحروف      (dad + sad), when both are radicals

classical_governance_v2.KHALIL_LETTER_PAIRS implements only 2 and 3. Rules 1 and 4 are absent.
This measures what adding them excludes from the 9,114-root inventory, and lists concrete roots,
so the gain (or absence of it) is visible rather than asserted.

Usage: python khalil_fidelity.py
"""
import sys
from collections import Counter

sys.path.insert(0, '/workspace/hf_v19_2_release')
sys.path.insert(0, '/workspace/hf_v19_2_release/models')

DIAC = __import__('re').compile(r'[\u064b-\u0652\u0670\u0640]')


def strip(s):
    return DIAC.sub('', s)


# 1..4, transcribed from the text. Order-insensitive: al-Khalil states co-occurrence bans.
SPECIFIED = {
    'hamza+ghayn (al-Ayn 19418)': {('ء', 'غ'), ('غ', 'ء')},
    'qaf+kaf (al-Ayn 19594)': {('ق', 'ك'), ('ك', 'ق')},
    'jim+qaf (al-Ayn 19594)': {('ج', 'ق'), ('ق', 'ج')},
    'dad+sad (al-Ayn, asl al-huruf)': {('ض', 'ص'), ('ص', 'ض')},
}


def main():
    import nrmp_vocab as nv
    V = next(v for k, v in vars(nv).items() if isinstance(v, type) and 'MorphemicVocab' in k)
    vocab = V('/workspace/hf_v19_2_release/data/rootformer_v12_arabic_blueprint.json')

    from classical_governance_v2 import KHALIL_LETTER_PAIRS, AlKhalilV2
    print(f'[*] inventory: {vocab.num_roots} roots')
    print(f'[*] code currently implements: {sorted(KHALIL_LETTER_PAIRS)}\n')

    implemented = set(map(tuple, KHALIL_LETTER_PAIRS))
    roots = vocab.roots_list
    real = [(i, r) for i, r in enumerate(roots)
            if not str(r).startswith('<') and len(strip(r)) == 3]

    print(f'{"specified rule":<40}{"in code":>9}{"roots excluded":>16}')
    print('-' * 65)
    total_new = set()
    for name, pairs in SPECIFIED.items():
        incode = pairs <= implemented
        hit = [(i, r) for i, r in real
               if any((a, b) in pairs for a, b in zip(strip(r), strip(r)[1:]))]
        if not incode:
            total_new |= {i for i, _ in hit}
        print(f'{name:<40}{("yes" if incode else "NO"):>9}{len(hit):>16}')
        if hit and not incode:
            print(f'      roots the MISSING rule excludes: '
                  f'{[r for _, r in hit][:12]}')

    print(f'\n[*] roots newly excluded by the two missing rules: {len(total_new)}')
    if total_new:
        print(f'    examples: {[roots[i] for i in sorted(total_new)][:20]}')

    # does the shipped mask already catch them by another route?
    shipped = AlKhalilV2(vocab).mask()
    print(f'\n[*] shipped AlKhalilV2 mask size: {len(shipped)}')
    missed = [i for i in total_new if i not in shipped]
    print(f'[*] of the newly-specified roots, NOT caught by any shipped rule: {len(missed)}')
    if missed:
        print(f'    {[roots[i] for i in sorted(missed)][:20]}')

    # --- verification against the inventory's own evidence, as al-Khalil argues from attestation
    print('\n[*] al-Khalil argues from attestation ("daliiluhu annahum awqa`u...").')
    print('    observed adjacent letter pairs in the inventory:')
    obs = Counter()
    for _, r in real:
        s = strip(r)
        for a, b in zip(s, s[1:]):
            obs[(a, b)] += 1
    for name, pairs in SPECIFIED.items():
        for p in sorted(pairs):
            print(f'      {name:<38} {p}  attested {obs.get(p, 0)}x')


if __name__ == '__main__':
    main()
