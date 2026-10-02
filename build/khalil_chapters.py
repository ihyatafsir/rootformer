#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
khalil_chapters.py -- parse al-'Ayn's CHAPTER STRUCTURE, not parenthesised strings.

My previous attempt read every parenthesised run containing «مستعمل» or «مهمل» as a root triple.
That was the wrong reading, and it produced false verdicts: sh-ayn-ra and dad-ayn-bad came out
"unused" when they are among the commonest roots in the language.

The actual structure of al-'Ayn is two-level:

  (1) ROOT CHAPTERS -- a bab heading declaring the letters, then the permutation list with the
      chapter's attestation mark:

        باب العين والقاف والصاد (ع ق ص، ق ع ص، ق ص ع، ص ع ق، ص ق ع مستعملات)
        (ع ض ه مستعمل فقط)
        (... مهملا)

      For these, the mark IS a verdict on the listed permutations: مستعملات / مستعمل فقط ->
      they are roots; مهملا -> they are not.

  (2) LETTER-COMBINATION STATEMENTS -- a different construction entirely, saying that a letter
      with a SET of letters forms no root:

        العين مع هذه الحروف: الغين والهاء والحاء والخاء مهملا
        "the ayn with these letters -- ghayn, ha', ha', kha' -- is unused"

      These say NOTHING about sh-ayn-ra. Reading them as root triples was the bug.

This separates the two and builds the attestation set only from (1).

Usage: python khalil_chapters.py
"""
import json
import re
import sys
from collections import Counter

DIAC = re.compile(r'[\u064b-\u0652\u0670\u0640\u06d6-\u06ed]')
LET = r'[\u0621-\u064a]'


def norm(s):
    s = DIAC.sub('', s)
    for a, b in (('أ', 'ا'), ('إ', 'ا'), ('آ', 'ا'), ('ٱ', 'ا'), ('ى', 'ي'), ('ة', 'ه'),
                 ('ؤ', 'و'), ('ئ', 'ي')):
        s = s.replace(a, b)
    return s


def main():
    t = re.sub(r'\s+', ' ', norm(open(sys.argv[1] if len(sys.argv) > 1 else
                                      '/workspace/heritage_foundations/Al_Khalil_Al_Ayn.txt',
                                      encoding='utf-8', errors='ignore').read()))

    # ---- (1) root chapters -----------------------------------------------------------------
    chap = re.findall(r'باب\s+([^()]{3,80}?)\(([^()]{2,160}?)\)', t)
    attested, single, unused = [], [], []
    for heading, body in chap:
        perms = []
        for g in re.findall(r'((?:%s\s+){1,3}%s)' % (LET, LET), body):
            letters = re.findall(LET, g)
            if 2 <= len(letters) <= 4:
                perms.append(letters)
        if not perms:
            continue
        if 'مستعملات' in body or 'مستعمل' in body:
            (single if 'فقط' in body or 'مستعملان' in body else attested).extend(perms)
        elif 'مهمل' in body:
            unused.extend(perms)

    def uniq(xs):
        seen, out = set(), []
        for x in xs:
            k = tuple(x)
            if k not in seen:
                seen.add(k)
                out.append(x)
        return out

    attested, single, unused = uniq(attested), uniq(single), uniq(unused)
    good = uniq(attested + single)

    # ---- (2) letter-combination statements (separated, NOT treated as roots) ---------------
    combos = re.findall(r'(%s+)\s+مع\s+هذه الحروف\s*:?\s*([^.]{0,120}?)\s*مهملا?' % LET, t)
    combo_letters = []
    for lead, rest in combos:
        combo_letters.append((lead, re.findall(LET, rest)))

    print(f"=== al-'Ayn CHAPTER STRUCTURE ===")
    print(f'  root chapters parsed            : {len(chap)}')
    print(f'  attested permutation triples    : {len(good)}')
    print(f'  expressly unused (muhmal)       : {len(unused)}')
    print(f'  letter-combination statements   : {len(combo_letters)}  (kept separate)')
    print()
    print('  sample ATTESTED roots:', [' '.join(x) for x in good[:12]])
    print('  sample UNUSED   roots:', [' '.join(x) for x in unused[:8]])
    print('  sample combos (NOT roots):',
          [(l, ''.join(r)) for l, r in combo_letters[:4]])
    print()
    # ---- the sanity test that failed before -----------------------------------------------
    for probe in (['ش', 'ع', 'ر'], ['ع', 'ر', 'ض'], ['ع', 'ج', 'ب'], ['ع', 'ل', 'م']):
        in_good = tuple(probe) in {tuple(x) for x in good}
        in_bad = tuple(probe) in {tuple(x) for x in unused}
        print(f'  {"".join(probe):<6} attested={in_good}  unused={in_bad}')

    json.dump({'attested': good, 'single': single, 'unused': unused,
               'combinations': [{'lead': l, 'with': r} for l, r in combo_letters],
               'counts': {'chapters': len(chap), 'attested': len(good), 'unused': len(unused)}},
              open('/workspace/khalil_chapters.json', 'w'), ensure_ascii=False, indent=1)
    print('\nwrote /workspace/khalil_chapters.json')


if __name__ == '__main__':
    main()
