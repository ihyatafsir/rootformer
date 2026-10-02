#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
khalil_attestation.py -- extract al-Khalil's permutation-attestation table from Kitab al-'Ayn.

HOW AL-KHALIL STRIPS TO A ROOT
------------------------------
al-'Ayn is organised BY ROOT. Each entry declares the root's radicals as separate letters and then
enumerates its PERMUTATIONS (al-taqalib) with an attestation mark:

    (ع ج، ج ع مستعملان)        both permutations in use
    (ع ض ه مستعمل فقط)          only this one in use
    (... مهملا)                 unused

and he states the stripping criterion outright:

    "والألف التي في اسحنكك واقشعر واسحنفر واسبكر ليست من أصل البناء، وإنما أدخلت هذه الألفات"
    -- the alif in ishankaka / iqsha'arra is NOT part of the root structure; it was INSERTED.

So the root is what remains after the inserted letters are removed, and -- crucially -- the result
is VALIDATED against the attested permutations. A candidate root whose permutation is marked
"important/muhmal" (unused) is not a root, however plausible the segmentation looks.

This is the POSITIVE counterpart to the letter-pair prohibition rules. The project implemented only
those prohibitions (qaf+kaf, jim+qaf) and never used the attestation table at all -- which is why
the analyser happily produces roots like عنن from "عنهم" (a preposition + pronoun).

Output: khalil_permutation_attestation.json
  { "attested": [[c1,c2,c3], ...], "single": [...], "unused": [[...]], "unused_letters": [...] }

Usage: python khalil_attestation.py
"""
import json
import re
import sys
from collections import Counter

DIAC = re.compile(r'[\u064b-\u0652\u0670\u0640\u06d6-\u06ed]')
LETTER = r'[\u0621-\u064a]'
AR = re.compile(LETTER)


def norm(s):
    s = DIAC.sub('', s)
    for a, b in (('أ', 'ا'), ('إ', 'ا'), ('آ', 'ا'), ('ٱ', 'ا'), ('ى', 'ي'), ('ة', 'ه'),
                 ('ؤ', 'و'), ('ئ', 'ي')):
        s = s.replace(a, b)
    return s


def main():
    path = ('/home/grem3/Documents/deepseek-harness/default-workspace/'
            'rootformer/corpus/basran/Al_Khalil_Al_Ayn.txt')
    t = re.sub(r'\s+', ' ', norm(open(path, encoding='utf-8', errors='ignore').read()))

    attested, single, unused, unused_letters = [], [], [], []
    # a declaration is a parenthesised run of letter-groups followed by an attestation mark
    for m in re.finditer(r'\(([^()]{0,120}?)\)', t):
        body = m.group(1).strip()
        if not re.search(r'مستعمل|مهمل', body):
            continue
        marks = ('مستعملات', 'مستعملان', 'مستعمل فقط', 'مستعمل', 'مهملا', 'مهمل', 'فقط')
        clean = body
        for w in marks:
            clean = clean.replace(w, ' ')
        groups = re.findall(r'((?:[\u0621-\u064a]\s+){1,4}[\u0621-\u064a])', clean)
        perms = []
        for g in groups:
            letters = re.findall(LETTER, g)
            if 2 <= len(letters) <= 4:
                perms.append(letters)
        if not perms:
            continue
        if 'مستعمل فقط' in body or 'مستعملان' in body:
            single.extend(perms)
        elif 'مهمل' in body or 'مهملا' in body:
            unused.extend(perms)
            unused_letters.extend(perms)
        elif 'مستعملات' in body or 'مستعمل' in body:
            attested.extend(perms)

    # de-duplicate, preserving order
    def uniq(xs):
        seen, out = set(), []
        for x in xs:
            k = tuple(x)
            if k not in seen:
                seen.add(k)
                out.append(x)
        return out

    attested, single, unused = uniq(attested), uniq(single), uniq(unused)
    allatt = uniq(attested + single)

    print(f'[*] permutation declarations parsed from al-\'Ayn')
    print(f'    attested permutations (musta\'mal)      : {len(allatt)}')
    print(f'    explicitly unused (muhmal)              : {len(unused)}')
    print(f'    total distinct permutation records      : {len(allatt)+len(unused)}')
    print()
    print('    sample ATTESTED :', [' '.join(x) for x in allatt[:10]])
    print('    sample UNUSED   :', [' '.join(x) for x in unused[:10]])
    print()
    # which letter pairs does the attested set actually exercise?
    pairs = Counter()
    for x in allatt:
        for a, b in zip(x, x[1:]):
            pairs[(a, b)] += 1
    print(f'    adjacent letter pairs in the attested set: {len(pairs)}')
    print('    the most-attested pairs:')
    for k, v in pairs.most_common(10):
        print(f'       {"".join(k)}  {v}')

    out = {'attested': allatt, 'single_attested': single, 'unused': unused,
           'counts': {'attested': len(allatt), 'unused': len(unused)}}
    json.dump(out, open('/home/grem3/Documents/deepseek-harness/default-workspace/rootformer/results/khalil_permutation_attestation.json', 'w'),
              ensure_ascii=False, indent=1)
    print('\nwrote results/khalil_permutation_attestation.json')
    print('\nUSE: validate analyser output against this -- a candidate root whose permutation is')
    print('     marked muhmal is not a root, however plausible the segmentation looks.')


if __name__ == '__main__':
    main()
