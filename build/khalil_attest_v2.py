#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
khalil_attest_v2.py -- parse BOTH of al-'Ayn's chapter formats.

Catalogue of the 1,888 headings:
    359  with a parenthesised permutation list   -- the TWO-letter chapters
    446  with a mark, permutations DASH-separated -- the THREE-letter chapters
   1083  plain ("bab al-rajul", "bab alladhi mada") -- the word bab as a common noun, not a head

The three-letter chapters are the ones that carry triliteral roots, and their format is:

    باب العين والدال واللام معهما ع د ل- ع ل د- دلع مستعملات د ع ل ...

so the permutations are separated by DASHES and the chapter mark follows. Parsing only the
parenthesised form (my first two attempts) captured the two-letter chapters and missed every
triliteral root -- which is exactly why sh-ayn-ra came out absent.

Marks:  مستعملات  |  مستعمل فقط / مستعملان  |  مهملا

Usage: python khalil_attest_v2.py
"""
import json
import re
import sys
from collections import Counter

DIAC = re.compile(r'[\u064b-\u0652\u0670\u0640\u06d6-\u06ed]')
LET = r'[\u0621-\u064a]'
MARKS = r'(مستعملات|مستعمل فقط|مستعملان|مستعمل|مهملا|مهمل)'


def norm(s):
    s = DIAC.sub('', s)
    for a, b in (('أ', 'ا'), ('إ', 'ا'), ('آ', 'ا'), ('ٱ', 'ا'), ('ى', 'ي'), ('ة', 'ه'),
                 ('ؤ', 'و'), ('ئ', 'ي')):
        s = s.replace(a, b)
    return s


def perms_from(run):
    """Read 'X Y Z- X Z Y- ...' into letter triples."""
    out = []
    for chunk in run.split('-'):
        letters = re.findall(LET, chunk)
        if 2 <= len(letters) <= 4:
            out.append(letters)
    return out


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else \
        '/workspace/heritage_foundations/Al_Khalil_Al_Ayn.txt'
    t = re.sub(r'\s+', ' ', norm(open(path, encoding='utf-8', errors='ignore').read()))

    attested, single, unused = [], [], []
    n_dash = n_paren = 0

    # ---- FORMAT A: dash-separated, "mucahuma <perms> <mark>" -----------------------------
    # do NOT require the connective 'mucahuma' -- other chapters introduce the permutation run
    # differently, and requiring it captured only ~23% of the marked headings.
    triple = r'%s\s+%s\s+%s' % (LET, LET, LET)
    for m in re.finditer(r'((?:' + triple + r'\s*-\s*){0,6}' + triple + r')\s*' + MARKS, t):
        run, mark = m.group(1), m.group(2)
        ps = perms_from(run)
        if not ps:
            continue
        n_dash += 1
        if 'مهمل' in mark:
            unused.extend(ps)
        elif 'فقط' in mark or 'مستعملان' in mark:
            single.extend(ps)
        else:
            attested.extend(ps)

    # ---- FORMAT B: parenthesised (the two-letter chapters) -------------------------------
    for m in re.finditer(r'باب\s+[^()]{3,70}?\(([^()]{2,140}?)\)', t):
        body = m.group(1)
        mark = re.search(MARKS, body)
        if not mark:
            continue
        clean = body.replace(mark.group(1), ' ')
        ps = perms_from(clean)
        if not ps:
            continue
        n_paren += 1
        if 'مهمل' in mark.group(1):
            unused.extend(ps)
        elif 'فقط' in mark.group(1) or 'مستعملان' in mark.group(1):
            single.extend(ps)
        else:
            attested.extend(ps)

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
    goodset = {tuple(x) for x in good}
    badset = {tuple(x) for x in unused}

    print(f"=== al-'Ayn attestation, both formats ===")
    print(f'  dash-format chapters (3-letter) : {n_dash}')
    print(f'  paren-format chapters (2-letter): {n_paren}')
    print(f'  attested triples                : {len(good)}')
    print(f'  unused (muhmal)                 : {len(unused)}')
    print()
    print('  sample attested:', [' '.join(x) for x in good[:10]])
    print('  sample unused  :', [' '.join(x) for x in unused[:8]])
    print()
    print('  SANITY -- common roots that the earlier parse wrongly missed:')
    for probe in (['ش', 'ع', 'ر'], ['ع', 'ر', 'ض'], ['ع', 'ج', 'ب'], ['ع', 'ل', 'م'],
                  ['ع', 'د', 'ل'], ['ك', 'ت', 'ب']):
        p = tuple(probe)
        print(f'    {"".join(probe):<6} attested={p in goodset}  unused={p in badset}')

    json.dump({'attested': good, 'single': single, 'unused': unused,
               'counts': {'dash_chapters': n_dash, 'paren_chapters': n_paren,
                          'attested': len(good), 'unused': len(unused)}},
              open('/workspace/khalil_attest_v2.json', 'w'), ensure_ascii=False, indent=1)
    print('\nwrote /workspace/khalil_attest_v2.json')


if __name__ == '__main__':
    main()
