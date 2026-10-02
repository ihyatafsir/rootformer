#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
khalil_attest_v3.py -- anchor the permutation span on the chapter's OWN declared letters.

Attempt 3 matched any dash-separated letter triple followed by a mark, anywhere. That over-matched:
it swept in spans from the "al-ayn with X" enumeration sections, so the extracted set contained
things like د ه ع and ه ر ع while sh-ayn-ra, dad-ayn-ra, ayn-jim-ba, ayn-lam-mim, kaf-ta-ba were
all absent -- i.e. it was not the root set.

The anchor is the heading itself. al-'Ayn's chapter heading NAMES its letters:

    باب العين والدال واللام معهما  ع د ل- ع ل د- دلع مستعملات ...

so the declared letter set is {ع, د, ل}, and a permutation is admissible ONLY if its letters are
exactly that set. That single constraint removes the over-matching, and the heading also tells us
how many letters the root has (2, 3 or 4), which is the scope condition for al-Khalil's dhalq rule.

Usage: python khalil_attest_v3.py
"""
import json
import re
import sys
from collections import Counter

DIAC = re.compile(r'[\u064b-\u0652\u0670\u0640\u06d6-\u06ed]')
LET = r'[\u0621-\u064a]'
MARKS = r'(مستعملات|مستعمل فقط|مستعملان|مستعمل|مهملا|مهمل)'
# a chapter heading names its letters as  الX والY والZ ...
HEAD = re.compile(r'باب\s+((?:وال' + LET + r'+|ال' + LET + r'+)(?:\s+(?:وال' + LET + r'+|ال' + LET + r'+)){0,3})')


def norm(s):
    s = DIAC.sub('', s)
    for a, b in (('أ', 'ا'), ('إ', 'ا'), ('آ', 'ا'), ('ٱ', 'ا'), ('ى', 'ي'), ('ة', 'ه'),
                 ('ؤ', 'و'), ('ئ', 'ي')):
        s = s.replace(a, b)
    return s


def heading_letters(h):
    out = []
    for tok in re.findall(r'(?:وال|ال)(' + LET + r'+)', h):
        if tok:
            out.append(tok[0])                      # the letter name's first letter IS the letter
    return out


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else \
        '/workspace/heritage_foundations/Al_Khalil_Al_Ayn.txt'
    t = re.sub(r'\s+', ' ', norm(open(path, encoding='utf-8', errors='ignore').read()))

    attested, single, unused = [], [], []
    n_ch = 0
    for m in HEAD.finditer(t):
        letters = heading_letters(m.group(1))
        if not (2 <= len(letters) <= 4):
            continue
        want = tuple(sorted(letters))
        span = t[m.end():m.end() + 500]
        # a chapter may carry SEVERAL mark groups -- some permutations attested, others
        # "muhmalat" (unused), as in:  ...d-l- ... musta'malat   d-c-l- ... muhmalat
        marks = list(re.finditer(MARKS, span))
        if not marks:
            continue
        n_ch += 1
        pos = 0
        for mk in marks:
            run = span[pos:mk.start()]
            pos = mk.end()
            # strip the CONNECTIVE before the permutation run: "mucahuma" contributes its own
            # letters (m-ayn-h-m-alif) and made every chunk 6 letters long, so clean chapters
            # like ayn-dal-lam were silently rejected.
            for conn in ('معهما', 'معهن', 'معها', 'معهم', 'معه'):
                run = run.replace(conn, ' ')
            ps = []
            # the separator is a DASH *or* a COMMA depending on the chapter
            for chunk in re.split(r'[-،,]', run):
                got = re.findall(LET, chunk)
                if len(got) == len(letters) and tuple(sorted(got)) == want:
                    ps.append(got)
            if not ps:
                continue
            mark = mk.group(1)
            if 'مهمل' in mark:
                unused.extend(ps)
            elif 'فقط' in mark or 'مستعملان' in mark:
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
    goodset, badset = {tuple(x) for x in good}, {tuple(x) for x in unused}

    print("=== al-'Ayn attestation, anchored on the chapter's declared letters ===")
    print(f'  chapters with a usable permutation run : {n_ch}')
    print(f'  attested triples                       : {len(good)}')
    print(f'  unused (muhmal)                        : {len(unused)}')
    print()
    print('  sample attested:', [' '.join(x) for x in good[:10]])
    print('  sample unused  :', [' '.join(x) for x in unused[:8]])
    print()
    print('  SANITY -- the common roots the earlier parses missed:')
    hits = 0
    for probe in (['ش', 'ع', 'ر'], ['ع', 'ر', 'ض'], ['ع', 'ج', 'ب'], ['ع', 'ل', 'م'],
                  ['ع', 'د', 'ل'], ['ك', 'ت', 'ب'], ['ن', 'ص', 'ر'], ['ض', 'ر', 'ب'],
                  ['ف', 'ع', 'ل'], ['ق', 'و', 'ل']):
        p = tuple(probe)
        ok = p in goodset
        hits += ok
        print(f'    {"".join(probe):<6} attested={ok}  unused={p in badset}')
    print(f'\n  {hits}/10 common roots present')

    json.dump({'attested': good, 'single': single, 'unused': unused,
               'counts': {'chapters': n_ch, 'attested': len(good), 'unused': len(unused)}},
              open('/workspace/khalil_attest_v3.json', 'w'), ensure_ascii=False, indent=1)
    print('\nwrote /workspace/khalil_attest_v3.json')


if __name__ == '__main__':
    main()
