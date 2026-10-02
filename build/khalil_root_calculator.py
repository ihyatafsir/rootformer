#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
khalil_root_calculator.py -- al-Khalil's algorithm for CALCULATING possible roots.

WHY A CALCULATOR AND NOT A LOOKUP
---------------------------------
I had been testing segmentation with lexicon membership: "is this root in the 9,013 list?".
That cannot discriminate, because the lexicon contains نتت، كمم، عنن، بلل -- so 'kuntu' stays
k+n+t+t and 'fakum' stays k+m+m.

al-Khalil does not work by membership. He CALCULATES the possible roots from the letters:

  1. PERMUTE  the radicals (al-taqalib) -- a triliteral has up to 6 orders;
  2. MAKHRAJ   reject letters that share an articulation point (they do not co-occur);
  3. PAIRS     reject the pairs he names explicitly, e.g. qaf+kaf and jim+qaf
               "illa an takuna al-kalima mu'arraba min kalami al-cajam";
  4. DHALQ     a quadriliteral or quinqueliteral must contain one of ر ل ن ف ب م;
  5. ATTEST    mark the residue musta'mal / muhmal.

Steps 1-4 need NO attestation data -- they are pure computation, and step 5 is the empirical
record that my extraction only covered for the cayn sections.

Step 1 is what membership testing can never do: for 'kuntu' the correct root is كون, whose waw is
ELIDED in the surface. You cannot look that up; you have to restore the weak letter and test.

Usage: python khalil_root_calculator.py
"""
import itertools
import re
import sys

DIAC = re.compile(r'[\u064b-\u0652\u0670\u0640\u06d6-\u06ed]')
LETTERS = list('ابتثجحخدذرزسشصضطظعغفقكلمنهويء')
WEAK = 'وي'
# articulation points (makhraj), in al-Khalil's own order
MAKHRAJ = {
    'halq':      set('ءهعحغخ'),
    'jih':       set('قك'),
    'shajriya':  set('جشي'),
    'hafawiyya': set('ض'),
    'asaliyya':  set('صسز'),
    'nitiya':    set('طدت'),
    'lisawiyya': set('ظذث'),
    'dhalqiyya': set('رلن'),
    'shafawiyya': set('فبم'),
}
# al-Khalil's explicit prohibitions (Kitab al-'Ayn)
FORBIDDEN_PAIRS = {('ق', 'ك'), ('ك', 'ق'), ('ج', 'ق'), ('ق', 'ج')}
DHALQ = set('رلنفبم')


def strip_diac(s):
    return DIAC.sub('', s)


def makhraj_ok(root):
    """Two letters from the same makhraj do not co-occur in a root."""
    for a, b in itertools.combinations(root, 2):
        if a == b:
            continue          # the DOUBLED root (idgham): C2==C3 is legitimate
        for pt in MAKHRAJ.values():
            if a in pt and b in pt:
                return False, f'{a}+{b} share a makhraj'
    return True, ''


def pairs_ok(root):
    for a, b in zip(root, root[1:]):
        if (a, b) in FORBIDDEN_PAIRS:
            return False, f'al-Khalil forbids {a}+{b}'
    return True, ''


def dhalq_ok(root):
    if len(root) >= 4 and not any(c in DHALQ for c in root):
        return False, 'no dhalq/labial letter in a 4+ root'
    return True, ''


def is_possible(root):
    """al-Khalil's tests 2-4, in order."""
    r = strip_diac(root)
    if len(r) < 2 or any(c not in LETTERS for c in r):
        return False, 'not all radicals'
    if r[0] == r[1] and len(r) > 2:
        pass                      # C1==C2 impossible, but a doubled root is C2==C3
    if len(r) == 3 and r[0] == r[1]:
        return False, 'C1==C2 impossible in a triliteral'
    for test in (makhraj_ok, pairs_ok, dhalq_ok):
        ok, why = test(r)
        if not ok:
            return False, why
    return True, 'possible'


def calculate_roots(stem, n=3):
    """CALCULATE the possible roots for a stem, rather than looking them up.

    A stem shorter than the root means a weak letter was elided (i'lal), so restore it in every
    position and permute. That is how 'kuntu' reaches كون.
    """
    s = strip_diac(stem)
    base = [c for c in s if c in LETTERS]
    # a letter written as alif / alif maqsura / hamza may STAND FOR a weak radical (al-qalb)
    SUBST = {'ا': WEAK, 'ى': WEAK, 'ء': WEAK, 'ئ': WEAK, 'ؤ': WEAK}
    out = set()

    def expand(letters):
        """every reading of the letters, allowing qalb substitution"""
        pools = [[]]
        for c in letters:
            opts = [c] + (list(SUBST[c]) if c in SUBST else [])
            pools = [p + [o] for p in pools for o in opts]
        return pools

    readings = expand(base)
    for rd in readings:
        # GENERAL REDUCTION: take ANY subset of the surface letters as the radical material,
        # then restore weak letters to reach n. This covers, in one rule:
        #   affixes        (kuntu -> drop the ta' -> k-n)
        #   elision        (kuntu -> restore the waw -> k-w-n)
        #   qalb           (qala -> alif read as waw -> q-w-l)
        for k in range(max(0, n - 2), min(len(rd), n) + 1):
            for keep in itertools.combinations(range(len(rd)), k):
                kept = [rd[i] for i in keep]
                need = n - k
                if need == 0:
                    pools = [kept]
                else:
                    pools = []
                    for combo in itertools.product(WEAK, repeat=need):
                        for pos in itertools.combinations(range(n), need):
                            cand = list(kept)
                            for i, at in enumerate(sorted(pos)):
                                cand.insert(at, combo[i])
                            pools.append(cand)
                for pool in pools:
                    for perm in set(itertools.permutations(pool)):
                        r = ''.join(perm)
                        ok, _ = is_possible(r)
                        if ok:
                            out.add(r)
    return out


def main():
    print("=== al-Khalil's calculation, on the words the lexicon could not fix ===\n")
    cases = [('كنت', 'كون', 'kana + -tu: the waw is ELIDED in the surface'),
             ('فكم', None, 'fa- + kam -- a particle, no root at all'),
             ('عنهم', None, 'can + hum -- a particle + pronoun'),
             ('قال', 'قول', 'the waw transposed to alif'),
             ('باع', 'بيع', 'the ya transposed to alif'),
             ('حقائق', 'حقق', 'the doubled cayn separated')]
    for stem, want, note in cases:
        got = calculate_roots(stem)
        hit = (want in got) if want else None
        mark = 'FOUND' if hit else ('n/a' if want is None else 'MISS')
        shown = sorted(got)[:6]
        print(f'  {stem:<8} -> {mark:<6} {note}')
        print(f'           candidates: {shown}{" ..." if len(got) > 6 else ""}'
              f'   ({len(got)} total)')
        if want:
            print(f'           wanted {want}')

    print('\n=== the tests themselves, on al-Khalil\'s own examples ===')
    for r, expect in (('قك', False), ('جق', False), ('حضثج', False), ('حقق', True),
                      ('كتب', True), ('ضرب', True)):
        ok, why = is_possible(r)
        print(f'  {r:<6} possible={ok!s:<5} {"ok" if ok == expect else "XX"}  {why}')


if __name__ == '__main__':
    main()
