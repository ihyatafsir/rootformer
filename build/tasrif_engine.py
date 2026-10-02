#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tasrif_engine.py -- ONE morphological engine, both directions, with conditioned operations.

WHY THIS REPLACES THE PREVIOUS APPROACH
---------------------------------------
Every bug in the earlier modules had the same shape: a rule applied where its CONDITION did not
hold (qalb applied to C1 where the operation is hadhf; qalb applied where a weak letter follows
and the transposition is blocked; a prothetic added to an already-pronounceable stem). Those were
not gaps in the sources -- they were design errors from building each rule as a STRING
TRANSFORMATION WITH AN AD-HOC GUARD.

The classical operations are OPERATIONS WITH CONDITIONS over phonological structure. So:

  * the pattern is parsed into an explicit sequence of (letter, vowel) -- the rules must be able
    to ASK "is the weak radical vowelled, and what precedes it";
  * each operation declares its own condition and is applied in the tradition's order:
        ibdal / idgham (structure)  ->  qalb (colour)  ->  hadhf (deletion)  ->  i'rab (final)
  * ANALYSIS IS THE INVERSE OF THE SAME GENERATOR, not a separate repair ladder. The two
    directions therefore cannot disagree -- which is what broke thara -> thur earlier.

Usage:
  python tasrif_engine.py            # hand-checkable cases + a corpus round trip
"""
import argparse
import glob
import json
import re
import sys
from collections import Counter

sys.path.insert(0, '/workspace/hf_v19_2_release')
sys.path.insert(0, '/workspace/hf_v19_2_release/models')

DIAC = re.compile(r'[\u064b-\u0652\u0670\u0640\u06d6-\u06ed]')
FATHA, DAMMA, KASRA, SUKUN, SHADDA = '\u064e', '\u064f', '\u0650', '\u0652', '\u0651'
ARABIC = set('ابتثجحخدذرزسشصضطظعغفقكلمنهويءأإآؤئى')
WEAK = set('وايى')
SLOT = {'ف': 0, 'ع': 1, 'ل': 2}
MUdARIA = set('يتنأ')


def strip_diac(s):
    return DIAC.sub('', s)


# ---------------------------------------------------------------------------- structure
class Pattern:
    """A wazn as explicit (letter, vowel) elements. The rules query this; they never guess."""

    def __init__(self, name):
        self.name = name
        self.elems = []                      # list of [char, vowel]
        raw = name or ''
        i = 0
        while i < len(raw):
            ch = raw[i]
            i += 1
            if ch not in ARABIC:
                continue
            v = ''
            while i < len(raw) and raw[i] not in ARABIC:
                v += raw[i]
                i += 1
            self.elems.append([ch, v])

    def slot_positions(self):
        return [i for i, (c, _) in enumerate(self.elems) if c in SLOT]

    def vowel_before(self, idx):
        """The vowel carried by the element immediately before idx (or '')."""
        return self.elems[idx - 1][1] if idx > 0 else ''

    def literal_after(self, idx):
        for c, v in self.elems[idx + 1:]:
            if c not in SLOT:
                return c
        return None


class Root:
    @staticmethod
    def klass(r):
        r = strip_diac(r)
        if len(r) != 3:
            return 'OTHER'
        if any(c in 'ءأإآؤئ' for c in r):
            return 'MAHMUZ'
        w = [i for i, c in enumerate(r) if c in WEAK]
        if len(w) >= 2:
            return 'LAFIF'
        if not w:
            return 'MUDAAF' if r[1] == r[2] else 'SALIM'
        return {0: 'MITHAL', 1: 'AJWAF', 2: 'NAQIS'}[w[0]]


# ---------------------------------------------------------------------------- operations
class Operation:
    """An operation declares its condition; the engine only calls it when it applies."""
    name = 'op'

    def applies(self, state, ctx):
        return False

    def apply(self, state, ctx):
        return state


class IbdalHamza(Operation):
    """al-ibdal: hamza substitution. A hamza radical is written in its seat."""
    name = 'ibdal'

    def applies(self, state, ctx):
        return ctx['class'] == 'MAHMUZ'

    def apply(self, state, ctx):
        return state          # the inventory stores hamza roots in surface form already


class Idgham(Operation):
    """al-idgham: a doubled root (C2 == C3) collapses to one geminate."""
    name = 'idgham'

    def applies(self, state, ctx):
        return ctx['class'] == 'MUDAAF'

    def apply(self, state, ctx):
        out = []
        i = 0
        while i < len(state):
            ch, v, role, k = state[i]
            if (role == 'slot' and k == 1 and i + 1 < len(state)
                    and state[i + 1][3] == 2 and state[i + 1][0] == ch):
                trail = state[i + 1][1]
                nxt = state[i + 2][1] if i + 2 < len(state) else ''
                out.append([ch, SHADDA + (nxt or ''), 'slot', 1])
                i += 2
                continue
            out.append([ch, v, role, k])
            i += 1
        return out


class Qalb(Operation):
    """al-qalb, with the conditions al-Mumti' states.

    Rule (Ibn 'Usfur, al-Mumti' fi al-Tasrif):
        "al-dammah is part of the waw, al-kasrah part of the ya', al-fathah part of the alif"
        -> a weak radical takes the COLOUR of the vowel before it.
    Conditions, all of which the earlier string-based version ignored:
        * applies to C2 / C3 only -- for MITHAL the operation is HADHF, not qalb;
        * requires a preceding vowel;
        * is BLOCKED when a weak literal of the same kind follows in the pattern -- Arabic
          avoids the meeting of two weak letters ("they dislike the meeting of the ya' and the
          waw"), which is why 'ayn + fu'ul is 'uyun, not 'uwun.
    """
    name = 'qalb'
    COLOUR = {DAMMA: 'و', KASRA: 'ي', FATHA: 'ا'}

    def applies(self, state, ctx):
        if ctx['class'] not in ('AJWAF', 'NAQIS', 'LAFIF'):
            return False
        for i, (ch, v, role, k) in enumerate(state):
            if role == 'slot' and k in (1, 2) and ch in WEAK and self._preceding_vowel(state, i):
                if not self._blocked(state, i, ch):
                    return True
        return False

    @staticmethod
    def _preceding_vowel(state, i):
        j = i - 1
        while j >= 0:
            if state[j][1]:
                return state[j][1]
            j -= 1
        return ''

    def _blocked(self, state, i, ch):
        want = self.COLOUR.get(self._preceding_vowel(state, i), ch)
        for ch2, v2, role2, k2 in state[i + 1:]:
            if role2 == 'lit':
                return ch2 in WEAK and ch2 == want
        return False

    def apply(self, state, ctx):
        out = [list(e) for e in state]
        for i, (ch, v, role, k) in enumerate(out):
            if role != 'slot' or k not in (1, 2) or ch not in WEAK:
                continue
            # CONDITION: a NAQIS (C3-weak) root is transposed in the PERFECT (ramaya -> rama)
            # but KEEPS its weak letter in the imperfect (yamdi, not yamda). The mudara'ah
            # prefix is what distinguishes them.
            if k == 2 and ctx.get('pattern_first_is_mudaria'):
                # NAQIS imperfect: the weak final carries no written vowel, and the 'ayn
                # harmonises with it -- yamdi (ya-m-DI), not ya-m-da. The vowel before the
                # weak letter matches its colour: kasrah before ya', dammah before waw.
                if i > 0:
                    out[i - 1][1] = KASRA if ch == 'ي' else DAMMA
                out[i][1] = ''
                continue
            pv = self._preceding_vowel(out, i)
            if not pv or self._blocked(out, i, ch):
                continue
            new = self.COLOUR.get(pv, ch)
            if new == ch:
                continue
            # CONDITION: refuse the transposition if it would put this weak letter adjacent to an
            # identical one -- the same 'meeting of two weak letters' the tradition forbids.
            nxt = out[i + 1][0] if i + 1 < len(out) else ''
            prv = out[i - 1][0] if i > 0 else ''
            if new in WEAK and (nxt == new or prv == new):
                continue
            out[i][0] = new
            out[i][1] = ''                # alif and the weak letters carry no vowel
        return out


class IdghamMumtad(Operation):
    """MUDAAF in a pattern whose alif falls BETWEEN C2 and C3 (sighat muntaha al-jumu').

    Adjacent idgham cannot fire here -- the slots are not neighbours. The tradition separates the
    two radicals and transposes the cayn (C2) to a HAMZA on a seat:

        maddad  + mafa'il  ->  mada'id     (not mamadd)
        haqqaq  + mafa'il  ->  haqa'iq     (not mahaaqq)

    The seat follows the vowel: kasrah -> ئ, dammah -> ؤ.
    """
    name = 'idgham_mumtad'

    def applies(self, state, ctx):
        if ctx['class'] != 'MUDAAF':
            return False
        pos = {k: i for i, (c, v, role, k) in enumerate(state) if role == 'slot'}
        if 0 not in pos or 1 not in pos:
            return False
        # CONDITION (this is what I had backwards): the long vowel must sit between the FA slot
        # (C1) and the CAYN slot (C2) -- as in mafa'il (m-f-alif-cayn-lam). In fi'al the alif
        # sits between the cayn and the lam, which is ordinary and licenses nothing.
        if pos[0] + 1 >= pos[1]:
            return False
        # AND the fa slot must not be INITIAL -- otherwise fa'il (fa-alif-cayn-lam) also
        # qualifies, and haqq + fa'il must give haaqq, not haa'.
        if pos[0] == 0:
            return False
        between = state[pos[0] + 1:pos[1]]
        return any(c in 'او' for c, v, r, k in between)

    def apply(self, state, ctx):
        out = [list(e) for e in state]
        pos = {k: i for i, (c, v, role, k) in enumerate(out) if role == 'slot'}
        i2 = pos[1]
        # the vowel that determines the hamza seat is the one on the cayn slot
        v = out[i2][1]
        seat = 'ئ' if v == KASRA or v == '' else ('ؤ' if v == DAMMA else 'ئ')
        out[i2][0] = seat
        out[i2][1] = v
        return out


class Hadhf(Operation):
    """al-hadhf: deletion.

      MITHAL -- the assimilated C1 cannot stand after a vowelled mudara'ah prefix, so it drops
                (wacada -> ya'idu, not ya-w'idu).
      AJWAF  -- the medial weak letter drops in the jussive; that form is produced by the
                apocope layer (iktifa_apocope), not here.
    """
    name = 'hadhf'

    def applies(self, state, ctx):
        if ctx['class'] != 'MITHAL':
            return False
        if not state or state[0][0] not in WEAK:
            return False
        # a mudara'ah prefix carrying a vowel
        return len(state) > 1 and state[0][1] != '' and ctx['pattern_first_is_mudaria']

    def apply(self, state, ctx):
        out = [list(e) for e in state]
        for i in (1, 2, 3):               # the C1 slot sits shortly after the prefix
            if out[i][0] in WEAK and out[i][3] == 0:
                out[i][0] = ''
                out[i][1] = ''
                break
        return out


# ---------------------------------------------------------------------------- the engine
class TasrifEngine:
    OPERATIONS = [IbdalHamza(), IdghamMumtad(), Idgham(), Qalb(), Hadhf()]

    def __init__(self, vocab=None):
        self.vocab = vocab
        self._roots3 = None

    # ---- forward -----------------------------------------------------------------------
    def generate(self, root, pattern_name):
        r = strip_diac(root)
        if len(r) != 3:
            return None
        pat = Pattern(pattern_name)
        if not pat.elems:
            return None
        ctx = {'class': Root.klass(r), 'pattern': pat, 'root': r,
               'pattern_first_is_mudaria': bool(pat.elems) and pat.elems[0][0] in MUdARIA
               and pat.elems[0][1] != ''}
        state = []
        for ch, v in pat.elems:
            if ch in SLOT:
                state.append([r[SLOT[ch]], v, 'slot', SLOT[ch]])
            else:
                state.append([ch, v, 'lit', None])
        for op in self.OPERATIONS:
            if op.applies(state, ctx):
                state = op.apply(state, ctx)
        surface = ''.join(c + v for c, v, _, _ in state if c)
        # ORTHOGRAPHY (al-Mumti', bab al-khatt): a final alif that stands for a yaa' radical is
        # written alif maqsura -- ramaya -> rama written with ى, not ا.
        if surface.endswith('ا') and r[2] == 'ي' and ctx['class'] in ('NAQIS', 'LAFIF'):
            surface = surface[:-1] + 'ى'
        return surface, ctx['class']

    # ---- inverse: analysis IS the generator, run backwards ------------------------------
    # Brute force (9,114 roots x 125 patterns) is 1.1M generations per word -- unusable. The
    # tractable inverse is TEMPLATE MATCHING: align the pattern's literal letters against the
    # surface, read candidate radicals off the slot positions, then VALIDATE by generating.
    # A slot may hold a transposed weak letter, so at a weak position we try the weak family.
    WEAK_FAMILY = {'ا': 'واي', 'و': 'اوي', 'ي': 'واي', 'ى': 'واي'}

    def candidate_pairs(self, surface, patterns, max_hits=8):
        target = strip_diac(surface)
        if not target:
            return []
        pats = [p if isinstance(p, Pattern) else Pattern(p) for p in patterns]
        hits = []
        for pat in pats:
            lits = sum(1 for c, _ in pat.elems if c not in SLOT)
            if len(target) != len(pat.elems) - lits + 3:
                pass                      # length may shift under idgham; do not hard-reject
            cands = self._align(target, pat)
            for root in cands:
                g = self.generate(root, pat.name)
                if g and strip_diac(g[0]) == target:
                    if (root, pat.name) not in hits:
                        hits.append((root, pat.name))
                    if len(hits) >= max_hits:
                        return hits
        return hits

    def _align(self, target, pat):
        """Walk the surface and the pattern together; return candidate radical triples."""
        results = [[]]

        def rec(pi, si, acc):
            if len(acc) == 3:
                # the rest of the pattern must be literals matching the remaining surface
                for c, _ in pat.elems[pi:]:
                    if c in SLOT:
                        return
                    if si >= len(target) or target[si] != c:
                        return
                    si += 1
                if si == len(target):
                    results.append(list(acc))
                return
            if pi >= len(pat.elems):
                return
            c, _ = pat.elems[pi]
            if c in SLOT:
                # (a) one surface letter = one radical  (the normal case)
                if si < len(target):
                    ch = target[si]
                    for cand in self.WEAK_FAMILY.get(ch, ch):
                        rec(pi + 1, si + 1, acc + [cand])
                # (b) HADHF: the radical was DELETED (mithal / ajwaf), so a slot consumes
                #     no surface letter at all -- only a weak letter can be deleted this way
                for cand in 'وي':
                    rec(pi + 1, si, acc + [cand])
                # (c) IDGHAM: one surface letter stands for TWO identical radicals
                if si < len(target) and len(acc) == 1:
                    rec(pi + 2, si + 1, acc + [target[si], target[si]])
            else:
                # idgham deletes one of a doubled pair; allow skipping a literal
                if si < len(target) and target[si] == c:
                    rec(pi + 1, si + 1, acc)

        rec(0, 0, [])
        out, seen = [], set()
        for r in results:
            k = ''.join(r)
            if k not in seen:
                seen.add(k)
                out.append(k)
        return out[:64]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=8000)
    args = ap.parse_args()
    eng = TasrifEngine()

    print('=== hand-checkable cases (each traceable to al-Mumti\') ===')
    checks = [('قول', 'فَعَلَ', 'قَالَ'), ('بيع', 'فَعَلَ', 'بَاعَ'), ('دعو', 'فَعَلَ', 'دَعَا'),
              ('خوص', 'فِعَال', 'خِيَاص'), ('طور', 'فِعَال', 'طِيَار'), ('عين', 'فُعُول', 'عُيُون'),
              ('وعد', 'يَفْعُلُ', 'يعد'), ('مدد', 'فَعَلَ', 'مَدَّ')]
    ok = 0
    for root, pat, expect in checks:
        got = eng.generate(root, pat)
        s = got[0] if got else None
        good = strip_diac(s or '') == strip_diac(expect)
        ok += good
        print(f'  {root:<6}+{pat:<9}-> {str(s):<12} expect {expect:<12} {"PASS" if good else "FAIL"}')
    print(f'  {ok}/{len(checks)} hand checks')

    if not getattr(sys.modules.get('nrmp_vocab'), '__name__', None):
        import nrmp_vocab as nv
    import nrmp_vocab as nv
    V = next(v for k, v in vars(nv).items() if isinstance(v, type) and 'MorphemicVocab' in k)
    vocab = V('/workspace/hf_v19_2_release/data/rootformer_v12_arabic_blueprint.json')
    d = json.load(open('/workspace/hf_v19_2_release/data/rootformer_v12_arabic_blueprint.json'))
    i2t = d['vocab_id_to_token']
    lo, hi = d['partitions']['classical_awzan']
    patterns = [i2t[str(i)].split('wazn_')[-1].rstrip('>') for i in range(lo, hi)]

    files = (sorted(glob.glob('/workspace/scholastic_sanitized/*.txt'))
             + sorted(glob.glob('/workspace/andalusian_canon_sanitized/*.txt')))[:6]
    seen = set()
    checked = good = 0
    by_class, by_class_ok = Counter(), Counter()
    fails = []
    for f in files:
        txt = open(f, encoding='utf-8', errors='ignore').read()
        for w in re.split(r'\s+', txt):
            if not re.search(r'[\u0600-\u06FF]', w):
                continue
            p, r, wz, s = vocab.encode_word(w)
            rn, wn = vocab.id2root.get(r, ''), vocab.id2wazn.get(wz, '')
            pat = wn.split('wazn_')[-1].rstrip('>')
            if not pat or len(strip_diac(rn)) != 3 or (rn, pat) in seen:
                continue
            seen.add((rn, pat))
            g = eng.generate(rn, pat)
            if not g:
                continue
            cls = Root.klass(rn)
            checked += 1
            by_class[cls] += 1
            # use the ENGINE's inverse, not the old separate ladder
            pairs = eng.candidate_pairs(g[0], patterns)   # ALL 125, not a slice
            hit = any(strip_diac(rr) == strip_diac(rn) and pp == pat for rr, pp in pairs)
            back = pairs[0][0] if pairs else '-'
            good += hit
            by_class_ok[cls] += hit
            if not hit and len(fails) < 8:
                fails.append((rn, pat, g[0], back))
            if checked >= args.n:
                break
        if checked >= args.n:
            break

    print(f'\n=== ROUND TRIP ({checked} attested pairs) ===')
    print(f'  recovered: {good}/{checked} = {100*good/max(checked,1):.1f}%')
    for k in sorted(by_class, key=lambda x: -by_class[x]):
        n_, o_ = by_class[k], by_class_ok[k]
        print(f'    {k:<9}{n_:>7}{o_:>8}{100*o_/max(n_,1):>7.1f}%')
    print('\n  sample failures:')
    for rn, pat, surf, back in fails:
        print(f'    {rn:<6}{pat:<12}-> {surf:<12}-> {back}')
    json.dump({'checked': checked, 'ok': good, 'by_class': dict(by_class),
               'by_class_ok': dict(by_class_ok), 'fails': fails},
              open('/workspace/tasrif_engine.json', 'w'), ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
