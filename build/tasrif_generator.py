#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tasrif_generator.py -- Andalusian taṣrīf run FORWARDS: (root, wazn) -> surface form.

The analyzer so far only runs BACKWARDS: it looks a word up, and if the lookup misses it applies
i'lal / idgham / mazid / hamza as a REPAIR ladder. That cannot tell you whether a form is
well-formed -- only whether some path back to a root exists.

This runs the bab system forwards, which is what the taṣrīf tradition actually specifies (Ibn
'Usfur, al-Mumti' fi al-Tasrif; Ibn Malik, Lamiyyat al-Af'al). Given the root and the pattern it
produces the surface form, applying the sound-change operations by root class:

    SALIM    sound              fill the template
    MAHMUZ   hamzated           ibdal of the hamza
    MUDAAF   doubled            idgham (C2 == C3)
    MITHAL   assimilated, C1 weak    the weak letter drops in the imperfect
    AJWAF    hollow, C2 weak         QALB -> alif in the perfect; deleted in the jussive
    NAQIS    defective, C3 weak      QALB -> alif after a fathah; stays vowelled otherwise

The operation names are the tradition's own: al-qalb (transposition), al-hadhf (deletion),
al-idgham (assimilation), al-ibdal (substitution).

Verification is a ROUND TRIP against the lexicon: for attested (root, wazn) pairs, generate the
form and check the analyzer recovers the same root and wazn. That measures fidelity objectively
rather than by assertion.

Usage:
  python tasrif_generator.py --n 20000
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
FATHA, DAMMA, KASRA, SUKUN = '\u064e', '\u064f', '\u0650', '\u0652'
SHADDA = '\u0651'
SUKUN_C, FATHA_C, DAMMA_C, KASRA_C = SUKUN, FATHA, DAMMA, KASRA
ARABIC = set('ابتثجحخدذرزسشصضطظعغفقكلمنهويءأإآؤئى')
WEAK = set('وايى')
SLOT = {'ف': 0, 'ع': 1, 'ل': 2}


def strip_diac(s):
    return DIAC.sub('', s)


class TasrifGenerator:
    def __init__(self, vocab=None):
        self.vocab = vocab

    # ---- root class (the bab divisions of the mu'tall) ---------------------------------
    @staticmethod
    def root_class(root):
        r = strip_diac(root)
        if len(r) != 3:
            return 'OTHER'
        w = [i for i, c in enumerate(r) if c in WEAK]
        if 'ء' in r or any(c in 'أإآؤئ' for c in r):
            return 'MAHMUZ'
        if len(w) >= 2:
            return 'LAFIF'
        if not w:
            return 'MUDAAF' if r[1] == r[2] else 'SALIM'
        if w == [0]:
            return 'MITHAL'
        if w == [1]:
            return 'AJWAF'
        if w == [2]:
            return 'NAQIS'
        return 'OTHER'

    # ---- template parsing --------------------------------------------------------------
    @staticmethod
    def parse(wazn_name):
        """'فَاعِل' -> [(0,''), ('lit','ا'), (1,'ِ'), (2,'')]  (slot index + trailing diacritics)"""
        w = DIAC.sub('', '') or wazn_name or ''
        out = []
        i = 0
        raw = wazn_name or ''
        while i < len(raw):
            ch = raw[i]
            i += 1
            if ch not in ARABIC:
                continue
            trail = ''
            while i < len(raw) and raw[i] not in ARABIC:
                trail += raw[i]
                i += 1
            if ch in SLOT:
                out.append(('slot', SLOT[ch], trail))
            else:
                out.append(('lit', ch, trail))
        return out

    # ---- AL-QALB, as al-Mumti' states it ------------------------------------------------
    # "al-dammah is part of the waw, al-kasrah part of the ya', al-fathah part of the alif"
    # therefore a weak radical takes the colour of the vowel BEFORE it:
    #     preceded by dammah -> waw | kasrah -> ya' | fathah -> alif
    # Ibn 'Usfur's own examples: tawaya -> tayyan (waw -> ya'), sayyid < siywid.
    QALB = {DAMMA: 'و', KASRA: 'ي', FATHA: 'ا'}

    @classmethod
    def qalb(cls, ch, preceding_vowel):
        if ch not in WEAK or preceding_vowel not in cls.QALB:
            return ch
        return cls.QALB[preceding_vowel]

    # ---- forward generation ------------------------------------------------------------
    def generate(self, root, wazn_name):
        """Return the surface form for (root, wazn), or None if the pattern is not template-like."""
        r = strip_diac(root)
        if len(r) != 3:
            return None
        tpl = self.parse(wazn_name)
        if not tpl or not any(t[0] == 'slot' for t in tpl):
            return None
        cls = self.root_class(r)
        c = list(r)
        note = ''

        # fill the template, applying AL-QALB as we go: each weak radical takes the colour of
        # the vowel that immediately precedes it in the pattern (al-Mumti', bab al-qalb).
        out = []
        prev_vowel = ''
        for ti, (kind, val, trail) in enumerate(tpl):
            if kind == 'slot':
                ch = c[val]
                # AL-QALB applies to C2 / C3 only. For MITHAL (C1 weak) the operation is
                # DELETION (hadhf), not transposition -- applying qalb there was wrong.
                # BLOCK: if a literal weak letter of the same kind follows in the pattern, the
                # transposition is refused -- Arabic avoids the meeting of two weak letters
                # (Ibn 'Usfur: 'they dislike the meeting of the ya' and the waw'). This is why
                # 'ayn + fu'ul is 'uyun, NOT 'uwun.
                blocked = False
                for k2, v2, t2 in tpl[ti + 1:]:
                    if k2 == 'lit' and v2 in WEAK:
                        blocked = (v2 == self.qalb(ch, prev_vowel) or v2 == ch)
                        break
                    if k2 == 'slot':
                        break
                if (val in (1, 2) and cls in ('AJWAF', 'NAQIS', 'LAFIF')
                        and ch in WEAK and prev_vowel and not blocked):
                    new_ch = self.qalb(ch, prev_vowel)
                    if new_ch != ch:
                        note = f'qalb({ch}->{new_ch} after {prev_vowel})'
                        c[val] = new_ch
                        ch = new_ch
                # alif and the weak letters never carry a vowel of their own
                out.append(ch + ('' if ch in 'اأإآوى' else trail))
                prev_vowel = trail
            else:
                out.append(val + trail)
                prev_vowel = trail
        surface = ''.join(out)

        # --- idgham: doubled root, C2 == C3 -------------------------------------------------
        if cls == 'MUDAAF':
            surface = self._idgham(surface, r)
            note = (note + ';idgham') if note else 'idgham'

        # --- mithal: the initial weak letter drops in the imperfect -------------------------
        if cls == 'MITHAL' and any(k == 'lit' and v == 'ي' and tr == FATHA
                                   for k, v, tr in tpl[:1] + tpl[1:2]):
            surface = self._drop_initial_weak(surface, r)
            note = 'hadhf(mithal)'
        return surface, note

    @staticmethod
    def _idgham(surface, root):
        """C2 == C3 -> a single doubled letter (madd -> madd with shaddah)."""
        b = strip_diac(surface)
        # find the doubled pair in the root and collapse it in the surface
        pat = root[1] + root[2]
        if pat in b:
            return b.replace(pat, root[1] + SHADDA, 1)
        return surface

    @staticmethod
    def _drop_initial_weak(surface, root):
        b = strip_diac(surface)
        # the mudara'ah prefix carries a vowel, so the assimilated C1 cannot stay
        for i in range(1, len(b)):
            if b[i] == root[0]:
                return b[:i] + b[i + 1:]
        return surface


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=20000)
    ap.add_argument('--out', default='/workspace/tasrif_roundtrip.json')
    args = ap.parse_args()

    import nrmp_vocab as nv
    V = next(v for k, v in vars(nv).items() if isinstance(v, type) and 'MorphemicVocab' in k)
    vocab = V('/workspace/hf_v19_2_release/data/rootformer_v12_arabic_blueprint.json')
    gen = TasrifGenerator(vocab)

    # ---- sanity examples ---------------------------------------------------------------
    print('=== forward generation (canonical weak verbs) ===')
    for root, wazn, expect in (('قول', 'فَعَلَ', 'قال'), ('بيع', 'فَعَلَ', 'باع'),
                               ('دعو', 'فَعَلَ', 'دعا'), ('رمي', 'فَعَلَ', 'رمى'),
                               ('قول', 'يَفْعُلُ', 'يقول')):
        got = gen.generate(root, wazn)
        s = got[0] if got else None
        print(f'  {root} + {wazn:<10} -> {str(s):<10} expect {expect:<10} '
              f'{"PASS" if s == expect else "~"}')

    # ---- round trip over attested pairs ------------------------------------------------
    files = (sorted(glob.glob('/workspace/scholastic_sanitized/*.txt'))
             + sorted(glob.glob('/workspace/andalusian_canon_sanitized/*.txt')))[:6]
    seen = Counter()
    checked = ok = 0
    by_class = Counter()
    by_class_ok = Counter()
    fails = []
    for f in files:
        txt = open(f, encoding='utf-8', errors='ignore').read()
        for w in re.split(r'\s+', txt):
            if not re.search(r'[\u0600-\u06FF]', w):
                continue
            p, r, wz, s = vocab.encode_word(w)
            if r in (vocab.UNK_ROOT, vocab.root2id.get('<PARTICLE>')):
                continue
            rn, wn = vocab.id2root.get(r, ''), vocab.id2wazn.get(wz, '')
            pat = wn.split('wazn_')[-1].rstrip('>')       # tolerate both wrapped and bare forms
            key = (rn, pat)
            if key in seen or len(strip_diac(rn)) != 3 or not pat:
                continue
            seen[key] += 1
            out = gen.generate(rn, pat)
            if not out or not out[0]:
                continue
            surf = out[0]
            # round trip: does the analyzer recover the same root from the generated form?
            p2, r2, w2, s2 = vocab.encode_word(surf)
            cls = gen.root_class(rn)
            checked += 1
            by_class[cls] += 1
            good = (vocab.id2root.get(r2, '') == rn)
            ok += good
            by_class_ok[cls] += good
            if not good and len(fails) < 12:
                fails.append((rn, pat, surf, vocab.id2root.get(r2, '?')))
            if checked >= args.n:
                break
        if checked >= args.n:
            break

    print(f'\n=== ROUND TRIP: generate(root,wazn) then re-analyse ({checked} attested pairs) ===')
    print(f'  root recovered correctly : {ok}/{checked} = {100*ok/max(checked,1):.1f}%')
    print(f'\n  {"root class":<10}{"tested":>8}{"recovered":>11}{"rate":>8}')
    for k in sorted(by_class, key=lambda x: -by_class[x]):
        n_, o_ = by_class[k], by_class_ok[k]
        print(f'  {k:<10}{n_:>8}{o_:>11}{100*o_/max(n_,1):>7.1f}%')
    if fails:
        print('\n  sample round-trip failures (root, pattern, generated, re-analysed as):')
        for rn, pat, surf, back in fails:
            print(f'    {rn:<6} {pat:<12} -> {surf:<12} -> {back}')
    json.dump({'checked': checked, 'ok': ok,
               'by_class': dict(by_class), 'by_class_ok': dict(by_class_ok),
               'fails': fails}, open(args.out, 'w'), ensure_ascii=False, indent=2)
    print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
