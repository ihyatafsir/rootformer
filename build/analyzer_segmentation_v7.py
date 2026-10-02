#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyzer_segmentation_v7.py -- al-Khalil's evidence ranks: al-'Ayn > lexicon, plus root calculation segmentation for decompose_arabic_word.

THE PROBLEM
-----------
decompose_arabic_word strips clitics greedily and roots whatever is left. The residue is often a
valid root, so nothing downstream can catch the error:

    بالله  -> root بلل   |  فكم -> root كمم  |  عنهم -> root عنن

THE FIX
-------
Invert the order: ENUMERATE the candidate (prefix, stem, suffix) splits, and accept one only when
its residue VALIDATES -- i.e. the engine can actually generate the stem from the (root, wazn) the
analyser assigns it. A root that cannot generate the stem it was read from is not that stem's root.

This wraps the tokenizer rather than rewriting decompose_arabic_word, so it is testable in
isolation and the hookup is one line:

    tok = SegmentAwareAnalyzer(tok)

Usage: python analyzer_segmentation.py
"""
import re
import sys

sys.path.insert(0, '/workspace/hf_v19_2_release')
sys.path.insert(0, '/workspace/hf_v19_2_release/models')

DIAC = re.compile(r'[\u064b-\u0652\u0670\u0640\u06d6-\u06ed]')

# al-huruf al-zawa'id, the mnemonic sa-altumuniha
PREFIXES = ['وال', 'فال', 'بال', 'كال', 'لل', 'ال', 'مست', 'يست', 'ين', 'يت', 'تن',
            'أ', 'ي', 'ن', 'ت', 'م', 'س', 'و', 'ف', 'ب', 'ك', 'ل']
SUFFIXES = ['هما', 'كما', 'هم', 'هن', 'كم', 'كن', 'نا', 'ني', 'ها', 'وا', 'ون', 'ين', 'ات',
            'ان', 'ية', 'تم', 'تن', 'ت', 'ه', 'ك', 'ي', 'ا', 'ن']
PARTICLES = {'من', 'في', 'إلى', 'على', 'عن', 'حتى', 'مع', 'إن', 'أن', 'كأن', 'لكن', 'ليت',
             'لعل', 'لا', 'ما', 'لم', 'لن', 'هل', 'قد', 'ثم', 'أو', 'أم', 'بل', 'كم', 'كيف',
             'أين', 'متى', 'إذا', 'إذ', 'لو', 'لولا', 'هذا', 'هذه', 'ذلك', 'تلك', 'الذي',
             'التي', 'هو', 'هي', 'أنا', 'نحن', 'أنت', 'هم', 'هن'}


def strip_diac(s):
    return DIAC.sub('', s)


class SegmentAwareAnalyzer:
    """Wrap a tokenizer so decompose_arabic_word validates its own segmentation."""

    def __init__(self, tok, verbose=False):
        self.tok = tok
        self.verbose = verbose
        try:
            from tasrif_engine import TasrifEngine
            self.engine = TasrifEngine(tok)
        except Exception:
            self.engine = None
        # THE MISSING CONSTRAINT: the engine generates a form for ANY triliteral root, so
        # "round-trips" alone is satisfied by nonsense roots (kuntu -> ka + n-t-t). The residue
        # must also be a REAL root. al-Khalil's attestation table supplies that.
        # THE EVIDENCE BASE is the blueprint's OWN root lexicon (9,013 roots). al-Khalil's
        # attestation table is only 2,483 entries -- too partial to confirm a root, though it can
        # still CONDEMN one. So: lexicon for positive evidence, al-'Ayn for negative.
        self.lexicon = {strip_diac(r) for r in getattr(tok, 'roots_set', set())}
        self.attested, self.unused, self.unused_pairs = set(), set(), set()
        try:
            import json
            d = json.load(open('/workspace/khalil_attest_v4.json'))
            # all lengths: the 2-letter entries are the PAIR record, which governs the muda''af
            self.attested = {tuple(t) for t in d.get('attested', [])}
            self.unused = {tuple(t) for t in d.get('unused', [])}
            self.unused_pairs = {t for t in self.unused if len(t) == 2}
        except Exception:
            pass
        self.stats = {'calls': 0, 'changed': 0}

    # -- delegate everything else ----------------------------------------------------------
    def __getattr__(self, k):
        return getattr(self.tok, k)

    def _generates(self, stem, root, wazn):
        """Does (root, wazn) generate this stem -- allowing the ELIDED reading?

        كنت is كان + -tu.  The surface stem is كن, because the alif of كان is deleted before the
        suffix (al-hadhf).  generate(كون, فَعَلَ) returns كان: the stem with one weak letter
        elided.  Exact equality therefore REJECTS the correct root, which is why every earlier
        version of this analyzer could not reach كون and settled for the invented نتت instead.
        The elision is the rule tasrif_engine already models, so it is admitted here: the stem
        must be the generated form with zero or more weak letters deleted, order preserved.
        """
        if not (self.engine and root and wazn):
            return False
        try:
            g = self.engine.generate(root, wazn)
        except Exception:
            return False
        if not g:
            return False
        gen, tgt = strip_diac(g[0]), strip_diac(stem)
        if gen == tgt:
            return True
        i = 0
        for ch in gen:
            if i < len(tgt) and tgt[i] == ch:
                i += 1
            elif ch in 'اويىء':
                continue                      # elided / transposed (al-hadhf, al-qalb)
            else:
                return False
        return i == len(tgt)

    def _candidate_roots(self, stem, wazn):
        """al-Khalil's CALCULATION, not a lookup.

        The tokenizer only ever proposes roots it can read off the surface, so a root whose
        radical was DELETED is unreachable: no amount of lexicon filtering finds كون in كنت.
        candidate_pairs() walks the wazn template against the surface and is allowed to spend a
        slot on a radical that left no letter behind (HADHF), then validates by generating --
        which is the generator run backwards, exactly as al-Khalil's taqalib + attestation is.
        """
        out = []
        if self.engine is None:
            return out
        try:
            from tasrif_engine import Pattern
        except Exception:
            return out
        pats = [wazn] if wazn else []
        for p in ('فَعَلَ', 'فَعِلَ', 'فَعُلَ'):
            if p not in pats:
                pats.append(p)
        for p in pats:
            if not isinstance(p, str):
                continue
            try:
                roots = self.engine._align(strip_diac(stem), Pattern(p))
            except Exception:
                continue
            # VALIDATE HERE, LOOSELY.  candidate_pairs() would have rejected كون for كنت: it
            # requires generate(root,wazn) to equal the stem exactly, and generate(كون,فَعَلَ) is
            # كان -- the stem with the alif elided.  The elision is the rule being modelled, so
            # the validator must admit it, and _generates() does.
            for r in roots:
                if (r, p) not in out and self._generates(stem, r, p):
                    out.append((r, p))
        return out

    # evidence strengths: al-'Ayn's own entry beats the blueprint lexicon, which is what finally
    # separates كون (attested) from كنن (lexicon only) and كم (attested particle) from كمم.
    MUHMAL, UNKNOWN, LEXICON, ATTESTED = -1, 0, 1, 2

    def _real_root(self, root):
        if not root:
            return self.UNKNOWN
        r = strip_diac(root)
        if len(r) != 3:
            return self.UNKNOWN
        rt = tuple(r)
        if rt in self.unused:
            return self.MUHMAL               # al-Khalil marks this permutation unused
        if rt in self.attested:
            return self.ATTESTED            # al-'Ayn enters it -- outranks the lexicon
        # THE MUDA''AF GATE.  A doubled root C1 C2 C2 IS the doubling of the ordered pair
        # (C1,C2), and al-'Ayn's two-letter chapters enumerate exactly which pairs are used:
        #   باب التاء والنون  ت ن  يستعمل فقط تن:      -> the pair ن ت is M U H M A L
        # so نتت -- the tokenizer's doubling of the stem نت -- is not a root of the language,
        # even though the blueprint's 9,013-root lexicon lists it.
        # The gate is applied ONLY to C2==C3.  It must not touch C1==C3 roots: al-'Ayn enters
        #    نتن inside باب التاء والنون itself («نتن ينتن نتنا ... وهذه المادة من الثلاثي»),
        # so the pair record plainly does not govern them, and reducing them to their skeleton
        # wrongly condemned نتن، ثلث، and the quadriliterals قنقل، قرقل، لغلغ.
        if r[1] == r[2] and (r[0], r[1]) in self.unused_pairs:
            return self.MUHMAL
        if r in self.lexicon:
            return self.LEXICON           # in the blueprint lexicon, but nothing else vouches
        return self.UNKNOWN

    PARTICLE_BONUS = 8

    def _score(self, prefix, stem, suffix, root, wazn):
        is_particle = strip_diac(stem) in PARTICLES
        if is_particle:
            # A recognised particle IS the analysis.  Its letters are not radicals, so no root
            # may be read off it -- awarding the particle bonus AND a root bonus on the same stem
            # is what let the doubled كمم (score 23) outbid the particle كم (score 7).
            return self.PARTICLE_BONUS - len(prefix) - len(suffix)
        sc = 0
        if self._generates(stem, root, wazn):
            sc += 10                      # the residue round-trips
        elif root:
            sc -= 6                       # a root that cannot generate its own stem
        strength = self._real_root(root)
        if strength == self.ATTESTED:
            sc += 8                       # al-'Ayn enters this root
        elif strength == self.LEXICON:
            sc += 3                       # in the lexicon only -- weaker
        elif strength == self.MUHMAL:
            sc -= 12                      # al-Khalil says this permutation is unused
        sc -= len(prefix) + len(suffix)   # penalise every clitic removed
        return sc

    def decompose_arabic_word(self, word):
        base = self.tok.decompose_arabic_word(word)
        if self.engine is None:
            return base
        self.stats['calls'] += 1
        clean = self.tok.clean_arabic(word)
        if not clean:
            return base
        # only re-segment when the greedy result claims a ROOT; particles are already fine
        bp, br, bw, bs = base
        if not br or (br and bw is None and bp is None and bs is None):
            return base

        # THE BASELINE GETS NO ROUND-TRIP BONUS. The analyser read the root FROM the word, so
        # regenerating it is vacuous -- every 3-letter word "round-trips" its own root. The
        # round trip is only informative for a SPLIT candidate, where the residue must generate
        # the stem. Awarding it to the baseline is what let kamm win over fa- + kam.
        best, best_sc = base, 0
        for pre in [''] + PREFIXES:
            if pre and not clean.startswith(pre):
                continue
            rest = clean[len(pre):]
            if len(rest) < 2:
                continue
            for suf in [''] + SUFFIXES:
                if suf and not rest.endswith(suf):
                    continue
                stem = rest[:len(rest) - len(suf)] if suf else rest
                if len(strip_diac(stem)) < 2:
                    continue
                try:
                    p, r, w, s = self.tok.decompose_arabic_word(stem)
                except Exception:
                    continue
                if not r:
                    continue
                # the tokenizer's own reading, plus every root al-Khalil's calculation reaches
                # for this stem -- including the ones whose radical was elided
                if strip_diac(stem) in PARTICLES:
                    readings = [(r, w)]      # a particle has no radicals to calculate from
                else:
                    readings = [(r, w)] + [c for c in self._candidate_roots(stem, w)
                                           if c != (r, w)]
                # CONSERVATIVE RULE: re-segment only on POSITIVE evidence -- the new residue's
                # root must be attested in al-'Ayn. Requiring merely "not condemned" lets a
                # nonsense root through whenever it is simply uncovered by the (partial) table,
                # which is how kuntu became ka + n-t-t.
                # a PARTICLE residue is its own evidence -- it is not a triliteral root, so the
                # root-attestation gate must not apply to it (that is why fa- + kam was skipped)
                for r2, w2 in readings:
                    if not r2:
                        continue
                    if strip_diac(stem) not in PARTICLES and self._real_root(r2) <= 0:
                        continue
                    sc = self._score(pre, stem, suf, r2, w2)
                    if sc > best_sc:
                        best_sc, best = sc, (pre or None, r2, w2, suf or None)
        if best != base:
            self.stats['changed'] += 1
            if self.verbose:
                print(f'  {word}: {base} -> {best}')
        return best


def main():
    import nrmp_vocab as nv
    V = next(v for k, v in vars(nv).items() if isinstance(v, type) and 'MorphemicVocab' in k)
    vocab = V('/workspace/hf_v19_2_release/data/rootformer_v12_arabic_blueprint.json')
    tok = vocab.base_tok
    wrapped = SegmentAwareAnalyzer(tok, verbose=True)

    print('=== greedy vs validated ===')
    for w in ['بالله', 'فكم', 'عنهم', 'كنت', 'والحقائق', 'بمصانعهم', 'فترددت', 'والكتاب', 'علم']:
        base = tok.decompose_arabic_word(w)
        fix = wrapped.decompose_arabic_word(w)
        flag = 'CHANGED' if base != fix else ''
        print(f'  {w:<12} greedy root={str(base[1]):<8} wazn={str(base[2]):<10} | '
              f'validated root={str(fix[1]):<8} wazn={str(fix[2]):<10} {flag}')
    print(f'\n  changed {wrapped.stats["changed"]} of {wrapped.stats["calls"]}')


if __name__ == '__main__':
    main()
