#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyzer_segmentation_v4.py -- validated (al-Ayn attestation v4) segmentation for decompose_arabic_word.

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
        self.attested, self.unused = set(), set()
        try:
            import json
            d = json.load(open('/workspace/khalil_attest_v4.json'))
            self.attested = {tuple(t) for t in d.get('attested', []) if len(t) == 3}
            self.unused = {tuple(t) for t in d.get('unused', []) if len(t) == 3}
        except Exception:
            pass
        self.stats = {'calls': 0, 'changed': 0}

    # -- delegate everything else ----------------------------------------------------------
    def __getattr__(self, k):
        return getattr(self.tok, k)

    def _generates(self, stem, root, wazn):
        if not (self.engine and root and wazn):
            return False
        try:
            g = self.engine.generate(root, wazn)
        except Exception:
            return False
        return bool(g) and strip_diac(g[0]) == strip_diac(stem)

    def _real_root(self, root):
        if not root:
            return None
        r = strip_diac(root)
        if len(r) != 3:
            return None
        rt = tuple(r)
        if rt in self.unused:
            return False                  # al-Khalil marks this permutation unused
        if r in self.lexicon:
            return True                   # a root of the language (blueprint lexicon)
        if rt in self.attested:
            return True                   # additionally attested in al-'Ayn
        return None

    def _score(self, prefix, stem, suffix, root, wazn):
        sc = 0
        is_particle = strip_diac(stem) in PARTICLES
        if self._generates(stem, root, wazn):
            sc += 10                      # the residue round-trips
        elif root and not is_particle:
            # a particle has no wazn, so it can never "generate" -- the penalty is for a ROOT
            # that fails to generate its stem, which is a different situation entirely.
            sc -= 6
        real = self._real_root(root)
        if real is True:
            sc += 6                       # attested in al-'Ayn
        elif real is False:
            sc -= 12                      # al-Khalil says this permutation is unused
        sc -= len(prefix) + len(suffix)   # penalise every clitic removed
        if is_particle:
            sc += 8                       # a real particle beats an invented root
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
                # CONSERVATIVE RULE: re-segment only on POSITIVE evidence -- the new residue's
                # root must be attested in al-'Ayn. Requiring merely "not condemned" lets a
                # nonsense root through whenever it is simply uncovered by the (partial) table,
                # which is how kuntu became ka + n-t-t.
                # a PARTICLE residue is its own evidence -- it is not a triliteral root, so the
                # root-attestation gate must not apply to it (that is why fa- + kam was skipped)
                if strip_diac(stem) not in PARTICLES and self._real_root(r) is not True:
                    continue
                sc = self._score(pre, stem, suf, r, w)
                if sc > best_sc:
                    best_sc, best = sc, (pre or None, r, w, suf or None)
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
