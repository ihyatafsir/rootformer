#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
segment_validator.py -- the tool the mis-segmentation actually needs.

WHY A NEW TOOL
--------------
The analyser's error is not that it produces bad roots. It strips clitics greedily and then roots
whatever is left, and the residue is often a perfectly good root:

    بالله  -> root بلل     (b- + Allah)
    فكم    -> root كمم     (fa- + kam)
    عنهم   -> root عنن     (can + hum)
    كنت    -> root كنن     (kana + -tu)

بلل، كمم، عنن are all valid roots, so al-Khalil's attestation gate CANNOT catch these -- it only
rejects non-roots, which are rare (0.2%). The fault is the segmentation, so the fix has to be in
the segmentation.

THE TOOL
--------
Invert the order of operations. Instead of stripping and then rooting:

    ENUMERATE candidate (prefix, stem, suffix) splits from the known clitic inventories,
    ROOT each stem, and
    SCORE the split -- accepting it only when the residue validates as a word form.

Scoring prefers, in order:
    1. a split whose stem is an attested root form (root in the attestation table, wazn known)
    2. FEWER clitics removed  -- minimal stripping is the null hypothesis
    3. a stem that is itself a recognised particle (so can + hum wins over roooting canhum)

Usage:
  python segment_validator.py            # tests the known failures + measures on the corpus
"""
import json
import re
import sys
from collections import Counter

sys.path.insert(0, '/workspace/hf_v19_2_release')
sys.path.insert(0, '/workspace/hf_v19_2_release/models')

DIAC = re.compile(r'[\u064b-\u0652\u0670\u0640\u06d6-\u06ed]')
AR = re.compile(r'[\u0600-\u06FF]')

# al-huruf al-zawa'id, gathered in the mnemonic sa-altumuniha:  س أ ل ت م و ن ي ه ا
PREFIXES = ['وال', 'فال', 'بال', 'كال', 'لل', 'ال', 'وال', 'مست', 'يست', 'ين', 'يت', 'تن',
            'أ', 'ي', 'ن', 'ت', 'م', 'س', 'و', 'ف', 'ب', 'ك', 'ل']
SUFFIXES = ['هما', 'كما', 'هم', 'هن', 'كم', 'كن', 'نا', 'ني', 'ها', 'وا', 'ون', 'ين', 'ات',
            'ان', 'ية', 'تم', 'تن', 'ت', 'ه', 'ك', 'ي', 'ا', 'ن']
PARTICLES = {'من', 'في', 'إلى', 'على', 'عن', 'حتى', 'مع', 'إن', 'أن', 'كأن', 'لكن', 'ليت',
             'لعل', 'لا', 'ما', 'لم', 'لن', 'هل', 'قد', 'ثم', 'أو', 'أم', 'بل', 'كم', 'كيف',
             'أين', 'متى', 'إذا', 'إذ', 'لو', 'لولا', 'هذا', 'هذه', 'ذلك', 'تلك', 'الذي',
             'التي', 'هو', 'هي', 'أنا', 'نحن', 'أنت', 'هم', 'هن'}


def strip_diac(s):
    return DIAC.sub('', s)


# --------------------------------------------------------------------------------------------
# VISIBILITY (pillar 3): the two handlers below swallowed their exception.  They now report it on
# stderr and still fall back to exactly the same value, so a degraded validator cannot pass for a
# working one.  Rate-limited (one line per site per run).
# --------------------------------------------------------------------------------------------
_WARNED_SITES = set()


def _warn_once(site, message):
    if site not in _WARNED_SITES:
        _WARNED_SITES.add(site)
        print(f'[segment_validator] {message}', file=sys.stderr)


class SegmentValidator:
    def __init__(self, vocab, attested_path=None):
        self.vocab = vocab
        # THE DECISIVE TEST: a (root, wazn) pair is only real if the taSrif engine actually
        # GENERATES the stem from it. balal is a good root, but no pattern turns it into
        # 'ballah' -- which is b- + Allah. Without this check the no-strip option wins on the
        # mere fact that the residue happens to be a root.
        try:
            from tasrif_engine import TasrifEngine, strip_diac as _sd
            self.engine = TasrifEngine(vocab)
        except Exception as exc:
            # VISIBILITY: this is the DECISIVE test -- with engine=None no (root, wazn) can be
            # generated, so every split is rejected by a test that never ran.  Default unchanged.
            _warn_once('engine', f'tasrif_engine unavailable -- the decisive test is INERT: {exc!r}')
            self.engine = None
        self.attested = set()
        if attested_path:
            try:
                d = json.load(open(attested_path))
                for t in d.get('attested', []):
                    if len(t) == 3:
                        self.attested.add(tuple(t))
            except Exception as exc:
                # VISIBILITY: an unread record left self.attested EMPTY, so the attestation half
                # of the score silently stopped contributing.  Default (empty set) unchanged.
                _warn_once('attested',
                           f'attestation record {attested_path!r} unreadable: {exc!r} -- '
                           f'self.attested is EMPTY for this run')

    def _analyze(self, stem):
        """Root + wazn for a bare stem, or (None, None)."""
        p, r, wz, s = self.vocab.encode_word(stem)
        rn = self.vocab.id2root.get(r, '')
        wn = self.vocab.id2wazn.get(wz, '')
        if rn.startswith('<') or len(strip_diac(rn)) != 3:
            return None, None
        return rn, wn.split('wazn_')[-1].rstrip('>')

    def _generates(self, stem, root, wazn):
        """Does the engine reproduce the stem from (root, wazn)? The decisive validity test."""
        if not (self.engine and root and wazn):
            return False
        g = self.engine.generate(root, wazn)
        return bool(g) and strip_diac(g[0]) == strip_diac(stem)

    def _score(self, prefix, stem, suffix, root, wazn=None):
        """Higher is better. The engine round trip dominates every other term."""
        sc = 0
        # DECISIVE: the residue must be generable from (root, wazn)
        if self._generates(stem, root, wazn):
            sc += 10
        elif root:
            sc -= 6          # a root that cannot generate the stem is a wrong segmentation
        sc -= len(prefix) + len(suffix)               # penalise every clitic removed
        if strip_diac(stem) in PARTICLES:
            sc += 8                                   # a real particle beats an invented root
        if root and tuple(strip_diac(root)) in self.attested:
            sc += 2
        return sc

    def segment(self, word):
        """Enumerate splits and return them best-first."""
        w = strip_diac(word)
        out = []
        for pre in [''] + PREFIXES:
            if pre and not w.startswith(pre):
                continue
            rest = w[len(pre):]
            if len(rest) < 2:
                continue
            for suf in [''] + SUFFIXES:
                if suf and not rest.endswith(suf):
                    continue
                stem = rest[:len(rest) - len(suf)] if suf else rest
                if len(strip_diac(stem)) < 2:
                    continue
                root, wazn = self._analyze(stem) if len(strip_diac(stem)) >= 3 else (None, None)
                out.append({'prefix': pre, 'stem': stem, 'suffix': suf,
                            'root': root, 'wazn': wazn,
                            'score': self._score(pre, stem, suf, root, wazn)})
        # de-dup on the split itself, best-first
        seen, uniq = set(), []
        for c in sorted(out, key=lambda x: -x['score']):
            k = (c['prefix'], c['stem'], c['suffix'])
            if k not in seen:
                seen.add(k)
                uniq.append(c)
        return uniq


def main():
    import nrmp_vocab as nv
    V = next(v for k, v in vars(nv).items() if isinstance(v, type) and 'MorphemicVocab' in k)
    bp = sys.argv[1] if len(sys.argv) > 1 else \
        '/workspace/hf_v19_2_release/data/rootformer_v12_arabic_blueprint.json'
    print(f'[*] blueprint: {bp}')
    vocab = V(bp)
    sv = SegmentValidator(vocab, '/workspace/khalil_attest_v3.json')
    print(f'[*] validator loaded: {len(sv.attested)} attested roots, '
          f'{len(PREFIXES)} prefixes, {len(SUFFIXES)} suffixes\n')

    print('=== the known mis-segmentations ===')
    for w in ['بالله', 'فكم', 'عنهم', 'كنت', 'والحقائق', 'بمصانعهم', 'فترددت']:
        cands = sv.segment(w)
        old = sv._analyze(w)
        top = cands[0] if cands else None
        print(f'  {w:<12} greedy gave root={old[0] or "-":<7} | validator best: '
              f'{top["prefix"] or "-":<3}+{top["stem"]:<9}+{top["suffix"] or "-":<3} '
              f'(root={top["root"] or "-"}) score={top["score"]}')

    print('\n=== the decisive test: does it prefer the split over the invented root? ===')
    good = 0
    tests = [('بالله', 'ب', 'الله'), ('فكم', 'ف', 'كم'), ('عنهم', '', 'عن'),
             ('والحقائق', 'وال', 'حقائق'), ('بمصانعهم', 'ب', 'مصانع')]
    for w, want_pre, want_stem in tests:
        cands = sv.segment(w)
        top = cands[0] if cands else {}
        ok = (top.get('prefix') == want_pre and strip_diac(top.get('stem', '')) == want_stem)
        good += ok
        print(f'  {w:<12} best={top.get("prefix","")}+{top.get("stem","")}+{top.get("suffix","")}'
              f'   want={want_pre}+{want_stem}   {"PASS" if ok else "FAIL"}')
    print(f'\n  {good}/{len(tests)}')

    json.dump({'prefixes': PREFIXES, 'suffixes': SUFFIXES,
               'attested_loaded': len(sv.attested)},
              open('/workspace/segment_validator.json', 'w'), ensure_ascii=False, indent=1)
    print('\nwrote /workspace/segment_validator.json')


if __name__ == '__main__':
    main()
