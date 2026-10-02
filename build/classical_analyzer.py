#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
classical_analyzer.py -- put the ANALYZER under the grammarians' algorithms.

The shipped analyzer fails on 3.8% of positions: encode_word() delegates to
decompose_arabic_word(), and if the returned root is not literally in the 9,114-root inventory it
is thrown away as <UNK>. No derivational rules are applied at all.

That is exactly the gap the taṣrīf tradition fills. Ibn Mālik's students and the Andalusian
morphologists (Ibn ʿUsfūr, Abū Ḥayyān, al-Shāṭibī) specify the sound<->weak alternations (iʿlāl),
hamza substitution (ibdāl), and the doubled-root (idghām) and augmented (mazīd) patterns. Applied
as a canonicalisation ladder over the root inventory, they recover roots the lookup misses.

What this does, in order, per word:
  0. particle / exact lookup                       (as shipped)
  1. hamza canonicalisation            أ إ آ ؤ ئ ٱ -> ء / ا
  2. weak-letter (iʿlāl) expansion     each و ي ا ى -> {و, ي, ا}, look up
  3. doubled-root (idghām) expansion   C1C2 -> C1C2C2
  4. augmented (mazīd) trimming        4-letter roots: drop the added letter, retry
  5. clitic stripping + retry          ال / و / ف / ب / ك / ل + pronoun suffixes

Reports UNK rate before and after, and which rule recovered each case, so the gain is attributable
rather than asserted.

Usage: python classical_analyzer.py --n 40000
"""
import argparse
import json
import re
import sys
from collections import Counter

sys.path.insert(0, '/workspace/hf_v19_2_release')
sys.path.insert(0, '/workspace/hf_v19_2_release/models')

DIAC = re.compile(r'[\u064b-\u0652\u0670\u0640]')
AR = re.compile(r'[\u0600-\u06FF]')
HAMZA = {'أ': 'ء', 'إ': 'ء', 'آ': 'ء', 'ؤ': 'ء', 'ئ': 'ء', 'ٱ': 'ا', 'ى': 'ي'}
WEAK = ('و', 'ي', 'ا')
CLITICS = ['ال', 'وال', 'فال', 'بال', 'كال', 'لل', 'و', 'ف', 'ب', 'ك', 'ل', 'س']
SUFFIXES = ['ها', 'هما', 'هم', 'هن', 'كما', 'كم', 'نا', 'ني', 'ه', 'ك', 'ت', 'ا', 'وا', 'ون',
            'ين', 'ات', 'ان', 'ية', 'يه']


def strip(s):
    return DIAC.sub('', s).strip()


class ClassicalAnalyzer:
    """Derivational canonicalisation over the root inventory (Khalilian + Andalusian taṣrīf)."""

    def __init__(self, vocab):
        self.v = vocab
        self.root2id = vocab.root2id
        self.stats = Counter()

    # ---------------------------------------------------------------- rules
    def _lookup(self, cand):
        return self.root2id.get(cand)

    def rule_hamza(self, root):
        c = ''.join(HAMZA.get(ch, ch) for ch in root)
        if c != root:
            rid = self._lookup(c)
            if rid is not None:
                return rid, 'hamza', c
        return None, None, None

    def rule_weak(self, root):
        """iʿlāl: a weak radical may surface as و/ي/ا; try every substitution."""
        idx = [i for i, ch in enumerate(root) if ch in WEAK or ch in HAMZA]
        if not idx:
            return None, None, None
        base = [HAMZA.get(ch, ch) for ch in root]
        for i in idx:
            for w in WEAK:
                if base[i] == w:
                    continue
                cand = ''.join(base[:i] + [w] + base[i + 1:])
                rid = self._lookup(cand)
                if rid is not None:
                    return rid, 'ilal', cand
        return None, None, None

    def rule_idgham(self, root):
        """Doubled root: a 2-letter surface root expands to C1C2C2."""
        if len(root) == 2:
            cand = root + root[-1]
            rid = self._lookup(cand)
            if rid is not None:
                return rid, 'idgham', cand
        return None, None, None

    def rule_mazid(self, root):
        """4-letter surface form: one letter is an augment; drop each in turn."""
        if len(root) == 4:
            for i in range(4):
                cand = root[:i] + root[i + 1:]
                rid = self._lookup(cand)
                if rid is not None:
                    return rid, 'mazid', cand
        return None, None, None

    def rule_clitic(self, word):
        w = strip(word)
        for pre in sorted(CLITICS, key=len, reverse=True):
            if w.startswith(pre) and len(w) > len(pre) + 2:
                _, r, wz, s = self.v.base_tok.decompose_arabic_word(w[len(pre):])
                if r:
                    rid = self.root2id.get(HAMZA.get(r[0], r[0]) + r[1:]) \
                        if len(r) == 3 else None
                    if rid is not None:
                        return rid, f'clitic:{pre}', r
        for suf in sorted(SUFFIXES, key=len, reverse=True):
            if w.endswith(suf) and len(w) > len(suf) + 2:
                _, r, wz, s = self.v.base_tok.decompose_arabic_word(w[:-len(suf)])
                if r and len(r) == 3:
                    rid = self.root2id.get(r)
                    if rid is not None:
                        return rid, f'suffix:{suf}', r
        return None, None, None

    RULES = ('rule_hamza', 'rule_weak', 'rule_idgham', 'rule_mazid')

    def analyze(self, word):
        """Return (root_id, how) where how is 'shipped', a rule name, or 'UNK'."""
        clean = strip(word)
        if not clean:
            return self.v.UNK_ROOT, 'UNK'
        tag = f'<P:{clean}>'
        if tag in self.root2id:
            return self.root2id[tag], 'particle'
        p, r, wz, s = self.v.base_tok.decompose_arabic_word(clean)
        if r:
            rid = self.root2id.get(r)
            if rid is not None:
                return rid, 'shipped'
            cand = ''.join(HAMZA.get(ch, ch) for ch in r)
            rid = self.root2id.get(cand)
            if rid is not None:
                return rid, 'shipped'
        # ---- derivational ladder ----
        surface = r if r else clean
        for name in self.RULES:
            rid, how, _ = getattr(self, name)(surface)
            if rid is not None:
                return rid, how
        rid, how, _ = self.rule_clitic(clean)
        if rid is not None:
            return rid, how
        if not r:
            # shipped behaviour: no root at all is a function word, not an analysis failure
            return self.root2id.get(f'<P:{clean}>', self.root2id['<PARTICLE>']), 'particle'
        return self.v.UNK_ROOT, 'UNK'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=40000)
    ap.add_argument('--out', default='/workspace/classical_analyzer.json')
    args = ap.parse_args()

    import glob
    import nrmp_vocab as nv
    V = next(v for k, v in vars(nv).items() if isinstance(v, type) and 'MorphemicVocab' in k)
    vocab = V('/workspace/hf_v19_2_release/data/rootformer_v12_arabic_blueprint.json')
    ca = ClassicalAnalyzer(vocab)

    files = (sorted(glob.glob('/workspace/scholastic_sanitized/*.txt'))
             + sorted(glob.glob('/workspace/andalusian_canon_sanitized/*.txt')))[:6]
    words = []
    for f in files:
        txt = open(f, encoding='utf-8', errors='ignore').read()
        words += [w for w in re.split(r'\s+', txt) if AR.search(w)]
    words = words[:args.n]
    print(f'[*] sampled {len(words)} words from {len(files)} works\n')

    shipped_unk = new_unk = fixed_by_rule = 0
    how_counts = Counter()
    for w in words:
        s_unk = (vocab.encode_word(w)[1] == vocab.UNK_ROOT)     # the REAL shipped behaviour
        shipped_unk += s_unk
        rid, how = ca.analyze(w)
        how_counts[how] += 1
        new_unk += (rid == vocab.UNK_ROOT)
        if s_unk and rid != vocab.UNK_ROOT:
            fixed_by_rule += 1

    n = len(words)
    print(f'UNK rate, shipped analyzer     : {100*shipped_unk/n:6.2f}%  ({shipped_unk}/{n})')
    print(f'UNK rate, classical canonical  : {100*new_unk/n:6.2f}%  ({new_unk}/{n})')
    fixed = shipped_unk - new_unk
    print(f'failures fixed                 : {fixed}  '
          f'({100*fixed/max(shipped_unk,1):.1f}% of the shipped failures)\n')
    print('how each word was resolved:')
    for k, c in how_counts.most_common(12):
        print(f'   {k:<20} {c:>7}  {100*c/n:5.2f}%')

    json.dump({'n': n, 'shipped_unk': shipped_unk, 'classical_unk': new_unk,
               'fixed_by_rule': fixed_by_rule,
               'reduction_pct': 100 * (shipped_unk - new_unk) / max(shipped_unk, 1),
               'resolution': dict(how_counts)},
              open(args.out, 'w'), indent=2)
    print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
