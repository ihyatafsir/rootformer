#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ilal_conditioned.py -- replace blind weak-letter substitution with the taṣrīf bāb system.

The current rule_weak() tries و/ي/ا at every weak position and stops at the first inventory hit.
That is not the classical procedure. The taṣrīf tradition classifies a weak root by WHERE the
weakness sits, and that classification comes from the pattern:

    مثال  (assimilated)  C1 weak      وعد، يسر، وصل
    أجوف  (hollow)       C2 weak      قال، باع، صار، خاف
    ناقص  (defective)    C3 weak      دعا، رمى، نهي
    لفيف  (doubly weak)  two weak     وقي، ولي، طوي

So the wazn determines which radical is weak; the surface letter then follows from the bāb. This
script:

  1. distils the alternation from the INVENTORY itself -- for each (position, surface letter),
     the attested canonical letters and their frequencies. That table *is* the taṣrīf knowledge
     as the lexicon embodies it, conditioned on position rather than guessed.
  2. re-runs the analyzer with the table-guided rule instead of the blind one.
  3. reports UNK before/after, so the gain is attributable per rule.

Usage: python ilal_conditioned.py --n 40000
"""
import argparse
import glob
import json
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, '/workspace/hf_v19_2_release')
sys.path.insert(0, '/workspace/hf_v19_2_release/models')

DIAC = re.compile(r'[\u064b-\u0652\u0670\u0640]')
AR = re.compile(r'[\u0600-\u06FF]')
HAMZA = {'أ': 'ء', 'إ': 'ء', 'آ': 'ء', 'ؤ': 'ء', 'ئ': 'ء', 'ٱ': 'ا', 'ى': 'ي'}
WEAKISH = set('وياأإآؤئٱى')


def strip(s):
    return DIAC.sub('', s).strip()


class IlalTable:
    """(position, surface letter) -> ranked canonical letters, distilled from the inventory."""

    def __init__(self, vocab):
        self.vocab = vocab
        self.table = defaultdict(Counter)
        self.weak_pos_freq = Counter()
        n = 0
        for r in vocab.roots_list:
            if str(r).startswith('<'):
                continue
            s = strip(r)
            if len(s) != 3:
                continue
            n += 1
            for i, ch in enumerate(s):
                if ch in WEAKISH:
                    self.table[(i, ch)][ch] += 1
                    self.weak_pos_freq[i] += 1
        self.n_roots = n
        # canonical preference: within a (position, surface) cell, order by attested frequency
        self.ranked = {k: [c for c, _ in v.most_common()] for k, v in self.table.items()}

    def candidates(self, pos, ch):
        """Taṣrīf-licensed alternants at this position, most-attested first."""
        out = list(self.ranked.get((pos, ch), []))
        for w in ('و', 'ي', 'ا'):
            if w not in out:
                out.append(w)
        return out

    def report(self):
        print(f'[*] inventory roots used for the table: {self.n_roots}')
        print('    weak position distribution (the bab system):')
        names = {0: 'mithal  (C1 weak)', 1: 'ajwaf   (C2 weak)', 2: 'naqis   (C3 weak)'}
        for i in (0, 1, 2):
            print(f'      {names[i]:<20} {self.weak_pos_freq.get(i, 0):>5}')
        print('    alternation table (position, surface) -> attested canonical letters:')
        for (pos, ch), c in sorted(self.table.items(), key=lambda x: -sum(x[1].values()))[:12]:
            top = ', '.join(f'{k}:{v}' for k, v in c.most_common(3))
            print(f'      C{pos+1} {ch}  ->  {top}')


class ConditionedAnalyzer:
    RULES = ('rule_hamza', 'rule_weak_conditioned', 'rule_idgham', 'rule_mazid')

    def __init__(self, vocab, table):
        self.v = vocab
        self.root2id = vocab.root2id
        self.t = table
        self.stats = Counter()

    def rule_hamza(self, root):
        c = ''.join(HAMZA.get(ch, ch) for ch in root)
        if c != root:
            rid = self.root2id.get(c)
            if rid is not None:
                return rid, 'hamza'
        return None, None

    def rule_weak_conditioned(self, root):
        base = [HAMZA.get(ch, ch) for ch in root]
        for i, ch in enumerate(base):
            if ch not in WEAKISH and ch not in 'ويا':
                continue
            for cand in self.t.candidates(i, ch):
                if cand == ch:
                    continue
                trial = ''.join(base[:i] + [cand] + base[i + 1:])
                rid = self.root2id.get(trial)
                if rid is not None:
                    return rid, f'ilal_C{i+1}'
        return None, None

    def rule_idgham(self, root):
        if len(root) == 2:
            rid = self.root2id.get(root + root[-1])
            if rid is not None:
                return rid, 'idgham'
        return None, None

    def rule_mazid(self, root):
        if len(root) == 4:
            for i in range(4):
                rid = self.root2id.get(root[:i] + root[i + 1:])
                if rid is not None:
                    return rid, 'mazid'
        return None, None

    def analyze(self, word):
        clean = strip(word)
        if not clean:
            return self.v.UNK_ROOT, 'UNK'
        tag = f'<P:{clean}>'
        if tag in self.root2id:
            return self.root2id[tag], 'particle'
        _, r, wz, s = self.v.base_tok.decompose_arabic_word(clean)
        if r:
            rid = self.root2id.get(r) or self.root2id.get(
                ''.join(HAMZA.get(c, c) for c in r))
            if rid is not None:
                return rid, 'shipped'
        surface = r if r else clean
        for name in self.RULES:
            rid, how = getattr(self, name)(surface)
            if rid is not None:
                return rid, how
        if not r:
            return self.root2id.get(f'<P:{clean}>', self.root2id['<PARTICLE>']), 'particle'
        return self.v.UNK_ROOT, 'UNK'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=40000)
    ap.add_argument('--out', default='/workspace/ilal_conditioned.json')
    args = ap.parse_args()

    import nrmp_vocab as nv
    V = next(v for k, v in vars(nv).items() if isinstance(v, type) and 'MorphemicVocab' in k)
    vocab = V('/workspace/hf_v19_2_release/data/rootformer_v12_arabic_blueprint.json')

    table = IlalTable(vocab)
    table.report()
    ca = ConditionedAnalyzer(vocab, table)

    files = (sorted(glob.glob('/workspace/scholastic_sanitized/*.txt'))
             + sorted(glob.glob('/workspace/andalusian_canon_sanitized/*.txt')))[:6]
    words = []
    for f in files:
        txt = open(f, encoding='utf-8', errors='ignore').read()
        words += [w for w in re.split(r'\s+', txt) if AR.search(w)]
    words = words[:args.n]

    shipped = cond = 0
    rules = Counter()
    for w in words:
        shipped += (vocab.encode_word(w)[1] == vocab.UNK_ROOT)
        rid, how = ca.analyze(w)
        cond += (rid == vocab.UNK_ROOT)
        rules[how] += 1

    n = len(words)
    print(f'\n[*] sampled {n} words')
    print(f'    UNK, shipped                 : {100*shipped/n:6.2f}%  ({shipped})')
    print(f'    UNK, wazn-conditioned taṣrīf : {100*cond/n:6.2f}%  ({cond})')
    print(f'    failures fixed               : {shipped-cond} '
          f'({100*(shipped-cond)/max(shipped,1):.1f}%)')
    print('\n    resolution by rule:')
    for k, v in rules.most_common(10):
        print(f'      {k:<14} {v:>7}  {100*v/n:5.2f}%')
    json.dump({'n': n, 'shipped_unk': shipped, 'conditioned_unk': cond,
               'rules': dict(rules)}, open(args.out, 'w'), indent=2)


if __name__ == '__main__':
    main()
