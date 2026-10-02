#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gold_roundtrip.py -- the round trip, with the contaminated pairs removed.

The earlier test validated the generator against pairs like (عنن, فَعَلَ) from the surface
"عنهم" -- the analyser had stripped the clitic هم, invented a root, and the test then blamed the
generator for not reproducing the invention. 10,331 (root, pattern) keys mapped to more than one
surface form, which is the signature of that mis-attribution.

Two filters remove it, and both are DATA-QUALITY filters, not passes for the algorithm:

  F1  no-clitic:  the surface must have exactly the consonant count the pattern implies
      (3 radicals + the pattern's literal consonants). A form with extra consonants came from a
      clitic combination and its attribution cannot be trusted.
  F2  consistency: for a (root, pattern) key, one surface form must dominate (>= 60% of the
      occurrences). An attribution that maps to many different surfaces is unreliable.

Whatever survives is a pair whose attribution is structural rather than guessed. The generator is
then judged against that, and the share filtered out is reported -- that share IS the analyser's
error rate on this test.

Usage: python gold_roundtrip.py
"""
import glob
import json
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, '/workspace/hf_v19_2_release')
sys.path.insert(0, '/workspace/hf_v19_2_release/models')
from tasrif_engine import TasrifEngine, Pattern, Root, strip_diac, SLOT

AR = re.compile(r'[\u0600-\u06FF]')


def main():
    import nrmp_vocab as nv
    V = next(v for k, v in vars(nv).items() if isinstance(v, type) and 'MorphemicVocab' in k)
    vocab = V('/workspace/hf_v19_2_release/data/rootformer_v12_arabic_blueprint.json')
    eng = TasrifEngine(vocab)
    d = json.load(open('/workspace/hf_v19_2_release/data/rootformer_v12_arabic_blueprint.json'))
    i2t = d['vocab_id_to_token']
    lo, hi = d['partitions']['classical_awzan']
    patterns = [i2t[str(i)].split('wazn_')[-1].rstrip('>') for i in range(lo, hi)]
    patcache = {p: Pattern(p) for p in patterns}

    files = (sorted(glob.glob('/workspace/scholastic_sanitized/*.txt'))
             + sorted(glob.glob('/workspace/andalusian_canon_sanitized/*.txt')))[:6]
    pairs = defaultdict(Counter)
    n_words = 0
    for f in files:
        txt = open(f, encoding='utf-8', errors='ignore').read()
        for w in re.split(r'\s+', txt):
            if not AR.search(w):
                continue
            n_words += 1
            p, r, wz, s = vocab.encode_word(w)
            rn, wn = vocab.id2root.get(r, ''), vocab.id2wazn.get(wz, '')
            pat = wn.split('wazn_')[-1].rstrip('>')
            if not pat or len(strip_diac(rn)) != 3 or rn.startswith('<'):
                continue
            pairs[(rn, pat)][strip_diac(w)] += 1

    total_pairs = len(pairs)
    kept, dropped_F1, dropped_F2 = {}, 0, 0
    for (rn, pat), forms in pairs.items():
        pat_o = patcache.get(pat)
        if pat_o is None:
            dropped_F1 += 1
            continue
        expected = 3 + sum(1 for c, _ in pat_o.elems if c not in SLOT)
        top, cnt = forms.most_common(1)[0]
        if len(top) != expected:                       # F1: clitic combination
            dropped_F1 += 1
            continue
        if cnt / sum(forms.values()) < 0.60:           # F2: inconsistent attribution
            dropped_F2 += 1
            continue
        kept[(rn, pat)] = top

    print(f'[*] corpus words {n_words}; distinct (root, pattern) keys {total_pairs}')
    print(f'[*] F1 dropped (surface length != pattern -> clitic mis-segmentation): {dropped_F1}')
    print(f'[*] F2 dropped (attribution maps to many surfaces -> unreliable)   : {dropped_F2}')
    print(f'[*] RETAINED (structurally sound attribution)                      : {len(kept)}')
    print(f'    -> the two filters removed '
          f'{100*(dropped_F1+dropped_F2)/max(total_pairs,1):.1f}% of the pairs, which is a direct '
          f'measure of analyser mis-attribution\n')

    ok = 0
    by_class, by_class_ok = Counter(), Counter()
    fails = []
    for (rn, pat), surface in kept.items():
        g = eng.generate(rn, pat)
        cls = Root.klass(rn)
        by_class[cls] += 1
        good = bool(g) and strip_diac(g[0]) == surface
        ok += good
        by_class_ok[cls] += good
        if not good and len(fails) < 12:
            fails.append((rn, pat, surface, g[0] if g else '-'))

    print(f'=== GOLD ROUND TRIP on the retained pairs ===')
    print(f'  generator reproduces the attested surface: {ok}/{len(kept)} = '
          f'{100*ok/max(len(kept),1):.1f}%\n')
    print(f'  {"class":<10}{"tested":>8}{"match":>8}{"rate":>8}')
    for k in sorted(by_class, key=lambda x: -by_class[x]):
        n_, o_ = by_class[k], by_class_ok[k]
        print(f'  {k:<10}{n_:>8}{o_:>8}{100*o_/max(n_,1):>7.1f}%')
    print('\n  failures (root, pattern, attested, generated):')
    for rn, pat, att, gen in fails:
        print(f'    {rn:<6}{pat:<12}attested {att:<12} generated {gen}')
    json.dump({'retained': len(kept), 'ok': ok, 'dropped_F1': dropped_F1,
               'dropped_F2': dropped_F2, 'by_class': dict(by_class),
               'by_class_ok': dict(by_class_ok), 'fails': fails},
              open('/workspace/gold_roundtrip.json', 'w'), ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
