#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
circularity_check.py -- are the round-trip failures the GENERATOR's fault, or the ANALYSER's?

The round trip is circular: (root, wazn) pairs are taken from vocab.encode_word() on corpus
words, and the result is validated by that same encode_word(). So an analyser mislabelling is
indistinguishable from a generation error -- and the "failing text" may simply be grammatically
wrong, or wrongly analysed, rather than the grammarians' algorithms being wrong.

This separates the two by looking at the ACTUAL corpus words behind each failing pair:

  * print the surface word(s) that produced the pair
  * print the analyser's attribution (root, wazn)
  * print what the engine generates for that pair
  * check whether the generated form is itself ATTESTED in the corpus

A failure where the generated form IS attested elsewhere points at the analyser/label.
A failure where it is NOT attested, and the corpus word is a different shape, points at the
generator.

Usage: python circularity_check.py
"""
import glob
import json
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, '/workspace/hf_v19_2_release')
sys.path.insert(0, '/workspace/hf_v19_2_release/models')

from tasrif_engine import TasrifEngine, Root, strip_diac

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

    files = (sorted(glob.glob('/workspace/scholastic_sanitized/*.txt'))
             + sorted(glob.glob('/workspace/andalusian_canon_sanitized/*.txt')))[:6]
    # index corpus word forms and their analyser attribution
    surface_of = defaultdict(Counter)      # (root, pattern) -> surface forms
    attested = Counter()                   # bare surface forms in the corpus
    words = []
    for f in files:
        txt = open(f, encoding='utf-8', errors='ignore').read()
        words += [w for w in re.split(r'\s+', txt) if AR.search(w)]
    print(f'[*] corpus words: {len(words)}')
    for w in words:
        b = strip_diac(w)
        attested[b] += 1
        p, r, wz, s = vocab.encode_word(w)
        rn, wn = vocab.id2root.get(r, ''), vocab.id2wazn.get(wz, '')
        pat = wn.split('wazn_')[-1].rstrip('>')
        if pat and len(strip_diac(rn)) == 3:
            surface_of[(rn, pat)][b] += 1

    # ---- the MUDAAF failures, examined one by one ---------------------------------------
    print('\n=== MUDAAF pairs the round trip flagged as failures ===')
    shown = 0
    for (rn, pat), forms in sorted(surface_of.items(), key=lambda x: -sum(x[1].values())):
        if Root.klass(rn) != 'MUDAAF':
            continue
        g = eng.generate(rn, pat)
        if not g:
            continue
        gen = strip_diac(g[0])
        top = forms.most_common(1)[0][0]
        # a failure means the analyser does not recover the root from the GENERATED form
        back = strip_diac(vocab.id2root.get(vocab.encode_word(g[0])[1], ''))
        if back == strip_diac(rn):
            continue
        shown += 1
        gen_attested = attested.get(gen, 0)
        print(f'  root={rn:<5} pattern={pat:<11}')
        print(f'      corpus form(s)   : {list(forms)[:4]}')
        print(f'      analyser says    : root={back or "(none)"} for the generated form')
        print(f'      engine generates : {gen}   (attested in corpus: {gen_attested}x)')
        # what does the analyser call the top corpus form?
        p2, r2, w2, s2 = vocab.encode_word(top)
        print(f'      analyser on "{top}" : root={vocab.id2root.get(r2,"?")} '
              f'wazn={vocab.id2wazn.get(w2,"?").split("wazn_")[-1].rstrip(">")}')
        print()
        if shown >= 10:
            break

    # ---- how often is the surface itself labelled inconsistently? ------------------------
    print('=== does the analyser agree with itself on the same surface? ===')
    multi = [(k, v) for k, v in surface_of.items() if len(v) > 1]
    print(f'  (root, pattern) keys seen with more than one surface form: {len(multi)}')
    return


if __name__ == '__main__':
    main()
