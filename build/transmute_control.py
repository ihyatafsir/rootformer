#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
transmute_control.py -- Does the shipped v19.2 "transmutation" actually depend on the model?

The audit claims `MasterSovereignTransmuter.transmute_proposition(hidden_states, ar_text)`
never reads its `hidden_states` argument, so the English output is a pure function of the
input string plus hardcoded dictionaries/regexes.

This script tests that directly, using the REAL published weights:
  A) real hidden states from the backbone
  B) all-zero hidden states of identical shape
  C) hidden_states = None
  D) hidden states from a DIFFERENT sentence (mismatched content)

If A == B == C == D for every input, the neural network cannot be influencing the output.
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import transmute_quickstart as _tq  # noqa: E402

_loader = None
for _name in ('load_v19_2_engine', 'load_v19_engine', 'load_engine'):
    if hasattr(_tq, _name):
        _loader = getattr(_tq, _name)
        break
if _loader is None:
    raise SystemExit('no engine loader found in transmute_quickstart.py')

TEXTS = [
    'العلم نور يضيء العقل ويهدي إلى الحق',
    'رأس الحكمة مخافة الله',
    'اليقين لا يزول بالشك',
    'إنما الأعمال بالنيات وإنما لكل امرئ ما نوى',
    # deliberately out-of-distribution: words absent from the hardcoded lexicon
    'القط يشرب الحليب في الصباح',
    'تويتر منصة اجتماعية حديثة',
]


def hs_for(ar_model, vocab, text, device):
    tuples = vocab.encode_sentence(text)
    if not tuples:
        tuples = [(vocab.NONE_PREFIX, vocab.BOS_ROOT, vocab.NONE_WAZN, vocab.NONE_SUFFIX)]
    p = torch.tensor([[t[0] for t in tuples]], dtype=torch.long, device=device)
    r = torch.tensor([[t[1] for t in tuples]], dtype=torch.long, device=device)
    w = torch.tensor([[t[2] for t in tuples]], dtype=torch.long, device=device)
    s = torch.tensor([[t[3] for t in tuples]], dtype=torch.long, device=device)
    with torch.no_grad():
        emb = ar_model.morphemic_embed(p, r, w, s)
        out = ar_model.backbone(inputs_embeds=emb, output_hidden_states=True)
    return out.hidden_states


def main():
    ar_model, engine, vocab, device = _loader()
    print('\n' + '=' * 78)
    print('CONTROL: is the transmutation output a function of the hidden states?')
    print('=' * 78)
    all_identical = True
    for text in TEXTS:
        real = hs_for(ar_model, vocab, text, device)
        zeros = tuple(torch.zeros_like(h) for h in real)
        other = hs_for(ar_model, vocab, 'رأس الحكمة مخافة الله', device)

        with torch.no_grad():
            a = engine.transmute_proposition(real, text)
            b = engine.transmute_proposition(zeros, text)
            c = engine.transmute_proposition(None, text)
            try:
                d = engine.transmute_proposition(other, text)
            except Exception as e:  # noqa: BLE001
                d = f'<{type(e).__name__}>'

        same = (a == b == c == d)
        all_identical &= same
        print(f'\n  INPUT   : {text}')
        print(f'  real hs : {a}')
        print(f'  zero hs : {b}')
        print(f'  None    : {c}')
        print(f'  other hs: {d}')
        print(f'  => hidden-state-independent: {same}')

    print('\n' + '=' * 78)
    print(f'VERDICT: output ignores hidden states for every input tested: {all_identical}')
    print('=' * 78)


if __name__ == '__main__':
    main()
