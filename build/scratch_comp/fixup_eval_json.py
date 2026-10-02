#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Recompute `unigram_bits_per_word_compositional` inside already-written *_eval.json files.

The first version of unigram_symbol_bits() charged 0 bits for special-root targets, so it was
not comparable to bits/word; the function was fixed and this patches the stored JSONs."""
import glob
import json
import sys

import torch

sys.path.insert(0, '/workspace/scratch_comp')
import scratch_lm_comp as S  # noqa: E402

D = sys.argv[1] if len(sys.argv) > 1 else '/workspace/scratch_comp/sf_data_9490'
meta = json.load(open(f'{D}/meta.json'))
data = torch.load(f'{D}/streams.pt', weights_only=False)
tables = S.Tables(meta)
splits = ['val', 'test_gen', 'test_deriv']
ub = S.unigram_bits(data, splits)
usb = S.unigram_symbol_bits(data, splits, tables)
print('type unigram        :', ub)
print('compositional unigram:', usb)
for fn in sorted(glob.glob(f'{D}/*_eval.json')):
    j = json.load(open(fn))
    j['unigram_bits_per_word_rootTYPE'] = ub
    j['unigram_bits_per_word_compositional'] = usb
    json.dump(j, open(fn, 'w'), indent=2, ensure_ascii=False)
    print('patched', fn)
