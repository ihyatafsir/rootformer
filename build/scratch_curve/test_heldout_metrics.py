#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test_heldout_metrics.py -- call curve_lm.held_out_metrics() directly on the 6k checkpoint.

The training run reported `deriv real-seen 0.00`, and the earlier grep suggested a slicing
bug.  Rather than reason about it, this calls the real function and prints the raw floats.
test-deriv real_seen_root top1 must be ~1.468% (the published value) if the function is right.
"""
import json
import os
import sys

import torch

sys.path.insert(0, '/workspace/scratch_curve')
sys.path.insert(0, '/workspace/hf_v19_2_release')
sys.path.insert(0, '/workspace/hf_v19_2_release/models')

import curve_lm as C  # noqa: E402

D = '/workspace/sf_data'
meta = json.load(open(f'{D}/meta.json'))
print('[*] loading', flush=True)
data = torch.load(f'{D}/streams.pt', weights_only=False)
import scratch_lm as S  # noqa: E402

dev = torch.device('cpu')
model = S.MorphemicLM((meta['n_roots'], meta['n_awzan'],
                       meta['n_prefixes'], meta['n_suffixes'])).to(dev)
model.load_state_dict(torch.load('/workspace/scratch_lm_run/sf_morph.pt',
                                 map_location=dev, weights_only=True))
model.eval()

for nsent in (400, 2000, 8000):
    m = C.held_out_metrics(model, data, meta['n_roots'], meta['special_root_ids'],
                           dev, nsent, seed=11)
    print(f'\n=== held_out_metrics(n_sent={nsent}) ===')
    print(f"  bits/word      {m['bits_per_word']:.4f}")
    for h in ('p', 'r', 'w', 's'):
        print(f"  {h}: top1 {m[h]['top1']*100:6.3f}%  top5 {m[h]['top5']*100:6.3f}%  "
              f"n={m[h]['n']}")
    print(f"  root_cats      {json.dumps(m['root_cats'], indent=None)}")
    print(f"  REAL-SEEN top1 (published 1.468) -> "
          f"{m['root_cats']['real_seen_root']['top1']*100:.3f}%  "
          f"n={m['root_cats']['real_seen_root']['n']} (published 19753)")
    print(f"  SPECIAL   top1 (published 39.202) -> "
          f"{m['root_cats']['special']['top1']*100:.3f}%  "
          f"n={m['root_cats']['special']['n']} (published 17818)")
