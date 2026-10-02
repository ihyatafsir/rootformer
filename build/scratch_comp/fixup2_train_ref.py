#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Adds the TRAIN split to unigram_root_acc inside the stored *_eval.json files, so the fair
baseline sits beside the train-vs-held-out per-head comparison too."""
import glob
import json
import sys

import torch

sys.path.insert(0, '/workspace/scratch_comp')
import scratch_lm_comp as S  # noqa: E402

D = '/workspace/scratch_comp/sf_data_9490'
meta = json.load(open(f'{D}/meta.json'))
data = torch.load(f'{D}/streams.pt', weights_only=False)
tables = S.Tables(meta)
ref = S.unigram_root_acc(data, ['train'], tables)['train']
print(json.dumps(ref, indent=2))
for fn in sorted(glob.glob(f'{D}/*_eval.json')):
    j = json.load(open(fn))
    if 'unigram_root_acc' in j:
        j['unigram_root_acc']['train'] = ref
        json.dump(j, open(fn, 'w'), indent=2, ensure_ascii=False)
        print('patched', fn)
