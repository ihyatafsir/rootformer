#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""roundtrip_refine.py -- how much of the gold-tuple round-trip failure is punctuation only,
and how often the held-out target is a control/particle root (excluded from the metric)."""
import os
import sys
from collections import Counter

import torch

sys.path.insert(0, '/workspace/hf_v19_2_release')
sys.path.insert(0, '/workspace/hf_v19_2_release/models')
os.environ.setdefault('ROOTFORMER_VALIDATED_SEG', '0')
import nrmp_vocab as nv  # noqa: E402

V = next(v for k, v in vars(nv).items() if isinstance(v, type) and 'MorphemicVocab' in k)
vocab = V('/workspace/hf_v19_2_release/data/rootformer_v12_arabic_blueprint.json')
rb = torch.load('/workspace/rca_prompt/fp32/val_stream_rebuild.pt', map_location='cpu')
streams, surfaces = rb['streams'], rb['surfaces']
P, R, W, S = streams
N = P.numel()

specials = sorted({vocab.PAD_ROOT, vocab.BOS_ROOT, vocab.EOS_ROOT, vocab.UNK_ROOT,
                   vocab.root2id['<PARTICLE>']} |
                  {i for i, r in enumerate(vocab.roots_list) if r.startswith('<P:')})
spec = set(specials)
particle_roots = {i for i, r in enumerate(vocab.roots_list) if r.startswith('<P:')}


def arabic_only(s):
    return ''.join(ch for ch in s if '\u0600' <= ch <= '\u06FF' or ch == '\u0640')


stat = Counter()
samples = []
for i in range(1, N):
    r = int(R[i])
    stat['total_word_events'] += 1
    if r in spec:
        stat['target_is_control_or_particle'] += 1
        if r == vocab.root2id['<PARTICLE>']:
            stat['target_root_<PARTICLE>'] += 1
        elif r == vocab.UNK_ROOT:
            stat['target_root_<UNK>'] += 1
        elif r in (vocab.PAD_ROOT, vocab.BOS_ROOT, vocab.EOS_ROOT):
            stat['target_root_PAD/BOS/EOS'] += 1
        else:
            stat['target_root_<P:..>_particle'] += 1
        continue
    stat['target_is_radical'] += 1
    d = vocab.decode_word(int(P[i]), r, int(W[i]), int(S[i]))
    s = surfaces[i]
    if d == s:
        stat['roundtrip_exact'] += 1
    else:
        stat['roundtrip_fail'] += 1
        if d == arabic_only(s):
            stat['roundtrip_fail_punctuation_only'] += 1
        else:
            stat['roundtrip_fail_real'] += 1
            if len(samples) < 15:
                samples.append({'corpus': s, 'realiser': d,
                                'root': vocab.roots_list[r],
                                'wazn': vocab.id2wazn[int(W[i])],
                                'prefix': vocab.id2prefix[int(P[i])],
                                'suffix': vocab.id2suffix[int(S[i])]})

out = {k: int(v) for k, v in stat.items()}
out['target_control_or_particle_pct'] = 100.0 * stat['target_is_control_or_particle'] / max(N - 1, 1)
out['roundtrip_fail_pct_of_radical'] = 100.0 * stat['roundtrip_fail'] / max(stat['target_is_radical'], 1)
out['roundtrip_fail_real_pct_of_radical'] = 100.0 * stat['roundtrip_fail_real'] / max(stat['target_is_radical'], 1)
out['roundtrip_fail_punctuation_only_pct_of_failures'] = (
    100.0 * stat['roundtrip_fail_punctuation_only'] / max(stat['roundtrip_fail'], 1))
out['real_failure_samples'] = samples
import json
print(json.dumps(out, ensure_ascii=False, indent=2))
open('/workspace/rca_prompt/roundtrip_refine.json', 'w').write(json.dumps(out, ensure_ascii=False, indent=2))
