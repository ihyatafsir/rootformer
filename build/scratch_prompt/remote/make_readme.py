#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""make_readme.py -- write README.md for /workspace/scratch_prompt/."""
import os

HERE = os.path.dirname(os.path.abspath(__file__))

TXT = """# scratch_prompt -- prompt-test of sf_morph.pt

Read-only on `scratch_lm.py`, `nrmp_vocab.py`, the blueprint, checkpoints and
`/workspace/nrmp_cache_9490`.  Everything here writes only under
`/workspace/scratch_prompt/`.  Nothing here touches the GPU: one CPU pass per script.

## Reproduce

```bash
cd /workspace/scratch_prompt
P=/workspace/venvs/rootformer/bin/python

$P verify_ids.py                 # is the live vocab's id space the checkpoint's? (yes, appends)
$P repro_check.py                # reproduces all 6 published heldout_root_eval numbers exactly
$P probe2.py acc                 # per-head acc, stratified; SLOW, and its train_seen row is buggy
$P fixups.py                     # the correct per-head table + unigram-on-same-positions
$P vs_unigram.py                 # marginal probability of the model's own predictions
$P root_marginal.py              # exact root marginals, constant/unigram baselines, special bucket
$P collapse_analysis.py          # distinct tuples emitted, top-5 share, P(<PARTICLE>), argmax changes
$P prompt_at_positions.py testgen 12 7        # 12 verbatim held-out-work prompts
$P prompt_at_positions.py heldout 20 7        # 20 held-out-root prompts from test_deriv
$P prompt_predict.py interactive              # one context per line on stdin
```

`probe2.py acc` is superseded by `fixups.py` (its summary table mis-printed the weighted
float in the position column and its unigram block crashed on `torch.isin`); it is kept
because its raw bucket logs are the source of the stratification weights.

## Key results

* per-head top-1/top-5: train root 19.07%, train_seen root 19.19%, test_gen root 20.18%,
  test_deriv root 17.50%  ->  **no overfitting**
* unigram on the same positions: test_gen root 18.91% top-1, test_deriv 16.58%
* unigram restricted to REAL (non-special) root targets: **0.000% / 0.000%**;
  the model: **1.490% / 5.228%** (test_gen), **1.468% / 4.916%** (test_deriv)
* one tuple is 80% of every top-1 prediction on every split; true distinct tuples are 5,000-6,600
* the `special` bucket (ids 0-70 = 5 controls + 66 `<P:...>`) is 46% of all target
  positions; `<PARTICLE>` alone is 40.3% of it and the model scores 41.471% on the bucket
  -> frequency, not skill; `<UNK>` is a legitimate token, not a collapse
* `test_gen/heldout_root n=0` is **structural**: `test_gen` is defined as test sentences
  containing no held-out root, so that cell is identically empty.  `test_deriv` is the arm
  that does contain them (4,602 tokens).

## Gotcha

`sf_morph.pt` was trained under a 9,114-root / 130-wazn id space.  The live `nrmp_vocab.py`
(rebuilt 2026-10-01 18:07) now exposes 9,490 roots / 142 awzan.  New ids are APPENDED so old
ids are stable (verified), but the live encoder can still emit ids the checkpoint cannot
embed -- 245 distinct new root ids appear in 16,991 tokens across 12,000 sampled corpus
sentences.  `prompt_predict.check_enc()` and `prompt_at_positions.safe()` trim at the first
out-of-range id; any future decoding of this checkpoint must do the same.
"""

open(os.path.join(HERE, '..', 'README.md'), 'w', encoding='utf-8').write(TXT)
print('wrote README.md')
