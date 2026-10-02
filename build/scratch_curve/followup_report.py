#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""followup_report.py -- REPORT_CURVE.md: learning curve + exposure analysis.

Answers the question the first report explicitly left open: is the low root accuracy
under-training, or is it where this configuration plateaus?
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
R = lambda n: json.load(open(os.path.join(HERE, n), encoding='utf-8'))


def pct(x, n=2):
    return f'{x*100:.{n}f}%'


curve = R('curve120k.json')['curve']
exp = R('../scratch_prompt/remote/exposure_vs_acc.json')

L = []
A = L.append
A('# Follow-up: is the root head under-trained, or has it plateaued?')
A('')
A('The first report left this open. `sf_morph.pt` has train root top-1 19.19% and held-out '
  'root top-1 20.18% — no overfitting — but "no overfitting" is compatible with both '
  '"still climbing" and "flat ceiling". `curve_lm.py` trains the **same architecture on the '
  'same data** (`/workspace/sf_data`, byte-unchanged: n_roots=9114, n_awzan=130, '
  '25,000,003 words) for **120,000 steps** — 20× the original run — evaluating every 10,000.')
A('')
A('## Learning curve')
A('')
A('| step | bits/word | root top1 | root top5 | prefix | wazn | suffix | '
  'test_deriv real-seen root | test_deriv special |')
A('|---|---|---|---|---|---|---|---|---|')
for r in curve:
    A(f"| {r['step']:,} | {r['bits_per_word']:.3f} | {pct(r['r_top1'])} | {pct(r['r_top5'])} | "
      f"{pct(r['p_top1'])} | {pct(r['w_top1'])} | {pct(r['s_top1'])} | "
      f"{pct(r['test_deriv_real_seen_root_top1'])} | {pct(r['test_deriv_special_top1'])} |")
b, e = curve[0], curve[-1]
A('')
A(f"**Over the whole 20× run:** bits/word **{b['bits_per_word']:.3f} → {e['bits_per_word']:.3f} "
  f"({e['bits_per_word']-b['bits_per_word']:+.3f})** — a real, steady, monotone information gain "
  f"of about half a bit per word. Root top-1 **{pct(b['r_top1'])} → {pct(e['r_top1'])} "
  f"({(e['r_top1']-b['r_top1'])*100:+.2f} pt)**. Root top-5 "
  f"**{pct(b['r_top5'])} → {pct(e['r_top5'])} ({(e['r_top5']-b['r_top5'])*100:+.2f} pt)**.")
A('')
A('### The two curves decouple')
A('')
A('This is the finding. The model is **still learning** — but it is not learning lexical '
  'identity:')
A('')
A('* **bits/word falls monotonically and does not flatten** (13.748 → 13.230, no plateau in '
  'the last 40k steps). The model has spare capacity and is using it.')
A('* **root top-1 is flat from 10k to 120k** (19.22% → 19.98%, +0.76 pt across 110,000 '
  'additional steps; the step-to-step noise is larger than most of those increments).')
A('* **`test_deriv` real-seen root is flat**: 0.97% → 1.81%, i.e. it never moves off the '
  'floor it already sat on at 6,000 steps (1.468%).')
A('* **the `special` bucket is saturated from the first evaluation**: 40.53% at 10k → 41.06% '
  'at 120k. Function words are a frequency lookup and are learned immediately.')
A('')
A('So the extra capacity went into the *structure* — affixation, function-word class, local '
  'constraints — and essentially none of it into *which root this word has*.')
A('')
A('## Exposure analysis: the model holds no per-sentence information')
A('')
A('`exposure_vs_acc.py` replays `scratch_lm.train`\'s sampler (`random.Random(0)`, `bs=16`, '
  '`max_sent=64`, 6,000 steps) to count how many times each training sentence was drawn.')
A('')
A('| times seen | sentences | positions | root top1 | root top5 | root top1 real | '
  'root top1 special |')
A('|---|---|---|---|---|---|---|')
for r in exp['exposure_buckets']:
    lab = f"{r['lo']}+" if r['hi'] > 10**8 else (f"{r['lo']}" if r['lo'] == r['hi']
                                                 else f"{r['lo']}-{r['hi']}")
    A(f"| {lab} | {r['sentences']:,} | {r['positions']:,} | {pct(r['root_top1'])} | "
      f"{pct(r['root_top5'])} | {pct(r['root_top1_real'])} | {pct(r['root_top1_special'])} |")
c = exp['never_seen_control']
A(f"| **never drawn** (control) | {c['sentences']:,} | {c['positions']:,} | "
  f"{pct(c['root_top1'])} | {pct(c['root_top5'])} | {pct(c['root_top1_real'])} | — |")
A('')
A(f"**{exp['total_seen_sentences']:,} distinct training sentences were drawn, and the "
  f"distribution is almost entirely 1×**: because `batches_morph` uses `rng.sample` *without "
  f"replacement* and discards the leftover partial batch, 92,496 were seen exactly once, "
  f"1,719 twice, 22 three-to-four times.")
A('')
A('**Accuracy is flat in exposure** — 19.20% (1×), 18.80% (2×), 20.77% (3-4×) — and the '
  f"control of **{c['sentences']:,} sentences the sampler never drew scores "
  f"{pct(c['root_top1'])}**, statistically identical to the ones it did. Drawing a sentence "
  'adds nothing. There is no memorisation to speak of, and no "it just needs more repeats" '
  'mechanism available.')
A('')
A('## Verdict (revised)')
A('')
A('The first report\'s verdict — "trained too little to have an opinion" — is **half right, '
  'and the correction matters.**')
A('')
A('* It is not overfitting: still true, and now much stronger. 20× more training produced no '
  'train/held-out divergence.')
A('* It was under-trained at 6,000 steps: true, and quantified — bits/word was still falling.')
A('* **But the root head does not improve with training budget.** 110,000 extra steps bought '
  '+0.76 pt root top-1 and +2.54 pt root top-5, while bits/word fell 0.52. The configuration '
  'learns the distribution of *structure* and saturates on *lexical identity* almost '
  'immediately.')
A('')
A('So "run it longer" will not fix the 1% root accuracy. The bottleneck is not budget. It is '
  'either the root head\'s optimisation or the shared encoder\'s ability to carry lexical '
  'information — which is a **structural** question about this design, not a training-length '
  'question.')
A('')
A('## The experiment that would settle it')
A('')
A('**Shrink the root vocabulary and see if the head can learn at all.** Re-labelling to the '
  'top ~500 roots (about 60% of real-root positions at ~1,250× less cardinality) and training '
  'the same 120k steps would separate the two live hypotheses:')
A('')
A('* root accuracy jumps to a plausible level → the head/encoder work; 9,114 classes at this '
  'data scale is the binding constraint (a data-per-class problem, and the morphemic scheme '
  'would still be viable with fewer, or compositionally-encoded, classes)')
A('* root accuracy stays at its floor → the root slot is not being optimised or not being '
  'carried by the shared trunk, which is an architecture bug to find')
A('')
A('This is worth the GPU time precisely because the answer changes what to build next, and '
  'neither the curve nor the exposure analysis can distinguish the two.')
A('')

open(os.path.join(HERE, '..', 'REPORT_CURVE.md'), 'w', encoding='utf-8').write('\n'.join(L))
print('wrote REPORT_CURVE.md')
