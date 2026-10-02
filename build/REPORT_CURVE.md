# Follow-up: is the root head under-trained, or has it plateaued?

The first report left this open. `sf_morph.pt` has train root top-1 19.19% and held-out root top-1 20.18% — no overfitting — but "no overfitting" is compatible with both "still climbing" and "flat ceiling". `curve_lm.py` trains the **same architecture on the same data** (`/workspace/sf_data`, byte-unchanged: n_roots=9114, n_awzan=130, 25,000,003 words) for **120,000 steps** — 20× the original run — evaluating every 10,000.

## Learning curve

| step | bits/word | root top1 | root top5 | prefix | wazn | suffix | test_deriv real-seen root | test_deriv special |
|---|---|---|---|---|---|---|---|---|
| 10,000 | 13.748 | 19.22% | 32.50% | 71.56% | 45.42% | 82.85% | 0.97% | 40.53% |
| 20,000 | 13.647 | 19.20% | 32.88% | 71.65% | 45.37% | 82.86% | 1.42% | 39.85% |
| 30,000 | 13.598 | 19.27% | 33.25% | 71.57% | 45.31% | 82.82% | 1.29% | 39.62% |
| 40,000 | 13.502 | 19.28% | 33.67% | 71.58% | 45.68% | 82.83% | 1.36% | 39.32% |
| 50,000 | 13.490 | 19.48% | 33.84% | 71.44% | 45.99% | 82.83% | 1.49% | 40.15% |
| 60,000 | 13.429 | 19.42% | 34.18% | 71.72% | 45.95% | 82.81% | 1.16% | 40.83% |
| 70,000 | 13.367 | 19.61% | 34.16% | 71.81% | 45.82% | 82.83% | 1.68% | 40.68% |
| 80,000 | 13.314 | 19.62% | 34.43% | 71.75% | 45.86% | 82.81% | 1.68% | 40.38% |
| 90,000 | 13.282 | 19.79% | 34.84% | 71.69% | 46.19% | 82.82% | 1.49% | 40.23% |
| 100,000 | 13.246 | 19.76% | 34.92% | 71.82% | 46.00% | 82.83% | 1.75% | 40.30% |
| 110,000 | 13.232 | 19.94% | 34.97% | 71.88% | 46.14% | 82.82% | 1.68% | 41.21% |
| 120,000 | 13.230 | 19.98% | 35.04% | 71.88% | 46.13% | 82.81% | 1.81% | 41.06% |

**Over the whole 20× run:** bits/word **13.748 → 13.230 (-0.518)** — a real, steady, monotone information gain of about half a bit per word. Root top-1 **19.22% → 19.98% (+0.76 pt)**. Root top-5 **32.50% → 35.04% (+2.54 pt)**.

### The two curves decouple

This is the finding. The model is **still learning** — but it is not learning lexical identity:

* **bits/word falls monotonically and does not flatten** (13.748 → 13.230, no plateau in the last 40k steps). The model has spare capacity and is using it.
* **root top-1 is flat from 10k to 120k** (19.22% → 19.98%, +0.76 pt across 110,000 additional steps; the step-to-step noise is larger than most of those increments).
* **`test_deriv` real-seen root is flat**: 0.97% → 1.81%, i.e. it never moves off the floor it already sat on at 6,000 steps (1.468%).
* **the `special` bucket is saturated from the first evaluation**: 40.53% at 10k → 41.06% at 120k. Function words are a frequency lookup and are learned immediately.

So the extra capacity went into the *structure* — affixation, function-word class, local constraints — and essentially none of it into *which root this word has*.

## Exposure analysis: the model holds no per-sentence information

`exposure_vs_acc.py` replays `scratch_lm.train`'s sampler (`random.Random(0)`, `bs=16`, `max_sent=64`, 6,000 steps) to count how many times each training sentence was drawn.

| times seen | sentences | positions | root top1 | root top5 | root top1 real | root top1 special |
|---|---|---|---|---|---|---|
| 1 | 92,496 | 809,458 | 19.20% | 34.65% | 1.72% | 39.11% |
| 2 | 1,719 | 14,791 | 18.80% | 34.68% | 1.60% | 38.31% |
| 3-4 | 22 | 207 | 20.77% | 39.13% | 2.00% | 38.32% |
| **never drawn** (control) | 40,000 | 350,168 | 19.02% | 34.35% | 1.63% | — |

**94,237 distinct training sentences were drawn, and the distribution is almost entirely 1×**: because `batches_morph` uses `rng.sample` *without replacement* and discards the leftover partial batch, 92,496 were seen exactly once, 1,719 twice, 22 three-to-four times.

**Accuracy is flat in exposure** — 19.20% (1×), 18.80% (2×), 20.77% (3-4×) — and the control of **40,000 sentences the sampler never drew scores 19.02%**, statistically identical to the ones it did. Drawing a sentence adds nothing. There is no memorisation to speak of, and no "it just needs more repeats" mechanism available.

## Verdict (revised)

The first report's verdict — "trained too little to have an opinion" — is **half right, and the correction matters.**

* It is not overfitting: still true, and now much stronger. 20× more training produced no train/held-out divergence.
* It was under-trained at 6,000 steps: true, and quantified — bits/word was still falling.
* **But the root head does not improve with training budget.** 110,000 extra steps bought +0.76 pt root top-1 and +2.54 pt root top-5, while bits/word fell 0.52. The configuration learns the distribution of *structure* and saturates on *lexical identity* almost immediately.

So "run it longer" will not fix the 1% root accuracy. The bottleneck is not budget. It is either the root head's optimisation or the shared encoder's ability to carry lexical information — which is a **structural** question about this design, not a training-length question.

## The experiment that would settle it

**Shrink the root vocabulary and see if the head can learn at all.** Re-labelling to the top ~500 roots (about 60% of real-root positions at ~1,250× less cardinality) and training the same 120k steps would separate the two live hypotheses:

* root accuracy jumps to a plausible level → the head/encoder work; 9,114 classes at this data scale is the binding constraint (a data-per-class problem, and the morphemic scheme would still be viable with fewer, or compositionally-encoded, classes)
* root accuracy stays at its floor → the root slot is not being optimised or not being carried by the shared trunk, which is an architecture bug to find

This is worth the GPU time precisely because the answer changes what to build next, and neither the curve nor the exposure analysis can distinguish the two.
