# scratch_lm.py — from-scratch NRMT-P language model, 6000 steps

Pod `6r0naag3vvrfpr` (213.173.108.47:22515), GPU RTX PRO 4500 Blackwell 32,623 MiB.
Python `/workspace/venvs/rootformer/bin/python` (torch 2.8.0+cu128).
Script run **unmodified**: `/workspace/hf_v19_2_release/scratch_lm.py`
(md5 `71db165f8f9d09f8725402834b793a3f`, identical to the copies in `lisan_root_fix/before_release/` and `taxpin/`).

Command: `python scratch_lm.py train --steps 6000 --batch 16 --lr 6e-4`

## 0. Prior artifacts found, and what was reused

| artifact | mtime | content | used? |
|---|---|---|---|
| `/workspace/sf_data/streams.pt` (300 MB) + `meta.json` | Oct 1 07:19 | **prepared dataset, 25,000,003 train words** | **REUSED** (no re-prepare) |
| `/workspace/sf_morph.json` / `.pt` | Oct 1 07:58 | prior **80,000-step** result + weights | reused as reference; preserved (`prior_80k_sf_morph.*`) |
| `/workspace/sf_morph_80k.json` | Oct 1 08:20 | byte-identical duplicate of `sf_morph.json` | redundant copy |
| `/workspace/sf_data_gov`, `sf_governed.*` | Oct 1 10:07–10:59 | different "governed LM" variant | not used |

No prior run log existed. The prepared dataset was **not** rebuilt.

> **Caveat (important):** the on-disk dataset was prepared at 07:19 against the **previous**
> vocabulary revision — `n_roots=9114, n_awzan=130` — while the current release blueprint
> (modified 18:07, after the awzan + lisan root fixes) yields `roots=9490, awzan=142`.
> So these numbers describe the 9114/130 revision. Re-prepare cost is modest
> (`encode_sentence` ≈ 47,600 words/s → ~9 min for 25 M words, plus file I/O); it was not
> done because reuse was instructed.

## 1. Corpus and vocabulary

Sources (all present): `scholastic_sanitized` (22 files), `andalusian_canon_sanitized` (12),
`heritage_foundations` (20), `rootformer_v12/v18_next_root_morph/data` (54),
`rootformer_v12/raw_translations` (80), `rootformer_v12/v17_deepseek_flash/data` (62) — 250 files.
Split **by work (file)**, deduped by diacritic-stripped sentence, held-out works test=16 val=9.

| split | sentences | words |
|---|---|---|
| train | 2,542,630 | **25,000,003** (capped at 25 M) |
| val | 2,000 | 19,336 |
| test_gen | 4,000 | 39,757 |
| test_deriv | 4,000 | 45,688 |

Vocabulary: **roots 9,114** (ids 0–70 are special: `<UNK>`, `<PARTICLE>`, `<P:…>`; 8,115 seen in train),
**wazn 130** (39 seen), **prefixes 26** (24 seen), **suffixes 22** (21 seen).
Model 32,723,712 params (~32.7 M), d=512, 8 layers, 8 heads.

## 2. Root-holdout construction and leak check

Construction (from the code, confirmed in `meta.json`): holdout roots are chosen **from the training
files only**, by *sentence containment* over a 120 k-sentence probe, **least-containing first**, stopping
once ~10 % of probed sentences are covered (cap 600). Non-special names only, exactly 3 consonants,
containment ≥ 20. Result: **38 content roots** held out. Then every train/val sentence whose root set
intersects the holdout is dropped; test works are partitioned into `test_gen` (no held-out root) and
`test_deriv` (contains them).

Independent leak check (`audit_data.py`, full pass, no sampling):

| split | sentences | sentences containing a held-out root | held-out root occurrences | distinct held-out roots |
|---|---|---|---|---|
| train | 2,542,630 | **0** | **0 / 25,000,003** | 0 |
| val | 2,000 | **0** | 0 / 19,336 | 0 |
| test_gen | 4,000 | **0** | 0 / 39,757 | 0 |
| test_deriv | 4,000 | **4,000** | 4,602 / 45,688 | **38 / 38** |

`held-out ∩ special_ids = ∅`. **The holdout is leak-free** and `test_deriv` is a genuine unseen-root set.

## 3. Baseline sanity check

* The unigram reference is built from **train** counts only and applied to the **same held-out splits**
  the model is scored on. Independently recomputed bits/word matched the script **to 6 decimals**:
  `val 13.825222  test_gen 13.615963  test_deriv 15.599057`; root acc matched identically.
  (The unigram pass skips the last sentence of each split — 1 sentence, 11 positions on test_gen — so
  `n=35746` vs the model's `35757`. Negligible.)
* **Caveat found in the reference itself:** the script's unigram *root-accuracy* predictor uses the
  globally most frequent root and global top-5, which are **all special tokens** (top-1 = id 4;
  top-5 = ids 3,4,5,6,20). Its `real` column is therefore **0.000 by construction** — it can never emit
  a real root. "Model 1.49 % vs unigram 0.00 %" on real roots is not a meaningful ratio. The fair
  context-free reference for real roots is the most frequent **real** root (id 283):
  **test_gen 2.076 % top-1 / 7.651 % top-5; test_deriv 1.555 % / 6.244 %** (computed here).

## 4. Results — 6000 steps vs the script's own references

**Bits per word (unweighted; lower is better)** — model **beats** unigram on all three splits:

| split | model | unigram | delta | relative |
|---|---|---|---|---|
| val | **12.9304** | 13.8252 | −0.8949 | −6.5 % |
| test_gen | **12.9855** | 13.6160 | −0.6305 | −4.6 % |
| test_deriv | **14.7404** | 15.5991 | −0.8587 | −5.5 % |

**Next-root accuracy (top-1 / top-5)**

| split | subset | model top-1 | model top-5 | unigram top-1/top-5 | majority-real-root top-1/top-5 |
|---|---|---|---|---|---|
| test_gen | all | 0.2009 | 0.3459 | 0.1875 / 0.3000 | — |
| test_gen | **real** | **0.0149** | 0.0523 | **0.0000 / 0.0000** | **0.0208 / 0.0765** |
| test_gen | special | 0.4147 | 0.6835 | 0.4033 / 0.6452 | — |
| test_deriv | all | 0.1745 | 0.3080 | 0.1641 / 0.2698 | — |
| test_deriv | **real** | **0.0121** | 0.0407 | **0.0000 / 0.0000** | **0.0155 / 0.0624** |
| test_deriv | special | 0.3920 | 0.6661 | 0.3839 / 0.6312 | — |

**Against the fair (majority real root) baseline the 6000-step model is BELOW it on real-root top-1.**
The "all" column is carried by specials (~44 % of positions, near-deterministic).

## 5. Derivational-family holdout — the thesis answer

Scored on exactly the held-out roots (my `eval_holdout_roots.py`; it sub-splits the script's `real`
column, and the split reconciles exactly with the script's own numbers):

| checkpoint | split | held-out-root top-1 | held-out-root top-5 | n positions |
|---|---|---|---|---|
| **6000 steps** | test_deriv | **0.000000** | **0.000000** | **4,117** |
| 80,000 steps | test_deriv | **0.000000** | **0.000000** | **4,117** |

`test_gen` contains 0 held-out-root positions (as designed), so the comparison base is clean.
**The model predicts an unseen root zero times, at both 6 k and 80 k steps — not once even in the top-5
of 9,114 classes.**

**Why — proved, not assumed.** Held-out roots occur **0 times** in the training stream, so:

* their **input** embedding rows are never looked up. Measured against a `torch.manual_seed(0)`
  replay of the init: `e_r` rows for held-out roots differ by exactly the global weight-decay shrink
  (mean |Δ| 1.429e-2, 1.79 %), **identical to the 961 other never-seen roots** (1.429e-2) — i.e. zero
  gradient, decay only;
* their **output** rows are never a target, so cross-entropy only ever pushes them *down* as negative
  classes (softmax mass on an untargeted class is > 0). Those rows grew from |w| mean 0.0220 to
  **0.1087 (5×)**, tracking the other never-seen roots (0.1082).

A per-root softmax can therefore never emit an unseen root type. **The holdout, as built, cannot test
the morphemic generalisation claim** — it returns a structurally-forced 0 %. Testing it would need a
compositional output (predict the root's slots) rather than a 9,114-way root classifier.

## 6. Loss curve, wall-clock, VRAM, NaN/Inf

The script prints only **one batch's** loss every 500 steps, and its val evaluation only fires every
10,000 steps (so at `--steps 6000` it never fires mid-run). The 12 printed points:

```
step   500   1000   1500   2000   2500   3000   3500   4000   4500   5000   5500   6000
loss 7.1097 6.7013 6.9867 7.2014 7.3476 6.4390 7.1766 6.4722 6.7040 6.9144 6.5956 6.6691
```

Single-batch noise (σ ≈ 0.29) swamps the trend: first-half mean 6.964 → second-half mean 6.755
(≈ −0.21). This is a *weighted* sum (root + ½·wazn + ¼·prefix + ¼·suffix) and is not comparable to the
unweighted bits/word above. The end-of-run eval is the meaningful learning signal.

* **Wall-clock:** launch `2026-10-01T22:12:45Z` → last write `22:15:22Z` = **157 s end-to-end**;
  6000-step training loop **119 s** (avg 50.37 steps/s, 6000/50.37).
* **Peak VRAM:** **≈ 1.77 GB** (1702/1764/1774 MiB observed by `nvidia-smi` polling during the loop;
  eval-only process peak 606 MiB at 1 Hz). Card has 32,623 MiB → **~5.4 % of the GPU**.
* **No NaN/Inf** anywhere (scanned log + result JSON); process exit 0. Only stderr output is a benign
  `UserWarning` at `scratch_lm.py:565` about `float(loss)` on a `requires_grad` tensor.

## 7. Prior 80,000-step run (same data, for reference)

bits/word val 12.316, test_gen 12.533, test_deriv 14.296 (unigram 13.825/13.616/15.599).
Real-root top-1: test_gen 0.0222, test_deriv 0.0169 — i.e. ~1.07× the fair majority-real-root baseline
(test_gen 2.08 % / test_deriv 1.55 %), versus the frozen-Qwen line's 5.67 % vs 3.551 % ≈ 1.60×.

## 8. Verdict

1. **Does it beat context-free unigram on bits/word?** Yes, but modestly: −0.63 to −0.89 bits/word
   (4.6–6.5 %). It learned *some* structure.
2. **Does it beat unigram on next-root accuracy?** Only against the script's degenerate real-root
   reference (0.000). Against a fair majority-real-root baseline the 6000-step model is **below** it
   (1.49 % vs 2.08 %; 1.21 % vs 1.55 %). At 80 k steps it is marginally above (~1.07–1.32×).
3. **Does it generalise to unseen roots?** **No — 0 / 4,117 top-1 and top-5, at both 6 k and 80 k steps**,
   and the design makes that outcome structurally unavoidable (unseen root types are only ever trained
   as negative classes). The derivational-family holdout does not support the thesis; it needs a
   compositional root decoder to be a real test.

Artifacts (pod `/workspace/scratch_lm_run/`, mirrored locally): `train_6000.log`,
`sf_morph.json` (**new 6000-step result**), `sf_morph.pt` (new weights),
`audit_data.{py,log,json}` (leak + independent unigram), `heldout_root_eval_6000.json`,
`heldout_root_eval_80000.json`, `heldout_root_eval.log`, `vram_eval.csv`, `zerograd_check.py`,
`prior_80k_sf_morph.{json,pt}`, `prior_sf_data_meta.json`.

The pod's original `/workspace/sf_morph.json` and `sf_morph.pt` (80 k) were **restored byte-identical**
(md5 `92a7761153d80ae3c394f55942aabdf8`); no existing source or checkpoint was modified.
