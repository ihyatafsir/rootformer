# Per-step loss / perplexity trace — rootformer v19_2 NRMT head

Setup: `/workspace/hf_v19_2_release`, `nrmt_train.py` **md5 41341576c1c3f0a52c0398832cb6ddce (UNCHANGED — zero edits)**,
checkpoint `rootformer_v19_2_synthesis_ar_backbone.awzan142.roots9490.tok10052.safetensors`
(md5 `3335a3d39091535d0fdd047dca842715` ✓), cache `/workspace/nrmp_cache_9490`,
`--head-init remap --batch-size 32`, GPU RTX PRO 4500 Blackwell 32 GB.

## No logging edit was required

`nrmt_train.py` already has a `--probe-out` flag (line 202) that writes a **per-step** JSONL:
`step, loss, grad_norm, p_ss, lr, root, wazn, prefix, suffix[, impossible][, orbit]`
(lines 563–568, flushed every 25 steps, closed at end). Every step is captured, so the requested
logging edit was **not** made: `nrmt_train.py` md5 is byte-identical before and after both runs.
Only a new external runner (`run_trace.sh`) and an analysis script were added.

## Runs

| run | command tail | steps | eval-every | rc | wall | peak VRAM |
|---|---|---|---|---|---|---|
| 1 | `--steps 5000 --batch-size 32 --eval-every 500 --tag ppl_trace` | 5000 | 500 | 0 | 278.45 s | 3001 MiB (2.93 GiB) |
| 2 | `--steps 20000 --batch-size 32 --eval-every 1000 --tag ppl_trace` | 20000 | 1000 | 0 | 583.66 s | 2742 MiB (2.68 GiB) |

Effective throughput ≈ 18 it/s (run 1, incl. ~130 s startup) and ≈ 34 it/s (run 2).

## Traces (deliverables)

- `ppl_trace_5000.jsonl` / `ppl_trace_5000.tsv` — 5000 rows, contiguous steps 1..5000
- `ppl_trace_20000.jsonl` / `ppl_trace_20000.tsv` — 20000 rows, contiguous steps 1..20000

TSV columns: `step, loss, ppl(=exp(loss)), root, wazn, prefix, suffix, impossible, grad_norm, lr, p_ss`.
(`impossible` is blank on steps whose batch drew no forbidden-root targets — not missing data.)

## NaN / Inf audit

**Zero NaN and zero Inf** in any field of either trace (5000 + 20000 steps).
JSONL grep for `nan|inf` → 0 hits. Losses stayed finite; max loss over both runs = **90.53**.

## Step 2–4 anomaly — RECURS (and is worse than the prior 65.78 / 9977)

| step | run1 loss | run1 grad_norm | run2 loss | run2 grad_norm | impossible |
|---|---|---|---|---|---|
| 1 | 11.659 | 1407.0 | 11.698 | 1392.1 | 0.027 / 0.030 |
| 2 | **86.973** | **12410.7** | **90.533** | **12137.1** | 267.3 / 299.1 |
| 3 | 75.763 | 8934.7 | 72.831 | 9540.6 | 283.5 / 236.0 |
| 4 | 37.116 | 5740.2 | 27.387 | 4051.5 | 115.0 / — |
| 5 | 13.027 | 737.1 | 18.969 | 2974.2 | — / 53.3 |

Reproduces every prior run: the warm/remapped logits start oversized and the impossibility hinge
`0.1 * relu(root_logits - margin)^2` (margin −2.0) is quadratic in that oversize, so the penalty
itself explodes (`impossible` 267–299 at step 2). Recovery is fast and monotone: grad_norm falls
below the 1.0 clip after step 1, loss is back under 20 by step 5 and under 12 by step 6.
grad_norm > 100 only on steps 1–11 (run 1: steps 1–11; run 2: 1–11). `impossible` settles to ~0.03
by step ~200 and stays there. This is worse than the reported 65.78 because the warm start here is
`--head-init remap` (better-calibrated warm logits ⇒ larger hinge activation), not scratch
(scratch peaked at 146). **No stop condition was triggered** (peak 90.53 < 100, no NaN/Inf).

## Train curve

Window means (run 2, same-seed trajectory shape as run 1):

| steps | 1–1000 | 4–5k | 8–9k | 12–13k | 16–17k | 19–20k |
|---|---|---|---|---|---|---|
| mean loss | 6.910 | 3.763 | 3.139 | 2.829 | 2.678 | 2.646 |
| mean PPL | 1002.6 | 43.07 | 23.08 | 16.92 | 14.55 | 14.10 |

- run 1: min loss **3.5120** @ step 4195 (PPL 33.52); final step loss 3.7904 (PPL 44.27).
- run 2: min loss **2.2216** @ step 18249 (PPL 9.222); final step loss 2.6083 (PPL 13.576).
- Train PPL **flattens at ≈ step 4500** in run 1 (last-500 mean 3.897, last-100 3.887).
  Over 20000 steps it keeps creeping down but the per-1000 decrement collapses from −0.20
  (5k) to −0.005/−0.01 (17k–20k), i.e. a second, much flatter plateau from ≈ step 17000.
  Mean train PPL ≈ 14.1 at step 20000.

## Val metrics at every eval point (run 1, eval-every 500)

| step | ALL acc@1 | NOVEL acc@1 | ALL PPL | NOVEL PPL | wazn CE / acc@1 | prefix acc@1 | suffix acc@1 | extra_norm |
|---|---|---|---|---|---|---|---|---|
| 500 | 5.42% | 5.28% | 1197.7 | 1225.9 | 1.849 / 49.55% | 73.93% | 83.43% | 6.76 |
| 1000 | 5.85% | 5.62% | 1881.0 | 1947.8 | 2.029 / 46.35% | 73.90% | 83.21% | 13.01 |
| 1500 | 5.58% | 5.38% | 2594.6 | 2703.8 | 2.241 / 44.12% | 73.90% | 83.00% | 16.60 |
| 2000 | 5.82% | 5.60% | 3265.4 | 3419.5 | 2.434 / 42.91% | 73.88% | 82.65% | 18.91 |
| 2500 | 5.62% | 5.42% | 3932.3 | 4138.0 | 2.659 / 42.04% | 73.87% | 82.30% | 20.33 |
| 3000 | 5.79% | 5.58% | 4391.7 | 4608.5 | 2.857 / 40.14% | 73.86% | 81.95% | 21.10 |
| 3500 | 5.94% | 5.66% | 4800.7 | 5063.8 | 3.052 / 39.85% | 73.86% | 81.70% | 21.44 |
| 4000 | 5.88% | 5.65% | 5025.9 | 5298.5 | 3.207 / 39.78% | 73.82% | 81.68% | 21.54 |
| 4500 | 5.95% | 5.75% | 5155.7 | 5433.4 | 3.280 / 39.65% | 73.84% | 81.60% | 21.56 |
| 5000 | 5.93% | 5.72% | 5167.1 | 5448.8 | 3.290 / 39.48% | 73.83% | 81.57% | 21.56 |

## Val metrics (run 2, eval-every 1000) — 5000 → 20000

| step | ALL acc@1 | NOVEL acc@1 | ALL PPL | NOVEL PPL | wazn CE / acc@1 | prefix acc@1 | suffix acc@1 | extra_norm |
|---|---|---|---|---|---|---|---|---|
| 5000 | 5.56% | 5.30% | 10529.5 | 11414.9 | 3.264 / 38.35% | 73.73% | 80.78% | 31.35 |
| 10000 | 5.72% | 5.40% | 35069.9 | 38586.2 | 4.392 / 37.99% | 73.48% | 79.76% | 39.61 |
| 15000 | 5.71% | 5.41% | 59649.8 | 66425.5 | 5.419 / 37.21% | 73.26% | 78.69% | 41.17 |
| 20000 | 5.67% | 5.32% | 65394.3 | 72995.9 | 5.785 / 36.85% | 73.08% | 78.18% | 41.18 |

(Full 20-point table: `ppl_trace_20000.results.json` → `history`.)

## PPL and accuracy DIVERGE — the prior 600-step observation reproduces, far more extremely

Yes. Root **acc@1 is scale-invariant and stays flat** (ALL 5.42–5.95%, no trend; NOVEL 5.28–5.75%),
while **val root CE/PPL rises without bound**: 1197.7 → 5167.1 by step 5000, and → **65394** by step
20000 (NOVEL 72996). The prior 600-step report (PPL 854→1138 while acc@1 rose) is confirmed and
extrapolated: a 55× PPL increase with flat accuracy.

The morph heads **degrade monotonically in val** (despite improving in train):
wazn CE 1.849 → 5.785 and wazn acc@1 49.55% → 36.85%; suffix acc@1 83.43% → 78.18%;
prefix acc@1 ~flat 73.9% → 73.1%.
Meanwhile train-window root CE falls to ~2.18 and train-window wazn CE to ~0.7 —
so the components "that actually move" are moving *opposite* ways in train and val.

**Mechanism evidence:** `extra_norm` (extra-conditioning feature norm) grows monotonically with the
val CE, then both saturate together at ≈ step 15000 (`extra_norm` 6.76→41.18; `ln(PPL)` == val root CE
7.57→11.09). corr(extra_norm, val root CE) = **+0.9757**; corr(step, val root CE) = +0.9295;
corr(extra_norm, val acc@1) = +0.52 (spurious). This is consistent with the extra-conditioning branch
inflating logit *magnitude*: argmax (hence acc@1/acc@5) is invariant to that scale, while CE — and
therefore PPL — grows with it. The model is not learning better, it is becoming confidently wrong;
the reported PPL is a miscalibration/scale artifact, not a capability signal. (Correlational: logits
were not saved; extra_norm is the best available proxy.)

## Integrity

`nrmt_train.py`, `nrmp_vocab.py`, the blueprint, both `.roots9490*.safetensors` checkpoints and all of
`/workspace/nrmp_cache_9490` are byte-identical / mtime-untouched (all mtimes predate the runs).
No module was edited, so the seven suites were not re-run. GPU returned to 2 MiB used.
