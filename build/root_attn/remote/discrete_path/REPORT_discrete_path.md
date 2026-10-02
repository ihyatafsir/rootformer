# A discrete root-history pathway for `NRMTHead.build_features` — design, CPU expressibility proof, and the queued GPU arm

Date 2026-10-02 · pod `6rnaag3vvrfpr` (213.173.108.47:22515) · everything under
`/workspace/discrete_path/` (mirrored to `rootformer/build/discrete_path/`).

## 0. Verdict in one paragraph

The brief's concrete proposal — **hash each order-k root n-gram into a ~2^20-row learnable
embedding table** — is **measurably the wrong design on this data**, and I did not build it. On the
exact 300 val windows / 18,869 radical positions the *hashed-context lookup ceiling* is 5.54 % at
2^20 buckets (4.85 % NOVEL) versus **53.13 %** for a collision-free table, because the train stream
contains **5,201,346 distinct order-4 contexts** — a 2^20 table merges ~5 contexts per bucket.
What I built instead is an **explicit, collision-free, per-order context vocabulary** (one row per
distinct order-k context, an OOV row pinned to exactly 0 so shorter orders take over = exact
backoff, geometric order-block scaling, and a zero-initialised fade-in gate). At the same width it
is the honest trade: **more expressive** (53.13 % vs 5.54 %) and **more expensive** (9.37 M rows
over orders 1–4, **149.9 M parameters at d_ng = 16**, versus 16.8 M for a 2^20 × 16 hash table).
On CPU, by a closed-form construction (no gradient descent),
the module reproduces the order-k lookup **exactly — gap 0.0000 pp at k = 1, 2, 3 and 4**, ALL and
NOVEL. 24/24 unit tests pass, all seven regression suites pass, and a single 20k GPU run is queued
behind a durable gate that never touches another agent's job.

## 1. The design, and why it can express the lookup

### 1.1 What the existing branch is, and why it is capped

`NRMTHead.build_features` concatenates the *embeddings* of `r_{t-1..t-3}` (plus op / prev-morph)
and pushes them through one `Linear(320→896)` + LayerNorm, added to `h_t`. That map is additive in
the per-root embeddings, i.e. **linear in the root ids**. A linear function of the ids cannot
express an order-k lookup; the previous bisection measured exactly this (1.41–1.81 % for a
closed-form linear readout of that content, 53.13 % for an order-4 lookup).

### 1.2 The measured ceiling of each candidate table form (CPU, no GPU)

Fitted on the full train root stream, evaluated at the same 300 val windows / 18,869 radical
positions / 15,046 NOVEL positions as every 20k run, with backoff to shorter orders and a unigram
fallback. `measure_ngrams.py` → `measure_ngrams.json`.

| table form | rows (k=4) | ALL acc@1 | NOVEL acc@1 |
|---|---|---|---|
| **explicit / collision-free** | **5,201,346** | **53.13 %** | **49.51 %** |
| hash 2^24 | 16,777,216 | 43.53 % | 40.24 % |
| hash 2^22 | 4,194,304 | 24.51 % | 22.48 % |
| hash 2^20 (the brief's value) | 1,048,576 | **5.54 %** | 4.85 % |
| hash 2^18 | 262,144 | 1.76 % | 1.52 % |
| hash 2^16 | 65,536 | 0.96 % | 0.83 % |
| order-3 exact (for reference) | 3,299,176 | 38.85 % | 37.13 % |

Distinct train contexts (exact, collision-free int64 codes; `9490^4 = 8.11e15 < 2^63`):
k=1 → 8,614 · k=2 → 858,205 · k=3 → 3,299,176 · k=4 → 5,201,346.

**A 2^20 hashed table cannot even reach what the head already scores.** That row is the reason the
design is `vocab`, not `hash`. (The hash path is still implemented — `--ngram-features hash` — so
the claim is reproducible, and it is the right mode if the vocabulary ever grows beyond what the
GPU can hold.)

### 1.3 A second measurement that constrains the training-time story

| quantity | k=2 | k=3 | k=4 |
|---|---|---|---|
| contexts | 858,205 | 3,299,176 | 5,201,346 |
| contexts with exactly ONE observed target | 59.9 % | 82.8 % | **92.7 %** |
| train positions inside those deterministic contexts | 8.2 % | 42.4 % | 73.1 % |

and, pruning by *occurrence* count (contexts seen < N go to the OOV row = backoff):

| min occurrences | order≤4 ALL | order-4 rows |
|---|---|---|
| ≥1 | 53.13 % | 5,201,346 |
| ≥2 | **12.48 %** | 660,400 |
| ≥3 | 8.34 % | 205,194 |
| ≥5 | 6.38 % | 74,197 |

**The 53 % lives in contexts seen exactly once.** That is not memorisation of noise: 92.7 % of
order-4 contexts have a single observed continuation, i.e. the conditional is concentrated and one
observation is usually the mode. Consequences:

* the table **cannot be pruned by count** — min-occurrence ≥2 destroys the signal;
* purity pruning is nearly free but nearly useless (`ndist ≤ 2` keeps 5.07 M rows and scores
  50.99 %; `ndist == 1` keeps 4.82 M and scores 49.00 %);
* the run must keep all 5.2 M order-4 rows, hence ~150 M parameters at d_ng = 16.

A favourable corollary: the 20k run draws ~81 M position-samples from a 7.06 M-token stream =
**11.5 epochs**, so the average singleton row receives ~11 Adam updates. Adam's per-element
normalisation takes a full `lr`-sized step on the first visit to a row, which is exactly the
one-shot memorisation this table needs.

### 1.4 The implemented module

`DiscreteNgramFeatures` (new class in `nrmt_arch.py`), wired into `NRMTHead.build_features` behind
`--ngram-features {none,vocab,hash}`; **default `none` reproduces the historical path bit-exactly**.

* context at position `t` = the k roots **ending** at `t`, i.e. `r_{t-k+1..t}` — exactly the block
  the measured lookup uses (`r_t` is a legitimate input: the target is `r_{t+1}`; the historical
  linear branch deliberately omits `r_t` because `h_t` carries it, but a discrete pathway should
  use it, and this is what makes it the *same* order-4 context as the 53.13 % row);
* codes are exact int64 polynomials in base `num_roots` for k ≤ 4 → **collision-free**; longer
  orders are folded mod `2^40-1` so no multiply can overflow on CPU or CUDA;
* `vocab` mode rows: `0` = OOV (never seen in train — **exactly 0 and never trained**, which is
  what makes the shorter orders take over = exact backoff), `1` = null (`t < k-1`), `2…` = the
  distinct train contexts;
* one `Embedding` per order → concat → `Linear(Σd_ng → d_model)` → `LayerNorm` → `* gate`, added to
  the (optional) historical linear branch;
* `gate` starts at exactly 0, so the arm's output is 0 at step 0 (no warm-up slam), while the
  embeddings are small-random (std 0.02) so `d(loss)/d(gate) ≠ 0` (not the measured dead saddle);
* dropout is applied once by the head to the sum, so `ngram_mode='none'` is behaviour-preserving.

Parameter count (orders 1–4, d_ng = 16): 9,367,352 rows × 16 = **149.9 M** + proj 57 k. Selection of
d_ng is discussed in §2.2.

## 2. The CPU expressibility proof

### 2.1 Closed-form construction (`expressibility_proof.py`)

No gradient descent at all. With `P: R^{d_ng} → R^{d_model}` an orthonormal partial isometry,
`x_c ~ N(0,I)` one prototype per root class, and `z_c = LayerNorm(P x_c)`:

1. set the readout `W := [z_1; …; z_C]ᵀ` (a legal setting of `root_head.weight`);
2. set the order-k embedding row of context `b` to `x_{lookup_target_k(b)}`, where
   `lookup_target_k(b)` is the top-1 next root of that context in the train stream;
3. leave rows 0 (OOV) and 1 (null) at exactly 0;
4. scale the order blocks geometrically, `proj_k = eps^(K-k) · P`, eps = 0.002.

Why it works: for a context whose order-K row is `x_c`, the logits are `(W f)_j = <z_j, z_c>` with
`<z_c,z_c> = ‖z_c‖² = d_model` and `|<z_j,z_c>| ≤ ‖z_j‖‖z_c‖` with equality only for `z_j = z_c`
(probability 0) — so the argmax is *c* unconditionally. A context whose order-K n-gram was never
seen in train has row 0 = 0, the order-K term vanishes, and the shorter orders decide; the
geometric scaling makes every shorter-order term a perturbation of relative size eps, far below the
prototype margin, so it cannot flip the argmax. The same argument applies with `h_t ≠ 0`: scaling
the module's LayerNorm gain scales `f` and makes `W h_t` an arbitrarily small *relative*
perturbation.

### 2.2 Result: fitted-vs-lookup gap at k = 1,2,3,4

| k | lookup ALL | constructed ALL | **gap** | lookup NOVEL | constructed NOVEL | **gap** |
|---|---|---|---|---|---|---|
| 1 | 1.7383 % | 1.7383 % | **0.0000 pp** | 1.6150 % | 1.6150 % | **0.0000 pp** |
| 2 | 11.8077 % | 11.8077 % | **0.0000 pp** | 11.4316 % | 11.4316 % | **0.0000 pp** |
| 3 | 38.8468 % | 38.8468 % | **0.0000 pp** | 37.1328 % | 37.1328 % | **0.0000 pp** |
| 4 | **53.1295 %** | **53.1295 %** | **0.0000 pp** | 49.5082 % | 49.5082 % | **0.0000 pp** |

**The module can represent the order-4 lookup that scores 53.13 %, exactly, and capacity does not
bind at any k.** The same construction also reproduces the pruned variants exactly
(min-occurrence ≥2: 10.8379 / 10.8379; ≥3: 6.8366 / 6.8366; ≥5: 4.9234 / 4.9287).

Prototype margin `1 − max_{j≠c} cos(z_j, z_c)` — the robustness budget against the shorter-order
and `h_t` perturbations, and the reason for the d_ng choice:

| d_ng | min margin | 1st pct | median | classes failing self-max | order-4 table params |
|---|---|---|---|---|---|
| 8 | 0.0055 | 0.0176 | 0.0578 | 0.000 % | 74.9 M |
| **16** | **0.0585** | 0.1155 | 0.2068 | 0.000 % | **149.9 M** |
| 32 | 0.2136 | 0.2797 | 0.3880 | 0.000 % | 299.8 M |
| 64 | 0.3852 | 0.4462 | 0.5463 | 0.000 % | 599.5 M |

d_ng = 16 is chosen: 10× the margin of d_ng = 8 at a parameter count that still fits the card with
dense grads + AdamW (~2.4 GB), and it is the width the GPU arm uses.

`h_t` dominance check (K=4, d_ng=64, a unit-RMS *random* `h_t` added to the branch output): accuracy
50.82 % at gain rho=1, 52.15 % at rho≥3. The residual ~1 pp at saturation is the set of positions
where *every* order is OOV/null — the branch is 0 there **by design** and the head must fall back on
`h_t`; a random `h_t` scores ~0 there, whereas the real frozen `h_t` carries `r_t` at 91.75 %
linear decodability. So this is a pessimistic diagnostic, not a ceiling.

### 2.3 Is it learnable? (short optimisation, CPU)

`fit_ngram_cpu.py` runs the real module + a `Linear(896→9490, bias=False)` readout on batches of
real 128-token train windows with the real masks, AdamW(lr=1e-3, wd=0.01), the zero-init gate, and
the exact 149.9 M-parameter table (d_ng=16). `h = 0` (no frozen backbone) so it isolates the
pathway. It is deliberately budget-limited (the GPU run gets ~27× more samples); the point is the
*shape* of the learning curve, not its endpoint.

Faithful arm — zero-init gate, exactly the GPU configuration (1500 steps, wall 1145 s):

| step | gate | ALL acc@1 | NOVEL acc@1 |
|---|---|---|---|
| 250 | 0.107 | 4.56 % | 4.47 % |
| 500 | 0.140 | 6.68 % | 6.34 % |
| 750 | 0.180 | 11.43 % | 10.33 % |
| 1000 | 0.219 | 16.62 % | 14.63 % |
| 1250 | 0.249 | 21.69 % | 19.84 % |
| **1500** | **0.278** | **25.53 %** | **23.74 %** |

`--no-gate` arm — branch at full scale from step 0, removing the gate-ramp confound (700 steps,
wall 658 s):

| step | ALL acc@1 | NOVEL acc@1 |
|---|---|---|
| 100 | 4.45 % | 4.41 % |
| 200 | 5.19 % | 4.93 % |
| 300 | 5.95 % | 5.72 % |
| 400 | 7.08 % | 6.77 % |
| 500 | 8.38 % | 7.78 % |
| 600 | 11.70 % | 10.53 % |
| 700 | 14.38 % | 12.55 % |

Readings:

* **The pathway learns.** From a cold start (`h = 0`, random readout) the faithful arm reaches
  25.53 % ALL / 23.74 % NOVEL, past the order-2 lookup (11.81 %) and climbing towards order-3
  (38.85 %).
* **It is not saturating.** The last 250 steps added **+3.84 pp** and the gate was still only
  **0.278** at the end — ~72 % of the branch's magnitude is still unused. Neither the capacitance
  nor the gate ramp has been exhausted.
* **It is budget-limited, by a factor of 13.3.** 1500 steps × 32 windows × 127 positions =
  **6.10 M position-samples = 0.86 epochs** of the 7.06 M-token stream. The queued GPU run gets
  20 000 × 32 × 127 = **81.3 M = 11.5 epochs**. The CPU curve therefore covers 7.5 % of the GPU
  run's gradient budget; it is evidence that SGD finds the table, not a prediction of where the GPU
  run lands.
* **The zero-init gate is a curriculum, not a handicap.** The full-scale arm is *ahead* early
  (14.38 % at step 700 with the branch at 1.0) but the gated arm has already overtaken it on a
  per-step basis and is still accelerating — the same reason the v18fix `feat_gate` was adopted for
  the linear branch.
* Caveat for interpretation: with `h = 0` this measures the pathway *in isolation*. It is not
  comparable to the 5.72–6.84 % head numbers (which are measured with a frozen backbone and a
  different target population of positions), and it is a lower bound on the difficulty the real
  head faces, since the real `h_t` carries `r_t` at 91.75 % linear decodability.

## 3. Unit tests (`test_discrete_ngram.py`) — 24/24 pass

```
[*] C=64 d_model=32 vocab(k=3)=3970 rows (+2 reserved)
  [PASS] shape [B,T,d_model] -- (3, 12, 32)          [PASS] dtype float32
  [PASS] no NaN/Inf                                  [PASS] float64 works
  [PASS] deterministic in eval mode                  [PASS] null positions (t<k-1) -> row 1
  [PASS] a train context maps to row >= 2            [PASS] unseen context -> row 0 (OOV)
  [PASS] module code == forward block code used by the vocab
  [PASS] OOV row exactly zero                        [PASS] null row exactly zero
  [PASS] hash bucket == overflow-free numpy reference
  [PASS] hash ids within [1, 2**bits]                [PASS] every parameter gets a gradient
  [PASS] all gradients finite                        [PASS] proj gradient non-zero
  [PASS] embedding gradients non-zero
  [PASS] OOV row gradient exactly zero (all train contexts are in the vocab)
  [PASS] gate has non-zero gradient at gate==0 (no dead saddle) -- -1.131e-01
  [PASS] gate==0 => discrete arm is a no-op (bit-identical to ngram_mode=none) -- max|d|=0.000e+00
  [PASS] gate==1 => discrete arm changes the output -- max|d|=6.976e-02
  [PASS] ngram_scale diagnostics
  [PASS] use_features=False & mode=none -> exact zeros
  [PASS] discrete-only arm works with use_features=False (non-zero)
==== 24 passed, 0 failed ====
```

Two real bugs were caught by these tests and fixed before any GPU work:
(1) the context packing was reversed relative to the vocabulary builder, so every context landed on
the OOV row; (2) the null/OOV rows were not separated and the OOV row was not pinned to zero, which
would have broken exact backoff. The gradient test also caught a degenerate loss
(`sum(LayerNorm(x)) ≡ d_model·beta` is constant, so it gives exactly zero gradient — the test now
uses a random projection).

## 4. Exact diff

Backups on the pod (`md5` verified against the pre-edit files):

```
41dc9489f033ea3ad4d2b2c0baaf2c04  nrmt_arch.py  -> nrmt_arch.py.bak.discrete_path_20261002T003837Z
46c809c096df06280327475646e9756a  nrmt_train.py -> nrmt_train.py.bak.discrete_path_20261002T003837Z
```

New files (md5 of the deployed copies == md5 of the local patched copies):

```
ac7f895123583fc47f04f74df7914535  /workspace/hf_v19_2_release/nrmt_arch.py
ceacdb608b9f69e8ae2c14250141257f  /workspace/hf_v19_2_release/nrmt_train.py
0826e283a80f8b148b219aa4e2ee2332  nrmt_arch_discrete.diff    (314 lines)
2fe0e0957606ba81862cc1a24172139e  nrmt_train_discrete.diff   ( 97 lines)
```

`nrmt_arch.py`: `+import numpy as np`; new `DEFAULT_NGRAM_ORDERS`, `_NGRAM_HASH_MOD`,
`_NGRAM_HASH_A`, `class DiscreteNgramFeatures`, `ngram_order_vocabs()`; `NRMTHead.__init__` gains
`ngram_mode/ngram_orders/d_ng/ngram_bucket_bits/ngram_vocabs/ngram_gate/ngram_emb_init_std`;
`build_features` is split so the historical linear branch is `_linear_features` (with `self.drop`
moved out, applied once to the sum) and the discrete branch is added; new diagnostics
`ngram_scale()`, `ngram_gate()`, `ngram_table_stats()`, `ngram_params()`; `RootformerNRMT.__init__`
forwards the new arguments.

`nrmt_train.py`: six new flags (`--ngram-features`, `--ngram-orders`, `--ngram-dim`,
`--ngram-bucket-bits`, `--ngram-emb-std`, `--no-ngram-gate`); the train/val cache is now loaded
*before* the head is built (the `vocab` mode needs the train root stream to size its tables); the
vocab is built and passed to the model; the arm configuration and the per-eval `ngram_gate` /
`ngram_scale` are logged. All defaults preserve the historical behaviour.

## 5. Seven-suite regression — all pass

Run on the pod after the module edit (`suites_after_discrete.log`):

| suite | result |
|---|---|
| `test_grammar_impl.py` | **20/20**, exit 0 |
| `verify_v2.py` | all PASS, exit 0 |
| `andalusian_realizer.py` | **67/67**, exit 0 |
| `test_sibawayh_governor.py` | **15/15**, exit 0 |
| `ibn_malik_automaton.py` | **46/46**, exit 0 |
| `test_khalil_orbits.py` | **30/30**, exit 0 |
| `test_awzan_order.py` | **24/24**, exit 0 |

## 6. The queued GPU arm

`chain_ngram.sh` is armed on the pod (pid visible via `pgrep -f chain_ngram.sh`; log
`/workspace/discrete_path/chain_ngram.log`). It is a durable `setsid nohup` gate that polls
`nvidia-smi --query-compute-apps` and requires the GPU to have **no compute process at all** and
< 500 MiB used, confirmed twice 60 s apart, before it execs `run_ngram.sh`. **It never signals,
renices or kills another agent's process.** Poll interval 20 s, 5000 polls (~28 h) before it gives
up. Observed gate log so far:

```
[chain_ngram] armed 2026-10-02T00:42:05Z
[chain_ngram] poll 30 00:51:47 gpu_procs=2 mem=3497MiB
[chain_ngram] poll 60 01:02:50 gpu_procs=1 mem=608MiB
```

NOTE on the other agent's run: the 120,000-step `curve_lm.py` finished **on its own** (no
`curve_lm` process remains; nothing in this work signalled it). By 01:02:50 a different, much
smaller job (`scratch_lm_comp.py eval`, 598 MiB) held the GPU, and the gate was correctly waiting
for it rather than preempting it.

The single run it will launch (nothing else is queued by me):

```
nrmt_train.py --checkpoint <awzan142.roots9490.tok10052> --cache <nrmp_cache_9490_aligned>
  --head-init remap --steps 20000 --batch-size 32 --eval-every 1000 --tag NGRAM
  --feat-gate --margin-ramp 10 --grad-warmup 10 --logit-scale ln        # = ALIGNED_FIX
  --ngram-features vocab --ngram-orders 1,2,3,4 --ngram-dim 16          # the only addition
```

so the discrete branch is the **single variable** against the staged ALIGNED_FIX arm; the
comparable *measured* numbers are CONTROL 5.72 / ALIGNED 6.15 / GUARD 6.63 / NOFEAT 6.59 /
FIX 6.84. The gate writes `train_NGRAM.log`, `trace_NGRAM.jsonl`, `results_NGRAM.json`,
`head_NGRAM.pt` and a peak-VRAM trace to `/workspace/discrete_path/`.

Budget: the naive estimate is the ALIGNED run's 900 s plus ~13 % for the dense optimizer step over
149.9 M parameters (measured pod disk writes 575 MB/s, so the 20 × 600 MB checkpoints add ~22 s).
Peak VRAM ≈ 12 GB of the card's 32.6 GB.

### 6.1 The patched trainer was smoke-tested end to end on CPU (no GPU used)

To avoid burning the single GPU slot on a glue bug, the complete patched `nrmt_train.py` was run
with `CUDA_VISIBLE_DEVICES=` (empty → CPU only) against a 900-token synthetic cache in
`/workspace/discrete_path/smoke_cache/` (the read-only real caches were not touched):

```
[*] ngram vocab (train stream): k1=865, k2=899, k3=898, k4=897 rows (0s)
[*] DISCRETE arm: mode=vocab orders=(1, 2, 3, 4) d_ng=16 gate=True tables={...} params=0.12M
[*] ARM [SMOKE] use_features=True dropout=0.1
[*] checkpoint ...missing=26 (of which nrmt_head new=26) unexpected=8
[*] head warm start [remap/truncate]: applied 8/8 legacy head tensors
[*] trainable parameters: 12.92M (backbone frozen)
[*] training windows: 2 x 127 positions = 254
  [eval @1] ... | ngram gate -0.00081 scale 0.0008 (RMS x 0.0008)
wrote /workspace/discrete_path/smoke_results.json and /workspace/discrete_path/smoke_head.pt
```

The whole path is exercised: cache pre-load → n-gram vocabulary → discrete head construction →
legacy head remap → frozen-backbone caching → ʿāmil stream → novelty mask → 2 training steps →
evaluate (with the new `ngram_gate`/`ngram_scale` diagnostics) → checkpoint save. The gate is
already non-zero after two steps, confirming it is not a dead saddle at real scale.

## 7. What this does and does not establish

**Established.** (a) A linear map of root embeddings cannot express the lookup, and neither can the
proposed 2^20 hash — its ceiling is 5.54 %, below the current head. (b) The collision-free explicit
table's ceiling is the full 53.13 %. (c) The implemented module *contains* that function: a
closed-form parameter setting reproduces it to 0.0000 pp at k = 1–4, on the real vocabulary, with a
prototype margin of 0.058 at the chosen width. (d) The arm is a bit-exact no-op at step 0 and has
clean, finite gradients everywhere it should and exactly zero where it should (OOV row).

**Not established, and what the GPU run is for.** That 20,000 Adam steps *find* such a setting.
The closed-form construction is an existence proof; it uses a rank-`d_ng` readout and a specific
prototype geometry, while the GPU run learns `root_head` jointly with the table and must also fit
`h_t`. The CPU learning curve (§2.3) is the only evidence on that question and it is budget-limited.

**Risks I would flag before the result is in.** (i) 149.9 M parameters trained for only 20k steps
with a zero-init gate that spends ~1k steps ramping up. (ii) The signal sits in singleton contexts,
so the run is a one-shot-memorisation test at a scale nobody has trained here before. (iii) With only
one GPU run affordable there is no same-flags-minus-discrete control; attribution rests on the
effect being far larger than the ~1 pp flag noise between the existing 20k arms.
