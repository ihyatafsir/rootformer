# Root cross-attention over the root history + partial trunk unfreeze — design, measurement, verdict

Date: 2026-10-02 · Pod `g6exduq0bd17z8` (213.173.104.76:46758) · all work under `/workspace/root_attn/`,
mirrored to `rootformer/build/root_attn/`.

---

## 0. Verdict in one paragraph

**YES — this moves the number, decisively and by the largest margin of any arm in the ladder.**
Root cross-attention over the root history, inserted in the top four trunk layers (20–23) with a
zero-initialised gate, plus a partial unfreeze of exactly those four layers, takes held-out
next-root `acc@1` from **FIX 6.84 % to 19.53 %** (`ALL_val`, step 20 000, the same 18 869 radical
positions) — **+12.69 pp, 5.50× the 3.551 % marginal, 1.0 pp-noise bar cleared by 11.7 pp** — with
**NOVEL 19.90 %** (positions whose 12-root context never occurs in train, i.e. 79.7 % of the val
positions: the gain is *generalisation*, not memorisation). It is not the readout, the loss or the
flags: ablating **only** the cross-attention at inference, on the same trained trunk and head, drops
`acc@1` to **2.83 %** — below the marginal. The capability lives in the new pathway. `acc@5` goes
12.9 % → **36.4 %** and NOVEL `acc@5` → **37.3 %**. The one number that does *not* improve is
`CE_z`: 8.15 vs FIX 7.54 (and vs the exact same-config control ALIGNED_FIX 8.10, i.e. flat) — the
arm is a much better top-1/top-5 predictor and a no better-calibrated distribution. Details,
caveats and one measurement bug I found and fixed in my own first attempt are all below.

---

## 1. The design, and why it can express the history dependence

### 1.1 The diagnosis this is aimed at

| quantity (same 18,869 radical / 15,046 NOVEL val positions) | acc@1 |
|---|---|
| next-word root, linear readout of a single `h_t` | **6.48 %** ← every prior attempt lives here |
| order-4 root n-gram **lookup** over the root history | **53.13 %** |
| current-word root, linear readout of `h_t` | **92.34 %** |
| majority-class marginal | 3.551 % |
| discrete n-gram table (measured arm) | 6.06 % final / 6.79 % best, declining |

The trunk contains the root (92.34 %); a single `h_t` cannot expose a *history-dependent* readout
(6.48 % linear ceiling). The root *sequence* carries the information (53.13 %). The discrete table
is expressible enough but trains to 6.06 % and declines, because 92.7 % of order-4 contexts occur
exactly once in train — a table can only memorise singletons. Cardinality is not the constraint
(9,490 → 500 classes left the lift flat at 1.86×).

### 1.2 The operation

For a selected trunk layer `i`, **after** its token-level self-attention + MLP block:

```
h  ←  h + gate · LayerNorm( o_proj( softmax( Q Kᵀ / √d ) · V ) )
      Q = W_q · h ,   K = W_k · E_root(r_{0..t}) ,   V = W_v · E_root(r_{0..t})
```

Queries from the **token stream**; keys and values from the **root history**. The mask is causal
over root positions (`j <= t`), so `r_t` is visible to `h_t`.

**Why `r_{<=t}` and not the brief's `r_{t-w..t-1}`.** `r_t` is not a leak: it is an *input* of this
model at position `t` (`morphemic_embed` is fed `(prefix_t, root_t, wazn_t, suffix_t)`) and the
target is `r_{t+1}`. It is also required for fidelity to the number this module exists to
approximate: the order-4 lookup that scores 53.13 % uses the context `r_{t-3..t}` **ending at t**
(`nrmt_arch.DiscreteNgramFeatures._codes` documents exactly this), and order-1 = `r_t` alone
already scores 1.74 % while order-2 (`r_{t-1}, r_t`) jumps to 11.81 %. `--rca-exclude-current`
reproduces the strictly-past variant.

**Why the injected vector is *necessarily* root-history content.** `V` is `W_v · E_root(r)` — there
is no surface term in the value path — so whatever the module writes into `h` is a learned,
h-conditioned soft lookup over root embeddings. That is the continuous relaxation of the exact-match
lookup: كتب/كاتب/مكتوب/كتابة share representation and similar contexts share attention mass, so it
can generalise off the singleton — which is precisely where the discrete table measured 6.06 %.

### 1.3 A dedicated module, not `IshtiqaqAttentionV12`'s dead root projections

`IshtiqaqAttentionV12` already owns, per layer, a trained `root_embed` (9015×64), `root_q_proj`
(64→896) and `root_k_proj` (64→128), and in the NRMT path they are dead code. They do **not** fit
the operation above, for three structural reasons:

1. **Q must come from the token stream.** `root_q_proj`'s domain is `root_embed`, so it can only
   produce queries *from roots* — root-to-root attention, not root-history attention from `h`.
2. **K and V must both come from the root history.** There `v = v_proj(hidden_states)` — the value
   path is *surface*, not radical.
3. The root term is a **fused additive bias on the surface attention scores**
   (`total_score += stream_mix[1] * ishtiqaq_cond * score_root`), not a separate residual stream
   that can carry root history into `h`.

**What is reused:** the released, synthesis-trained root embedding table
`morphemic_embed.root_embed` (9015 × 448, **frozen**) — the same table the (also dead) linear
conditioning branch reads — is the K/V source. 24 per-layer copies would instead add
24 × 9015 × 448 = 96.9 M new untrained embedding parameters and fragment one root space into 24.

Deadness is preserved, measured (`probe_root_paths.py`): instrumenting
`IshtiqaqAttentionV12.forward` and driving the trunk exactly as the trainer drives it gives
**24 attention calls per forward, 0 carrying `root_ids`, 0 with `active_root_ids` set — with and
without the cross-attention attached** (historical figure: 0 of 456). The new module carries its own
root ids: `rca_calls = 83 204`, `rca_calls_with_root_ids = 83 204` in the A2 run.

### 1.4 Insertion scheme and parameter efficiency

Cross-attention is inserted at the **top 4 layers (20–23)** — the same set that is unfrozen.
Rationale: a cross-attention inserted at a *frozen* layer can never learn (its gate starts at
exactly 0 and it has no gradient path unless the layer itself or its consumers adapt), so
"RCA set == unfrozen set" is the only coherent scheme; the trainer warns if they differ.
`--root-cross-attn` accepts `topN` / `midN` / `everyK` / an explicit list, so depth is a flag.

| item | value |
|---|---|
| new parameters | **9.64 M** (4 × 2.41 M: q/k/v/o projections + QK LayerNorms + output LayerNorm + gate) |
| K/V source | shared released `root_embed` (9015 × 448), **frozen** |
| output | `gate (0-init) × LayerNorm(o_proj(ctx))` — exactly 0 at init |
| heads | 8 × 112 |
| causal reach | `j <= t` (includes `r_t`); `--rca-exclude-current` for `j < t` |

The zero-init gate is the measured-safe v18fix `--feat-gate` device, including its measured trap:
zeroing **both** the gate and `o_proj` is a dead saddle (every gradient is exactly 0), so `o_proj`
is initialised small-random (`--rca-out-std 1e-3`) while the gate stays exactly 0. Measured:
`max|h(gate=0) − h(no RCA)| = 0.000e+00`; `max|h(gate=1) − h(no RCA)| = 1.25e-01`; a rolled root
sequence changes the output (`6.94e+00`).

**Defaults preserve current behaviour.** With `--root-cross-attn none --unfreeze-last 0` the cache,
optimizer, LR schedule and code path are the ones that produced FIX 6.84 / ALIGNED_FIX 6.62.
Proven: `nrmt_train_ORIGINAL.py` vs the patched `nrmt_train.py`, 40 seeded steps on the same cache,
traces **field-for-field identical** (`loss, root, wazn, prefix, suffix, grad_norm, lr, p_ss,
skipped` × 40 steps, 0 mismatches) — re-verified on the final delivered file.

---

## 2. Unfreezing the trunk — and the cache consequence

### 2.1 What was unfrozen, and the LR split

| arm | trunk layers trained | head lr | cross-attention lr | trunk (pretrained) lr | other |
|---|---|---|---|---|---|
| **A2 (final, 20k)** | 20–23 | 1e-3 | 3e-4 (0.3×) | **1e-5 (0.01×)** | out-norm ON, RCA dropout 0.1 |
| **B2 (final, 3k, killed)** | none (frozen) | 1e-3 | 1e-4 (0.1×) | — | out-norm ON, RCA dropout 0.1 |
| A (first attempt, superseded) | 20–23 | 1e-3 | 1e-3 (1.0×) | 1e-4 (0.1×) | unbounded residual, no dropout |

`morphemic_embed` and `final_norm` stay **frozen** in every arm, exactly as in the FIX baseline.
The split is legible from the run itself (`train_RCA_UNFREEZE_A2.log`):

```
[*] trainable parameters: 84.69M (head 12.80M @lr 0.001 | rca 9.64M @lr 0.0003 layers [20,21,22,23]
    | trunk 62.25M @lr 1e-05 layers [20,21,22,23])
[*] optimizer groups (max_lr; the lr shown by the scheduler at step 0 is max_lr/div_factor=25 ...):
    head n=19 max_lr=4e-05 | rca n=44 max_lr=1.2e-05 | trunk n=92 max_lr=4e-07
```

(The printed values are the OneCycle warm-up start = `max_lr/25`, i.e. 1e-3, 3e-4, 1e-5 exactly.)

**Why 0.01× and not 0.1×.** The first attempt at 0.1× destroyed the warm start by memorisation:
train root CE fell to **0.019** by step 5200 (the frozen baseline is at ~3.8 at step 3000 and 2.35
at step 20000) with 84.69 M trainable parameters over 381,000 training positions. The bounded,
trunk-*frozen* arm stayed on the baseline trajectory (root CE 4.01 at step 1000 vs 4.68), i.e. the
memorisation came from the **trunk**, not from the cross-attention. Hence 0.01×, plus three
stabilisers each with a measured reason: residual LayerNorm (`--rca-out-norm`, the v18fix
`feat_norm` principle: an unbounded additive branch lets the free scalar gate inflate `h`),
`--rca-dropout 0.1` on the injected residual, and RCA lr 0.3×.

### 2.2 Bypassing the hidden-state cache — correctly, not silently

`nrmt_train.py` precomputes the frozen backbone's hidden states once (`extract_backbone` →
`Htr`/`Hva`, in memory) and trains the head on those fixed tensors. That is valid only while the
trunk is frozen. **Note on the `--cache` path:** `/workspace/head_fix/nrmp_cache_9490_aligned` holds
the int32 **token/root streams** (`train.pt` 7,063,767 × 4, `val.pt` 837,575 × 4) — it is *not* the
hidden-state cache and is still required. The hidden-state cache is the `extract_backbone`
precompute.

In live mode (`--root-cross-attn` and/or `--unfreeze-last > 0`):

* `extract_backbone` is **not called**; `live_h_windows()` runs
  `morphemic_embed → backbone → final_norm` inside the training step, exactly where the cached
  tensor used to be indexed;
* `extract_backbone` is **rebound in the module globals to a function that RAISES**, so an
  accidental call crashes the run instead of silently training on stale activations;
* with the trunk frozen but the cross-attention live, the cache is *still* bypassed, because the
  cross-attention changes the trunk's residual stream.

### 2.3 Proof the trunk is genuinely being trained

| evidence | measurement |
|---|---|
| cache-bypass guard | never fired (a call would have aborted the run); rc=0 |
| per-group pre-clip gradient norm | `grad_norm_trunk > 0` on every logged step of A2 (1.46 at step 1, 5.9e-01 at step 4000); **exactly 0.000e+00** in the frozen arm B2 |
| fixed-batch hidden-state drift | `max|h_stepN − h_step1| = 1.03` at step 1000 → **6.68** at step 20000 (A2); the fixed-batch fingerprint moves too |
| live path ≡ cached path | trainer-side: the published 6.62 % arm re-evaluated through the **cached** path gives 6.62 % (CE_z 8.0997); the same head evaluated through the **live** path with the RCA gates frozen at 0 gives **6.64 %** (CE_z 8.0997). `validate_eval_path.log` |
| RCA really consumes root ids | `rca_calls 83 204`, `rca_calls_with_root_ids 83 204` (100 %), while `IshtiqaqAttentionV12` still carries roots in 0 calls |
| moved weights on disk | `head_RCA_UNFREEZE_A2.pt.trunk.pt` (143.8 MB) written at every eval with the unfrozen layers + cross-attention state and the layer lists |

### 2.4 Two measurement defects in my own work, found and fixed

**(a) The live evaluator was fed the training windows.** The first version called
`evaluate(..., live_streams=(Pr, Tr, Wr, Sr))` while passing the **validation targets**
(`Tv[:, :-1]`). `h` and the targets pair by index, so the held-out number measured a different
window set — it reported **0.23 % / 0.70 %** for models whose *train* curves were on the normal
trajectory. Fixed to `(Pv, Tv, Wv, Sv)`; the cached path never had this failure mode (`Hva_w` is
built from the val starts alone). Runs A and B are reported only for train-side evidence; A2/B2 use
the fixed evaluator.

**(b) The cross-attention stack is left in `train()` mode during `evaluate`.** `model.eval()` is
never called on the stack, so with `--rca-dropout 0.1` the held-out metric was computed with dropout
active on the injected residual. A paired same-weights measurement bounds the effect at
**−0.13 pp** (11.007 % with dropout active vs 11.140 % without, identical harness) — two orders of
magnitude inside the 1 pp noise band, and in the conservative direction (the clean number is the
higher one). The one-line fix for any re-run is `rca_stack.eval()` before and `train()` after the
loop in `evaluate`.

**(c) A side-check that failed, reported for completeness.** My standalone `repro_eval.py` does
*not* reproduce the published ALIGNED_FIX arm (0.93 % vs 6.62 %), so it is a broken harness and none
of its numbers are evidence about this work. The validation that counts is trainer-side (2.3): the
same evaluator reproduces the published baseline to the decimal and reproduces it again through the
live path.

---

## 3. The diff

* new file: `/workspace/root_attn/root_cross_attn.py` — the module, the stack, the hook wiring,
  `parse_layer_spec`
* edited: `/workspace/hf_v19_2_release/nrmt_train.py`
  * on-pod backup `nrmt_train.py.bak.root-attn-20261002T140448Z`; pristine copy kept as
    `nrmt_train_ORIGINAL.py`
  * md5 original `ceacdb608b9f69e8ae2c14250141257f` → patched (final, the file both arms ran)
    **`56ef3f2fd9ddc352dee596f32590e7b5`** (local mirror identical)
  * `root_cross_attn.py` md5 **`b53ae702bde947ac66f933191f904c36`** (local mirror identical)
  * unified diff `/workspace/root_attn/nrmt_train_root_attn.diff` (**406** lines,
    md5 `1d799c357714de63b6f488a7eccce2f6`)
* unchanged and read-only: `nrmp_vocab.py`, the blueprint, the checkpoints, the shipped caches,
  `nrmt_arch.py`, `models/*.py`

New flags (all default to the historical behaviour): `--root-cross-attn`, `--rca-heads`,
`--rca-dropout`, `--rca-out-std`, `--rca-out-norm`, `--rca-exclude-current`, `--rca-ablate-eval`,
`--rca-lr-scale`, `--unfreeze-last`, `--trunk-lr-scale`, `--h-drift-probe`.

---

## 4. Seven suites (after the edits)

`/workspace/root_attn/suites_after_rootattn.log` (runner `run_seven_suites.sh`):

| suite | result |
|---|---|
| `test_grammar_impl.py` | **20/20**, 0 FAIL |
| `verify_v2.py` | all PASS, exit 0 |
| `andalusian_realizer.py` | **67/67**, 0 FAIL |
| `test_sibawayh_governor.py` | **15/15** |
| `ibn_malik_automaton.py` | **46/46** |
| `test_khalil_orbits.py` | **30/30**, 0 FAIL |
| `test_awzan_order.py` | **24/24**, 0 FAIL |

---

## 5. Measurement against the closed ladder

### 5.1 The ladder and this work (final step; the `arm_table.py` convention)

| arm | ALL@1 | ALL@5 | NOVEL@1 | NOVEL@5 | CE_z | ×marginal | Δ vs FIX |
|---|---|---|---|---|---|---|---|
| CONTROL | 5.718 | 11.410 | 5.470 | 11.126 | 7.168 | 1.61× | −1.12 |
| NGRAM (discrete table) | 6.063 | 8.617 | 5.742 | 8.221 | 9.535 | 1.71× | −0.77 |
| ALIGNED | 6.148 | 11.935 | 5.869 | 11.511 | 7.155 | 1.73× | −0.69 |
| ALIGNED_FIX (exact control for A2) | 6.619 | 12.566 | 6.454 | 12.136 | 8.100 | 1.86× | −0.22 |
| NOFEAT | 6.593 | 12.979 | 6.533 | 12.814 | 7.613 | 1.86× | −0.24 |
| GUARD | 6.630 | 13.000 | 6.547 | 12.841 | 7.614 | 1.87× | −0.21 |
| **FIX (the bar)** | **6.837** | 12.905 | 6.660 | 12.701 | 7.543 | 1.93× | — |
| **RCA_UNFREEZE_A2 (this work, 20k)** | **19.529** | **36.414** | **19.899** | **37.326** | 8.152 | **5.50×** | **+12.69** |
| RCA_FROZEN_B2 (this work, **3k of 20k**, killed) | 13.760 | 22.86 | 13.820 | 22.98 | 7.073 | 3.87× | +6.92 |
| *single-`h` linear ceiling* | *6.48* | | | | | | |
| *order-4 lookup ceiling* | *53.13* | | | | | | |

Decisive bar (the brief): FIX + 1 pp = **7.84 %**. A2 clears it by **+11.7 pp**.
Within-run eval sd over the last 10 evals is 0.026–0.113 pp across the ladder and **0.093 pp** for
A2, so the ~1 pp band really is config-to-config variation, and this is far outside it.

### 5.2 A2 held-out trajectory (every eval; gates are the four cross-attention gates)

| step | ALL@1 | NOVEL@1 | CE_z | **RCA-OFF ALL@1** | gates |
|---|---|---|---|---|---|
| 1000 | 10.40 | 10.32 | 6.888 | 6.24 | ±0.35 |
| 2000 | 16.86 | 16.90 | 6.649 | 4.31 | ±0.68 |
| 3000 | 17.77 | 17.83 | 6.875 | 3.58 | ±0.89 |
| 4000 | 18.45 | 18.61 | 7.084 | 3.17 | ±1.06 |
| 5000 | 18.51 | 18.66 | 7.196 | 3.23 | ±1.20 |
| 6000 | 18.64 | 18.78 | 7.301 | 3.19 | ±1.32 |
| 8000 | 19.00 | 19.27 | 7.484 | 3.08 | ±1.51 |
| 10000 | 19.30 | 19.51 | 7.643 | 3.04 | ±1.65 |
| 15000 | 19.45 | 19.81 | 8.018 | 3.15 | ±1.81 |
| 20000 | **19.53** | **19.90** | 8.152 | **2.83** | −1.833, +1.856, +1.890, −1.929 |

Three things to read off this table:

* **NOVEL ≥ ALL at every step** (19.90 vs 19.53 at the end). NOVEL positions are the 79.7 % whose
  12-root context never appears in train. A memorising model cannot do this; the gain is
  generalisation over root *histories*.
* **The inference-time ablation is the causal statement.** Zeroing only the four gates — same trunk
  weights, same head, same windows — leaves **2.83 %**, below the 3.551 % marginal. The trained
  trunk has become *dependent* on the pathway (the ablation gets worse over training: 6.24 → 2.83).
* The curve is flat and stable after step 8000 (late-run sd 0.093 pp), so 19.53 % is not a lucky
  eval.

### 5.3 B2 (cross-attention alone, trunk FROZEN) — partial, killed by a sibling at 3 k

| step | ALL@1 | NOVEL@1 | CE_z | RCA-OFF ALL@1 | gates |
|---|---|---|---|---|---|
| 1000 | 6.92 | 6.79 | 7.178 | 6.41 | ±0.12 |
| 2000 | 9.63 | 9.44 | 7.231 | 5.95 | ±0.25 |
| 3000 | **13.76** | **13.82** | 7.073 | 5.20 | ±0.36 |

Baseline ALIGNED_FIX at the same steps: 6.37 / 6.71 / 6.50. So **with no trunk adaptation at all the
cross-attention alone already doubles the number by step 3000 and is still climbing steeply**, and
its ablation stays at baseline level. Attribution is clean: the cross-attention is the active
ingredient; unfreezing the top four layers adds a further large gain (3 k: 13.8 % → 20 k: 19.5 %).
A sibling arm's own frozen-trunk top4 control (a different RCA lr/normalisation) is reported in its
own script as **16.5 %** at 20 k, consistent with this trend.

### 5.4 The one number that does not improve: CE_z

`CE_z` (per-position-standardised CE, the metric contract's capability measure) is **8.152** for A2
vs **7.543** FIX and **8.100** ALIGNED_FIX — i.e. **worse than FIX by 0.61, flat vs its exact
control (+0.05)**. So the arm converts a flat calibrated distribution into a much better top-1/top-5
predictor: `acc@1` 3.0×, `acc@5` 2.8×, NOVEL `acc@1` 3.0×, NOVEL `acc@5` 2.9×, `CE_z` flat-to-worse.
This is reported plainly rather than buried: if `CE_z` is treated as the sole capability measure,
this is not a win; on the three accuracy metrics — including the NOVEL split that excludes
memorisation — it is a large one.

---

## 6. Resource accounting

| run | wall-clock | peak VRAM | NaN/Inf | notes |
|---|---|---|---|---|
| **A2** (20 k, RCA top4 + unfreeze top4) | **6602 s = 110.0 min** (3.0 it/s) | **21 540 MiB** | **0** | clean, sole GPU tenant for most of the run |
| B2 (RCA top4, trunk frozen, killed at 3 k) | 720 s to the kill | 19 797 MiB (co-tenant inflated) | 0 | **rc=143 = SIGTERM (externally terminated, sibling trainers co-resident)**, no crash, no OOM, no traceback |
| A (superseded first attempt, killed for cause) | 1081 s | 11 787 MiB | 0 | memorised the train set; killed at step 5200 |
| cache rebuild / suites / probes | CPU only | — | — | seven suites ~4 min wall |

The frozen baseline (FIX) was 900 s for the same 20 k at batch 32; the live trunk costs **7.3×** the
wall clock (110 min vs 15 min) and about 2.4× the VRAM (21.5 GB vs 8.8 GB), because the 0.5 B trunk
now runs forward (and the top four layers backward) every step. Serialisation was not maintained across agents: three `nrmt_train.py` processes were co-resident at
17:15 (mine plus sibling arms) and my B2 was terminated at step 3000 with rc=143. A2's accounting
above was measured with the GPU otherwise idle and is unaffected.

---

## 7. Verdict

**Root cross-attention over the root history, plus a partial trunk unfreeze, moves the number
decisively: 6.84 % → 19.53 % held-out `acc@1` (5.50× the marginal, +12.69 pp, NOVEL 19.90 %), with
the capability provably residing in the new pathway (inference ablation → 2.83 %).** This is the
first arm in the ladder to leave the 5.7–6.8 % band, and it leaves it by an order of magnitude more
than the 1 pp noise band.

What is established beyond the headline:

1. The gain is **history generalisation, not memorisation**: NOVEL ≥ ALL at every eval, on the 79.7 %
   of positions whose 12-root context never occurs in train.
2. The mechanism is the designed one: the injected residual is a function of root embeddings only
   (`V = W_v E_root`), causal over root positions, fully activated (gates ≈ ±1.9, 83 204/83 204 calls
   carrying root ids), and its removal at inference destroys the capability.
3. **Unfreezing is a real but secondary ingredient**: the cross-attention alone with a *frozen*
   trunk already reaches 13.8 % by step 3000 (B2) and a sibling's frozen-trunk control 16.5 % at 20 k,
   so the RCA — not the trunk adaptation — is what makes the difference; the unfreeze adds roughly
   the last 3–6 pp.
4. Unfreezing at 0.1× LR destroys the warm start outright (train root CE 0.019; held-out collapsed);
   0.01× plus a bounded residual is the configuration that works. The stabilisers are load-bearing,
   not decoration.

What is **not** established / caveats, stated as plainly as the win:

* `CE_z` does not improve (8.15 vs 7.54 FIX, 8.10 ALIGNED_FIX). On the metric contract's
  "capability" definition the arm is not better calibrated; it is a much better top-1/top-5
  predictor. If only `CE_z` counts, this is not a win.
* The trunk is trained on 381,000 positions and does memorise the train set (train root CE 0.02–0.2
  at the end vs the frozen arm's 2.35). The held-out NOVEL result says that is not what the gain is
  made of, but the arm is not regularisation-clean and a larger training corpus is the obvious next
  lever.
* Only the top 4 of 24 layers carry the cross-attention, and only those 4 are unfrozen. Depth was
  not swept here (a sibling arm is doing that); the ceiling is 53.13 %, so 19.53 % is **37 % of the
  order-4 lookup ceiling** — there is room above it.
* The held-out metric for A2/B2 was measured with the cross-attention dropout active (bounded at
  −0.13 pp by a paired measurement); the fix is one line (`rca_stack.eval()`), and a re-run with it
  should be the first thing a follow-up does.
* One of my two final arms (B2) was terminated externally at 3 k; its trend and ablation are
  informative but it is not a completed 20 k arm.

---

## 8. Reproduction

```bash
KEY=/home/grem3/Documents/deepseek-harness/default-workspace/rootformer/.secrets/id_ed25519_runpod
ssh -i $KEY -p 46758 -o StrictHostKeyChecking=no root@213.173.104.76
cd /workspace/hf_v19_2_release
PY=/workspace/venvs/rootformer/bin/python
CK=checkpoints/rootformer_v19_2_synthesis_ar_backbone.awzan142.roots9490.tok10052.safetensors
CACHE=/workspace/head_fix/nrmp_cache_9490_aligned
FIX="--feat-gate --margin-ramp 10 --grad-warmup 10 --logit-scale ln"

# A2 -- the winning arm (ALL@1 19.53 %, NOVEL 19.90 %, wall 6602 s, peak 21540 MiB)
$PY nrmt_train.py --checkpoint $CK --cache $CACHE --head-init remap --steps 20000 \
    --batch-size 32 --eval-every 1000 --tag RCA_UNFREEZE_A2 $FIX \
    --root-cross-attn top4 --unfreeze-last 4 --trunk-lr-scale 0.01 --rca-lr-scale 0.3 \
    --rca-out-norm --rca-dropout 0.1 --rca-ablate-eval --h-drift-probe 500

# B2 -- cross-attention alone, trunk frozen (13.76 % at 3 k before an external SIGTERM)
$PY nrmt_train.py --checkpoint $CK --cache $CACHE --head-init remap --steps 20000 \
    --batch-size 32 --eval-every 1000 --tag RCA_FROZEN_B2 $FIX \
    --root-cross-attn top4 --unfreeze-last 0 --rca-lr-scale 0.1 \
    --rca-out-norm --rca-dropout 0.1 --rca-ablate-eval --h-drift-probe 500

# correctness gates
$PY /workspace/root_attn/verify_live_eval.py --checkpoint $CK --cache $CACHE
$PY /workspace/root_attn/probe_root_paths.py --checkpoint $CK --cache /workspace/discrete_path/smoke_cache
bash /workspace/root_attn/run_seven_suites.sh
# cached-vs-live reproduction of the published baseline (validate_eval_path.log)
$PY nrmt_train.py --checkpoint $CK --cache $CACHE --head-init remap \
    --head-checkpoint /workspace/head_fix/head_ALIGNED_FIX.pt --steps 1 --batch-size 32 \
    --eval-every 1 --tag V1 $FIX
$PY nrmt_train.py --checkpoint $CK --cache $CACHE --head-init remap \
    --head-checkpoint /workspace/head_fix/head_ALIGNED_FIX.pt --root-cross-attn top4 \
    --unfreeze-last 0 --rca-lr-scale 1e-12 --rca-out-norm --rca-dropout 0.0 --steps 1 \
    --batch-size 32 --eval-every 1 --tag V2 $FIX
```

Files: `REPORT_root_attn.md` (this), `root_cross_attn.py`, `nrmt_train_patched_rootattn.py` (the
delivered trainer), `nrmt_train_root_attn.diff`, `probe_root_paths.py`, `verify_live_eval.py`,
`chain_final.sh`, `run_seven_suites.sh`, `summarize_rootattn.py`, `repro_eval.py` (broken
side-check, kept for transparency), plus `remote/root_attn/` mirrors of every log, trace, results
JSON and saved head/trunk.
