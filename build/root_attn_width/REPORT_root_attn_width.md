# RCA width / residual-bound experiment — findings

**Owner:** subagent on session-a4f1047e. **Pod:** `g6exduq0bd17z8` @ 213.173.104.76:46758.
**Artifacts:** `/workspace/root_attn_width/` (mirrored to `rootformer/build/root_attn_width/`).
**No shipped file was modified:** `nrmt_train.py` md5 `56ef3f2fd9ddc352dee596f32590e7b5` and
`root_cross_attn.py` md5 `b53ae702bde947ac66f933191f904c36` — both recorded in the run's own log.

---

## TL;DR — verdict

**The frozen trunk is the limiting factor, not bandwidth.**

| lever | Δ acc@1 | Δ acc@5 |
|---|---|---|
| **Residual bound** (`--rca-out-norm`), 4 layers, frozen | **+2.54** (matched @15k) | +5.30 |
| **Width** 896 → 1792 (+9.64 M params), bound ON, frozen | **+0.22** (matched @20k) | +0.71 |
| **Depth** 4 → 8 RCA layers, bound OFF, frozen | **+0.13** (matched @15k) | +1.56¹ |
| Unfreeze top-4 trunk layers, bound ON² | +0.36 | −2.99 |

¹ acc@5 row for depth is `DEPTH_T8`@20k vs `CTL`@15k (CTL has no 20k eval — it was SIGTERM'd);
the acc@1 figure uses the matched @15k pair. ² `RCA_UNFREEZE_A2` differs from `RCA_NORM` in **three**
ways (`--unfreeze-last 4`, `--rca-lr-scale 0.3`, `--rca-dropout 0.1`), so this row is **not** a
single-variable contrast — it is quoted only to size the trunk-adaptation lever.

Doubling the pathway's *width* (or its *depth*) is worth a fifth of a point; bounding the injected
residual is worth 2.5 points. `RCA_W2N` finishes at **19.39 %** — squarely inside the brief's
"stays ≈16–19 %" branch, not "rising toward 53 %".

**And the brief's lever #1 does not exist as described:** the live root channel is **448 of 896
(50 %)**, not 64 of 896 (7 %). See §1 — this is the single most important correction here.

---

## 1. The brief's lever #1 rests on a misidentified tensor

The brief states the root channel is `(896, 64)` — "64 of 896 dimensions ≈ 7 % of model width".

| what | shape | live? |
|---|---|---|
| `backbone.layers.{0..23}.self_attn.root_q_proj.weight` | (896, 64) | **NO — dead code** |
| `backbone.layers.{0..23}.self_attn.root_k_proj.weight` | (128, 64) | **NO — dead code** |
| `backbone.layers.{0..23}.self_attn.root_embed.weight` | (9015, 64) | **NO — dead code** |
| `morphemic_embed.root_embed.weight` — the RCA's K/V source | **(9490, 448)** | **YES** |

Three independent confirmations:

1. **Checkpoint tensor headers** (`safetensors.safe_open` on
   `rootformer_v19_2_synthesis_ar_backbone.awzan142.roots9490.tok10052.safetensors`): the
   `(896,64)` tensors exist, but `morphemic_embed.root_embed.weight` is `(9490, 448)`.
2. **The parent's own probe**, `root_attn/probe_root_paths.out`:
   `the dual-stream root path is STILL dead: 0 of 24 calls carry roots (historical: 0 of 456)` —
   `IshtiqaqAttentionV12`'s per-layer root branch is never fed. Its `root_q_proj (896,64)` is
   exactly the tensor cited by the brief.
3. **The control run's banner**: `d_root=448, heads=8x112` — `RootCrossAttentionStack` is
   constructed with `model.morphemic_embed.root_embed`.

Consequences:

- The live root channel is **448/896 = 50 %**. "Widen 64 → 128/256" would have been a **narrowing**
  (448 → 128).
- Widening is **information-theoretically void** here: `k_proj`/`v_proj` read a **frozen 448-dim
  embedding**, so the q·k bilinear form has rank ≤ 448 at *any* projection width. Extra width buys
  capacity, not information rate. The brief's inference — "you cannot carry a 53 % signal through a
  7 % channel" — does not apply to this architecture.

The width lever was therefore run under the only definition the architecture admits, and is
labelled a **capacity** test throughout.

## 2. What was built, and proof that nothing shipped changed

Two generated copies (`patch_module.py`, `patch_trainer.py`; 4 + 5 exact-match edits; `.diff` files
kept). The shipped files are untouched, so the parent's arms and any further `chain_final.sh` arm
read exactly what they loaded.

`--rca-dim W` = the cross-attention's **internal width**: `q_proj 896→W`, `k_proj/v_proj 448→W`,
`o_proj W→896`, per-head dim `W/heads`. `W = 0` (default) → `W = d_model` → bit-identical to shipped.

| `--rca-dim` | per layer | module (4 layers) | head dim | q_proj | k/v_proj | o_proj |
|---|---|---|---|---|---|---|
| 0 / 896 (shipped) | 2.408897 M | **9.636 M** | 112 | (896,896) | (896,448) | (896,896) |
| **1792 (this run)** | 4.817793 M | **19.271 M** | 224 | (1792,896) | (1792,448) | (896,1792) |

**Parameter cost: +9.636 M**, i.e. trainable total 22.43 M → **32.08 M (+43 %)**.
(The brief's "module was 9.64 M at 64" is the 4-layer total at the shipped width; reproduced here as
9.6356 M.)

**CPU equivalence proof** (`equiv_check.py`, exit 0, run before any GPU time was taken):

- params bitwise equal at `d_attn=None` and `d_attn=896`, `out_norm` off **and** on;
- forward bitwise equal at `gate=0.37`, both cases;
- `gate=0` forward is **exactly 0** (the no-op-at-init property is preserved);
- `num_heads*head_dim != d_attn` raises (`d_attn=1001` → `8*125 != 1001`).

The first version of this test reported three failures; all three were bugs in the *test*, not the
module (RNG not reset per construction; `gate=0` made the forward comparison vacuous at 0 == 0;
`d_attn=1000` genuinely *is* divisible by 8). Kept on the record: a green test that is green for the
wrong reason is worse than a red one.

**Probe fork validated:** `probe_width.py` (the parent's read-only dynamics probe, re-pointed at the
width module and rebuilt at the checkpoint's recorded `rca_dim`) reproduces the parent's
`probe_RCA_FROZEN_CTL_step9000.json` **exactly** — gates `[0.1709, 0.4531, 0.9922, 0.8320]`,
`max|Δh| = 21.156`, `h_rms = 1.0022`. So the fork changes nothing at the shipped width.

## 3. Arms

Every arm differs from `RCA_FROZEN_CTL` in exactly one variable. Trunk frozen (`--unfreeze-last 0`),
`--rca-lr-scale 1.0`, FIX flags (`--feat-gate --margin-ramp 10 --grad-warmup 10 --logit-scale ln`),
`--head-init remap`, aligned cache, 20 000 steps, batch 32, eval every 1000.

| arm | who | RCA layers | width | bound | dropout | trunk |
|---|---|---|---|---|---|---|
| `RCA_FROZEN_CTL` | parent | top4 | 896 | off | 0 | frozen |
| `RCA_DEPTH_T8` | parent | **top8** | 896 | off | 0 | frozen |
| `RCA_NORM` | parent | top4 | 896 | **ON** | 0 | frozen |
| **`RCA_W2N`** | **this work** | top4 | **1792** | **ON** | 0 | frozen |
| `RCA_UNFREEZE_A2` | parent | top4 | 896 | ON | 0.1 | top4 @ lr×0.01 |

`RCA_W2N` vs `RCA_NORM` isolates **width alone** (bound fixed ON). `RCA_NORM` vs `RCA_FROZEN_CTL`
isolates **the bound** (width fixed at 896). Each contrast is therefore single-variable, even though
`RCA_W2N` changes two things relative to `CTL`. The missing 2×2 cell (W=1792, bound **OFF**) was not
run — **stated as the attribution limit**.

## 4. Results — `ALL_val acc@1`, held out, `n = 18869`

### 4.1 The matrix, at every eval of the width arm's comparator

| step | CTL (896,off) | DEPTH_T8 (896,off,8L) | NORM (896,ON) | **W2N (1792,ON)** | W2N−NORM |
|---|---|---|---|---|---|
| 1000 | 13.57 | 14.92 | 17.27 | 17.23 | −0.04 |
| 2000 | 15.09 | 15.94 | 18.64 | 18.95 | +0.31 |
| 3000 | 16.02 | 16.13 | 18.95 | 18.92 | −0.03 |
| 4000 | 15.94 | 16.11 | 19.06 | 18.89 | −0.17 |
| 5000 | 16.20 | 16.60 | 18.89 | 19.05 | +0.16 |
| 6000 | 16.24 | 16.36 | 18.67 | 18.79 | +0.12 |
| 7000 | 16.33 | 16.49 | 18.81 | 19.11 | +0.30 |
| 8000 | 16.46 | 16.57 | 19.04 | 19.07 | +0.03 |
| 9000 | 16.47 | 16.77 | 19.07 | 19.14 | +0.07 |
| 10000 | 16.68 | 16.89 | 18.96 | 19.32 | +0.36 |
| 11000 | 16.50 | 16.85 | 18.89 | 19.23 | +0.34 |
| 12000 | 16.53 | 16.75 | 18.65 | 19.26 | +0.61 |
| 13000 | 16.52 | 16.86 | 18.95 | 19.18 | +0.23 |
| 14000 | 16.58 | 16.66 | 18.91 | 19.39 | +0.48 |
| 15000 | 16.54 | 16.67 | 19.08 | 19.40 | +0.32 |
| 16000 | — | — | 19.13 | 19.47 | +0.34 |
| 17000 | — | 16.67 | 19.14 | 19.42 | +0.28 |
| 18000 | — | 16.70 | 19.15 | 19.39 | +0.24 |
| 19000 | — | 16.72 | 19.18 | 19.39 | +0.21 |
| **20000** | — | **16.71** | **19.17** | **19.39** | **+0.22** |

`RCA_FROZEN_CTL` was terminated **deliberately by the parent** at step 15000 (rc=143 = SIGTERM,
`wall_s=5463`) to speed up `RCA_UNFREEZE_A2` — **not** by this work. Its `results_*.json` was never
written, but all 15 `ckpt/rca_step_{1000..15000}.pt` snapshots survive, so matched-step comparison
and probing remain valid.

`RCA_UNFREEZE_A2` final (20 000 steps, complete): acc@1 **19.53 %**, acc@5 36.41 %, CE_z 8.1519;
NOVEL-only 19.90 %; RCA-OFF 2.82 %; `nan_steps 0`.

### 4.2 acc@5 / CE_z — where the width does show up

| arm @20000 | acc@1 | acc@5 | CE_z |
|---|---|---|---|
| `RCA_FROZEN_CTL` (@15k) | 16.54 | 33.97 | 8.5335 |
| `RCA_DEPTH_T8` | 16.71 | 35.53 | — |
| `RCA_NORM` | 19.17 | 39.40 | — |
| **`RCA_W2N`** | **19.39** | **40.11** | **7.8456** |
| `RCA_UNFREEZE_A2` | 19.53 | 36.41 | 8.1519 |

`RCA_W2N` has the **best acc@5 of every arm** (40.11 vs NORM 39.40, A2 36.41) and the best CE_z
(7.8456 vs A2 8.1519). So the 2× width does buy something real — a correct root more often inside
the top-5 and better-calibrated logits — it just does not move acc@1 much. That is consistent with
the sharper attention measured in §4.4.

### 4.3 Width effect, honestly quantified

Across the 20 matched evals, mean Δ = **+0.21** acc@1. It is not uniform: first 10 evals mean
**+0.11**, last 10 mean **+0.33**, and the last **14 evals are all positive** (sign test
p = 2⁻¹⁴ ≈ 6 × 10⁻⁵ if signs were random — but the evals are highly correlated, so the effective
independent count is a handful; treat it as suggestive, not significant). For scale: a single
eval's binomial SE at p ≈ 0.19, n = 18869 is ≈ 0.29 points.

**Conclusion: doubling the pathway width is worth ≈ +0.2 acc@1, i.e. under one SE of a single eval,
against +2.54 for the bound.** Bandwidth is not the binding constraint.

### 4.4 Attention is NOT degenerate, and widening did not make it uniform

From `probe_width.py` on 8 fixed val windows (8192 rows/layer). `ent_norm` 1.0 = uniform,
0 = one-hot; `eff_pos` = exp(entropy).

| arm / step | ent_norm | eff_pos | max_w | frac rows max_w>0.9 | mass on current root | injected resid RMS | att logit max | max\|Δh\| |
|---|---|---|---|---|---|---|---|---|
| CTL @9000 | 0.4879 | 9.42 | 0.420 | 5.53 % | 0.0270 | 19.67 | 14.34 | 21.16 |
| **W2N @9000** | 0.4506 | 7.75 | 0.455 | 5.89 % | 0.0184 | 11.18 | 12.55 | 13.62 |
| T8 @9000 | 0.4643 | 8.66 | 0.448 | 6.78 % | 0.0299 | 12.75 | 13.85 | 23.81 |
| CTL @15000 | 0.4858 | 9.35 | 0.423 | 5.66 % | 0.0269 | 20.49 | 14.34 | 22.69 |
| **W2N @15000** | 0.4557 | 7.90 | 0.451 | 5.75 % | 0.0186 | 12.02 | 12.66 | 13.34 |
| T8 @15000 | 0.4674 | 8.76 | 0.446 | 6.51 % | 0.0297 | 13.22 | 13.78 | 23.31 |
| W2N @20000 | 0.4559 | 7.91 | 0.451 | 5.80 % | 0.0186 | 12.15 | 12.70 | 14.00 |

**Widening did not make attention uniform — the opposite.** At matched step 15000 the normalized
entropy *fell* (0.486 → 0.456), effective positions fell 9.35 → 7.90, and mean max weight *rose*
0.423 → 0.451. The distribution stays broad and genuinely shaped: about halfway from uniform to
one-hot, spread over ~8 positions, never collapsing; only 5.8 % of rows have a max weight > 0.9; and
the mass on the current root (0.019) is *below* the ~0.038 uniform-prior average, so it is not
simply copying the diagonal. Attention logits |·|max ≈ 12–15, so this is not a saturation artifact.

### 4.5 `h_drift` — the residual bound demonstrably worked

`h_drift` = `max|h − h_step1|` on a fixed batch; `h`'s own RMS is 1.00.

| arm | bound | @2k | @5k | @10k | @15k | final |
|---|---|---|---|---|---|---|
| `RCA_FROZEN_CTL` | off | 13.97 | 20.00 | 21.69 | 21.81 | 21.81 (@15k) |
| `RCA_DEPTH_T8` | off | 15.84 | 18.69 | 16.75 | 20.31 | 20.31 (@15k) |
| **`RCA_W2N`** | **ON** | **5.05** | 11.81 | 12.88 | 13.38 | **14.47** (@20k) |
| `RCA_UNFREEZE_A2` | ON | 1.91 | 3.30 | — | — | **6.68** (@20k) |

Early on the bound cuts the overshoot 4× (13.97 → 5.05 at step 2000) and at 15k it is 21.81 → 13.38
(−39 %). The independent probe agrees on its own fixed windows: `max|Δh|` 22.69 → 13.34 and injected
residual RMS 20.49 → 12.02 at step 15000. **So yes, the bound worked** — though note it does *not*
hold the residual at h's own scale: `W2N` still moves h by ~14× its RMS at 20k, and its gates
(|mean| 3.21) are 4.6× `CTL`'s (0.70), because LayerNorm removes the magnitude penalty so the gate
is free to grow. The unbounded arm's failure mode is throttled, not eliminated.

**Note:** the parent's `RCA_NORM` arm did **not** pass `--h-drift-probe` — its log has zero
`h_drift` occurrences — so there is no `h_drift` readout for the frozen+bounded+narrow cell from
that arm. `W2N` covers it.

### 4.6 Gates and gradient norms

| arm @20000 | gates | \|gate\| mean | gnorm head | gnorm rca | gnorm trunk |
|---|---|---|---|---|---|
| `RCA_FROZEN_CTL` (@15k) | +0.181, +0.510, +1.117, +0.986 | 0.698 | 1.94e-02 | 2.95e-02 | 0.0 |
| `RCA_DEPTH_T8` (@19k) | 8 gates, mixed signs | 0.458 | — | — | 0.0 |
| `RCA_NORM` (@15k) | +3.242, +3.180, −3.212, −3.283 | 3.229 | — | — | 0.0 |
| **`RCA_W2N`** | −3.236, +3.155, −3.198, +3.231 | **3.205** | 4.74e-03 | 1.90e-03 | 0.0 |
| `RCA_UNFREEZE_A2` | −1.833, +1.856, +1.891, −1.929 | 1.877 | — | — | live |

**Does the gate open faster with more width?** No — the gate trajectory is essentially identical to
the narrow bounded arm: `|mean|` 0.995 / 1.674 / 2.191 at steps 1k/2k/4k for `W2N`, versus
1.004 / 1.700 / 2.235 for `NORM` at the same steps (within 1 %). Width changes the gate's *final*
level by 3.205 vs 3.229 (−0.7 %). The gate saturates around step 14k in both and the run then
plateaus. Gradient norms decay monotonically to ~2e-3 (rca) by step 18k — the run is converged, not
still-learning, which is why "the trajectory is still rising" is not a live possibility.

## 5. Cost, wall-clock, VRAM, integrity

| | |
|---|---|
| wall-clock | **7218 s = 120.3 min** for 20 000 steps (≈2.8 it/s) |
| schedule | launched 19:00:01Z, finished 21:00Z, run **concurrently** with another agent's 32-batch arm for the whole time |
| the brief's estimate | 69 min alone at 4.8 it/s; contention cost ~1.75× |
| peak VRAM, this arm | **10 080 MiB** |
| peak VRAM, whole GPU | **22 840 MiB / 32 623** — the 2-arm configuration the parent had already measured at 21.5 GB |
| NaN / Inf | **0** non-finite loss steps, **0** NaN/Inf lines in the 20 000-row trace, **0** WARN lines |
| params | 32.08 M trainable (head 12.80 M + RCA 19.28 M); +9.64 M over the shipped control |

**Nothing of the parent's was killed or preempted.** The launcher was pid- and VRAM-gated
(`launch_width2.sh`) precisely so it could never become a third concurrent 32-batch arm: it waits
until ≤1 other trainer is resident *and* ≥11.5 GB is free, so peak occupancy stayed at the proven
2-arm level. It fired at 19:00:01Z when the GPU was momentarily empty.

**Seven regression suites** (not required — no shipped module was edited — but run anyway):

| suite | result |
|---|---|
| `test_grammar_impl.py` | **20/20** checks pass, 0 FAIL |
| `verify_v2.py` | **ALL v2 CHECKS PASS**, exit 0 |
| `andalusian_realizer.py` | **67/67** checks pass, 0 FAIL |
| `test_sibawayh_governor.py` | **15/15** checks pass |
| `ibn_malik_automaton.py` | **46/46** checks pass |
| `test_khalil_orbits.py` | **30/30** checks pass, 0 FAIL |
| `test_awzan_order.py` | **24/24** awzan-order checks pass, 0 FAIL |

Full log: `suites_after_width.log`.

## 6. Interpretation, against the brief's pre-registered rule

The brief fixed the reading in advance:

> *Accuracy rises toward the 53 % lookup ceiling → bandwidth was the limit.*
> *Accuracy stays ≈16–19 % → bandwidth is not the limit, and the ceiling is the frozen trunk itself.*

**The second branch is what happened.** `RCA_W2N` ended at **19.39 %**, inside the pre-registered
"stays ≈16–19 %" band, having bought 2× width for +0.22 and having plateaued from ~step 14000.
Every capacity knob tested moves the number by ≲0.3 points, while the residual *bound* moves it 2.5
and the pathway as a whole is worth 16.7 (RCA-off 3.05 % vs live 19.39 %). The frozen trunk — what
the head can read out of a trunk that was never trained for this objective — is the ceiling.

Two honest caveats:

1. **The strict 2×2 is incomplete.** (W=1792, bound **OFF**) was not run, so the width effect is
   measured only with the bound on. Given that the unbounded regime is the one where depth did
   nothing, and that the bounded regime is where a capacity effect would be most visible, this is
   the more informative cell to have — but it is an assumption, not a measurement.
2. **The width effect is genuinely non-zero, just small.** +0.22 acc@1 and +0.71 acc@5, positive in
   the last 14 consecutive evals, is more than noise-level nothing. It is not enough to change the
   verdict, but saying "width changes nothing at all" would overstate it.

## 7. Reproduction

```bash
# 0. equivalence proof, CPU only, before taking any GPU (exit 0 required)
/workspace/venvs/rootformer/bin/python /workspace/root_attn_width/equiv_check.py

# 1. the arm, detached and pid/VRAM-gated (never a 3rd concurrent 32-batch arm)
TAG=RCA_W2N RCA_DIM=1792 OUT_NORM=1 STEPS=20000 \
  setsid nohup bash /workspace/root_attn_width/launch_width2.sh &

# 2. read-only attention/gate/residual probe at the arm's own width
bash /workspace/root_attn_width/probe_run.sh /workspace/root_attn_width \
  W2N_step15000=/workspace/root_attn_width/ckpt/rca_step_15000.pt

# 3. trajectory tables (all arms, all evals)
/workspace/venvs/rootformer/bin/python /workspace/root_attn_width/collect.py \
  /workspace/root_attn_ctl/train_RCA_FROZEN_CTL.log:CTL \
  /workspace/root_attn_depth/logs/train_RCA_DEPTH_T8.log:DEPTH_T8 \
  /workspace/root_attn_norm/train_RCA_NORM.log:NORM \
  /workspace/root_attn_width/logs/train_RCA_W2N.log:W2N \
  /workspace/root_attn/train_RCA_UNFREEZE_A2.log:A2
```

## 8. Files

| path (pod) | what |
|---|---|
| `root_cross_attn_width.py` + `.diff` | the width module (generated; shipped module untouched) |
| `nrmt_train_width.py` + `.diff` | the trainer fork (generated; shipped trainer untouched) |
| `patch_module.py`, `patch_trainer.py`, `patch_probe.py` | deterministic, exact-match patchers |
| `equiv_check.py` | CPU bit-identity proof (`EQUIV_CHECK: PASS`) |
| `launch_width2.sh`, `run_width.sh` | gated launcher + per-arm runner |
| `probe_width.py`, `probe_run.sh` | read-only attention/gate/residual probe (fork validated) |
| `collect.py` | per-eval / per-step trajectory extractor |
| `logs/train_RCA_W2N.log`, `logs/arm_RCA_W2N.out` | the arm's full log |
| `trace_RCA_W2N.jsonl` | 20 000-step trace (loss, grad norms, h_drift) |
| `results_RCA_W2N.json` | final metrics + full history |
| `head_RCA_W2N.pt`, `head_RCA_W2N.pt.trunk.pt` | final head (51 MB) and RCA payload (38 MB) |
| `ckpt/rca_step_{1000..19000}.pt` | per-1000-step RCA snapshots (38 MB each) |
| `probe/probe_W2N_step{9000,15000,20000}.json` | attention entropy / effective positions |
| `suites_after_width.log` | the seven suites |
| `vram_RCA_W2N.txt` | peak VRAM (`total=22840 arm=10080`) |
