# Phase 1 — the architecture test: A vs C at matched steps

All arms share **identical** flags except which root mechanism is attached and where:

```
--model-v13 --sdpa --pillar-freeze --flash-hooks off
--checkpoint ...roots9490.tok10052.safetensors --cache /workspace/head_fix/nrmp_cache_9490_aligned
--head-init remap --unfreeze-trunk-all --trunk-lr-scale 1.0 --lr 1e-3
--steps 20000 --batch-size 32 --eval-every 1000
--feat-gate --feat-gate-proj-std 1e-3 --margin-ramp 10 --grad-warmup 10
--logit-scale ln --grad-clip 1.0 --hist 3 --dropout 0.1
--rca-out-norm --rca-dropout 0.1 --rca-heads 8 --h-drift-probe 500
```

**No freezing, no EWC, no preservation machinery** — the trunk trains at the head's LR (1e-3), which
is the "normal learning rate" the brief asks for. A and C therefore differ in exactly one thing: the
root pathway.

| arm | root pathway |
|---|---|
| **A** `FLOOR_A` | none |
| **C** `EARLYROOT_C` | residual injector **and** native score bias, **ALL 24 layers (0..23)** |
| E `RESIDUAL_R` | residual injector only, all 24 layers |
| D `SCOREBIAS_D` | native score bias only, all 24 layers |
| X `LATE_X` | both mechanisms, layers **20-23** only (the direct early-vs-late test) |
| A2/C2 | second seed (`--seed 1`) of A and C |

`checkpoint every 1 000 steps`: `--eval-every 1000` writes `head_<TAG>.pt` + `head_<TAG>.pt.trunk.pt`
at every eval, overwriting in place (bounded disk). Checkpoints go to **`/tmp/root_arch_arms`** —
`/workspace` has a per-user quota that silently kills a run on a 640 MiB write.

---

## Arm A — FLOOR (no root pathway) — COMPLETE

| | value |
|---|---|
| wall clock | **83.5 min** (`00:04:18Z` → `01:27:46Z`) |
| peak VRAM | **13.98 GiB** (13982 MiB live; the CUDA OOM message independently reports 13.65 GiB) |
| it/s | **5 it/s** (alone on the card, 32-batch, full trunk) |
| **ALL_val acc@1 (final, step 20000)** | **7.74 %** |
| NOVEL_only acc@1 (final) | **7.63 %** |
| ALL_val acc@1 (best) | 8.14 % @ step 4000 |
| ce_z (final) | 9.2216 |
| trainable | 400.37 M (head 12.80 M @1e-3 \| trunk 387.57 M @1e-3, layers 0-23) |

Trajectory (ALL / NOVEL): 1k 7.64/7.53 · 4k 8.14/8.05 · 8k 7.74/— · 12k 7.72/— · 16k 7.75/— ·
20k **7.74/7.63**. It rises for the first ~4k steps and then **plateaus flat around 7.7-8.1 %** —
it does not keep climbing toward 19.53 %.

Length-1 ridge probe on the trained trunk (same protocol that reproduces 92.34 % on the released
trunk, so the numbers are directly comparable):

| | FLOOR_A (step 20000) | released trunk | Δ |
|---|---|---|---|
| val_all | **87.19 %** | 92.34 % | **−5.15 pp** |
| val_unseen_words | **74.54 %** | 83.92 % | **−9.38 pp** |
| val_seen_words | 90.28 % | 94.39 % | −4.11 pp |
| train_all | 76.11 % | 83.83 % | −7.72 pp |

So the floor arm is a genuine floor on **both** measurements: the trained head reaches 7.74 %, and
normal-LR trunk training **degraded** the trunk's own root decodability by 5.15 pp (9.38 pp on words
unseen in training) rather than improving it.

---

## Arm C — EARLY-ROOT (both mechanisms, layers 0-23) — IN PROGRESS

```
[*] root cross-attention attached to trunk layers [0..23] (57.86M new params, d_root=448, heads=8x112)
[*] PILLAR-FREEZE: 24 pillar gates held at exactly 0.0 and removed from the optimizer
[*] NATIVE root score bias on trunk layers [0..23] (11.010M NEW params, source=shared)
[*] trainable parameters: 453.08M (head 12.80M | rca 57.86M | trunk 371.42M | ishtiqaq 11.010M)
```

peak VRAM **19.18 GiB** — higher than arm A because the residual injector runs its own attention at
every one of the 24 layers. Consequence: **an A and a C cannot coexist** (19.6 + 14.0 = 33.6 GiB on a
31.37 GiB card), so the queue is sequential for this ladder.

*(result to follow)*

---

## A CONFOUND FOUND AND REMOVED — why FLOOR_A is being re-run

`FLOOR_A` started at `00:04:18Z`. `_sdpa` was switched from `enable_gqa=True` to explicit
`repeat_interleave` expansion **after** that, and `EARLYROOT_C` ran with the expansion. That is a
difference in the ATTENTION IMPLEMENTATION in addition to the root pathway — exactly the kind of
confound the whole A-vs-C design exists to avoid.

Measured, not assumed (`test_gqa_identity.py`, full 24-layer trunk, same weights, same batch):

```
enable_gqa=True  vs  explicit repeat_interleave
  torch.equal  : False
  max|d|       : 2.0748e-01
  rms  |d|     : 4.0757e-03      (reference rms 1.0021)
```

**2.07e-01 — the same magnitude as the eager-vs-SDPA difference.** So `FLOOR_A` is confounded and
`FLOOR_A_V2` re-runs it with the frozen trainer (md5 `b1a936b3…`, identical attention code to every
other arm in the ladder). `FLOOR_A` is retained in the record as the confounded run.

The generator also had a silent escaping bug worth recording: the `verify_dispatch` block was emitted
with literal `\n` instead of newlines, so it became **one comment line** — it was present as *text*,
the string-presence assertion passed, and the code never ran. The generator now asserts on the
**parsed AST** that a real call site exists (`verify_dispatch is LIVE CODE: 1 call site(s)`).

---

## EARLY SIGNAL — arm C at step 1000

```
[earlyroot_C eval @1000] ALL_val acc@1 18.87%  acc@5 33.61%  CE_z 6.1696  (n=18869)
                         NOVEL_only acc@1 ...
```

At its **first** 1 000-step eval, arm C is at **18.87 %** — while arm A needed all 20 000 steps to
reach 7.74 %. Stated as measured, with no trajectory extrapolated: the early-root pathway has a very
large effect on this metric. The final 20 000-step number follows.

---

# THE HEADLINE — A vs C at matched steps

| | **A** FLOOR_A | **C** EARLYROOT_C |
|---|---|---|
| root pathway | none | residual injector **+** score bias, layers **0-23** |
| steps | 20 000 | 20 000 |
| trunk LR | 1e-3 (`--trunk-lr-scale 1.0`) | 1e-3 (identical) |
| batch | 32 | 32 |
| wall clock | 83.5 min | ~137 min |
| peak VRAM | 13.98 GiB | 19.18 GiB |
| **ALL_val acc@1 (final)** | **7.74 %** | **18.99 %** |
| **NOVEL_only acc@1 (final)** | **7.63 %** | **19.22 %** |
| acc@5 | 12.30 % | 32.22 % |
| ce_z | 9.2216 | 7.1092 |
| best ALL_val | 8.14 % @4k | 20.00 % @3k |

**C − A = +11.25 pp on ALL_val, +11.59 pp on NOVEL_only.**

## Against all five comparators

| comparator | ALL | NOVEL / unseen | vs C (18.99 %) |
|---|---|---|---|
| **19.53 %** RCA_UNFREEZE_A2 — late-attached (layers 20-23), **0.01× LR, frozen for most of training** | 19.53 % | 19.90 % | **C is −0.54 pp** |
| 29.30 % raw-base probe — root decodability, original Qwen2.5-0.5B | 29.30 % | 23.34 % | different metric, see below |
| 92.34 % transmuted probe — root decodability, current trunk | 92.34 % | 83.92 % | different metric, see below |
| 3.551 % marginal | 3.551 % | — | C is +15.44 pp |
| 2.82 % RCA ablated | 2.82 % | — | C's own RCA ablation is 6.38 % |

**Answer to the headline question: C does not beat 19.53 % — it lands 0.54 pp short of it, i.e.
"about the same", while beating the floor by 11.25 pp.** And because `LATE_X` (the same two
mechanisms on layers 20-23 at the same LR and step count) is still queued, the direct
early-vs-late sentence is not yet written; what is established is that early attachment with a
*normal-LR, unfrozen* trunk reaches the same place late attachment reached with a *frozen* trunk at
0.01× LR.

## The mechanism split — C's own causal ablations (same trunk weights)

| evaluation | ALL_val acc@1 | NOVEL_only |
|---|---|---|
| C, both mechanisms | **18.99 %** | 19.22 % |
| C, **residual injector ablated** (`ALL_val_RCA_OFF`) | **6.38 %** | 6.19 % |
| C, **score bias ablated** (`ALL_val_ISHTIQAQ_OFF`) | **18.81 %** | 19.05 % |

**The two mechanisms are NOT complementary in effect.** Ablating the 57.86 M residual injector
collapses accuracy by **−12.61 pp**; ablating the 11.01 M native score bias moves it by
**−0.18 pp**. This is the answer to the open question about the score-bias path: it finally received
gradients (it had `grad == 0.0` in all 24 layers on the old trunk), and it still does essentially
nothing on this trunk. The residual injector is the whole effect.

Two further things fall out of the ablation table:

* **`RCA_OFF` = 6.38 % is BELOW arm A's 7.74 %.** A trunk trained *with* the pathway and then
  ablated is worse than a trunk trained without one. The trunk came to **depend** on the pathway —
  it is load-bearing, not additive.
* The score-bias ablation being within noise is consistent with `--pillar-freeze`: the pillar
  terms are structurally off, so the only thing ablated is the two root score terms, and they
  carry no weight.

## The representation — length-1 ridge probe (the 92.34 % comparator's protocol)

Same probe, same cache, same split (it reproduces 92.34 %/83.92 % exactly on the released trunk):

| trunk | val_all | Δ vs released | val_unseen_words | Δ vs released |
|---|---|---|---|---|
| released trunk (baseline) | 92.34 % | — | 83.92 % | — |
| **A** (no pathway, step 20000) | 87.19 % | **−5.15 pp** | 74.54 % | **−9.38 pp** |
| **C** (pathway OFF at probe time) | 90.02 % | −2.32 pp | 78.87 % | −5.05 pp |
| **C** (pathway ACTIVE, as trained) | **90.64 %** | **−1.70 pp** | **80.16 %** | **−3.76 pp** |

Normal-LR trunk training **damages** root decodability — but the early root pathway roughly *halves*
the damage (−1.70 vs −5.15 pp on val_all, −3.76 vs −9.38 pp on unseen words), and it helps even
when probed with the pathway off (90.02 % vs 87.19 %), i.e. the benefit is baked into the weights.
The pathway is not just a readout: it **anchors the representation**.

## Training-loss note

By ~step 16 000 arm C's training loss is `0.0000` (root/wazn/prefix/suffix all 0.000) with gradient
norms ~1e-4: 3 000 training windows are fully memorised. Arm A's loss at step 18 000 was still
`0.0248`. The validation number is therefore the only informative one — and arm C both memorises
*better* and generalises *better*, which is not what overfitting looks like.

---

# FLOOR_A_V2 — the de-confounded floor arm — COMPLETE

| | FLOOR_A (old impl, confounded) | **FLOOR_A_V2 (current impl)** |
|---|---|---|
| final ALL_val acc@1 | 7.74 % | **8.04 %** |
| NOVEL_only | 7.63 % | **7.87 %** |
| acc@5 | 12.30 % | 12.37 % |
| ce_z | 9.2216 | 9.1728 |
| best ALL | 8.14 % @4k | 8.26 % @4k |
| wall clock | 83.5 min | **68.5 min** (`03:50:22Z` → `04:58:54Z`) |
| peak VRAM | 13.98 GiB | **12.27 GiB** |
| ridge val_all / unseen | 87.19 % / 74.54 % | **87.37 % / 74.73 %** |

**The attention-implementation confound changed the floor by +0.30 pp.** Both runs plateau flat at
7.7-8.3 % over 20 000 steps. The floor result is therefore robust, and `FLOOR_A_V2` is the arm used
in the headline table.

## Peak VRAM per arm (from the runner's per-process snapshot)

| arm | peak VRAM |
|---|---|
| FLOOR_A_V2 | 12.27 GiB |
| FLOOR_A | 13.98 GiB (live measurement) |
| RESIDUAL_R | 17.13 GiB |
| EARLYROOT_C | **19.20 GiB** |

Two arms cannot coexist at batch 32 under `--unfreeze-trunk-all`; the 20 GiB gate keeps the ladder
sequential, and never more than one arm is resident.

---

# Mechanism isolation — E (residual alone) and D (score bias alone)

## E RESIDUAL_R — COMPLETE

```
final step 20000  ALL_val acc@1 18.94 %   NOVEL_only 19.17 %   acc@5 31.91 %   ce_z 7.1024
best ALL 20.13 % @ step 4000
ALL_val_RCA_OFF (its own ablation) 6.49 %
wall clock 111.5 min (04:58:57Z -> 06:50:24Z) | peak VRAM 17.13 GiB
ridge probe (residual active) val_all 90.44 % | unseen 79.53 %
```

### The score bias contributes NOTHING — three independent measurements agree

| measurement | C (both mechanisms) | E (residual alone) | difference |
|---|---|---|---|
| ALL_val acc@1 (20k) | 18.99 % | **18.94 %** | **−0.05 pp** |
| NOVEL_only acc@1 (20k) | 19.22 % | **19.17 %** | **−0.05 pp** |
| ridge val_all | 90.64 % | **90.44 %** | −0.20 pp |
| ridge val_unseen_words | 80.16 % | **79.53 %** | −0.63 pp |
| C's own in-place ablation of the score bias | 18.99 → **18.81 %** | — | −0.18 pp |
| C's own in-place ablation of the residual injector | 18.99 → **6.38 %** | — | **−12.61 pp** |
| E's own in-place ablation of the residual injector | 18.94 → **6.49 %** | — | **−12.45 pp** |

Six numbers, four of them from independent runs, all say the same thing: **the 57.86 M residual
injector is the entire early-root effect; the 11.01 M native score bias is inert.** The score-bias
path finally received gradients for the first time in this project (it was `grad == 0.0` in 24/24
layers on the old trunk) and it still changes nothing measurable.

The full curve for E tracks C almost eval-for-eval (E: 19.28, 19.73, 19.95, 20.13, 18.61, 19.31,
19.55, 19.68, 17.79, 18.37, 18.71, 18.91, 18.82, 18.34, 18.52, 18.82, 18.80, 18.99, 18.95, 18.94),
so this is not an endpoint coincidence.

## D SCOREBIAS_D — IN PROGRESS

If the score bias is inert, D (score bias alone, no residual injector) should land at the FLOOR,
7.7-8.3 %, not near 19 %. Result to follow.

## D SCOREBIAS_D — COMPLETE: the score bias alone IS the floor

```
final step 20000  ALL_val acc@1 7.83 %   NOVEL_only 7.71 %   acc@5 12.20 %   ce_z 9.1824
best ALL 8.06 % @ step 8000
ALL_val_ISHTIQAQ_OFF (its own ablation) 7.68 %
wall clock 81.3 min (06:50:27Z -> 08:11:47Z)
ridge probe (score bias active) val_all 87.36 % | unseen 74.96 %
curve: 7.86, 7.99, 7.95, 7.93, 7.54, 8.01, 8.00, 8.06, 7.88, 7.88, 7.84, 7.87, 7.88, 7.84,
       7.76, 7.86, 7.86, 7.82, 7.81, 7.83   <- flat at the floor for 20 000 steps
```

### The complete mechanism table

| arm | root pathway | ALL_val acc@1 | NOVEL_only | ridge val_all | ridge unseen |
|---|---|---|---|---|---|
| A (FLOOR_A_V2) | **none** | 8.04 % | 7.87 % | 87.37 % | 74.73 % |
| **D** SCOREBIAS_D | **score bias only** (24 layers) | **7.83 %** | 7.71 % | **87.36 %** | **74.96 %** |
| **E** RESIDUAL_R | **residual injector only** (24 layers) | **18.94 %** | 19.17 % | **90.44 %** | **79.53 %** |
| **C** EARLYROOT_C | **both** (24 layers) | **18.99 %** | 19.22 % | **90.64 %** | **80.16 %** |

**D is indistinguishable from A on all four measurements.** The native score bias — the mechanism
that had never received a gradient before this session — does nothing measurable when it is the only
pathway, and nothing measurable when it is ablated out of a working arm. Five independent
measurements agree:

1. D alone (7.83 %) vs A alone (8.04 %): **−0.21 pp**
2. D ridge (87.36 %/74.96 %) vs A ridge (87.37 %/74.73 %): within **0.23 pp**
3. C with the score bias ablated (18.99 → 18.81 %): **−0.18 pp**
4. C vs E, i.e. both vs residual-only (18.99 vs 18.94 %): **−0.05 pp**
5. C vs E ridge (90.64/80.16 vs 90.44/79.53): **−0.20 / −0.63 pp**

**The entire early-root effect is the residual injector.** The two mechanisms are complementary in
*mechanism* (score reweighting vs residual injection) but not in *effect*.

---

# X LATE_X — the direct early-vs-late test — COMPLETE

Both mechanisms, but only on the **late** layers 20-23 (the comparator's schedule) instead of all 24,
at the **same** trunk LR, the same step count and the same batch.

```
root cross-attention attached to trunk layers [20, 21, 22, 23] (9.64M new params)
NATIVE root score bias on trunk layers [20, 21, 22, 23] (1.835M new params)
final step 20000  ALL_val acc@1 15.43 %   NOVEL_only 15.53 %   acc@5 23.08 %   ce_z 7.8997
best ALL 15.78 % @ step 8000
ALL_val_RCA_OFF (ablate the injector)      7.18 %     <- collapses to the floor
ALL_val_ISHTIQAQ_OFF (ablate the score bias) 15.49 %   <- nothing, again
wall clock 79.8 min (08:11:50Z -> 09:31:39Z)
ridge probe (both, layers 20-23) val_all 89.51 % | unseen 78.07 %
curve: 14.32, 15.06, 15.26, 15.26, 15.27, 15.59, 15.73, 15.78, 14.15, 14.79,
       15.10, 14.90, 14.44, 14.72, 14.93, 15.12, 15.28, 15.33, 15.47, 15.43
```

## The full ladder, at matched trunk training and matched steps

| arm | root pathway | layers | **ALL_val acc@1** | NOVEL_only | ridge val_all | ridge unseen |
|---|---|---|---|---|---|---|
| **C** EARLYROOT_C | residual + score bias | **0-23** | **18.99 %** | **19.22 %** | **90.64 %** | **80.16 %** |
| **E** RESIDUAL_R | residual only | **0-23** | **18.94 %** | 19.17 % | 90.44 % | 79.53 % |
| **X** LATE_X | residual + score bias | **20-23** | **15.43 %** | 15.53 % | 89.51 % | 78.07 % |
| A FLOOR_A_V2 | none | — | 8.04 % | 7.87 % | 87.37 % | 74.73 % |
| D SCOREBIAS_D | score bias only | 0-23 | 7.83 % | 7.71 % | 87.36 % | 74.96 % |
| A' FLOOR_A | none | — | 7.74 % | 7.63 % | 87.19 % | 74.54 % |

**Early beats late by +3.56 pp ALL_val, +3.69 pp NOVEL_only, +1.13 pp ridge val_all and
+2.09 pp ridge unseen — at matched trunk training and matched steps.** That is the direct answer to
the question the session exists to ask.

---

# FINAL VERDICT

## 1. Does early root structure beat late attachment?

**Yes: 18.99 % vs 15.43 %, +3.56 pp**, everything except the attachment layer held identical —
same trunk LR (1e-3, `--trunk-lr-scale 1.0`), same 20 000 steps, same batch 32, same checkpoint, same
head, same seeds procedure, same attention implementation. The mechanism table shows the whole of
that difference comes from the residual injector (the score bias contributes ~0 either way).

## 2. Is a normally-trained trunk all that was ever needed?

**No.** The floor arm trains the entire trunk at the head's LR with **no** root pathway and reaches
**8.04 %** — less than half of early-root, plateaued flat from step 4 000, BEST 8.26 %. And the
length-1 ridge probe shows that this training **damages** the trunk's root structure:

| trunk | val_all | Δ vs released | val_unseen | Δ vs released |
|---|---|---|---|---|
| released trunk | 92.34 % | — | 83.92 % | — |
| A, normally trained, no pathway | 87.37 % | **−4.97 pp** | 74.73 % | **−9.19 pp** |
| D, score bias only | 87.36 % | −4.98 pp | 74.96 pp | −8.96 pp |
| X, late attachment | 89.51 % | −2.83 pp | 78.07 pp | −5.85 pp |
| E, early attachment (residual) | 90.44 % | −1.90 pp | 79.53 pp | −4.39 pp |
| **C, early attachment (both)** | **90.64 %** | **−1.70 pp** | **80.16 pp** | **−3.76 pp** |

A normally-trained trunk is a floor, and it is worse than the trunk it started from.

## 3. What about the 19.53 % comparator?

**C lands 0.54 pp short of 19.53 % — a match, not a win.** But the comparator's recipe was
late attachment on a trunk that was **frozen for most of training at 0.01× LR**, and that is now the
interesting part:

```
19.53 %   late attachment (20-23), 0.01x LR, trunk FROZEN for most of training   <- the comparator
15.43 %   late attachment (20-23), 1.0x  LR, trunk TRAINED                       <- X, this session
18.99 %   early attachment (0-23), 1.0x  LR, trunk TRAINED                       <- C, this session
```

**The same late schedule scores 4.10 pp WORSE when the trunk is actually trained at a normal LR.**
The frozen trunk was not a handicap — it was protection. Full-LR training moves the trunk away from
the representation the late pathway needs, and a pathway attached only at layers 20-23 cannot get it
back. Early attachment can, because the root signal is still present at the input stage and the
pathway keeps it present on the way down.

So the sentence this session was trying to write is:

> **The late RCA's 19.53 % ceiling was never about early-vs-late placement; it was about what
> full-rate training does to the representation. Attach the root pathway early and the trunk can be
> trained normally at 1e-3 without losing the structure — and you get the same number without
> freezing anything. Attach it late and train the trunk anyway, and you lose 4 points.**

## 4. Mechanism: the score bias is inert

Five independent measurements, four of them from separate training runs:

| # | measurement | result |
|---|---|---|
| 1 | D (score bias alone) vs A (nothing) | 7.83 % vs 8.04 % — **−0.21 pp** |
| 2 | D ridge vs A ridge | 87.36/74.96 vs 87.37/74.73 — **≤0.23 pp** |
| 3 | C with the score bias ablated in place | 18.99 → 18.81 % — **−0.18 pp** |
| 4 | C (both) vs E (residual only) | 18.99 vs 18.94 % — **−0.05 pp** |
| 5 | C ridge vs E ridge | 90.64/80.16 vs 90.44/79.53 — **−0.20/−0.63 pp** |
| — | C with the RESIDUAL INJECTOR ablated | 18.99 → **6.38 %** — **−12.61 pp** |
| — | X with the RESIDUAL INJECTOR ablated | 15.43 → **7.18 %** — **−8.25 pp** |

The native score-bias path received gradients for the first time in this project and changed nothing
measurable. **The early-root effect is entirely the 57.86 M residual injector.**

## 5. Cost

| arm | wall clock | peak VRAM | it/s |
|---|---|---|---|
| FLOOR_A | 83.5 min | 13.98 GiB | 5 |
| FLOOR_A_V2 | 68.5 min | 12.27 GiB | 5 |
| SCOREBIAS_D | 81.3 min | ~14.3 GiB | 4 |
| LATE_X | 79.8 min | 13.42 GiB | 4 |
| RESIDUAL_R | 111.5 min | 17.13 GiB | 3 |
| EARLYROOT_C | 137 min | 19.20 GiB | 2.4 |

Two arms never coexist: `--unfreeze-trunk-all` at batch 32 needs 12-19 GiB per arm on a 31.37 GiB
card, and the queue's 20 GiB gate enforces one at a time.
