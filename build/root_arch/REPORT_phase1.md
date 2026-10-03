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
