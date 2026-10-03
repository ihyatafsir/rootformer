# Clean re-evaluation protocol for the arm ladder (CPU/GPU, no new code)

## Why

Two defects make the arms' own logged `acc@1` an unfair C-vs-X comparison:

1. **`rca_stack.eval()` is never called.** `head = model.nrmt_head` (`nrmt_train_v13_sdpa.py:791`)
   and `model.root_cross = rca_stack` (`:576`) are SIBLINGS under `model`, so `head.eval()` at
   `:955` cannot reach the stack. Each layer has `self.drop = nn.Dropout(dropout)`
   (`root_cross_attn.py:114`), and the arms pass `--rca-dropout 0.1`. **EARLYROOT_C therefore
   evaluates with 24 active dropout modules and LATE_X with 4** — so C's eval is ~6x noisier than
   X's. Expectation is unchanged (Dropout scales by 1/(1-p) in train mode), so this is a VARIANCE
   asymmetry, not a bias.
2. **Both arms are unseeded.** `--seed` defaults to `-1` and `runner.sh` passes none
   (`grep -c "seeded run"` == 0 in both logs). Train-mode dropout also consumes the global RNG, so
   every eval perturbs the training trajectory — differently for 24 vs 4 modules.

## The fix, using only existing flags

Run the trainer for **one step** on each arm's saved trunk, with dropout OFF and a FIXED seed.
`--rca-dropout 0.0` makes point 1 vacuous without touching any code.

```bash
PY=/workspace/venvs/rootformer/bin/python
TR=/workspace/root_arch/nrmt_train_v13_sdpa.py
CACHE=/workspace/head_fix/nrmp_cache_9490_aligned
COMMON="--model-v13 --flash-hooks off --sdpa --pillar-freeze \
  --cache $CACHE --head-init remap --unfreeze-trunk-all --trunk-lr-scale 1.0 \
  --lr 1e-3 --steps 1 --batch-size 32 --eval-every 1 \
  --feat-gate --feat-gate-proj-std 1e-3 --margin-ramp 10 --grad-warmup 10 \
  --logit-scale ln --grad-clip 1.0 --hist 3 --dropout 0.1 \
  --rca-out-norm --rca-heads 8 --h-drift-probe 0 \
  --rca-dropout 0.0 --seed 42 --rca-ablate-eval --ishtiqaq-ablate-eval"

# EARLYROOT_C (both mechanisms, all 24 layers)
$PY $TR $COMMON --root-cross-attn all \
  --ishtiqaq-root-bias all --ishtiqaq-root-source shared \
  --ishtiqaq-gate-init 0.0 --ishtiqaq-pillar-gate-init 0.0 --ishtiqaq-lr-scale 1.0 \
  --checkpoint /tmp/root_arch_arms/head_EARLYROOT_C.pt.trunk.pt \
  --tag REEVAL_C --out /workspace/root_arch/arms/reeval_C.json

# LATE_X (identical, only the two placement flags differ)
$PY $TR $COMMON --root-cross-attn top4 \
  --ishtiqaq-root-bias top4 --ishtiqaq-root-source shared \
  --ishtiqaq-gate-init 0.0 --ishtiqaq-pillar-gate-init 0.0 --ishtiqaq-lr-scale 1.0 \
  --checkpoint /tmp/root_arch_arms/head_LATE_X.pt.trunk.pt \
  --tag REEVAL_X --out /workspace/root_arch/arms/reeval_X.json
```

## Two things to know before trusting the numbers

* **`--steps 1` is ONE REAL TRAINING STEP, not a pure re-eval.** The eval block sits at `:1154`,
  *after* `opt.step()` at `:1103-1118`. That is acceptable here because it is applied
  **identically** to both arms, so the comparison stays matched — but do not describe it as
  "no training".
* **Compare WITHIN-arm ablation deltas, not the raw cross-arm diff.**
  `Δ_arm = acc@1(arm) - acc@1(arm_RCA_OFF)` is that arm's own causal pathway contribution. C and X
  train *different trunks* from the same init, so `acc@1(C) - acc@1(X)` conflates pathway placement
  with trunk divergence. `--rca-ablate-eval` / `--ishtiqaq-ablate-eval` produce the OFF numbers in
  the same run (fetch them from the log; they are not always in the JSON).

## Read the tie correctly

If both Δ are ~0, the pathway contributes nothing and the placement question is void. And if both
arms land at FLOOR_A's level (~7.7-8.1 %), that is **"both failed"**, not "placement does not
matter" — a tie at a no-pathway arm's level means there is no capability whose placement could
matter. FLOOR_A itself has no head-independent control and trains its trunk at 10x the LR the repo
documents as trunk-destroying, so it is not yet a certified floor either.

Residual limitation that no flag fixes: with one seed per arm, a small gap still carries
seed-to-seed training variance. Settling the placement question properly needs >=2 seeds per arm.
