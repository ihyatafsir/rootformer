#!/bin/bash
# launch_width.sh -- pid-gated launch of ONE width arm.
#
# WHY GATED: the parent's chain_final.sh is ALIVE and has a stage-2 arm queued
#   run_arm RCA_FROZEN_B2 --root-cross-attn top4 --unfreeze-last 0 --rca-lr-scale 0.1 \
#           --rca-out-norm --rca-dropout 0.1 --rca-ablate-eval --h-drift-probe 500
# which fires the instant its stage-1 arm (RCA_UNFREEZE_A2) exits.  Three concurrent 32-batch
# arms do NOT fit in 32.6 GB (peak was 21.5 GB with two), so racing it could OOM and kill the
# parent's own comparator.  This launcher therefore:
#   1. waits for BOTH visible arms (RCA_FROZEN_CTL and RCA_UNFREEZE_A2) to be gone;
#   2. sleeps 30 s so the chain's own stage-2 arm wins the race and takes the GPU;
#   3. runs a 15-step GPU SMOKE of the width code path through the REAL runner (attach /
#      forward / eval / RCA ablation / h_drift) and refuses to continue on error or non-finite;
#   4. launches ONE arm -- the width arm -- concurrently with whatever the chain started.
#      TWO arms is the configuration the parent already measured at 21.5 GB peak.  Never three.
#
# env: TAG (default RCA_W2), RCA_DIM (default 1792), OUT_NORM (default 0), STEPS (default 20000),
#      SKIP_SMOKE=1 to skip step 3.
set -u
OUT=/workspace/root_attn_width
TAG=${TAG:-RCA_W2}
RCA_DIM=${RCA_DIM:-1792}
OUT_NORM=${OUT_NORM:-0}
STEPS=${STEPS:-20000}
mkdir -p $OUT/logs
exec >> $OUT/logs/launch_${TAG}.out 2>&1
echo "=== launch_width.sh start $(date -u +%Y-%m-%dT%H:%M:%SZ) tag=$TAG dim=$RCA_DIM norm=$OUT_NORM ==="

wait_for () {
  while pgrep -f "$1" > /dev/null; do sleep 20; done
  echo "[gate] '$1' gone at $(date -u +%H:%M:%S)"
}
wait_for "tag RCA_FROZEN_CTL"
wait_for "tag RCA_UNFREEZE_A2"
echo "[gate] comparators done at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[gate] sleeping 30 s so the chain's stage-2 arm (RCA_FROZEN_B2) starts first"
sleep 30
echo "[gate] gpu: $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader)"
echo "[gate] concurrent trainer tags: $(ps -eo args= | grep -oE '\-\-tag [A-Za-z0-9_]+' | tr '\n' ' ')"

# ---- 15-step GPU smoke of the WIDTH path, through the real runner ----------------------
if [ "${SKIP_SMOKE:-0}" != "1" ]; then
  echo "[smoke] start $(date -u +%H:%M:%S)"
  rm -rf $OUT/smoke; mkdir -p $OUT/smoke
  TAG=RCA_W2_SMOKE RCA_DIM=$RCA_DIM OUT_NORM=$OUT_NORM STEPS=15 EVAL_EVERY=15 \
      OUTDIR=$OUT/smoke bash $OUT/run_width.sh > /dev/null 2>&1
  SRC=$?
  echo "[smoke] rc=$SRC"
  grep -E "root cross-attention attached|trainable parameters|TRUNK LIVE|\[proof\] step|\[eval @" \
      $OUT/smoke/logs/train_RCA_W2_SMOKE.log | cut -c1-460
  NF=$(grep -oE "non-finite lines in trace: [0-9]+" $OUT/smoke/logs/train_RCA_W2_SMOKE.log \
       | grep -oE "[0-9]+$")
  echo "[smoke] non-finite lines in trace: ${NF:-<not reported>}"
  BAD=0
  [ "$SRC" != "0" ] && { echo "[smoke] runner rc != 0"; BAD=1; }
  [ "${NF:-1}" != "0" ] && { echo "[smoke] non-finite lines present or unreported"; BAD=1; }
  grep -qiE "traceback" $OUT/smoke/logs/train_RCA_W2_SMOKE.log && { echo "[smoke] traceback present"; BAD=1; }
  grep -q "\[eval @" $OUT/smoke/logs/train_RCA_W2_SMOKE.log || { echo "[smoke] no eval line"; BAD=1; }
  if [ "$BAD" != "0" ]; then
    echo "[smoke] FAILED -- NOT launching the long arm.  tail:"
    tail -40 $OUT/smoke/logs/train_RCA_W2_SMOKE.log
    echo "LAUNCH_WIDTH_ABORTED"; exit 1
  fi
  echo "[smoke] PASS (d_attn=$RCA_DIM live on GPU, eval ran, zero non-finite, no traceback)"
fi

# ---- the real arm ----------------------------------------------------------------------
echo "[launch] $TAG at $(date -u +%H:%M:%SZ)"
TAG=$TAG RCA_DIM=$RCA_DIM OUT_NORM=$OUT_NORM STEPS=$STEPS OUTDIR=$OUT \
  setsid nohup bash $OUT/run_width.sh > $OUT/logs/arm_${TAG}.out 2>&1 < /dev/null &
echo "[launch] spawned; waiting 90 s to confirm it is training"
sleep 90
head -22 $OUT/logs/train_${TAG}.log
echo "LAUNCH_WIDTH_DONE $TAG"
