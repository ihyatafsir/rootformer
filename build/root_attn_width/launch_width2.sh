#!/bin/bash
# launch_width2.sh -- pid- and VRAM-gated launch of the WIDTH arm (RCA_W2, --rca-dim 1792).
#
# HISTORY OF THIS FILE: the first launcher (launch_width.sh) aborted its own smoke because it
# redirected the RUNNER's stdout to /dev/null and then looked for the runner's summary line
# ("non-finite lines in trace: N") inside the TRAINER's log.  The smoke itself PASSED -- banner
# `d_attn=1792, heads=8x224, q=(1792,896), k=(1792,448), o=(896,1792)`, 19.27M new params,
# rc=0, an eval at step 15, no traceback.  This version writes the runner's stdout to a file and
# checks the TRACE directly.
#
# GATING: the GPU is shared with the parent's concurrently running arms.  Never start a third
# 32-batch arm: wait until at most GATE_MAX_OTHERS of the gated tags remain AND free VRAM is
# above VRAM_MIN_MIB.
#
# env: TAG (RCA_W2), RCA_DIM (1792), OUT_NORM (0), STEPS (20000), VRAM_MIN_MIB (11500),
#      GATE_MAX_OTHERS (0), GATE_TAGS (space separated patterns; empty = no tag gate)
set -u
OUT=/workspace/root_attn_width
TAG=${TAG:-RCA_W2}
RCA_DIM=${RCA_DIM:-1792}
OUT_NORM=${OUT_NORM:-0}
STEPS=${STEPS:-20000}
VRAM_MIN_MIB=${VRAM_MIN_MIB:-11500}
GATE_MAX_OTHERS=${GATE_MAX_OTHERS:-1}
mkdir -p $OUT/logs
exec >> $OUT/logs/launch_${TAG}.out 2>&1
echo "=== launch_width2.sh start $(date -u +%Y-%m-%dT%H:%M:%SZ) tag=$TAG dim=$RCA_DIM norm=$OUT_NORM ==="
echo "[gate] max_others=$GATE_MAX_OTHERS vram_min=${VRAM_MIN_MIB}MiB (others = ALL nrmt_train processes except tag $TAG)"

# Count EVERY trainer process except this arm.  This is the guard that keeps us from ever
# becoming the third concurrent 32-batch arm: with two others resident, others=2 > 1 and we wait,
# regardless of how much VRAM looks free.
others () {
  ps -eo args= 2>/dev/null | grep -E 'nrmt_train[a-z_]*\.py' | grep -v "tag $TAG" | grep -vc grep
}

wait_for_room () {
  local i=0
  while [ $i -lt 720 ]; do          # up to ~2 h
    local o free
    o=$(others)
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
    if [ "${o:-9}" -le "$GATE_MAX_OTHERS" ] && [ "${free:-0}" -ge "$VRAM_MIN_MIB" ]; then
      echo "[gate] room at $(date -u +%H:%M:%S): other_gated=$o free=${free}MiB"
      return 0
    fi
    [ $((i % 10)) -eq 0 ] && echo "[gate $(date -u +%H:%M:%S)] other_gated=$o free=${free}MiB used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader) -- waiting"
    sleep 10; i=$((i + 1))
  done
  echo "[gate] TIMED OUT waiting for room"; return 1
}
wait_for_room || { echo "LAUNCH_WIDTH_ABORTED"; exit 1; }

# ---- 15-step GPU smoke, runner stdout captured to a file -------------------------------
if [ "${SKIP_SMOKE:-0}" != "1" ]; then
  echo "[smoke] start $(date -u +%H:%M:%S)"
  rm -rf $OUT/smoke; mkdir -p $OUT/smoke/logs
  TAG=RCA_W2_SMOKE RCA_DIM=$RCA_DIM OUT_NORM=$OUT_NORM STEPS=15 EVAL_EVERY=15 \
      OUTDIR=$OUT/smoke bash $OUT/run_width.sh > $OUT/smoke/runner.out 2>&1
  SRC=$?
  SLOG=$OUT/smoke/logs/train_RCA_W2_SMOKE.log
  TR=$OUT/smoke/trace_RCA_W2_SMOKE.jsonl
  echo "[smoke] rc=$SRC"
  grep -E "root cross-attention attached|trainable parameters|TRUNK LIVE|\[proof\] step|\[eval @" \
      "$SLOG" | cut -c1-300
  grep -E "non-finite lines|WARN lines|rc=" $OUT/smoke/runner.out
  BAD=0
  [ "$SRC" != "0" ] && { echo "[smoke] runner rc != 0"; BAD=1; }
  grep -qiE "traceback" "$SLOG" && { echo "[smoke] traceback"; BAD=1; }
  grep -q "\[eval @" "$SLOG" || { echo "[smoke] no eval line"; BAD=1; }
  grep -qE "d_attn=$RCA_DIM" "$SLOG" || { echo "[smoke] banner does not report d_attn=$RCA_DIM"; BAD=1; }
  if [ ! -s "$TR" ]; then echo "[smoke] trace missing/empty"; BAD=1; else
    NF=$(grep -ciE 'nan|inf' "$TR")
    echo "[smoke] nan/inf lines in trace: $NF ($(wc -l < "$TR") rows)"
    [ "$NF" != "0" ] && { echo "[smoke] non-finite trace entries"; BAD=1; }
  fi
  if [ "$BAD" != "0" ]; then
    echo "[smoke] FAILED -- NOT launching.  tail runner.out:"; tail -25 $OUT/smoke/runner.out
    echo "LAUNCH_WIDTH_ABORTED"; exit 1
  fi
  echo "[smoke] PASS (d_attn=$RCA_DIM live on GPU, eval ran, 0 non-finite, no traceback)"
fi

# ---- the real arm ------------------------------------------------------------------------
wait_for_room || { echo "LAUNCH_WIDTH_ABORTED"; exit 1; }
echo "[launch] $TAG at $(date -u +%H:%M:%SZ) free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader)"
TAG=$TAG RCA_DIM=$RCA_DIM OUT_NORM=$OUT_NORM STEPS=$STEPS OUTDIR=$OUT \
  setsid nohup bash $OUT/run_width.sh > $OUT/logs/arm_${TAG}.out 2>&1 < /dev/null &
sleep 120
echo "[launch] banner:"; grep -E "root cross-attention attached|trainable parameters" $OUT/logs/train_${TAG}.log | cut -c1-300
echo "[launch] progress: $(grep -oE 'step [0-9]+/20000' $OUT/logs/train_${TAG}.log | tail -1) $(grep -oE '[0-9.]+ it/s' $OUT/logs/train_${TAG}.log | tail -1)"
echo "LAUNCH_WIDTH_DONE $TAG"
