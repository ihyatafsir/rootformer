#!/bin/bash
# run_ishtiqaq_arm.sh -- RCA_NATIVE_A2: the RCA_UNFREEZE_A2 configuration PLUS the fixed NATIVE
# root score bias, EVERY other flag identical (single variable).
#
#   RCA_UNFREEZE_A2 (comparator, 19.53 % acc@1):
#     --root-cross-attn top4 --unfreeze-last 4 --trunk-lr-scale 0.01 --rca-lr-scale 0.3
#     --rca-out-norm --rca-dropout 0.1 --rca-ablate-eval --h-drift-probe 500
#     --feat-gate --margin-ramp 10 --grad-warmup 10 --logit-scale ln
#     --steps 20000 --batch-size 32 --eval-every 1000 --head-init remap
#   THIS ARM = the above + --ishtiqaq-root-bias top4 --ishtiqaq-root-source shared
#                                 --ishtiqaq-lr-scale 0.3 --ishtiqaq-ablate-eval
#
# The GPU is shared: two 32-batch arms are the measured-safe limit, so this script WAITS for a
# slot (at most one other real trainer) instead of starting a third.  The trainer is invoked by
# an absolute path whose argv still contains `nrmt_train.py --checkpoint`, so the established
# guards (`pgrep -f "nrmt_train[.]py --checkpoint"`) see it.
set -u
REL=/workspace/hf_v19_2_release
OUT=/workspace/ishtiqaq_check
PY=/workspace/venvs/rootformer/bin/python
CK=$REL/checkpoints/rootformer_v19_2_synthesis_ar_backbone.awzan142.roots9490.tok10052.safetensors
CACHE=/workspace/head_fix/nrmp_cache_9490_aligned
TAG=${TAG:-RCA_NATIVE_A2}
STEPS=${STEPS:-20000}
GATE_INIT=${GATE_INIT:-0.0}
SRC=${SRC:-shared}
PAT='nrmt_train[.]py --checkpoint'
MEMLIM=${MEMLIM:-22000}

# Count EVERY resident rootformer trainer, whoever owns it.  The narrow `nrmt_train[.]py`
# pattern misses the sibling trainers on this pod (`nrmt_train_width.py`, `nrmt_train_ewc.py`),
# so counting only that pattern would start a THIRD 32-batch arm by mistake.
count_trainers() {
  ps -eo args= 2>/dev/null \
    | grep -E 'venvs/rootformer/bin/python .*nrmt_train[a-z_]*\.py .*--checkpoint' \
    | grep -cv grep
}
count_gpu_procs() {
  nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c '[0-9]'
}
gpu_mem() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1; }

mkdir -p "$OUT"
echo "[ishtiqaq] start $(date -u +%Y-%m-%dT%H:%M:%SZ) tag=$TAG steps=$STEPS gate_init=$GATE_INIT src=$SRC"

for f in "$OUT/nrmt_train.py" "$OUT/ishtiqaq_root_bias.py"; do
  [ -f "$f" ] || { echo "[ishtiqaq] ABORT missing $f"; exit 90; }
done

# ---- wait for a GPU slot (at most ONE other trainer / one other GPU process) -----------------
SLOT=0
for i in $(seq 1 960); do
  N=$(count_trainers); G=$(count_gpu_procs); MEM=$(gpu_mem)
  if [ "$N" -le 1 ] && [ "$G" -le 1 ] && [ -n "${MEM:-}" ] && [ "$MEM" -lt "$MEMLIM" ]; then
    sleep 20
    N=$(count_trainers); G=$(count_gpu_procs); MEM=$(gpu_mem)
    if [ "$N" -le 1 ] && [ "$G" -le 1 ] && [ "$MEM" -lt "$MEMLIM" ]; then SLOT=1; break; fi
  fi
  sleep 15
done
if [ "$SLOT" -ne 1 ]; then echo "[ishtiqaq] TIMEOUT waiting for a slot"; exit 91; fi
echo "[ishtiqaq] slot free at $(date -u +%H:%M:%S) other_trainers=$N gpu_procs=$G mem=${MEM}MiB after $i polls"

# ---- VRAM peak poller ----------------------------------------------------------------------
( M=0; while true; do
    V=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    [ -n "${V:-}" ] && [ "$V" -gt "$M" ] 2>/dev/null && M=$V
    echo "$M" > "$OUT/vram_${TAG}.txt"
    sleep 2
  done ) &
SP=$!

T0=$(date +%s)
cd "$REL" || exit 92
setsid nohup $PY -u "$OUT/nrmt_train.py" \
    --checkpoint "$CK" --cache "$CACHE" --head-init remap \
    --steps "$STEPS" --batch-size 32 --eval-every 1000 --tag "$TAG" \
    --probe-out "$OUT/trace_${TAG}.jsonl" --out "$OUT/results_${TAG}.json" \
    --save "$OUT/head_${TAG}.pt" \
    --feat-gate --margin-ramp 10 --grad-warmup 10 --logit-scale ln \
    --root-cross-attn top4 --unfreeze-last 4 --trunk-lr-scale 0.01 \
    --rca-lr-scale 0.3 --rca-out-norm --rca-dropout 0.1 --rca-ablate-eval --h-drift-probe 500 \
    --ishtiqaq-root-bias top4 --ishtiqaq-root-source "$SRC" \
    --ishtiqaq-gate-init "$GATE_INIT" --ishtiqaq-pillar-gate-init 0.0 \
    --ishtiqaq-lr-scale 0.3 --ishtiqaq-ablate-eval \
    > "$OUT/train_${TAG}.log" 2>&1 < /dev/null &
P=$!
echo "[ishtiqaq] REMOTE_PID=$P"

while kill -0 $P 2>/dev/null; do
  sleep 60
  L=$(grep -E "^  \[eval" "$OUT/train_${TAG}.log" | tail -1)
  echo "[ishtiqaq $(date -u +%H:%M:%S)] ${L:0:400}"
done
wait $P; RC=$?
kill $SP 2>/dev/null; wait $SP 2>/dev/null
T1=$(date +%s); PEAK=$(cat "$OUT/vram_${TAG}.txt" 2>/dev/null)
echo "[ishtiqaq] rc=$RC wall_s=$((T1-T0)) wall_min=$(( (T1-T0)/60 )) peak_vram_mib=$PEAK"
grep -E 'trainable parameters|TRUNK LIVE|optimizer groups|RCA init|NATIVE root score bias' \
    "$OUT/train_${TAG}.log"
echo "[ishtiqaq] nan/inf steps: $(grep -ciE 'nan|inf' "$OUT/trace_${TAG}.jsonl" 2>/dev/null)"
tail -6 "$OUT/train_${TAG}.log"
echo ISHTIQAQ_ARM_DONE
