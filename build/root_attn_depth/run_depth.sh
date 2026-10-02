#!/bin/bash
# run_depth.sh -- ONE DEPTH arm of the root cross-attention experiment.
#
# SINGLE VARIABLE vs the parent's clean control /workspace/root_attn_ctl/run_frozen_ctl.sh
# (RCA_FROZEN_CTL, trunk FROZEN, --root-cross-attn top4 = layers [20,21,22,23], 16.5 %):
#
#     --root-cross-attn top4   ->   --root-cross-attn top8   (layers [16..23])
#     (RCA=top12 -> layers [12..23] for the second budget point)
#
# Everything else is byte-identical to run_frozen_ctl.sh: same checkpoint, same aligned cache,
# --steps 20000, --batch-size 32, --eval-every 1000, --head-init remap, the FIX flags
# (--feat-gate --margin-ramp 10 --grad-warmup 10 --logit-scale ln), --unfreeze-last 0,
# --trunk-lr-scale 0.1, --rca-lr-scale 1.0, NO --rca-out-norm, NO --rca-dropout, NO --rca-dim,
# --rca-ablate-eval --h-drift-probe 500.
#
# NO MODULE EDIT: `top8`/`top12` are already parsed by the SHIPPED
# /workspace/root_attn/root_cross_attn.py parse_layer_spec (md5 b53ae702bde947ac66f933191f904c36)
# driven by the SHIPPED /workspace/hf_v19_2_release/nrmt_train.py (md5 56ef3f2fd9ddc352dee596f32590e7b5).
# Both are READ-ONLY here.  Nothing under /workspace/root_attn or /workspace/hf_v19_2_release is written.
#
# GPU GATE (durable): waits until (a) at most GATE_MAX_OTHERS other nrmt_train arms are running and
# (b) free VRAM >= GATE_MIN_FREE_MIB, verified twice GATE_RECHECK_S apart.  Two concurrent 32-batch
# arms is the parent's measured-safe configuration (~21.5 GB of 32.6 GB); three risks OOM, so the
# gate never starts a third.  Another agent's processes are NEVER killed.
#
# env: TAG (req) RCA (top8) STEPS (20000) EVAL_EVERY (1000) OUTDIR (/workspace/root_attn_depth)
#      GATE_MAX_OTHERS (1) GATE_MIN_FREE_MIB (12000) GATE_RECHECK_S (10) SKIP_GATE (0)
set -u
cd /workspace/hf_v19_2_release || exit 90
OUT=${OUTDIR:-/workspace/root_attn_depth}
TAG=${TAG:?TAG required}
RCA=${RCA:-top8}
STEPS=${STEPS:-20000}
EVAL_EVERY=${EVAL_EVERY:-1000}
GATE_MAX_OTHERS=${GATE_MAX_OTHERS:-1}
GATE_MIN_FREE_MIB=${GATE_MIN_FREE_MIB:-12000}
GATE_RECHECK_S=${GATE_RECHECK_S:-10}
mkdir -p "$OUT" "$OUT/ckpt" "$OUT/logs"

CK=checkpoints/rootformer_v19_2_synthesis_ar_backbone.awzan142.roots9490.tok10052.safetensors
PY=/workspace/venvs/rootformer/bin/python
CACHE=/workspace/head_fix/nrmp_cache_9490_aligned

other_arms () {   # count nrmt_train arms whose --tag is NOT ours
  local n=0 c
  for p in $(pgrep -f 'nrmt_train[a-z_]*\.py' 2>/dev/null); do
    c=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null)
    case "$c" in
      *"--tag $TAG"*) : ;;
      *"--tag "*)     n=$((n+1)) ;;
    esac
  done
  echo "$n"
}
free_mib () {
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' '
}

echo "[depth] tag=$TAG rca=$RCA steps=$STEPS eval_every=$EVAL_EVERY out=$OUT"
echo "[depth] start=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[depth] md5 trainer=$(md5sum nrmt_train.py | cut -d' ' -f1) module=$(md5sum /workspace/root_attn/root_cross_attn.py | cut -d' ' -f1)"
echo "[depth] gpu=$(nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader)"

# ---- durable gate ---------------------------------------------------------------------
if [ "${SKIP_GATE:-0}" != "1" ]; then
  LASTLOG=0
  while true; do
    O=$(other_arms); F=$(free_mib)
    if [ "${O:-9}" -le "$GATE_MAX_OTHERS" ] 2>/dev/null && [ -n "${F:-}" ] && [ "$F" -ge "$GATE_MIN_FREE_MIB" ] 2>/dev/null; then
      sleep "$GATE_RECHECK_S"
      O2=$(other_arms); F2=$(free_mib)
      if [ "${O2:-9}" -le "$GATE_MAX_OTHERS" ] 2>/dev/null && [ -n "${F2:-}" ] && [ "$F2" -ge "$GATE_MIN_FREE_MIB" ] 2>/dev/null; then
        echo "[gate] PASS at $(date -u +%H:%M:%S): other_arms=$O2 free=${F2}MiB (>=${GATE_MIN_FREE_MIB})"
        break
      fi
      echo "[gate] recheck failed: other_arms=$O2 free=${F2}MiB -- still waiting"
    fi
    N=$(date +%s)
    if [ $((N - LASTLOG)) -ge 120 ]; then
      echo "[gate $(date -u +%H:%M:%S)] waiting: other_arms=$O free=${F}MiB running=[$(ps -eo args= | grep -oE '\-\-tag [A-Za-z0-9_]+' | tr '\n' ' ')]"
      LASTLOG=$N
    fi
    sleep 15
  done
fi

T0=$(date +%s)

# ---- peak-VRAM poller: total GPU peak, and THIS arm's own CUDA-process peak -------------
( T=0; A=0
  while true; do
    V=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    [ -n "${V:-}" ] && [ "$V" -gt "$T" ] 2>/dev/null && T=$V
    M=0
    for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null); do
      if tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | grep -q -- "--tag $TAG"; then
        U=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null \
            | awk -F', ' -v q="$p" '$1==q{print $2}')
        [ -n "${U:-}" ] && M=$((M + U))
      fi
    done
    [ "$M" -gt "$A" ] 2>/dev/null && A=$M
    echo "total=$T arm=$A" > $OUT/vram_${TAG}.txt
    sleep 2
  done ) &
SP=$!

# ---- per-eval snapshot of the RCA payload (the trainer overwrites .trunk.pt every eval) --
( L=""
  while true; do
    S=$(grep -oE '\[eval @[0-9]+' $OUT/logs/train_${TAG}.log 2>/dev/null | tail -1 | grep -oE '[0-9]+$')
    if [ -n "${S:-}" ] && [ "$S" != "$L" ]; then
      for i in $(seq 1 60); do
        a=$(stat -c %s $OUT/head_${TAG}.pt.trunk.pt 2>/dev/null || echo 0)
        sleep 1
        b=$(stat -c %s $OUT/head_${TAG}.pt.trunk.pt 2>/dev/null || echo 0)
        [ "$a" = "$b" ] && [ "$a" != "0" ] && break
      done
      cp -f $OUT/head_${TAG}.pt.trunk.pt $OUT/ckpt/rca_step_${S}.pt 2>/dev/null
      L=$S
    fi
    sleep 5
  done ) &
SS=$!

$PY nrmt_train.py --checkpoint $CK --cache $CACHE --head-init remap \
    --steps $STEPS --batch-size 32 --eval-every $EVAL_EVERY --tag $TAG \
    --probe-out $OUT/trace_${TAG}.jsonl --out $OUT/results_${TAG}.json --save $OUT/head_${TAG}.pt \
    --feat-gate --margin-ramp 10 --grad-warmup 10 --logit-scale ln \
    --root-cross-attn $RCA --unfreeze-last 0 \
    --trunk-lr-scale 0.1 --rca-lr-scale 1.0 --rca-ablate-eval --h-drift-probe 500 \
    > $OUT/logs/train_${TAG}.log 2>&1 < /dev/null &
P=$!
echo "[depth] REMOTE_PID=$P"
while kill -0 $P 2>/dev/null; do
  sleep 60
  echo "[depth $(date -u +%H:%M:%S)] $TAG :: $(grep -E '^  \[eval|^  step|^  \[proof' $OUT/logs/train_${TAG}.log | tail -1 | cut -c1-190)"
done
wait $P; RC=$?
kill $SP $SS 2>/dev/null; wait $SP $SS 2>/dev/null
T1=$(date +%s)
echo "[depth] tag=$TAG rc=$RC wall_s=$((T1-T0)) ($(( (T1-T0)/60 )) min) peak_vram=$(cat $OUT/vram_${TAG}.txt 2>/dev/null)"
echo "[depth] $(grep -E 'trainable parameters|root cross-attention attached|TRUNK LIVE|RCA init' $OUT/logs/train_${TAG}.log | tr '\n' ' ')"
echo "[depth] nan/inf lines in trace: $(grep -ciE 'nan|inf' $OUT/trace_${TAG}.jsonl 2>/dev/null)"
echo "[depth] WARN lines: $(grep -cE 'WARN' $OUT/logs/train_${TAG}.log)"
tail -4 $OUT/logs/train_${TAG}.log
echo "[depth] RESULT $TAG:"
$PY -c "
import json
h=json.load(open('$OUT/results_${TAG}.json'))['history'][-1]
print(' final step', h['step'], 'nan_steps', h.get('nan_steps'))
for k in ('ALL_val','ALL_val_RCA_OFF','NOVEL_only','NOVEL_only_RCA_OFF'):
    if k in h:
        v=h[k]; print(f\"  {k}: acc@1 {100*v['acc@1']:.2f}% acc@5 {100*v['acc@5']:.2f}% CE_z {v['ce_z']:.4f} (n={v['n']})\")
print(' rca_layers', h.get('rca_layers'), 'gates', h.get('rca_gate'), 'h_drift', h.get('h_drift'))
" 2>&1 | tail -8
echo "DEPTH_ARM_DONE $TAG"
