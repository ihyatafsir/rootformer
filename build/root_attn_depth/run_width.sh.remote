#!/bin/bash
# run_width.sh -- ONE arm of the WIDTH / RESIDUAL-BOUND experiment on the frozen trunk.
#
# Everything is IDENTICAL to the parent's clean single-flag control
#   /workspace/root_attn_ctl/run_frozen_ctl.sh  (tag RCA_FROZEN_CTL, 16.68 % @10k)
# except the two levers under test, which are turned on per-arm by env vars:
#
#   RCA_DIM=0     --rca-dim absent              -> shipped module, d_attn = d_model = 896
#   RCA_DIM=1792  --rca-dim 1792                -> WIDTH LEVER, 2x internal attention width
#   OUT_NORM=1    --rca-out-norm                -> RESIDUAL BOUND
#
# The SHIPPED trainer (/workspace/hf_v19_2_release/nrmt_train.py, md5 56ef3f2f...) and the
# SHIPPED module (/workspace/root_attn/root_cross_attn.py, md5 b53ae702...) are never touched:
# this script runs the generated /workspace/root_attn_width/nrmt_train_width.py, which imports
# the generated /workspace/root_attn_width/root_cross_attn_width.py, both proven bit-identical
# at the default width by equiv_check.py.
#
# Usage (detached):
#   TAG=RCA_W2 RCA_DIM=1792 OUT_NORM=0 setsid nohup bash run_width.sh &
set -u
cd /workspace/hf_v19_2_release || exit 90
OUT=${OUTDIR:-/workspace/root_attn_width}
TAG=${TAG:?TAG is required}
RCA_DIM=${RCA_DIM:-0}
OUT_NORM=${OUT_NORM:-0}
STEPS=${STEPS:-20000}
EVAL_EVERY=${EVAL_EVERY:-1000}
mkdir -p "$OUT" "$OUT/ckpt" "$OUT/logs"

CK=checkpoints/rootformer_v19_2_synthesis_ar_backbone.awzan142.roots9490.tok10052.safetensors
PY=/workspace/venvs/rootformer/bin/python
CACHE=/workspace/head_fix/nrmp_cache_9490_aligned

NORM_FLAG=""
[ "$OUT_NORM" = "1" ] && NORM_FLAG="--rca-out-norm"

echo "[width] tag=$TAG rca_dim=$RCA_DIM out_norm=$OUT_NORM steps=$STEPS eval_every=$EVAL_EVERY"
echo "[width] out=$OUT  starts=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[width] host=$(hostname) pytorch=$(nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader)"
echo "[width] md5 trainer=$(md5sum nrmt_train.py | cut -d' ' -f1) module=$(md5sum /workspace/root_attn/root_cross_attn.py | cut -d' ' -f1)"
T0=$(date +%s)

# ---- peak-VRAM poller: total, and THIS arm's own CUDA process ------------------------
( T=0; A=0
  while true; do
    V=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    [ -n "${V:-}" ] && [ "$V" -gt "$T" ] 2>/dev/null && T=$V
    M=0
    for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null); do
      if tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | grep -q "tag $TAG"; then
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

# ---- per-eval snapshotter of the RCA payload ----------------------------------------
( L=""
  while true; do
    S=$(grep -oE '\[eval @[0-9]+' $OUT/logs/train_${TAG}.log 2>/dev/null | tail -1 | grep -oE '[0-9]+$')
    if [ -n "${S:-}" ] && [ "$S" != "$L" ]; then
      for i in $(seq 1 40); do
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

PYTHONPATH=/workspace/hf_v19_2_release:/workspace/root_attn_width \
$PY /workspace/root_attn_width/nrmt_train_width.py \
    --checkpoint $CK --cache $CACHE --head-init remap \
    --steps $STEPS --batch-size 32 --eval-every $EVAL_EVERY --tag $TAG \
    --probe-out $OUT/trace_${TAG}.jsonl --out $OUT/results_${TAG}.json \
    --save $OUT/head_${TAG}.pt --feat-gate --margin-ramp 10 --grad-warmup 10 --logit-scale ln \
    --root-cross-attn top4 --unfreeze-last 0 \
    --trunk-lr-scale 0.1 --rca-lr-scale 1.0 --rca-ablate-eval --h-drift-probe 500 \
    --rca-dim $RCA_DIM $NORM_FLAG \
    > $OUT/logs/train_${TAG}.log 2>&1 < /dev/null
RC=$?
kill $SP $SS 2>/dev/null; wait $SP $SS 2>/dev/null
T1=$(date +%s)
echo "[width] rc=$RC wall_s=$((T1-T0)) ($(( (T1-T0)/60 )) min) peak_vram=$(cat $OUT/vram_${TAG}.txt 2>/dev/null)"
echo "[width] $(grep -E 'trainable parameters|root cross-attention attached|TRUNK LIVE|RCA init' $OUT/logs/train_${TAG}.log | tr '\n' ' ')"
echo "[width] non-finite lines in trace: $(grep -ciE 'nan|inf' $OUT/trace_${TAG}.jsonl 2>/dev/null)"
echo "[width] WARN lines: $(grep -cE 'WARN' $OUT/logs/train_${TAG}.log)"
echo "[width] last log lines:"; tail -4 $OUT/logs/train_${TAG}.log
echo "[width] RESULT:"
$PY -c "
import json
h=json.load(open('$OUT/results_${TAG}.json'))['history'][-1]
print(' final step', h['step'], 'nan_steps', h.get('nan_steps'))
for k in ('ALL_val','ALL_val_RCA_OFF','NOVEL_only','NOVEL_only_RCA_OFF'):
    if k in h:
        v=h[k]; print(f\"  {k}: acc@1 {100*v['acc@1']:.2f}% acc@5 {100*v['acc@5']:.2f}% CE_z {v['ce_z']:.4f} (n={v['n']})\")
print(' rca gates', h.get('rca_gate'), 'h_drift', h.get('h_drift'))
" 2>&1 | tail -8
echo "WIDTH_ARM_DONE $TAG"
