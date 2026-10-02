#!/bin/bash
# run_rootattn.sh -- the ROOT CROSS-ATTENTION + PARTIAL TRUNK UNFREEZE arm.
#
# Base config is EXACTLY run_aligned_fix.sh (the FIX flags on the index-aligned cache):
#   --feat-gate --margin-ramp 10 --grad-warmup 10 --logit-scale ln, --head-init remap,
#   20000 steps, batch 32, same checkpoint.  The ONLY added variables are:
#   --root-cross-attn top4  (h <- h + gate*CrossAttn(Q=Wq h, K=V=Wkv E_root(r_{<=t})))
#   --unfreeze-last 4       (layers 20..23 train at --lr * --trunk-lr-scale)
#   --rca-lr-scale 1.0      (the NEW cross-attention trains at --lr)
#   --rca-ablate-eval       (inference-time causal ablation of the RCA at every eval)
#   --h-drift-probe 500     (proof: trunk h for a fixed batch drifts from step 1)
set -u
cd /workspace/hf_v19_2_release || exit 90
OUT=/workspace/root_attn; mkdir -p $OUT
CK=checkpoints/rootformer_v19_2_synthesis_ar_backbone.awzan142.roots9490.tok10052.safetensors
PY=/workspace/venvs/rootformer/bin/python
CACHE=/workspace/head_fix/nrmp_cache_9490_aligned
TAG=${TAG:-ROOTATTN}
RCA=${RCA:-top4}
UF=${UF:-4}
TRUNK_LR=${TRUNK_LR:-0.1}

if [ "$(pgrep -f 'nrmt_train[.]py --checkpoint' | wc -l)" -ne 0 ]; then
  echo "[rootattn] ABORT: another nrmt_train.py is already running"; exit 91
fi

echo "[rootattn] tag=$TAG rca=$RCA unfreeze_last=$UF trunk_lr_scale=$TRUNK_LR"
echo "[rootattn] cache=$CACHE"
echo "[rootattn] starts=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
T0=$(date +%s)

# peak-VRAM poller (the only GPU process by construction)
( M=0; while true; do
    V=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    [ -n "${V:-}" ] && [ "$V" -gt "$M" ] 2>/dev/null && M=$V
    echo "$M" > $OUT/vram_${TAG}.txt
    sleep 1
  done ) &
SP=$!

setsid nohup $PY nrmt_train.py --checkpoint $CK --cache $CACHE --head-init remap \
    --steps 20000 --batch-size 32 --eval-every 1000 --tag $TAG \
    --probe-out $OUT/trace_${TAG}.jsonl --out $OUT/results_${TAG}.json \
    --save $OUT/head_${TAG}.pt --feat-gate --margin-ramp 10 --grad-warmup 10 --logit-scale ln \
    --root-cross-attn $RCA --unfreeze-last $UF \
    --trunk-lr-scale $TRUNK_LR --rca-lr-scale 1.0 --rca-ablate-eval --h-drift-probe 500 \
    > $OUT/train_${TAG}.log 2>&1 < /dev/null &
P=$!
echo "[rootattn] REMOTE_PID=$P"

while kill -0 $P 2>/dev/null; do
  sleep 60
  L=$(grep -E "^  \[eval|^  step|^  \[proof" $OUT/train_${TAG}.log | tail -1)
  echo "[rootattn $(date -u +%H:%M:%S)] ${L:0:230}"
done
wait $P; RC=$?
kill $SP 2>/dev/null; wait $SP 2>/dev/null
T1=$(date +%s)
PEAK=$(cat $OUT/vram_${TAG}.txt 2>/dev/null)
echo "[rootattn] rc=$RC wall_s=$((T1-T0)) peak_vram_mib=$PEAK"
echo "[rootattn] $(grep -E 'trainable parameters|TRUNK LIVE|optimizer groups|RCA init' $OUT/train_${TAG}.log | tr '\n' ' ')"
echo "[rootattn] nan/inf lines: $(grep -ciE 'nan|inf' $OUT/trace_${TAG}.jsonl 2>/dev/null)"
tail -8 $OUT/train_${TAG}.log
echo "[rootattn] RESULTS:"
$PY -c "
import json
h=json.load(open('$OUT/results_${TAG}.json'))['history'][-1]
print(' final step', h['step'], 'nan_steps', h.get('nan_steps'))
for k in ('ALL_val','ALL_val_RCA_OFF','NOVEL_only','NOVEL_only_RCA_OFF'):
    if k in h:
        v=h[k]; print(f\"  {k}: acc@1 {100*v['acc@1']:.2f}% acc@5 {100*v['acc@5']:.2f}% CE_z {v['ce_z']:.4f} (n={v['n']})\")
print(' rca gates', h.get('rca_gate'), 'h_drift', h.get('h_drift'))
" 2>&1 | tail -8
echo ROOTATTN_DONE
