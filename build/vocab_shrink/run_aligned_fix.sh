#!/bin/bash
# run_aligned.sh -- the item-1 arm: EXACT CONTROL configuration, only --cache changed to the
# index-aligned rebuild.  Directly comparable to CONTROL 5.72 / GUARD 6.63 / FIX 6.84.
set -u
cd /workspace/hf_v19_2_release || exit 90
OUT=/workspace/head_fix; mkdir -p $OUT
CK=checkpoints/rootformer_v19_2_synthesis_ar_backbone.awzan142.roots9490.tok10052.safetensors
PY=/workspace/venvs/rootformer/bin/python
CACHE=/workspace/head_fix/nrmp_cache_9490_aligned
TAG=ALIGNED_FIX

# never run two trainers at once
if [ "$(pgrep -f 'nrmt_train[.]py --checkpoint' | wc -l)" -ne 0 ]; then
  echo "[aligned] ABORT: another nrmt_train.py is already running"; exit 91
fi

echo "[aligned] cache=$CACHE"
echo "[aligned] starts=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
T0=$(date +%s)

# peak-VRAM poller (the only GPU process by construction)
( M=0; while true; do
    V=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    [ -n "${V:-}" ] && [ "$V" -gt "$M" ] 2>/dev/null && M=$V
    echo "$M" > $OUT/vram_ALIGNED.txt
    sleep 1
  done ) &
SP=$!

setsid nohup $PY nrmt_train.py --checkpoint $CK --cache $CACHE --head-init remap \
    --steps 20000 --batch-size 32 --eval-every 1000 --tag $TAG \
    --probe-out $OUT/trace_${TAG}.jsonl --out $OUT/results_${TAG}.json \
    --save $OUT/head_${TAG}.pt --feat-gate --margin-ramp 10 --grad-warmup 10 --logit-scale ln > $OUT/train_${TAG}.log 2>&1 < /dev/null &
P=$!
echo "[aligned] REMOTE_PID=$P"

while kill -0 $P 2>/dev/null; do
  sleep 45
  L=$(grep -E "^  \[eval|^  step" $OUT/train_${TAG}.log | tail -1)
  echo "[aligned $(date -u +%H:%M:%S)] ${L:0:170}"
done
wait $P; RC=$?
kill $SP 2>/dev/null; wait $SP 2>/dev/null
T1=$(date +%s)
PEAK=$(cat $OUT/vram_ALIGNED.txt 2>/dev/null)
echo "[aligned] rc=$RC wall_s=$((T1-T0)) peak_vram_mib=$PEAK"
echo "[aligned] $(grep -E 'trainable parameters|training windows' $OUT/train_${TAG}.log | tr '\n' ' ')"
tail -6 $OUT/train_${TAG}.log
echo "[aligned] RESULTS:"
$PY -c "
import json
h=json.load(open('$OUT/results_${TAG}.json'))['history'][-1]
print(' final step', h['step'])
for k in ('ALL_val','NOVEL_only'):
    v=h[k]; print(f\"  {k}: acc@1 {100*v['acc@1']:.2f}% acc@5 {100*v['acc@5']:.2f}% CE_z {v['ce_z']:.4f} (n={v['n']})\")
"
echo ALIGNED_DONE
