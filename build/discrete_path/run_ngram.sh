#!/bin/bash
# run_ngram.sh -- the ONE GPU arm authorised for this work.
#
# Configuration: the best-known arm (ALIGNED_FIX: the aligned cache + the FIX flags
#   --feat-gate --margin-ramp 10 --grad-warmup 10 --logit-scale ln)
# PLUS the DISCRETE root-history pathway (--ngram-features vocab, orders 1..4, d_ng=16).
# The discrete branch is the ONLY addition vs ALIGNED_FIX, so the comparison is single-variable.
#
# Why `vocab` and not `hash`: measure_ngrams.py measured the hashed-table CEILING on the same
# 18,869 val positions -- 2**20 buckets (the value the brief proposed) reaches only 5.54 %,
#  2**22 -> 24.51 %, 2**24 -> 43.53 %, while the collision-free table reaches 53.13 %.  With
#  5,201,346 distinct order-4 contexts, hashing is the wrong side of the trade.
#
# One trainer at a time; never preempts another agent's process.
set -u
cd /workspace/hf_v19_2_release || exit 90
OUT=/workspace/discrete_path; mkdir -p $OUT
CK=checkpoints/rootformer_v19_2_synthesis_ar_backbone.awzan142.roots9490.tok10052.safetensors
PY=/workspace/venvs/rootformer/bin/python
CACHE=/workspace/head_fix/nrmp_cache_9490_aligned
TAG=NGRAM

if [ "$(pgrep -f 'nrmt_train[.]py --checkpoint' | wc -l)" -ne 0 ]; then
  echo "[ngram] ABORT: another nrmt_train.py is already running"; exit 91
fi
echo "[ngram] cache=$CACHE"
echo "[ngram] ckpt=$CK"
echo "[ngram] starts=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
T0=$(date +%s)

( M=0; while true; do
    V=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    [ -n "${V:-}" ] && [ "$V" -gt "$M" ] 2>/dev/null && M=$V
    echo "$M" > $OUT/vram_${TAG}.txt
    sleep 2
  done ) &
SP=$!

setsid nohup $PY nrmt_train.py --checkpoint $CK --cache $CACHE --head-init remap \
    --steps 20000 --batch-size 32 --eval-every 1000 --tag $TAG \
    --feat-gate --margin-ramp 10 --grad-warmup 10 --logit-scale ln \
    --ngram-features vocab --ngram-orders 1,2,3,4 --ngram-dim 16 \
    --probe-out $OUT/trace_${TAG}.jsonl --out $OUT/results_${TAG}.json \
    --save $OUT/head_${TAG}.pt > $OUT/train_${TAG}.log 2>&1 < /dev/null &
P=$!
echo "[ngram] REMOTE_PID=$P"

while kill -0 $P 2>/dev/null; do
  sleep 45
  L=$(grep -E "^  \[eval|^  step" $OUT/train_${TAG}.log | tail -1)
  echo "[ngram $(date -u +%H:%M:%S)] ${L:0:200}"
done
wait $P; RC=$?
kill $SP 2>/dev/null; wait $SP 2>/dev/null
T1=$(date +%s)
echo "[ngram] rc=$RC wall_s=$((T1-T0)) peak_vram_mib=$(cat $OUT/vram_${TAG}.txt 2>/dev/null)"
grep -E 'trainable parameters|ngram vocab|DISCRETE arm|training windows' $OUT/train_${TAG}.log
tail -5 $OUT/train_${TAG}.log
echo "[ngram] FINAL:"
$PY - <<'PYEOF'
import json, glob
for p in sorted(glob.glob('/workspace/discrete_path/results_NGRAM.json')):
    h = json.load(open(p))['history'][-1]
    print(' step', h['step'])
    for k in ('ALL_val', 'NOVEL_only'):
        v = h[k]
        print(f"  {k}: acc@1 {100*v['acc@1']:.2f}% acc@5 {100*v['acc@5']:.2f}% "
              f"CE_z {v['ce_z']:.4f} (n={v['n']})")
    print('  ngram_gate', h.get('ngram_gate'), 'ngram_scale', h.get('ngram_scale'),
          'feat_scale', h.get('feat_scale'))
PYEOF
echo NGRAM_DONE
