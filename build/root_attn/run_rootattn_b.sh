#!/bin/bash
# run_rootattn_b.sh -- run B: the ROOT CROSS-ATTENTION ALONE, with the trunk FROZEN.
#
# WHY THIS IS RUN B AND NOT ANOTHER UNFREEZE ARM
# ---------------------------------------------
# Run A (RCA top4 + unfreeze top4) answered the combined question: it is a catastrophic
# FAILURE -- by step 1000 held-out acc@1 was 0.28 % and train root CE had fallen to 1.77
# (frozen baseline: 4.68), i.e. the 84.69 M trainable parameters memorised the 381,000
# training positions outright.  The remaining question is therefore attribution, and the
# single most informative arm is the one that removes the confound:
#
#   RCA top4, trunk FROZEN, head at the SAME 1e-3 as the FIX baseline.
#
# If the RCA can expose root history to the frozen trunk, that is the decisive move the
# eight prior arms could not make, and it cannot be memorisation THROUGH THE TRUNK.
# If it cannot, the structural candidate is dead independently of the unfreeze.
#
# Stabilisers relative to Run A, each with a measured reason:
#   --rca-out-norm   the injected residual is magnitude-bounded (v18fix `feat_norm` principle)
#   --rca-dropout 0.1 the new pathway cannot memorise singleton contexts unpenalised
#   --rca-lr-scale 0.1 the new pathway learns at 1e-4, not at the head's 1e-3
# The trunk is UNTOUCHED (--unfreeze-last 0); the hidden-state cache is still bypassed because
# the RCA changes the trunk's residual stream, so h must be recomputed.
set -u
cd /workspace/hf_v19_2_release || exit 90
OUT=/workspace/root_attn; mkdir -p $OUT
CK=checkpoints/rootformer_v19_2_synthesis_ar_backbone.awzan142.roots9490.tok10052.safetensors
PY=/workspace/venvs/rootformer/bin/python
CACHE=/workspace/head_fix/nrmp_cache_9490_aligned
TAG=${TAG:-RCA_FROZEN}
RCA=${RCA:-top4}

if [ "$(pgrep -f 'nrmt_train[.]py --checkpoint' | wc -l)" -ne 0 ]; then
  echo "[rootattn-b] ABORT: another nrmt_train.py is already running"; exit 91
fi
echo "[rootattn-b] tag=$TAG rca=$RCA trunk=FROZEN out_norm=on rca_dropout=0.1 rca_lr_scale=0.1"
echo "[rootattn-b] starts=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
T0=$(date +%s)
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
    --root-cross-attn $RCA --unfreeze-last 0 --rca-out-norm --rca-dropout 0.1 \
    --rca-lr-scale 0.1 --rca-ablate-eval --h-drift-probe 500 \
    > $OUT/train_${TAG}.log 2>&1 < /dev/null &
P=$!
echo "[rootattn-b] REMOTE_PID=$P"
while kill -0 $P 2>/dev/null; do
  sleep 60
  L=$(grep -E "^  \[eval|^  step|^  \[proof" $OUT/train_${TAG}.log | tail -1)
  echo "[rootattn-b $(date -u +%H:%M:%S)] ${L:0:230}"
done
wait $P; RC=$?
kill $SP 2>/dev/null; wait $SP 2>/dev/null
T1=$(date +%s); PEAK=$(cat $OUT/vram_${TAG}.txt 2>/dev/null)
echo "[rootattn-b] rc=$RC wall_s=$((T1-T0)) peak_vram_mib=$PEAK"
echo "[rootattn-b] $(grep -E 'trainable parameters|TRUNK LIVE|optimizer groups|RCA init' $OUT/train_${TAG}.log | tr '\n' ' ')"
echo "[rootattn-b] nan/inf steps: $(grep -ciE 'nan|inf' $OUT/trace_${TAG}.jsonl)"
tail -6 $OUT/train_${TAG}.log
echo ROOTATTN_B_DONE
