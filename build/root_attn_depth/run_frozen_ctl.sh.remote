#!/bin/bash
# run_frozen_ctl.sh -- CLEAN SINGLE-VARIABLE CONTROL for the ROOTATTN collapse.
#
# Run A (tag ROOTATTN, /workspace/root_attn/run_rootattn.sh) added TWO things at once:
#     --root-cross-attn top4   AND   --unfreeze-last 4 --trunk-lr-scale 0.1
# and its held-out metric read 0.28 / 0.24 / 0.23 %.
#
# This run changes EXACTLY ONE variable against Run A:
#     --unfreeze-last 4  ->  --unfreeze-last 0      (trunk FROZEN; no trunk optimizer group)
#
# IDENTICAL to Run A, deliberately including the things Run B (RCA_FROZEN) changed:
#     same checkpoint, same aligned cache, same --steps 20000, same --batch-size 32,
#     same --eval-every 1000, same --head-init remap, same FIX flags
#     (--feat-gate --margin-ramp 10 --grad-warmup 10 --logit-scale ln),
#     same --rca-lr-scale 1.0  (NOT 0.1), NO --rca-out-norm, NO --rca-dropout,
#     same --rca-ablate-eval --h-drift-probe 500.
#     --trunk-lr-scale 0.1 is kept verbatim even though it is inert with an empty trunk group.
# The RCA module itself is the very same /workspace/root_attn/root_cross_attn.py Run A used
# (the trainer hard-codes that sys.path entry); nothing under /workspace/root_attn is touched.
#
# NO MODULE WAS EDITED: --unfreeze-last already exists (it is the flag the previous agent added).
#
# Same trainer code as the currently running chain (nrmt_train.py md5 56ef3f2fd9ddc352dee596f32590e7b5),
# i.e. it carries the live-evaluator BUGFIX (val streams, not train streams).  Run A's 0.23-0.28 %
# was measured with the buggy stream.
set -u
cd /workspace/hf_v19_2_release || exit 90
OUT=/workspace/root_attn_ctl; mkdir -p $OUT $OUT/ckpt
CK=checkpoints/rootformer_v19_2_synthesis_ar_backbone.awzan142.roots9490.tok10052.safetensors
PY=/workspace/venvs/rootformer/bin/python
CACHE=/workspace/head_fix/nrmp_cache_9490_aligned
TAG=${TAG:-RCA_FROZEN_CTL}
RCA=${RCA:-top4}
# NOTE: the "another nrmt_train.py is already running" guard of run_rootattn.sh is deliberately
# NOT used here: the parent's chain_final.sh (tag RCA_UNFREEZE_A2) holds the GPU and this control
# is run CONCURRENTLY rather than preempting it.  VRAM: ~10.3 GiB in use + ~9.5 GiB here of 31.9 GiB.
echo "[frozen-ctl] tag=$TAG rca=$RCA unfreeze_last=0 trunk=FROZEN rca_lr_scale=1.0 out_norm=off dropout=0"
echo "[frozen-ctl] cache=$CACHE"
echo "[frozen-ctl] starts=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
T0=$(date +%s)

# peak-VRAM poller
( M=0; while true; do
    V=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    [ -n "${V:-}" ] && [ "$V" -gt "$M" ] 2>/dev/null && M=$V
    echo "$M" > $OUT/vram_${TAG}.txt
    sleep 1
  done ) &
SP=$!

# per-1000-step RCA/trunk checkpoint snapshotter (the trainer overwrites .trunk.pt at every eval,
# so the RCA's gate / attention degeneracy trajectory is otherwise lost)
( L=""
  while true; do
    S=$(grep -oE '\[eval @[0-9]+' $OUT/train_${TAG}.log 2>/dev/null | tail -1 | grep -oE '[0-9]+$')
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

setsid nohup $PY nrmt_train.py --checkpoint $CK --cache $CACHE --head-init remap \
    --steps 20000 --batch-size 32 --eval-every 1000 --tag $TAG \
    --probe-out $OUT/trace_${TAG}.jsonl --out $OUT/results_${TAG}.json \
    --save $OUT/head_${TAG}.pt --feat-gate --margin-ramp 10 --grad-warmup 10 --logit-scale ln \
    --root-cross-attn $RCA --unfreeze-last 0 \
    --trunk-lr-scale 0.1 --rca-lr-scale 1.0 --rca-ablate-eval --h-drift-probe 500 \
    > $OUT/train_${TAG}.log 2>&1 < /dev/null &
P=$!
echo "[frozen-ctl] REMOTE_PID=$P"

while kill -0 $P 2>/dev/null; do
  sleep 60
  L=$(grep -E "^  \[eval|^  step|^  \[proof" $OUT/train_${TAG}.log | tail -1)
  echo "[frozen-ctl $(date -u +%H:%M:%S)] ${L:0:200}"
done
wait $P; RC=$?
kill $SP $SS 2>/dev/null; wait $SP $SS 2>/dev/null
T1=$(date +%s)
PEAK=$(cat $OUT/vram_${TAG}.txt 2>/dev/null)
echo "[frozen-ctl] rc=$RC wall_s=$((T1-T0)) peak_vram_mib=$PEAK"
echo "[frozen-ctl] $(grep -E 'trainable parameters|TRUNK LIVE|optimizer groups|RCA init' $OUT/train_${TAG}.log | tr '\n' ' ')"
echo "[frozen-ctl] nan/inf lines in trace: $(grep -ciE 'nan|inf' $OUT/trace_${TAG}.jsonl 2>/dev/null)"
grep -cE 'WARN' $OUT/train_${TAG}.log | sed 's/^/[frozen-ctl] WARN lines: /'
tail -4 $OUT/train_${TAG}.log
echo "[frozen-ctl] RESULTS:"
$PY -c "
import json
h=json.load(open('$OUT/results_${TAG}.json'))['history'][-1]
print(' final step', h['step'], 'nan_steps', h.get('nan_steps'))
for k in ('ALL_val','ALL_val_RCA_OFF','NOVEL_only','NOVEL_only_RCA_OFF'):
    if k in h:
        v=h[k]; print(f\"  {k}: acc@1 {100*v['acc@1']:.2f}% acc@5 {100*v['acc@5']:.2f}% CE_z {v['ce_z']:.4f} (n={v['n']})\")
print(' rca gates', h.get('rca_gate'), 'h_drift', h.get('h_drift'))
" 2>&1 | tail -8
echo FROZEN_CTL_DONE
