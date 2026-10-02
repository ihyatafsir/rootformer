#!/bin/bash
# Per-step PPL trace runner (zero edits to nrmt_train.py; uses its built-in --probe-out).
# Usage: run_trace.sh <STEPS> <EVAL_EVERY> <TAG>
set -u
STEPS="${1:-5000}"
EVAL="${2:-500}"
TAG="${3:-5000}"
OUTDIR=/workspace/ppl_trace
mkdir -p "$OUTDIR"
cd /workspace/hf_v19_2_release || exit 90

# --- GPU VRAM sampler (0.5 s) ---
( while true; do
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null
    sleep 0.5
  done ) > "$OUTDIR/vram_${TAG}.txt" &
SAMPLER=$!

START=$(date +%s.%N)
{
  echo "[run_trace] tag=$TAG steps=$STEPS eval_every=$EVAL start=$START"
  echo "[run_trace] cmd: python nrmt_train.py --checkpoint checkpoints/rootformer_v19_2_synthesis_ar_backbone.awzan142.roots9490.tok10052.safetensors --cache /workspace/nrmp_cache_9490 --head-init remap --steps $STEPS --batch-size 32 --eval-every $EVAL --tag ppl_trace --probe-out $OUTDIR/ppl_trace_${TAG}.jsonl"
} | tee "$OUTDIR/train_${TAG}.log"

/workspace/venvs/rootformer/bin/python nrmt_train.py \
  --checkpoint checkpoints/rootformer_v19_2_synthesis_ar_backbone.awzan142.roots9490.tok10052.safetensors \
  --cache /workspace/nrmp_cache_9490 \
  --head-init remap \
  --steps "$STEPS" --batch-size 32 --eval-every "$EVAL" --tag ppl_trace \
  --probe-out "$OUTDIR/ppl_trace_${TAG}.jsonl" \
  --out "$OUTDIR/ppl_trace_${TAG}.results.json" \
  --save "$OUTDIR/ppl_trace_${TAG}_head.pt" \
  2>&1 | tee -a "$OUTDIR/train_${TAG}.log"
RC=${PIPESTATUS[0]}
END=$(date +%s.%N)

kill $SAMPLER 2>/dev/null
wait $SAMPLER 2>/dev/null

echo "[run_trace] rc=$RC start=$START end=$END wall_s=$(awk -v a="$START" -v b="$END" 'BEGIN{printf "%.3f", b-a}')" \
  | tee -a "$OUTDIR/train_${TAG}.log"
echo "$STEPS $EVAL $TAG $RC $START $END" > "$OUTDIR/run_${TAG}.meta"
exit $RC
