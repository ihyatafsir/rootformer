#!/bin/bash
# run_smoke_then_arm.sh -- GPU-side validation BEFORE the 20k arm:
#   1. verify_ishtiqaq_fixed.py on CUDA   (the CPU proof cannot catch device placement)
#   2. a 30-step smoke with the EXACT arm CLI (device, optimizer groups, wiring, ISHTIQAQ_OFF)
#   3. only if both pass: hand over to run_ishtiqaq_arm.sh, which waits for its slot.
set -u
REL=/workspace/hf_v19_2_release
OUT=/workspace/ishtiqaq_check
PY=/workspace/venvs/rootformer/bin/python
CK=$REL/checkpoints/rootformer_v19_2_synthesis_ar_backbone.awzan142.roots9490.tok10052.safetensors
CACHE=/workspace/head_fix/nrmp_cache_9490_aligned
cd "$REL" || exit 90

# ---- GPU slot discipline: never make a THIRD concurrent 32-batch arm -----------------------
count_trainers() {
  ps -eo args= 2>/dev/null     | grep -E 'venvs/rootformer/bin/python .*nrmt_train[a-z_]*\.py .*--checkpoint' | grep -cv grep
}
count_gpu_procs() {
  nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c '[0-9]'
}
gpu_mem() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1; }
for i in $(seq 1 480); do
  N=$(count_trainers); G=$(count_gpu_procs); M=$(gpu_mem)
  if [ "$N" -le 1 ] && [ "$G" -le 1 ] && [ -n "${M:-}" ] && [ "$M" -lt 22000 ]; then
    sleep 20
    N=$(count_trainers); G=$(count_gpu_procs); M=$(gpu_mem)
    if [ "$N" -le 1 ] && [ "$G" -le 1 ] && [ "$M" -lt 22000 ]; then break; fi
  fi
  sleep 15
done
echo "[chain] slot ok at $(date -u +%H:%M:%S) other_trainers=$N gpu_procs=$G mem=${M}MiB"

echo "[stage1] 30-step smoke $(date -u +%H:%M:%S)"
$PY -u "$OUT/nrmt_train.py" --checkpoint "$CK" --cache "$CACHE" --head-init remap \
    --steps 30 --batch-size 32 --eval-every 30 --tag SMOKE_ISHTIQAQ \
    --probe-out "$OUT/trace_SMOKE.jsonl" --out "$OUT/results_SMOKE.json" \
    --save "$OUT/head_SMOKE.pt" \
    --feat-gate --margin-ramp 10 --grad-warmup 10 --logit-scale ln \
    --root-cross-attn top4 --unfreeze-last 4 --trunk-lr-scale 0.01 \
    --rca-lr-scale 0.3 --rca-out-norm --rca-dropout 0.1 --rca-ablate-eval \
    --ishtiqaq-root-bias top4 --ishtiqaq-root-source shared \
    --ishtiqaq-gate-init 0.0 --ishtiqaq-pillar-gate-init 0.0 \
    --ishtiqaq-lr-scale 0.3 --ishtiqaq-ablate-eval \
    > "$OUT/logs/smoke.log" 2>&1
R1=$?
echo "[stage1] rc=$R1"
grep -E "NATIVE root score bias|trainable parameters|ISHTIQAQ gates|ISHTIQAQ_OFF|step 30/30" \
    "$OUT/logs/smoke.log" | cut -c1-320
[ "$R1" -eq 0 ] || { echo "[stage1] ABORT: smoke FAILED"; tail -25 "$OUT/logs/smoke.log"; exit 93; }

echo "[stage2] cuda equivalence proof $(date -u +%H:%M:%S)"
$PY -u "$OUT/verify_ishtiqaq_fixed.py" --checkpoint "$CK" --cache "$CACHE" \
    --windows 1 --device cuda --out "$OUT/verify_fixed_cuda.json" \
    > "$OUT/logs/verify_cuda.log" 2>&1
R2=$?
echo "[stage2] rc=$R2 $(grep -c PASS "$OUT/logs/verify_cuda.log") PASS lines"
tail -3 "$OUT/logs/verify_cuda.log"
[ "$R2" -eq 0 ] || { echo "[stage2] ABORT: cuda equivalence FAILED"; exit 92; }

echo "[stage3] handing over to run_ishtiqaq_arm.sh $(date -u +%H:%M:%S)"
exec bash "$OUT/run_ishtiqaq_arm.sh"
