#!/bin/bash
# run_verify.sh -- durable launcher for the fixed-native-path equivalence proof.
set -u
cd /workspace/ishtiqaq_check || exit 90
export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
PY=/workspace/venvs/rootformer/bin/python
CK=/workspace/hf_v19_2_release/checkpoints/rootformer_v19_2_synthesis_ar_backbone.awzan142.roots9490.tok10052.safetensors
CACHE=/workspace/head_fix/nrmp_cache_9490_aligned
echo "[run_verify] start $(date -u +%Y-%m-%dT%H:%M:%SZ) pid=$$"
$PY -u verify_ishtiqaq_fixed.py \
    --checkpoint "$CK" --cache "$CACHE" --windows 1 --device cpu \
    --out /workspace/ishtiqaq_check/verify_fixed.json
echo "[run_verify] rc=$? at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo VERIFY_DONE
