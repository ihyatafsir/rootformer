#!/bin/bash
# run_native_eval.sh -- CPU-only held-out acc@1 decomposition of the native root path.
set -u
cd /workspace/ishtiqaq_check || exit 90
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
PY=/workspace/venvs/rootformer/bin/python
CK=/workspace/hf_v19_2_release/checkpoints/rootformer_v19_2_synthesis_ar_backbone.awzan142.roots9490.tok10052.safetensors
CACHE=/workspace/head_fix/nrmp_cache_9490_aligned
echo "[native_eval] start $(date -u +%Y-%m-%dT%H:%M:%SZ) pid=$$"
$PY -u repro_native_eval.py --checkpoint "$CK" --cache "$CACHE" \
    --head /workspace/head_fix/head_ALIGNED_FIX.pt --layers top4 --device cpu \
    --out /workspace/ishtiqaq_check/native_eval.json
echo "[native_eval] rc=$? at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo NATIVE_EVAL_DONE
