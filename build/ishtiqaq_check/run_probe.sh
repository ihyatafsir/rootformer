#!/bin/bash
# run_probe.sh -- durable launcher for the native-root-path diagnostic probe.
set -u
cd /workspace/ishtiqaq_check || exit 90
PY=/workspace/venvs/rootformer/bin/python
CK=/workspace/hf_v19_2_release/checkpoints/rootformer_v19_2_synthesis_ar_backbone.awzan142.roots9490.tok10052.safetensors
CACHE=/workspace/head_fix/nrmp_cache_9490_aligned
echo "[run_probe] start $(date -u +%Y-%m-%dT%H:%M:%SZ) pid=$$"
$PY -u probe_native_root_path.py \
    --checkpoint "$CK" --cache "$CACHE" --windows 2 --device cpu \
    --out /workspace/ishtiqaq_check/probe_native.json
echo "[run_probe] rc=$? at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo PROBE_DONE
