#!/usr/bin/env bash
# chain_probe.sh -- the decodability probe, run ALONE on the GPU.  Nothing else of mine runs here.
cd /workspace/alt_base
export HF_HOME=/workspace/alt_base/cache
PY=/workspace/venvs/rootformer/bin/python
echo "=== [$(date -u +%H:%M:%S)] PROBE (original Qwen2.5-0.5B, no morphemic adaptation) ==="
ARMS_INCL=2 NEED_FREE_MIB=5000 ./gate_gpu.sh $PY -u alt_probe.py --tag orig --max-train-words 400000
echo "=== [$(date -u +%H:%M:%S)] probe rc=$? ==="
