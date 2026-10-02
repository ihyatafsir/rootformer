#!/usr/bin/env bash
cd /workspace/alt_base
export HF_HOME=/workspace/alt_base/cache
PY=/workspace/venvs/rootformer/bin/python
echo "=== [$(date -u +%H:%M:%S)] verify_alignment ==="
timeout 900 $PY verify_alignment.py --out /workspace/alt_base/verify_alignment.json
echo "=== [$(date -u +%H:%M:%S)] alt_probe (ORIGINAL untransmuted base) ==="
timeout 2400 $PY alt_probe.py --tag orig
echo "=== [$(date -u +%H:%M:%S)] chain done rc=$? ==="
