#!/usr/bin/env bash
cd /workspace/alt_base
export HF_HOME=/workspace/alt_base/cache
PY=/workspace/venvs/rootformer/bin/python
echo "=== [$(date -u +%H:%M:%S)] LIVE BENCH (grad-ckpt, batch 4) ==="
ARMS_INCL=3 NEED_FREE_MIB=2500 ./gate_gpu.sh $PY -u alt_train.py \
  --tag BENCH --cache /workspace/alt_base/cache/rootqwen --live --grad-ckpt \
  --steps 6000 --bench 12 --batch-size 4 --eval-every 0 \
  --train-windows 6000 --val-windows 32 --logit-scale ln \
  --trunk-lr-scale 1.0 --no-features --root-cross-attn none \
  --out /workspace/alt_base/results_BENCH.json --save /workspace/alt_base/head_BENCH.pt
echo "=== [$(date -u +%H:%M:%S)] bench done ==="
