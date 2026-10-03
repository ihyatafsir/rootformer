#!/usr/bin/env bash
# probe (fast) then live-training benchmark, strictly sequential so only one of mine is on the GPU.
cd /workspace/alt_base
export HF_HOME=/workspace/alt_base/cache
PY=/workspace/venvs/rootformer/bin/python
echo "=== [$(date -u +%H:%M:%S)] PROBE (original base, plain HF) ==="
ARMS_INCL=3 NEED_FREE_MIB=3000 ./gate_gpu.sh $PY -u alt_probe2.py --tag orig \
  --max-train-words 180000 --bs 48
echo "=== [$(date -u +%H:%M:%S)] LIVE BENCH ==="
ARMS_INCL=3 NEED_FREE_MIB=3000 ./gate_gpu.sh $PY -u alt_train.py \
  --tag BENCH --cache /workspace/alt_base/cache/rootqwen --live \
  --steps 6000 --bench 8 --batch-size 8 --eval-every 0 \
  --train-windows 3000 --val-windows 64 --logit-scale ln \
  --no-features --root-cross-attn none \
  --out /workspace/alt_base/results_BENCH.json --save /workspace/alt_base/head_BENCH.pt
echo "=== [$(date -u +%H:%M:%S)] chain_pb done ==="
