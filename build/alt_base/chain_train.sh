#!/usr/bin/env bash
# chain_train.sh -- run arm A then arm C, each behind the durable GPU gate.
cd /workspace/alt_base
PY=/workspace/venvs/rootformer/bin/python
export HF_HOME=/workspace/alt_base/cache
COMMON="--cache /workspace/alt_base/cache/rootqwen --steps 6000 --batch-size 16 \
 --eval-every 1000 --val-windows 320 --train-windows 4000 --feat-gate --margin-ramp 10 \
 --grad-warmup 10 --logit-scale ln --rca-out-norm"

echo "=== [$(date -u +%H:%M:%S)] waiting for slot: ARM A ==="
./gate_train.sh $PY alt_train.py $COMMON --tag A --no-features --root-cross-attn none \
  --out /workspace/alt_base/results_A.json --save /workspace/alt_base/head_A.pt \
  --probe-out /workspace/alt_base/trace_A.jsonl || echo "ARM A FAILED rc=$?"

echo "=== [$(date -u +%H:%M:%S)] waiting for slot: ARM C ==="
./gate_train.sh $PY alt_train.py $COMMON --tag C --root-cross-attn all --score-bias false \
  --rca-lr-scale 1.0 --rca-ablate-eval \
  --out /workspace/alt_base/results_C.json --save /workspace/alt_base/head_C.pt \
  --probe-out /workspace/alt_base/trace_C.jsonl || echo "ARM C FAILED rc=$?"

echo "=== [$(date -u +%H:%M:%S)] CHAIN DONE ==="
