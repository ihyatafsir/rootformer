#!/usr/bin/env bash
cd /workspace/alt_base
export HF_HOME=/workspace/alt_base/cache
PY=/workspace/venvs/rootformer/bin/python
echo "=== SMOKE frozen A ==="
timeout 900 $PY alt_train.py --tag SMOKE --cache /workspace/alt_base/cache/rootqwen \
  --steps 12 --batch-size 4 --eval-every 6 --val-windows 8 --train-windows 24 \
  --feat-gate --margin-ramp 10 --grad-warmup 10 --logit-scale ln \
  --no-features --root-cross-attn none \
  --out /workspace/alt_base/results_SMOKE.json --save /workspace/alt_base/head_SMOKE.pt
echo "=== SMOKE frozen C (RCA all layers) ==="
timeout 900 $PY alt_train.py --tag SMOKEC --cache /workspace/alt_base/cache/rootqwen \
  --steps 12 --batch-size 4 --eval-every 6 --val-windows 8 --train-windows 24 \
  --feat-gate --margin-ramp 10 --grad-warmup 10 --logit-scale ln \
  --root-cross-attn all --rca-out-norm --rca-ablate-eval \
  --out /workspace/alt_base/results_SMOKEC.json --save /workspace/alt_base/head_SMOKEC.pt
echo "=== SMOKE done ==="
