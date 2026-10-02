#!/bin/bash
# Continuation queue: one GPU-window wait per TRAINING run; the (short, same-model) eval runs
# immediately after its own training in the same window.
cd /workspace/scratch_comp
PY=/workspace/venvs/rootformer/bin/python
D=/workspace/scratch_comp/sf_data_9490

wait_free () {
  local waited=0
  while true; do
    local busy mem
    busy=$(pgrep -fc "python nrmt_train.py --checkpoint" 2>/dev/null); busy=${busy:-0}
    mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); mem=${mem:-9999}
    if [ "$busy" = "0" ] && [ "$mem" -lt 800 ]; then
      sleep 8
      busy=$(pgrep -fc "python nrmt_train.py --checkpoint" 2>/dev/null); busy=${busy:-0}
      mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); mem=${mem:-9999}
      if [ "$busy" = "0" ] && [ "$mem" -lt 800 ]; then
        echo "[$(date -u +%H:%M:%S)] GPU FREE (mem=${mem}MiB) after waiting ${waited}s"
        return 0
      fi
    fi
    sleep 15
    waited=$((waited+15))
    if [ "$waited" -ge 5400 ]; then
      echo "[$(date -u +%H:%M:%S)] WAIT TIMEOUT ${waited}s"; return 1
    fi
  done
}

echo "QUEUE2 START $(date -u)"

wait_free || exit 1
echo "[$(date -u +%H:%M:%S)] EVAL comp6k (trained earlier)"
$PY scratch_lm_comp.py eval --root-head comp --tag comp6k --ckpt $D/comp6k.pt --out $D --free-max-pos 4000 > eval_comp6k.log 2>&1
echo "[$(date -u +%H:%M:%S)] EVAL comp6k rc=$?"

wait_free || exit 1
echo "[$(date -u +%H:%M:%S)] LAUNCH type6k"
$PY scratch_lm_comp.py train --root-head type --tag type6k --steps 6000 --batch 16 --lr 6e-4 --out $D > train_type6k.log 2>&1
echo "[$(date -u +%H:%M:%S)] TRAIN type6k rc=$?"
$PY scratch_lm_comp.py eval --root-head type --tag type6k --ckpt $D/type6k.pt --out $D > eval_type6k.log 2>&1
echo "[$(date -u +%H:%M:%S)] EVAL type6k rc=$?"

wait_free || exit 1
echo "[$(date -u +%H:%M:%S)] LAUNCH comp24k"
$PY scratch_lm_comp.py train --root-head comp --tag comp24k --steps 24000 --batch 16 --lr 6e-4 --out $D > train_comp24k.log 2>&1
echo "[$(date -u +%H:%M:%S)] TRAIN comp24k rc=$?"
$PY scratch_lm_comp.py eval --root-head comp --tag comp24k --ckpt $D/comp24k.pt --out $D --free-max-pos 4000 > eval_comp24k.log 2>&1
echo "[$(date -u +%H:%M:%S)] EVAL comp24k rc=$?"

echo "QUEUE2 DONE $(date -u)"
