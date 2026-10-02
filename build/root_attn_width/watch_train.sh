#!/bin/bash
# watch_train.sh TAG -- poll every 5 min with a short ssh; exit when the arm writes its results
# json (normal completion) or the runner prints WIDTH_ARM_DONE / a traceback.
TAG=${1:?TAG}
KEY=/home/grem3/Documents/deepseek-harness/default-workspace/rootformer/.secrets/id_ed25519_runpod
SSH="timeout 30 ssh -i $KEY -p 46758 -o StrictHostKeyChecking=no -o ServerAliveInterval=10 root@213.173.104.76"
for i in $(seq 1 120); do
  OUT=$($SSH "date -u +%H:%M:%S
echo -n 'free='; nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits
echo 'co-resident:' \$(ps -eo args= | grep -oE 'tag (RCA|F_)[A-Za-z0-9_]+' | sort -u | tr '\n' ' ')
echo -n 'w2n: '; grep -oE 'step [0-9]+/20000' /workspace/root_attn_width/logs/train_${TAG}.log 2>/dev/null | tail -1
echo 'evals:'; grep -oE '\[eval @[0-9]+\] ALL_val: acc@1 [0-9.]+% acc@5 [0-9.]+%' /workspace/root_attn_width/logs/train_${TAG}.log 2>/dev/null | tail -4
grep -oE 'RCA gates.*' /workspace/root_attn_width/logs/train_${TAG}.log 2>/dev/null | tail -1
grep -oE 'h_drift [0-9.eE+-]+$' /workspace/root_attn_width/logs/train_${TAG}.log 2>/dev/null | tail -1
ls /workspace/root_attn_width/results_${TAG}.json 2>/dev/null && echo RESULTS_JSON_PRESENT
grep -cE 'Traceback|nan_steps.: [1-9]' /workspace/root_attn_width/logs/train_${TAG}.log 2>/dev/null | sed 's/^/err_lines=/' " 2>/dev/null)
  echo "### poll $i $(date -u +%H:%M:%S)"; echo "$OUT"
  echo "$OUT" | grep -q RESULTS_JSON_PRESENT && { echo "WATCHER: DONE"; break; }
  echo "$OUT" | grep -qE 'err_lines=[1-9]' && { echo "WATCHER: ERROR DETECTED"; break; }
  sleep 300
done
echo "WATCHER_EXIT $(date -u +%H:%M:%S)"
