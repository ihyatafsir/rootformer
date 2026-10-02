#!/bin/bash
# watch_launch.sh TAG -- poll the pod with ONE short ssh per iteration (a dropped channel cannot
# kill the watch).  Exit as soon as the arm is training, or the launcher aborts.
TAG=${1:?TAG}
KEY=/home/grem3/Documents/deepseek-harness/default-workspace/rootformer/.secrets/id_ed25519_runpod
SSH="timeout 25 ssh -i $KEY -p 46758 -o StrictHostKeyChecking=no -o ServerAliveInterval=10 root@213.173.104.76"
for i in $(seq 1 300); do
  OUT=$($SSH "date -u +%H:%M:%S
echo -n 'others='; ps -eo args= | grep -E 'nrmt_train[a-z_]*\.py' | grep -v 'tag $TAG' | grep -vc grep
echo -n ' free='; nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits
echo '--- parent arms:'; ps -eo args= | grep -oE 'tag (RCA_[A-Za-z0-9_]+)' | sort -u | tr '\n' ' '; echo
echo '--- launcher tail:'; tail -2 /workspace/root_attn_width/logs/launch_${TAG}.out
echo '--- arm progress:'; grep -oE 'step [0-9]+/20000|[0-9.]+ it/s|\[eval @[0-9]+\] ALL_val: acc@1 [0-9.]+%' /workspace/root_attn_width/logs/train_${TAG}.log 2>/dev/null | tail -3" 2>/dev/null)
  echo "### poll $i $(date -u +%H:%M:%S)"; echo "$OUT"
  if echo "$OUT" | grep -qE 'step [0-9]+/20000'; then echo "WATCHER: ARM TRAINING"; break; fi
  echo "$OUT" | awk '/launch_width2.sh start/{n=NR} /LAUNCH_WIDTH_ABORTED/{m=NR} END{exit !(m>n)}' \
      && { echo "WATCHER: ABORTED"; break; }
  sleep 60
done
echo "WATCHER_EXIT $(date -u +%H:%M:%S)"
