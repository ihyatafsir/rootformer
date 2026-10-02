#!/bin/bash
# local monitor: poll the pod every 5 min, print a progress line, exit when the chain is done.
K=/home/grem3/Documents/deepseek-harness/default-workspace/rootformer/.secrets/id_ed25519_runpod
SSH="ssh -i $K -p 46758 -o StrictHostKeyChecking=no root@213.173.104.76"
while true; do
  S=$($SSH "for T in RCA_DEPTH_T8 RCA_DEPTH_T12; do L=/workspace/root_attn_depth/logs/train_\$T.log; if [ -f \$L ]; then printf '%s ' \$T; tail -1 \$L | cut -c1-150; fi; done; echo; grep -E 'wall_s|STAGE|DEPTH_ARM_DONE|CHAIN_DEPTH_DONE' /workspace/root_attn_depth/chain_depth.out | tail -3; echo GPU=\$(nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader | head -1)" 2>/dev/null | grep -v known_hosts)
  echo "=== $(date -u +%H:%M:%S) ==="; echo "$S"
  case "$S" in *CHAIN_DEPTH_DONE*) echo MONITOR_DONE; break;; esac
  sleep 300
done
