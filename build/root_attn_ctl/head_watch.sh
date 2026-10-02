#!/bin/bash
cd /workspace/root_attn_ctl
L=$(stat -c %Y head_RCA_FROZEN_CTL.pt 2>/dev/null || echo 0)
while true; do
  M=$(stat -c %Y head_RCA_FROZEN_CTL.pt 2>/dev/null || echo 0)
  if [ "$M" != "$L" ]; then
    for i in $(seq 1 40); do
      a=$(stat -c %s head_RCA_FROZEN_CTL.pt 2>/dev/null || echo 0); sleep 1
      b=$(stat -c %s head_RCA_FROZEN_CTL.pt 2>/dev/null || echo 0)
      [ "$a" = "$b" ] && [ "$a" != "0" ] && break
    done
    S=$(grep -oE "\[eval @[0-9]+" train_RCA_FROZEN_CTL.log | tail -1 | grep -oE "[0-9]+$")
    cp -f head_RCA_FROZEN_CTL.pt ckpt/head_step_${S}.pt
    L=$M
  fi
  sleep 3
done
