#!/bin/bash
# finalize_t12.sh -- local driver: wait for the top12 arm, capture run-end facts, run the
# three-way matched probes (CTL@15k, T8@15k, T12@15k) + T12 final, download, and print the
# full depth curve report.
set -u
cd /home/grem3/Documents/deepseek-harness/default-workspace/rootformer/build/root_attn_depth || exit 90
K=../../.secrets/id_ed25519_runpod
R=root@213.173.104.76
S="ssh -i $K -p 46758 -o StrictHostKeyChecking=no $R"
SCP="scp -i $K -P 46758 -o StrictHostKeyChecking=no"

for i in $(seq 1 260); do
  OUT=$($S "test -f /workspace/root_attn_depth/results_RCA_DEPTH_T12.json && echo YES || echo NO" 2>/dev/null)
  echo "[finalize-t12 $(date -u +%H:%M:%S)] $OUT"
  [ "$OUT" = "YES" ] && break
  sleep 60
done

$S "cd /workspace/root_attn_depth
echo '=== RUN-END FACTS (T12) ==='
grep -E 'wall_s|peak_vram|rc=|nan/inf|WARN' chain_depth.out | tail -8
grep -E 'attached to trunk|trainable parameters' logs/train_RCA_DEPTH_T12.log
echo '=== final eval line (T12) ==='
grep 'eval @20000' logs/train_RCA_DEPTH_T12.log | cut -c1-700
echo '=== gates per eval (T12) ==='
grep -oE '\[eval @[0-9]+\].*RCA gates [^|]+' logs/train_RCA_DEPTH_T12.log | sed -E 's/.*\[eval @([0-9]+)\].*RCA gates/step \1 gates/' | tail -6
echo '=== probes ==='
bash probe_depth.sh /workspace/root_attn_depth/ckpt/rca_step_15000.pt RCA_DEPTH_T12_step15000 12,13,14,15,16,17,18,19,20,21,22,23 2>&1 | grep -E 'gates:|layer '
bash probe_depth.sh /workspace/root_attn_depth/head_RCA_DEPTH_T12.pt.trunk.pt RCA_DEPTH_T12_final 12,13,14,15,16,17,18,19,20,21,22,23 2>&1 | grep -E 'gates:|layer '
echo '=== GPU now ==='; nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader" 2>&1 | grep -v known_hosts

$SCP $R:/workspace/root_attn_depth/logs/train_RCA_DEPTH_T12.log logs/ >/dev/null 2>&1
$SCP $R:/workspace/root_attn_depth/trace_RCA_DEPTH_T12.jsonl . >/dev/null 2>&1
$SCP $R:/workspace/root_attn_depth/chain_depth.out . >/dev/null 2>&1
$SCP $R:/workspace/root_attn_depth/results_RCA_DEPTH_T12.json . >/dev/null 2>&1
$SCP $R:"/workspace/root_attn_depth/probe_RCA_DEPTH_T12_final.json /workspace/root_attn_depth/probe_RCA_DEPTH_T12_step15000.json" . >/dev/null 2>&1

python3 depth_report.py --ctl-log ctl/train_RCA_FROZEN_CTL.log --ctl-trace ctl/trace_RCA_FROZEN_CTL.jsonl \
  --arm RCA_DEPTH_T8:logs/train_RCA_DEPTH_T8.log:trace_RCA_DEPTH_T8.jsonl \
  --arm RCA_DEPTH_T12:logs/train_RCA_DEPTH_T12.log:trace_RCA_DEPTH_T12.jsonl --max-step 20000
echo FINALIZE_T12_DONE
