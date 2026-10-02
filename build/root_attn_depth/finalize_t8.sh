#!/bin/bash
# finalize_t8.sh -- local driver: wait for the T8 arm to finish, capture the run-end facts,
# run the matched attention probes (CTL step15000, T8 step15000, T8 final), download the
# artifacts, and print the matched-step report including step 20000.
set -u
cd /home/grem3/Documents/deepseek-harness/default-workspace/rootformer/build/root_attn_depth || exit 90
K=../../.secrets/id_ed25519_runpod
R=root@213.173.104.76
S="ssh -i $K -p 46758 -o StrictHostKeyChecking=no $R"
SCP="scp -i $K -P 46758 -o StrictHostKeyChecking=no"

# ---- 1. wait for stage 1 to finish (results json is written after the final eval) ----------
for i in $(seq 1 200); do
  DONE=$($S "test -f /workspace/root_attn_depth/results_RCA_DEPTH_T8.json && echo YES || echo NO; grep -c 'DEPTH_ARM_DONE RCA_DEPTH_T8' /workspace/root_attn_depth/chain_depth.out 2>/dev/null" 2>/dev/null | tr '\n' ' ')
  echo "[finalize $(date -u +%H:%M:%S)] $DONE"
  case "$DONE" in YES*|*" 1") break;; esac
  sleep 60
done

# ---- 2. run-end facts + the three probes -------------------------------------------------
$S "cd /workspace/root_attn_depth
echo '=== RUN-END FACTS (T8) ==='
grep -E 'wall_s|peak_vram|trainable parameters|attached to trunk' chain_depth.out logs/train_RCA_DEPTH_T8.log | tail -8
echo '=== final eval line ==='
grep 'eval @20000' logs/train_RCA_DEPTH_T8.log | cut -c1-700
echo '=== vram file ==='; cat vram_RCA_DEPTH_T8.txt
echo '=== probes ==='
bash probe_depth.sh /workspace/root_attn_ctl/ckpt/rca_step_15000.pt RCA_FROZEN_CTL_step15000 20,21,22,23 2>&1 | grep -E 'gates:|layer '
bash probe_depth.sh /workspace/root_attn_depth/ckpt/rca_step_15000.pt RCA_DEPTH_T8_step15000 16,17,18,19,20,21,22,23 2>&1 | grep -E 'gates:|layer '
bash probe_depth.sh /workspace/root_attn_depth/head_RCA_DEPTH_T8.pt.trunk.pt RCA_DEPTH_T8_final 16,17,18,19,20,21,22,23 2>&1 | grep -E 'gates:|layer '
echo '=== who else is on the GPU now ==='
ps -eo args= | grep -oE 'nrmt_train[a-z_]*\.py.*--tag [A-Za-z0-9_]+' | grep -oE '\-\-tag [A-Za-z0-9_]+'
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader" 2>&1 | grep -v known_hosts

# ---- 3. download artifacts --------------------------------------------------------------
$SCP $R:/workspace/root_attn_depth/logs/train_RCA_DEPTH_T8.log logs/ >/dev/null 2>&1
$SCP $R:/workspace/root_attn_depth/trace_RCA_DEPTH_T8.jsonl . >/dev/null 2>&1
$SCP $R:/workspace/root_attn_depth/chain_depth.out . >/dev/null 2>&1
$SCP $R:/workspace/root_attn_depth/results_RCA_DEPTH_T8.json . >/dev/null 2>&1
$SCP $R:"/workspace/root_attn_depth/probe_RCA_DEPTH_T8_final.json /workspace/root_attn_depth/probe_RCA_DEPTH_T8_step15000.json /workspace/root_attn_depth/probe_RCA_FROZEN_CTL_step15000.json" . >/dev/null 2>&1

# ---- 4. full matched report -------------------------------------------------------------
python3 depth_report.py --ctl-log ctl/train_RCA_FROZEN_CTL.log --ctl-trace ctl/trace_RCA_FROZEN_CTL.jsonl \
  --arm RCA_DEPTH_T8:logs/train_RCA_DEPTH_T8.log:trace_RCA_DEPTH_T8.jsonl --max-step 20000
echo FINALIZE_T8_DONE
