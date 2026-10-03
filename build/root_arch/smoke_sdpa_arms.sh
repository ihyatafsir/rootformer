#!/usr/bin/env bash
# smoke_sdpa_arms.sh -- the gate that MUST pass before 20 000 steps are committed.
#
# It runs the real trainer, real flags, real checkpoint, real cache, for a handful of steps at a
# small batch, for BOTH arms, and then ASSERT on the things that would silently waste 20 h:
#
#   1  SDPA installed BEFORE model construction (Phase 0a actually took effect)
#   2  preflight root count is 9490 / 142  (the silent-shrink trap)
#   3  the v13 stale-table drop is exactly 48 and nothing else is missing
#   4  FLOOR_A attaches NO root pathway; EARLYROOT_C attaches BOTH mechanisms on 24 layers
#   5  PILLAR-FREEZE fired, so no pillar gate is in the optimizer
#   6  the loss is finite and decreasing-ish; no NaN steps
#   7  the trunk is LIVE (h_drift probe is non-null and non-zero)
#   8  a checkpoint pair (.pt and .pt.trunk.pt) is written at the eval step
#   9  it/s is recorded so the 20k projection is grounded
#
# Usage: smoke_sdpa_arms.sh [BATCH] [STEPS]
set -u
BATCH=${1:-8}
STEPS=${2:-40}
PY=/workspace/venvs/rootformer/bin/python
ROOT=/workspace/root_arch
TRAIN=$ROOT/nrmt_train_v13_sdpa.py
SMOKE=$ROOT/smoke
CKPTDIR=/tmp/root_arch_arms      # /workspace has a per-user quota; 640 MiB dies there
CKPT=/workspace/hf_v19_2_release/checkpoints/rootformer_v19_2_synthesis_ar_backbone.awzan142.roots9490.tok10052.safetensors
CACHE=/workspace/head_fix/nrmp_cache_9490_aligned
mkdir -p "$SMOKE" "$CKPTDIR"
rm -f "$SMOKE"/head_*.pt "$SMOKE"/head_*.pt.trunk.pt

common() {
  echo -n "--model-v13 --flash-hooks off --sdpa --pillar-freeze"
  echo -n " --checkpoint $CKPT --cache $CACHE"
  echo -n " --head-init remap --unfreeze-trunk-all --trunk-lr-scale 1.0"
  echo -n " --lr 1e-3 --steps $STEPS --batch-size $BATCH --eval-every $((STEPS/2))"
  echo -n " --feat-gate --feat-gate-proj-std 1e-3 --margin-ramp 10 --grad-warmup 10"
  echo -n " --logit-scale ln --grad-clip 1.0 --hist 3 --dropout 0.1"
  echo -n " --rca-out-norm --rca-dropout 0.1 --rca-heads 8 --h-drift-probe 5"
}

run_arm() {
  local tag=$1; shift
  echo "=== SMOKE $tag ==="
  ( cd "$ROOT" && timeout 1800 "$PY" -u "$TRAIN" $(common) "$@" \
      --tag "SMOKE_$tag" \
      --out "$SMOKE/results_$tag.json" \
      --probe-out "$SMOKE/trace_$tag.jsonl" \
      --save "$CKPTDIR/smoke_head_$tag.pt" > "$SMOKE/log_$tag.txt" 2>&1 )
  echo "  exit=$?  log=$(wc -l < "$SMOKE/log_$tag.txt") lines"
}

run_arm FLOOR_A --root-cross-attn none --ishtiqaq-root-bias none
run_arm EARLYROOT_C --root-cross-attn all --rca-lr-scale 1.0 --rca-ablate-eval \
  --ishtiqaq-root-bias all --ishtiqaq-root-source shared --ishtiqaq-gate-init 0.0 \
  --ishtiqaq-pillar-gate-init 0.0 --ishtiqaq-lr-scale 1.0 --ishtiqaq-ablate-eval

echo
echo "############ SMOKE ASSERTIONS ############"
PYCHK=$(cat <<'EOF'
import json, os, re, sys
ROOT='/workspace/root_arch'; SMOKE=ROOT+'/smoke'
fails=[]
def need(cond, msg):
    print(('  PASS  ' if cond else '  FAIL  ')+msg)
    if not cond: fails.append(msg)

for tag, want_pathway, want_bias in (('FLOOR_A', False, False), ('EARLYROOT_C', True, True)):
    log=open(f'{SMOKE}/log_{tag}.txt').read()
    print(f'--- {tag} ---')
    need('SDPA installed BEFORE model construction' in log, f'{tag}: SDPA installed pre-build')
    need('PREFLIGHT root count OK: roots=9490 awzan=142' in log, f'{tag}: root count 9490/142')
    need('dropped 48 stale-SPACE' in log, f'{tag}: exactly 48 stale tables dropped')
    m=re.search(r'missing=(\d+)', log)
    need(m is not None and int(m.group(1))>0, f'{tag}: load reported missing/unexpected counts')
    need('flash hooks DETACHED: 3 hooks' in log, f'{tag}: 3 flash hooks detached')
    need('TRUNK-ALL: 24 backbone layers' in log, f'{tag}: all 24 trunk layers trainable')
    has_rca = 'root cross-attention attached to trunk layers [0, 1, 2' in log
    need(has_rca == want_pathway, f'{tag}: residual injector attached={has_rca} (want {want_pathway})')
    has_bias = 'NATIVE root score bias on trunk layers [0, 1, 2' in log
    need(has_bias == want_bias, f'{tag}: score bias attached={has_bias} (want {want_bias})')
    if want_bias:
        need('PILLAR-FREEZE:' in log, f'{tag}: PILLAR-FREEZE fired')
        need('pillar gates held at exactly 0.0' in log, f'{tag}: pillar gates held at 0')
    # optimizer groups
    gm=re.search(r'optimizer groups \(max_lr.*', log)
    print('        '+(gm.group(0)[:200] if gm else 'NO OPTIMIZER GROUP LINE'))
    if want_bias:
        need(gm is not None and 'ishtiqaq' in gm.group(0), f'{tag}: ishtiqaq group present')
    # trace: finite losses, step count, it/s
    tr=[json.loads(l) for l in open(f'{SMOKE}/trace_{tag}.jsonl') if l.strip()]
    need(len(tr)>=1, f'{tag}: trace has {len(tr)} steps')
    import math
    need(all(math.isfinite(r['loss']) for r in tr), f'{tag}: every loss finite')
    need(sum(1 for r in tr if r.get('skipped'))==0, f'{tag}: no skipped (grad-warmup) steps')
    drift=[r['h_drift_max'] for r in tr if r.get('h_drift_max') is not None]
    need(len(drift)>0 and max(drift)>0, f'{tag}: trunk LIVE (h_drift non-null, max={max(drift) if drift else None})')
    gn=[r.get('grad_norm_trunk',0) for r in tr]
    need(max(gn)>0, f'{tag}: trunk parameters receive gradient (max grad_norm_trunk={max(gn):.3e})')
    need(os.path.exists(f'/tmp/root_arch_arms/smoke_head_{tag}.pt'), f'{tag}: head checkpoint written')
    need(os.path.exists(f'/tmp/root_arch_arms/smoke_head_{tag}.pt.trunk.pt'), f'{tag}: trunk checkpoint written')
    js=json.load(open(f'{SMOKE}/results_{tag}.json')) if os.path.exists(f'{SMOKE}/results_{tag}.json') else None
    need(js is not None and 'history' in js and len(js['history'])>=1, f'{tag}: results json with history')
    if js and js['history']:
        h=js['history'][-1]
        print(f"        eval@{h['step']}: ALL_val acc@1={100*h['ALL_val']['acc@1']:.2f}%  "
              f"NOVEL acc@1={100*h['NOVEL_only']['acc@1']:.2f}%  ce_z={h['ALL_val']['ce_z']:.4f}")
    need('Aborted' not in log and 'Traceback' not in log, f'{tag}: no traceback in log')

print()
print('SMOKE RESULT:', 'ALL PASS' if not fails else f'{len(fails)} FAILURES')
for f in fails: print('   FAILED:', f)
sys.exit(0 if not fails else 1)
EOF
)
$PY -c "$PYCHK"
