# NEXT — handoff for a fresh session

Read **`STATE.md` first** (verified facts + three retracted errors). This file is the action plan.

---

## 0. Reconnect

```
ssh -i rootformer/.secrets/id_ed25519_runpod -p 46758 -o StrictHostKeyChecking=no root@213.173.104.76
```
Pod `g6exduq0bd17z8`, ~20 h GPU budget. Python `/workspace/venvs/rootformer/bin/python`.
**Never `ssh -n`.** `scp -i <key> -P 46758 -o StrictHostKeyChecking=no`.

**Local:** `rootformer/` is a git repo, pushed to `github.com/ihyatafsir/rootformer` (private).
`src/` holds the md5-verified canonical files. **Backups:** six Drive restore points,
newest `gdrive:rootformer_backup_2026-10-03_final/`; HF holds the weights as safetensors.

**Antigravity** (a Gemini 3.8 Flash agent with its own weekly token budget) is drivable:
```
POST http://127.0.0.1:8787/api/messages   {"text": "..."}    -> submits
GET  http://127.0.0.1:8787/api/state                          -> {messages[], agent.busy}
POST http://127.0.0.1:8787/api/models/select  {"name":"Gemini 3.8 Flash High"}
```
It reads files in this workspace itself. Use it for **token-heavy reading**, not for pod work.

---

## 1. Do this FIRST — CPU, minutes, and it touches the thesis

**Re-run the qiyās induction with TWO conditions instead of four.**

`STATE.md` establishes, from both authors verbatim, that only **munḍabiṭ** and **muṭṭarid** are
required — **munʿakis and mutaʿaddī are not**. The build rejected **ʿilla = the root** on the
*mutaʿaddī* condition, citing `Mahsul:15719-15721`, a passage whose own response opens
*«قد بينا في كتبنا العقلية ما في هذين الوجهين من المغالطة»* — it **refutes** the position.

So: re-induce with two conditions and see whether the **root itself** becomes a valid ʿilla.
Files: `/workspace/qiyas/` (code + `QIYAS_REPORT.md`), corpus at `rootformer/corpus/`.
**This may restore the thesis-relevant ʿilla, and it needs no GPU.**

Then: **the full 9,220-root inverse search** — Task B's candidate set was restricted to 37 roots.

---

## 1b. LIVE STATUS — supersedes every estimate below. Read this first.

### THE FLOOR IS NOT 7.75 %. THE TRUNK IS DAMAGED. (this retracts the earlier live status)

The earlier version of this section said `FLOOR_A` "has effectively answered its question — the
floor is ~7.75 %". **That number is FLOOR_A's own head self-report, and it is misleading.**
Measured head-independently (fixed foreign head `head_ALIGNED_FIX.pt`, no root pathway, native root
ids nulled, same 300 val windows / 18,869 radical positions), `build/qiyas/head_probe_v13.py`:

```
FLOOR_A, step 13000 checkpoint   LIVE fp32   acc@1  3.498 %   acc@5 6.582 %   ce_z 9.1264
```

Comparators under the *same* protocol — i.e. the bar it should be held to:

```
FIX (frozen released trunk) 6.837 | ALIGNED_FIX 6.619 | GUARD 6.630 | NOFEAT 6.593
ALIGNED 6.148 | NGRAM 6.063 | CONTROL 5.718 | marginal (word-stream) 3.551
```

**FLOOR_A sits BELOW the marginal rate and at less than half the 6.6–6.8 % cluster.** Its
self-report (flat 7.64–8.14 %) is a *failure* signature, not capability. This is the arm-P pattern
again: self-report 16.34 % on a trunk measured at 2.676 %.

**Mechanism, measured — the trunk LR was 10x the documented trunk-destroying rate.** From
`trace_FLOOR_A.jsonl` (n=17,075): peak `1.000e-03`, held ~1e-3 through step 2000, still >1e-4 past
step 15000, decaying to 5.78e-05 by 17000. The repo documents **1e-4 (0.1x)** as the LR that
destroys the trunk.

**So the earlier inference inverts.** "A trained Arabic trunk does NOT simply learn roots" was
argued from the 7.75 % self-report and is retracted. What the trunk actually achieves is **3.498 %**,
below the 3.551 % marginal. Separately, the root pathway reaching ~19.5 % *on a damaged trunk* is a
stronger fact than it was before, not a weaker one.

**Scoping that keeps the main comparison valid:** every arm (A, C, R, D, both `_S2`) shares this LR
schedule via `common_flags`, so a damaged trunk is common to all of them and the **relative** C-vs-X
comparison still measures placement rather than damage. What is invalid is reading `A` as a healthy
floor, or reading C/X against A's damaged capability.

**Confound, not excluded:** `head_ALIGNED_FIX.pt` was fitted for a *different* trunk, so part of the
3.2 pp gap could be head/trunk mismatch. The clean control — a head fitted on FLOOR_A's own trunk,
or the root decodability probe — **has not been run**.

### !! DO NOT RESTART THE RUNNER. IT SURVIVED AND WILL LAUNCH C BY ITSELF. !!

**Verified 2026-10-03T01:21Z.** The queue process outlived the agent that started it — it is
detached (`ppid=1`, its own session) and still running:

```
ps -eo pid=,ppid=,etime=,args | grep 'runner.sh'
  430556  1  17:36  bash runner.sh run        <- ALIVE, detached, will launch the next stage
  405012  1  01:17  bash runner.sh one FLOOR_A  <- the launcher for the currently running arm
```

It is **gated, not stuck**: free VRAM is 18,136 MiB against the 20,000 MiB full-trunk gate, so it
holds while FLOOR_A runs and **launches `EARLYROOT_C` itself when FLOOR_A exits.** No human or agent
action is required.

**Why restarting would be actively harmful:** `runner.sh` takes no lock (grep for
`flock|lockfile|already running|kill -0` returns nothing) and `echo $$ > "$PIDFILE"` overwrites
unconditionally. A second `runner.sh run` would therefore run *concurrently* and could launch a
second full-trunk arm into 18.1 GiB — the exact ~22 MiB-margin OOM that already killed
`EARLYROOT_C` once (it OOM'd on a 28 MiB request). **A duplicate runner is the most likely way to
break this ladder.**

**To check progress, read — never restart:**

```bash
tail -3   /workspace/root_arch/queue/runner.log        # WAIT / LAUNCH / SLOT FREE lines
tail -5   /workspace/root_arch/queue/watch.log         # APPEARED / EXITED timeline
ls -la    /workspace/root_arch/arms/                   # which log_/trace_ files exist
grep -c 'eval @' /workspace/root_arch/arms/log_<TAG>.txt   # how many evals that arm has done
```

An arm is finished when `results_<TAG>.json` exists (written once, after the loop, at trainer
`:1245`) — **not** when the log stops growing.

*(Note: pid 413704, quoted in the older block below, was replaced by 430556 when `runner.sh` was
edited at 01:03:36 to add the two `_S2` seed arms. The older block is stale in that one detail.)*

### Runner state (updates the block below, which is stale)

```
pid 430556  bash runner.sh run          <- restarted; pid 413704 is GONE
runner.sh edited 01:03:36 -> md5 5683a6bccae0f661303fcec5c657b14b (was 66184fed...)
STAGES now SIX arms:
  unstarted=[EARLYROOT_C RESIDUAL_R SCOREBIAS_D LATE_X FLOOR_A_S2 EARLYROOT_C_S2]
```

`FLOOR_A_S2` and `EARLYROOT_C_S2` are **second seeds** (`--seed 1`); the runner's own comment:
*"the A-vs-C conclusion is the whole point of the session, so it must not rest on a single unseeded
run."* Good instrument — and note `build_meta` fixes the val windows with
`np.random.default_rng(1)` independently of `--seed`, so **every arm evaluates on the identical
windows** and the seed perturbs only the training trajectory. The A-vs-C comparison is now two-seed.

The runner is **gated, not stuck**: free VRAM 18,136 MiB against the 20,000 MiB full-trunk gate, so
it holds until the incumbent exits and then launches the next stage itself. No human action needed.
`FLOOR_A` is at ~step 17,000 of 20,000, ETA ~01:14 UTC.

*Stale below:* the pid-413704 block and the "~6 h" schedule (six arms is ~7 h).

---

## 2. The ladder (already staged, ~6 h of the 20 h)

```
1 A  FLOOR_A      no root pathway                     ~7.75 % PLATEAU (answered, see 1b)
2 C  EARLYROOT_C  residual + score bias, layers 0-23  QUEUED
3 E  RESIDUAL_R   residual alone, layers 0-23         QUEUED
4 D  SCOREBIAS_D  score bias alone, layers 0-23       QUEUED
5 X  LATE_X       both mechanisms, layers 20-23 ONLY  QUEUED   <- the direct early-vs-late test
```
All `--unfreeze-trunk-all --trunk-lr-scale 1.0 --lr 1e-3 --steps 20000 --batch-size 32`.

**The sentence the session is trying to write:** does root structure from the **input stage** beat
attaching it at **layer 20**? If it's a tie, everything concluded about the late RCA's ceiling was
a statement about a *frozen* trunk and the architecture was never the bottleneck.

**E and D matter more than they look:** the score-bias path has **never received a gradient** in
this project (`stream_mix[1].grad == 0.0` exactly, 24/24 layers; element 0 live 24/24). Report its
gate trajectory explicitly.

Then **1.5B scale** and the **derivational holdout** — the latter needs new engineering.

---

## 3. The thesis — still never tested

The claim: one root embedding shared across كتب / كاتب / مكتوب / كتابة generalises to **unseen
derivational forms**. Attempt 1 was structurally impossible (a softmax cannot emit an unseen root,
proved at 4,117 positions); attempt 2 ran at **0.141 epochs**. Qiyās demonstrated it as a
*computed capacity* (79.10 % over the FULL 9,220-root inventory vs 0.00 %; the older
84.48 % was on a 37-root candidate set — see STATE.md). It has **never** been demonstrated as a
*learned* one.

**This is the project's actual open question. Nothing else on this list is.**

---

## 4. Hard-won rules — each of these cost a run tonight

```
NEVER a third CUDA occupant. And note: two 32-batch FULL-TRUNK arms do not fit either
  (17.69 GiB each vs 10.6 GiB for --unfreeze-last 4; 31.37 GiB card). Sequential for full-trunk.
The guard `pgrep -f "nrmt_train[.]py --checkpoint"` MISSES nrmt_train_width.py / _ewc.py / _v13.py
  -- it under-counts by 1-2. Every gate on this pod has been wrong because of it. Use broadened ps AND VRAM.
Checkpoints to /tmp (pod 50 G overlay). /workspace is MooseFS (NETWORK): its "466 T free" is the
  CLUSTER's. ENSOSPC killed two arms at a write with NO traceback. `df` lies here.
NEVER `|| true` on a transfer or a check. One upload "succeeded" while transferring nothing.
NEVER kill another agent's process. (The coordinator SIGTERM'd an arm that had already produced
  the session's cleanest attribution, and separately killed a repliceation the parent had asked for.)
Bit-identical proofs before training: gate 0 == no pathway (torch.equal). Then an isolation
  ablation (roots rolled, trunk input FIXED -> output changes). Then, only then, stack mechanisms.
A SELF-REPORTING METRIC CAN HIDE DAMAGE. Arm P looked healthy at 16.34 % while its trunk was
  destroyed (head-independent 2.676 %). Always keep a head-independent control.
When a check finds NOTHING, suspect the check before the claim. Three "citation problem" flags were
  retracted for exactly this (see STATE.md).
```

---

## 5. Traps that silently produce wrong results

```
RootformerNRMT state dict is backbone.layers.*; V13 holds the CausalLM backbone.model.layers.*
  A naive load_state_dict reports 555 missing / 520 unexpected and LOADS NOTHING -> random model.
Local `src/` is missing validated_segmentation.py -> nrmp_vocab reports 9445 roots, not 9490.
  The POD is correct at 9490/142. Assert the root count before trusting any number.
root_cross_attn.py:136 `clamp_` mutates the CALLER's tensor; harmless while morphemic_embed is
  FROZEN (index_select builds no grad node), fatal the moment the input stage trains.
  Workaround in root_arch/rca_compat.py; upstream fix is clamp_ -> clamp.
152 M orphan params in live forward hooks (layers 1/11/14) in NO checkpoint and NO model.parameters().
  Now perturbation-proved detachable. `--flash-hooks off`.
Shipped forward is MIXED dtype: q_norm/k_norm are float32 (never cast), so QK^T runs fp32 while
  v is bf16. SDPA needs one dtype -> bf16-flash shipped, measured against an fp64 oracle.
`--eval-every 0` -> ZeroDivisionError. `evaluate` never calls rca_stack.eval() (dropout active at
  eval, -0.13 pp).
```

---

## 6. The numbers to hold

```
marginal (word-stream)      3.551 %      RCA_UNFREEZE_A2        19.53 %   (5.50x)
RCA ablated                 2.82 %       RCA_NORM (frozen)      19.17 %
FIX baseline                6.837 %      RCA_W2N                19.39 % (best acc@5 40.11, CE_z 7.8456)
FLOOR_A (no root pathway)   ~8 %         EWC_Q20K (0.1x LR)     ~19.1 % plateau
root decodability: raw base 29.30 %  |  transmuted trunk 92.34 %
  local isolated probe: 21.83 % @ layer 4  ->  13.00 % @ layer 23  (DECAYS; transmuted is monotone UP)
qiyas on forms the learned decoder scored 0.00 % on:
  FULL 9,220-root inventory  79.10 %  <- honest
  37-root candidate set      84.48 %  <- the older, inflated number (see STATE.md)
levers: bound +2.54 | unfreeze 0.01x +3.0 | width +0.22 | depth +0.13   (CE_z does NOT improve)
```

**Every lever that worked fixed HOW WE INJECT. None changed WHAT THE TRUNK REPRESENTS.**

---

## REFRAMING (parent decision, 2026-10-03) — the thesis is not a hypothesis to test

**The root generalising across patterns is not something to prove. It is what the tradition has
assumed for two thousand years** — al-qiyās and al-ishtiqāq are built on it, and every Arabic
lexicon is organised by it. Asking a held-out split to confirm the organising principle of Arabic
grammar is an ML habit applied where it doesn't belong.

**And the capacity is already demonstrated computationally:** qiyās realises unseen-root forms at
**84.48 %** where the learned decoder scored **0.00 %**. A rule doesn't need a held-out split to
generalise — it applies to whatever it accepts. Unseen forms are not the hard case for a rule-based
realiser; **they are the normal case.**

### So: DROP the root-disjoint retraining experiment

```
DROP   root-disjoint split + ~6 h retraining, framed as "proving generalisation"
       -> different question, expensive, and it answers something the tradition settled
KEEP   end-to-end realisation accuracy on held-out roots
       -> not a proof of a principle: a measurement of the SYSTEM.
          same corpus, no retraining, and it fails in ways you can fix.
```

### The pipeline is the deliverable

```
1  ANALYZE    word -> (P,R,W,S)        the analyzer -- fix by cause, from the grammarians
2  PREDICT    context -> next tuple     the tuple-LM; the objective already exists
3  REALIZE    (P,R,W,S) -> surface      tokenizer + tasrif + qiyas + the orthography cases
                                        (divine name = surface-preserving passthrough)
4  GENERATE   the loop
```

**Text training waits on stage 3, not on a proof.** A morphemic model's ceiling is its realiser:
if realisation is at 57.59 %, a tuple-LM produces well-formed tuples that render as the wrong word.
Fix the analyzer and realiser first, then train on top of them.

**The metric that matters: does it produce correct Arabic** — which is the only question the
tradition was ever answering.
