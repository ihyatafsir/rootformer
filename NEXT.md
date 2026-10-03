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

## 2. Watch the ladder (already running, ~6 h of the 20 h)

```
1 A  FLOOR_A      no root pathway                     RUNNING  ~8 % at step 6k, NOT climbing
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
