# Controlled experiment: root cross-attention with the trunk FROZEN

**Run tag** `RCA_FROZEN_CTL` · pod `g6exduq0bd17z8` (213.173.104.76:46758) · dirs `/workspace/root_attn_ctl/`
(mirrored to `rootformer/build/root_attn_ctl/`) · 2026-10-02, started 14:56:43Z.

---

## 1. Verdict (the answer)

**The second fixed interpretation holds: the collapse was the unfreeze, not the root
cross-attention.** With the trunk frozen, root cross-attention `top4` does not merely hold the
frozen baseline — it **more than doubles it**:

| step | ALL_val acc@1 | acc@5 | CE_z |
|---|---|---|---|
| 1000 | **13.57 %** | 26.00 % | 6.4730 |
| 2000 | **15.09 %** | 30.00 % | 6.8064 |
| 3000 | **16.02 %** | 31.17 % | 7.1874 |
| 4000 | **15.94 %** | 31.83 % | 7.4586 |
| 5000 | **16.20 %** | 32.42 % | 7.5896 |
| 6000 | **16.24 %** | 32.64 % | 7.7573 |
| 7000 | **16.33 %** | 32.84 % | 7.8415 |
| 8000 | **16.46 %** | 33.27 % | 7.9038 |
| 9000 | **16.47 %** | 33.38 % | 7.9954 |
| 10000 | **16.68 %** | 33.53 % | 8.0779 |

FIX baseline (frozen trunk, cached h), same steps and same evaluator:
6.37 → 6.71 → 6.50 → 6.54 → 6.29 → 6.58 → 6.44 → 6.42 → 6.64 → 6.58 % acc@1.
Ratio ≈ **2.5×**, flat-to-rising, no collapse at any point through the halfway mark.

There is no collapse, no plateau at 0.2–0.3 %, and no NaN. The pre-registered rule
("Still collapses ≈0.2–0.3 % → the RCA itself is destabilising") is **not** met.

The live direction therefore becomes the productive one the brief named: give root attention
real width, and keep the trunk out of the way (or adapt it under a regime that does not destroy
it — see §4).

---

## 2. The exact change (one variable)

Nothing was edited in any module: `--unfreeze-last` already exists (the flag the previous agent
added), so **no `.bak`, no module diff, and no seven-suite re-run is owed**. Both arms now run
`nrmt_train.py` md5 `56ef3f2fd9ddc352dee596f32590e7b5`, which contains the val-stream evaluator
fix.

Diff of the *run script* only (`config_diff_vs_run_ROOTATTN.txt`, comments stripped):

```diff
-OUT=/workspace/root_attn; mkdir -p $OUT
+OUT=/workspace/root_attn_ctl; mkdir -p $OUT $OUT/ckpt
-TAG=${TAG:-ROOTATTN}
-UF=${UF:-4}
+TAG=${TAG:-RCA_FROZEN_CTL}
 ...
-    --root-cross-attn $RCA --unfreeze-last $UF \
-    --trunk-lr-scale $TRUNK_LR --rca-lr-scale 1.0 --rca-ablate-eval --h-drift-probe 500 \
+    --root-cross-attn $RCA --unfreeze-last 0 \
+    --trunk-lr-scale 0.1 --rca-lr-scale 1.0 --rca-ablate-eval --h-drift-probe 500 \
```

i.e. the **only** substantive change is `--unfreeze-last 4 → 0`. Deliberately *not* changed
(these are what the earlier arm `RCA_FROZEN`/`RCA_FROZEN_B2` varied, which is why they cannot
attribute the collapse): `--rca-lr-scale` stays **1.0** (not 0.1), and there is **no**
`--rca-out-norm` and **no** `--rca-dropout`. Same checkpoint, same aligned cache
(`/workspace/head_fix/nrmp_cache_9490_aligned`), same `--steps 20000`, `--batch-size 32`,
`--eval-every 1000`, `--head-init remap`, `--feat-gate --margin-ramp 10 --grad-warmup 10
--logit-scale ln`, `--trunk-lr-scale 0.1` (kept verbatim; inert with an empty trunk group).
The RCA code is the same file `Run A` used, `/workspace/root_attn/root_cross_attn.py`
(imported, never modified).

Confirmation from the log that the freeze took effect:

```
[*] trainable parameters: 22.43M (head 12.80M @lr 0.001 | rca 9.64M @lr 0.001 layers [20, 21, 22, 23])
[*] optimizer groups: head n=19 max_lr=4e-05 | rca n=36 max_lr=4e-05      <- NO trunk group
  [proof] step 1: max|dh|=0.000000e+00 ... grad_norm head=1.103e+01 rca=1.827e-05 trunk=0.000e+00
```

`grad_norm_trunk` is **exactly 0.0 in all 5 250 logged steps** of the trace (0 non-zero of 5 250)
— the trunk provably received no update.

---

## 3. The confounded run, re-measured (this is why Run A "collapsed")

`head_ROOTATTN.pt` and `head_ROOTATTN.pt.trunk.pt` were saved at *every* eval, so Run A's exact
step-5000 model still existed on disk. Re-running those same weights through the fixed evaluator
(`eval_saved_head.py`) recovers its true number without retraining.

**Tool validation (two ground truths, no free parameters):**

| model | my re-evaluation | trainer log | 
|---|---|---|
| FIX head + released trunk (step 20000) | 6.635 / 12.603 / CE_z 8.0997 | 6.62 / 12.57 / 8.100 |
| CTL step-1000 head + RCA | 13.588 / 25.963 / CE_z 6.4725 | 13.57 / 26.00 / 6.4730 |

**Run A (RCA top4 + unfreeze top4 + trunk_lr_scale 0.1), step 5000, fixed evaluator:**

| configuration | ALL_val acc@1 | acc@5 | CE_z |
|---|---|---|---|
| Run A as saved (RCA live) | **9.900 %** | 18.750 % | 7.7023 |
| Run A, RCA ablated at inference (gates 0) | 1.654 % | 3.233 % | 9.7351 |
| Run A head on the *released* trunk | 1.903 % | 4.208 % | 8.9787 |
| **FIX baseline's own head** + Run A trunk, gates 0 | **2.411 %** | 5.241 % | 9.5876 |
| FIX baseline's own head + Run A trunk, RCA live | 11.055 % | 20.229 % | 7.8211 |
| FIX baseline head + released trunk (reference) | 6.635 % | 12.603 % | 8.0997 |

So Run A's real held-out accuracy at step 5000 was **9.90 %**, not 0.23 %. And the *unfreeze* is
demonstrably what damaged it: the FIX baseline's **own** head — nothing retrained — drops from
**6.635 % to 2.411 %** simply by being placed on Run A's trunk with the RCA switched off. That is
a head-independent measurement of trunk damage; the per-layer relative weight change is
‖ΔW‖/‖W‖ = 3.46 / 3.63 / 3.78 / 3.59 % for layers 20 / 21 / 22 / 23.

**Direct reproduction of the reported "collapse".** Feeding the *train* windows to `live_h()` while
pairing them with the *val* targets (the first patch version's bug) reproduces the log exactly:

| model | buggy evaluator | trainer's logged number |
|---|---|---|
| Run A step 5000 | **0.228 %** / 0.843 % / CE_z **13.7095** | 0.23 % / 0.84 % / CE_z 13.7138 |
| **this healthy frozen arm**, step 1000 | **0.366 %** / 1.431 % / CE_z 9.9474 | 13.57 % / 26.00 % / CE_z 6.4730 |

The 0.2–0.4 % band is a property of the broken evaluator, not of any model. The pre-stated
decision band "≈0.2–0.3 %" was therefore never a capability measurement on either arm.

---

## 4. RCA learning dynamics (required sanity check)

**Gradient norms** (from `trace_RCA_FROZEN_CTL.jsonl`, 5 250 steps):

| group | min | median | p90 | max | last |
|---|---|---|---|---|---|
| `grad_norm_rca` | 1.83e-05 | 5.33e-01 | 1.15e+00 | 1.24e+01 | 2.76e-01 |
| `grad_norm_head` | 3.02e-01 | 6.18e-01 | — | 1.10e+01 | 3.45e-01 |
| `grad_norm_trunk` | 0.0 | 0.0 | 0.0 | **0.0** | 0.0 |

The RCA gradient is neither vanishing nor exploding, it is the same order as the head's, and it
does not decay to zero. 0 skipped steps, 0 non-finite losses, 0 NaN/Inf tokens anywhere.
Train root CE falls 7.341 → 0.085 by step 5 250 (**this arm memorises the training positions too** —
that is not by itself the failure; the held-out metric still improves).

**The gate opens**, monotonically, and ends the run far from 0 (it starts at exactly 0.0, which is
what makes "attached but untrained" a numerical no-op):

| step | gates (layers 20,21,22,23) | mean abs |
|---|---|---|
| 1000 | +0.1003, +0.1643, +0.2673, +0.1574 | 0.1723 |
| 2000 | +0.1395, +0.2742, +0.5443, +0.3706 | 0.3321 |
| 3000 | +0.1546, +0.3362, +0.7122, +0.5163 | 0.4298 |
| 4000 | +0.1607, +0.3727, +0.8051, +0.6088 | 0.4868 |
| 5000 | +0.1636, +0.3960, +0.8612, +0.6746 | 0.5238 |

**Attention weights are NOT degenerate** (`probe_rca_dynamics.py`, 8 val windows; recorder is a
process-local monkeypatch, the RCA source is untouched). This is the question the brief asked and
the answer is unambiguous:

| checkpoint | entropy/log(n) (1.0 = uniform, 0 = one-hot) | effective positions attended | mean max weight | frac rows w/ max>0.9 | mass on current root (diagonal) | argmax = current root | argmax = first root |
|---|---|---|---|---|---|---|---|
| CTL step 2000 | 0.517–0.533 | 10.5–11.4 | 0.344–0.405 | 3.3–4.1 % | 0.027–0.032 | 2.8–6.1 % | 0.0–0.1 % |
| CTL step 4000 | 0.474–0.520 | 8.9–10.8 | 0.361–0.441 | 3.9–6.0 % | 0.025–0.030 | 2.5–5.5 % | 0.0–0.1 % |
| **Run A** (damaged trunk) step 5000 | 0.491–0.554 | 9.4–11.9 | 0.379–0.421 | 3.2–4.5 % | 0.023–0.027 | 2.5–3.3 % | 0.0 % |

The distribution is broad and genuinely shaped (half the way from uniform to one-hot), spreads
over ~9–12 root positions, never collapses onto a single position, and does **not** even
concentrate on the current root (diagonal mass ≈ 0.025–0.032, i.e. near the uniform prior over the
visible row; attention logits |·|max ≈ 11–15, so this is not a saturation problem either). **Run A's
attention is just as healthy** — so neither the collapse nor the "too narrow" hypothesis shows up
as degenerate attention.

**The one real structural red flag is residual magnitude, not attention.** The RCA's injected
update is unbounded (free scalar gate × un-normalised `o_proj`, `--rca-out-norm` off here):

| checkpoint | injected delta RMS | max‖h(live) − h(no RCA)‖ | RMS of h itself |
|---|---|---|---|
| CTL step 2000 | 14.2–19.2 | 17.6 | 1.00 |
| CTL step 4000 | 15.7–20.5 | 22.1 | 1.00 |
| Run A step 5000 | 4.8–6.7 | 9.75 | 1.00 |

measured on the same tensor the model actually reads (post-`final_norm`), so these are comparable
units: **the cross-attention moves the hidden state by up to 22× its own RMS.** That is a genuine
design smell and the reason `--rca-out-norm`/`--rca-dropout` exist — but it is *not* what collapsed
Run A: this arm carries the identical unbounded residual and improves on the baseline. Train root
CE at Run A's step 4000 was 0.028 vs 0.187 here, i.e. the unfrozen trunk was absorbing the residual
into the trunk weights.

---

## 5. Status of the previous run's checkpoints (neither was overwritten)

Read-only `torch.load` + a safetensors comparison; nothing was written to either file.

**`/workspace/root_attn/head_ROOTATTN.pt` — 51.20 MB — INTACT, but trunk-coupled.**
A plain dict of **19** tensors, 12.799 M params, fp32, **0 NaN, 0 Inf**, all shapes correct
(`root_head.weight (9490,896)`, `wazn_head (142,896)`, `cond_proj.0 (896,1344)`, `feat_gate ()` …).
It is not a corrupt or empty artifact. But it is *not* a drop-in usable head: evaluated on the
released trunk it gives **1.903 %** (vs 6.635 % for the FIX baseline's own head), because it was
trained against the RCA-modified activations of a trunk that had moved. As one third of the triple
`head + .trunk.pt` it reproduces its arm exactly (9.900 % fixed evaluator / 0.228 % old evaluator).
Verdict: **intact and self-consistent, but only meaningful together with its trunk payload.**

**`/workspace/root_attn/head_ROOTATTN.pt.trunk.pt` — 143.82 MB — YES, it holds the damaged trunk.**
Metadata: `{"unfrozen_layers":[20,21,22,23], "rca_layers":[20,21,22,23], "live":true,
"rca_exclude_current":false, "rca_heads":8, "trunk_lr_scale":0.1, "rca_lr_scale":1.0, "step":5000}`.
128 tensors: 36 `root_cross.*` (9.636 M) + 92 trunk tensors for layers 20–23 (62.253 M), bf16,
**0 NaN, 0 Inf**, gate values +0.12695 / −0.15625 / −0.30469 / −0.24707 (matching the log).
Damage quantified against the released checkpoint: ‖ΔW‖/‖W‖ = 3.46 / 3.63 / 3.78 / 3.59 % for
layers 20 / 21 / 22 / 23. Its trained `o_proj` has grown to RMS 0.072–0.091 from a 1e-3 init, i.e.
70–90× the init scale — the unbounded residual again. The head-independent proof that this trunk is
damaged is in §3: the FIX baseline's own head reads 6.635 % on the released trunk and **2.411 %**
on this one with the RCA ablated.

---

## 6. Pre-existing pod process (nothing was killed)

* `nrmt_train.py` PID **8716** named in the brief is **gone**. `nvidia-smi -L` shows a single GPU
  (RTX PRO 4500 Blackwell, 32 623 MiB).
* What was actually running: **`chain_final.sh` (PID 17136, started 14:52Z)** — the parent agent's
  chain — running arm **`RCA_UNFREEZE_A2`** (PID 17294): `--root-cross-attn top4 --unfreeze-last 4
  --trunk-lr-scale 0.01 --rca-lr-scale 0.3 --rca-out-norm --rca-dropout 0.1`. **I killed nothing
  and preempted nothing.**
* **Stated plainly:** this was a live run, not a leftover, and I ran my arm *concurrently* instead of
  idling ~70 min for it and then contending with the chain's next arm. I reported this to the parent
  before launching and offered to yield. Concurrency costs wall-clock only; the two processes share
  no state and the metrics are unaffected. Its trajectory (fixed evaluator, as a bonus data point)
  is 10.40 → 16.86 → 17.77 → 18.45 → 18.51 → 18.64 → 18.62 % acc@1 over steps 1000–7000:
  **unfreezing at `trunk_lr_scale 0.01` with a bounded residual does not collapse either.**

**Cost / resources.** Peak total GPU memory observed while both arms ran: **21 540 MiB / 32 623**
(A2's own ~10.3 GiB ⇒ this arm's own peak ≈ 11.2 GiB; previously measured isolated arms with the
same payload: 9 478 MiB frozen, 11 787 MiB unfrozen). GPU utilisation ~100 %.
Wall-clock: started 14:56:43Z, reached step 6000 at 15:35Z = 38.7 min under contention
(≈2.7 it/s). The identical single-process arm (Run A) ran 5 200 steps in 1 081 s = 4.8 it/s, i.e.
**20 000 steps is ≈69 min alone, not the ≈15 min the brief assumed — a ~4.6× underestimate.**
NaN: **none** (0 non-finite losses, 0 NaN/Inf in the trace, 0 WARN lines).

**Run status at hand-off:** alive and continuing under `setsid` toward 20 000 steps; per-1000-step
`head`/RCA snapshots are being captured in `/workspace/root_attn_ctl/ckpt/` (steps 1000–6000 so
far). `results_RCA_FROZEN_CTL.json` is written only when the run ends.

---

## 7. Regression suites

No module was edited, so the seven suites were not required and were not re-run. For the record the
post-RCA run of the seven suites already on disk (`/workspace/root_attn/suites_after_rootattn.log`)
passes: `test_grammar_impl.py` 20/20, `verify_v2.py` exit 0, `andalusian_realizer.py` 67/67,
`test_sibawayh_governor.py` 15/15, `ibn_malik_automaton.py` 46/46, `test_khalil_orbits.py` 30/30,
`test_awzan_order.py` 24/24.

---

## 8. Files

On the pod (`/workspace/root_attn_ctl/`) and mirrored locally (`rootformer/build/root_attn_ctl/`):

| file | what |
|---|---|
| `run_frozen_ctl.sh` | the run script (the one-variable delta) |
| `config_diff_vs_run_ROOTATTN.txt` | diff against `run_rootattn.sh` |
| `remote_logs/train_RCA_FROZEN_CTL.log` | full log incl. every eval line and gates |
| `remote_logs/trace_RCA_FROZEN_CTL.jsonl` | per-step loss / grad norms (head, rca, trunk) / h_drift |
| `remote_json/probe_rca_*.json`, `probe_CTL_step*.json` | attention / gate / residual diagnostics |
| `remote_json/eval_*.json` | all re-evaluations of the confounded run's checkpoints |
| `remote_json/ckpt_forensics.json` | checkpoint forensics |
| `ckpt_forensics.py`, `probe_rca_dynamics.py`, `eval_saved_head.py`, `repro_eval_bug.py` | the probes (all read-only) |
| `ckpt/rca_step_*.pt`, `ckpt/head_step_1000.pt` | mirrored snapshots (larger ones stay on the pod) |

**Not overwritten, not moved, not modified:** `/workspace/root_attn/head_ROOTATTN.pt`,
`/workspace/root_attn/head_ROOTATTN.pt.trunk.pt`, `/workspace/root_attn/root_cross_attn.py`,
`/workspace/root_attn/*.log`, `nrmp_vocab.py`, the blueprint, the checkpoints, the shipped caches.
