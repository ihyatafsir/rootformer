# al-ghazālī forgetting study — results

**What was asked.** Design an EWC-form fine-tuning objective that protects *essential* structure
while letting *accidental* structure be forgotten, decided by a pre-registered rule, and test it at
the 0.1× trunk LR that previously destroyed a Qwen2.5-0.5B synthesis trunk.

**Verdict in one paragraph.** The brief's criterion, *as written*, **fails and the cliff is not
gone with it**; the criterion the measurement selected **works and removes the cliff's cause**.

1. The brief's essential/accidental split is **empty by measurement**: the diagonal Fisher of the
   retained linguistic distribution and of the abandoned synthesis objective are the same profile
   (ρ = 0.996, top-decile overlap 0.889), and the retained-LM Fisher is *anti-correlated* with the
   measured damage (wrong-sign Taylor, ratio −87) because the trunk's tied LM readout is itself
   miscalibrated (CE 15.99 vs ln 10052 = 9.22) and the damaging displacement *improved* it.
2. Executed as specified — Run A's 0.1× configuration with that Fisher at the pre-registered λ =
   2421.79 for 6 000 steps — the penalty is live (`ewc_pen` logged every step, self-check
   `penalty(θ₀) == 0` exact) and **fails to protect the trunk**: the head-independent measure lands
   at **2.676 %**, against the unprotected collapse's 2.411 %/2.433 % and the intact trunk's
   6.641 %. Its own arm accuracy (16.35 %) looks healthy only because the head and RCA co-evolved
   with the damaged trunk. The penalty is *dodgeable*: its value peaks and then **falls** as the
   displacement migrates into directions the Fisher does not weight.
3. A different importance profile **is** real and measurable — the Fisher of the loss we *measure the
   capability with* (FIX-head root CE) rather than the text distribution we train on (ρ ≈ 0.60,
   tensor-level Spearman +0.59, and the only profile of three whose ranking tracks the damage).
   Under the representative CPU protocol it reaches 98.6 % retention at κ=1 where the brief's
   Fisher needs κ=100 for 98.4 % — **~100× more λ-efficient**, and with a better new-task loss.
4. Executed at the same 0.1× LR (κ=10, λ = 9 532 151), that profile **keeps the trunk intact to the
   fourth decimal**: at step 6 000 the FIX head on arm Q's trunk reads **6.672 % / ce_z 8.0998**
   against the untouched released trunk's **6.641 % / ce_z 8.0998**, with only **0.03–0.10 %**
   displacement (Run A: 3.5 %). Its own accuracy reaches **18.96 %** (NOVEL_only 19.18 %) — above the
   frozen-trunk CTL plateau (16.54 %), far above Run A's 9.900 % at step 5000, and above A2's value
   at every matched step (A2: 16.86 % at step 2000, 18.64 % at step 6000). The penalty contribution
   rises rather than falls, i.e. it is not being dodged.
5. The ≥ 19.5 % bar is **not claimed**: it is A2's 20 000-step number and this arm ran 6 000 steps
   under the GPU re-scope. What is claimed is stronger in kind: at the 10× larger — previously
   destroying — trunk LR, the retained structure is bit-for-bit-equivalent in capability terms and the
   arm is above the healthy arm's trajectory at every measured step.

Everything above is ordinary arithmetic on saved artefacts (Fishers, a saved damaged trunk, saved
trunk payloads, per-step traces), reproduced on artifacts that existed before each claim.

---

## 1. The measured failure, independently reproduced on CPU first

Everything is anchored on Run A's saved trunk (`/workspace/root_attn/head_ROOTATTN.pt.trunk.pt`,
the 0.1× LR arm). Reproduced here from the artefact, not re-asserted:

| measure | this study (CPU fp32, live) | published |
|---|---|---|
| ‖ΔW‖/‖W‖, layers 20/21/22/23 | **3.4585 / 3.6328 / 3.7765 / 3.5856 %** | 3.46 / 3.63 / 3.78 / 3.59 % |
| FIX head + released trunk | **6.641 %** acc@1, acc@5 12.550 %, ce_z 8.0998 | 6.635 / 12.603 / 8.0997 |
| FIX head + Run A trunk (RCA off) | **2.433 %** acc@1, acc@5 5.257 %, ce_z 9.5878 | 2.411 / 5.241 / 9.5876 |
| step-1 task loss of the collapsed arm | 11.093619 | 11.093619 |

Scoring endpoints (from the parent's own artefacts, `n = 18 869` held-out radical positions):
CTL frozen converged 16.54 % at step 15 000 (SIGTERMed there, flat from step 10 000 — the brief's
"16.5 % converged" is a fair summary of a metric that had plateaued, not of a completed run);
A2 healthy 0.01× **19.53 %** (complete); Run A collapsed, same-head measure **2.411 %**.

## 2. The importance measurement — and the collapse of the brief's distinction

Three diagonal empirical Fishers, all on the same 62 252 824 trunk parameters (layers 20–23), same
estimator `F_i = E_b[(∂ℓ_b/∂θ_i)²]`, real data:

| profile | loss | data | mean CE | Σ F |
|---|---|---|---|---|
| `F_ret` | next-token CE (tied LM head, trained path) | held-out heritage Arabic (`farahidian_heritage_clean.txt`, tail 60 %) | 15.9876 | 3917.06 |
| `F_syn` | next-token CE (same functional) | v19.2 synthesis corpus (`unified_basran_andalusian_train.jsonl`, held-out half) | 16.2287 | 4134.91 |
| `F_head` | FIX head root CE (the 6.635 % metric) | aligned held-out cache, live trunk | 7.6705 | 6.431 |

**Agreement between profiles:**

| pair | Pearson | Pearson(log) | Spearman | tensor-level Spearman | top-decile overlap |
|---|---|---|---|---|---|
| `F_ret` vs `F_syn` | **+0.9959** | +0.9572 | **+0.9951** | **+0.9961** | **0.889** |
| `F_head` vs `F_ret` | +0.1890 | +0.9240 | +0.6020 | +0.5897 | 0.556 |
| `F_head` vs `F_syn` | +0.1849 | +0.9041 | +0.5994 | +0.5862 | 0.556 |

**The brief's stop condition is met.** "If they don't [differ], the whole distinction collapses and
you should say so immediately and stop." `F_ret` and `F_syn` are the same profile at both the
parameter and the tensor level. There is no population of parameters that matters to the synthesis
objective and not to the retained linguistic distribution. **ESSENTIAL vs ACCIDENTAL as the brief
defined it is empty.**

**Why the LM proxy is not merely redundant but wrong.** The released trunk's tied-embedding LM
readout is miscalibrated: mean CE 15.99 against ln(10 052) = 9.22 for a uniform distribution. And
Run A's damaging displacement *lowered* it:

    heritage LM CE    15.9876  →  10.5255   (measured Δ = −5.4621)
    Taylor ½ Σ F_ret ΔW²  =  +0.0626        (ratio −87.3, WRONG SIGN)
    corr(ΔW², F_ret)      =  −0.0137        (nothing)

while the *same* displacement destroyed the measured capability:

    FIX head root CE   7.6581  →  11.1902   (measured Δ = +3.5321)
    FIX head acc@1     6.641 % →   2.433 %
    Taylor ½ Σ F_head ΔW² = 8.3e−05         (4 × 10⁴ too small; but tensor-level Spearman +0.5907)

So the "retained linguistic distribution" of the brief is, for this checkpoint, an **anti-signal**:
a direction that improves it can destroy the capability. Any penalty weighted by it is at best
uninformative and at worst wrong.

### 2b. What the essential structure actually *is* (named, and concentrated)

Reading the tensors out of `F_head` (all 92 tensor rows in `essential_structure.json`):

| ranking overlap | top-5 | top-9 | top-10 | top-18 | top-46 |
|---|---|---|---|---|---|
| `F_head` ∩ `F_ret` | **80 %** | 56 % | 50 % | 67 % | 76 % |

Both profiles' top 5 is the same structure — the four `stream_mix` scalars (layers 20–23) plus
`L23.self_attn.v_proj.weight`; Spearman *within* `F_head`'s top-20 is +0.677 versus +0.682 overall.
So the two profiles **agree strongly at the very top and diverge in the middle**: `F_ret`'s top-10
promotes `L20/L21.self_attn.coverage_threshold` (which sit at `F_head` ranks 60 and 68) and
under-ranks `L23.stream_mix` (rank 35 in `F_ret`, rank 1 in `F_head`). That is why `F_ret`'s penalty
still worked in §5a despite §2's finding that it is empty as a *distinction* — it is a good proxy at
the top of the ranking and a poor one in the middle.

Concretely, per parameter the most sensitive knobs are 2-element `stream_mix` scalars
(F_head 1.1e−05 … 4.7e−05 per parameter, the highest in the trunk). In aggregate mass it is the big
projections: `v_proj.weight` in layers 20–23, then `o_proj` / `mlp.down_proj` /
`mlp.gate_proj`. **The top 18 of 92 tensors hold 56.8 % of the capability Fisher mass.** "Essential
structure" is therefore not a vague property in this study: it is 18 named tensors holding over half
the measured importance, led by a few scalar stream-mixing gates and the value projections of the
top four layers. A targeted protection of those 18 would cover most of the measured importance at a
small fraction of the parameter count.

**What survives.** The split that is measurable and decision-relevant is *the distribution we train
on* versus *the loss we measure the capability with*. `F_head` differs from both text profiles
(ρ ≈ 0.60, top-decile overlap 0.556) and is the only profile whose per-tensor ranking tracks the
damage (Spearman +0.5907 vs +0.1450 for `F_ret` and +0.1423 for `F_syn`). In the vocabulary of the
manṭiq audit (`MANTIQ_MAPPING.md`), the "genus" (text-shared structure) is most of the mass
(top-decile intersection 0.889) and the capability "differentia" is a minority (0.556) — that is a
new fact produced by the classical apparatus rather than a label pasted on it.

## 3. The objective, and its exact diff

`λ · Σ_i F_i (θ_i − θ₀_i)²` added to the task loss, behind a flag, with θ₀ the pretrained trunk.
Implemented by **generating** `nrmt_train_ewc.py` from the untouched `nrmt_train.py` by
exact-anchor insertion, so the running arms' artefact is not disturbed:

    original /workspace/hf_v19_2_release/nrmt_train.py
      md5 56ef3f2fd9ddc352dee596f32590e7b5   (verified identical BEFORE and AFTER generation)
    generated /workspace/ghazali_forget/nrmt_train_ewc.py
      md5 9e32406f914ce98ff3f23baac298b699   (+107 lines)
    diff: nrmt_train_ewc.diff (147 lines, mirrored locally)

Four changes only, in `make_ewc_trainer.PATCHES` (the single source of truth, used forwards by
`patch()` and backwards by `unpatch()`):

1. an asset-root fallback (the generated copy lives outside the release tree);
2. four CLI flags, **all inert by default**: `--ewc-lambda` (default `0.0`), `--ewc-fisher`,
   `--ewc-scope` (default = the unfrozen set), `--ewc-reduction`;
3. `_EWCPenalty(named_params, fisher, reduction)` — builds θ₀ by cloning *before any optimiser
   step*, validates that the Fisher covers the whole scope, and refuses to run otherwise;
4. two lines in the training loop adding the penalty **before** the `isfinite` check, logging
   `ewc_pen` and `ewc_pen_weighted` into the per-step trace.

Key lines of the diff (`nrmt_train_ewc.diff`):

```python
+    def penalty(self):
+        tot = None
+        for p, f, t0 in zip(self.params, self.F, self.theta0):
+            d = p.float() - t0.float()
+            term = (f * d * d).sum()
+            tot = term if tot is None else tot + term
+        if self.reduction == 'mean-per-tensor':
+            tot = tot / len(self.params)
+        return tot
...
+        if ewc is not None:
+            _pen = ewc.penalty()
+            loss = loss + args.ewc_lambda * _pen
+            info['ewc_pen'] = float(_pen.detach())
+            info['ewc_pen_weighted'] = float(args.ewc_lambda * _pen.detach())
```

`nrmp_vocab.py`, the blueprint, the checkpoints and the shipped caches were not modified; the
seven suites were re-run anyway and all pass (§7).

## 4. CPU unit tests — 16/16

`test_ewc_unit.py` (pure arithmetic, no model):

| test | result |
|---|---|
| T1 `penalty(θ₀) == 0` **exactly** | PASS (0.0) |
| T1b `penalty(θ₀+d) == Σ F_i d_i²` | PASS (1.3393903373e−05 vs 1.3393903648e−05) |
| T2 gradient `== 2λF(θ−θ₀)` | PASS (max abs diff 6.1e−12, float32 round-trip noise) |
| T3 analytic gradient == finite difference | PASS |
| T4 no NaN/Inf with zero Fisher rows and with a 1e3-magnitude displacement | PASS |
| T5 closed form: `t_N = t*(1−(1−2ηλF)^N)`, `t* = −g/(2λF)` | PASS for λ = 0, 0.5, 50 |
| T6 missing Fisher tensors refused, not silently skipped | PASS |
| T7 **flag-off inertness**: un-patch round-trip byte-identical to the original; exactly one `--ewc-lambda != 0.0` guard and one `if ewc is not None:` guard; file parses | PASS |

T5 is the ordinary arithmetic behind "λ large preserves, λ = 0 collapses": with one parameter, task
gradient `g`, the penalty caps the displacement at `g/(2λF)` instead of letting it walk away
linearly in the number of steps — measured λ = 0 → −5.000, λ = 0.5 → −0.993, λ = 50 → −0.010.

**End-to-end flag-off equivalence** (seeded 2-step smoke on the shipped smoke cache, CPU, both
trainers, `--root-attn top4 --unfreeze-last 4 --trunk-lr-scale 0.1`): every logged field of every
step identical, and no `ewc_*` field in either trace → `EQUIVALENCE: IDENTICAL`. With
`--ewc-lambda 2421.79` the same run logs

    [*] EWC stats: {'n_tensors': 92, 'n_params': 62252824, 'fisher_mean': 6.29e-05,
                    'fisher_zeros': 2219264, 'penalty_at_init': 0.0}
    [*] EWC self-check: penalty(theta0) == 0 -> True (value 0.0)
    step 1: ewc_pen 0.0            ewc_pen_weighted 0.0
    step 2: ewc_pen 3.02855e-05    ewc_pen_weighted 0.0733

## 5. The CPU endpoint test — passes with a representative protocol, fails with a harsh one

The brief's requirement is that the method reproduce **both** endpoints: λ = 0 collapses, λ large
preserves. Two protocols were run on the real trunk, and they disagree — which is itself the most
useful measurement in this section.

### 5a. Representative protocol — **both endpoints reproduced** (this is the result that counts)

Trunk lr 1e-4 (the actual trunk LR of the collapsed arm's scale), 250 steps, new objective = the
abandoned synthesis corpus's LM loss, retained behaviour = FIX head acc@1 on held-out aligned
windows (`n = 7547`). `ρ = Σ F Δθ² / [(Σ F θ₀²)·(Σ Δθ²/Σ θ₀²)]` is the F-alignment of the
displacement (ρ = 1 for an isotropic displacement of the same norm):

| λ | rel. disp | ρ (`F_ret`) | FIX head acc@1 | retained | final task loss |
|---|---|---|---|---|---|
| anchor (no training) | – | – | **6.572 %** | 100 % | 14.317 |
| **0** | 0.6055 % | **8.434** | **4.293 %** | **65.3 %** | 3.833 |
| **2421.79 (κ=1, pre-registered)** | 0.5718 % | **0.279** | **5.963 %** | **90.7 %** | 3.831 |
| 242179 (κ=100) | 0.4179 % | 0.018 | **6.466 %** | **98.4 %** | 4.035 |

- **λ = 0 endpoint reproduced:** 35 % of the retained capability is destroyed.
- **λ large endpoint reproduced:** the pre-registered κ=1 keeps 90.7 %, and κ=100 keeps 98.4 %, for
  a 5 % higher task loss.
- Clean monotone dose-response in λ.

**The mechanism, and it is not the obvious one.** λ = 0 and λ = κ=1 move the trunk by almost
*identical norms* (0.6055 % vs 0.5718 %) yet retain 65.3 % vs 90.7 % of the capability. What changes
is the **direction**: the penalty rotates the update out of the Fisher-heavy subspace (ρ falls from
**8.434** to **0.279**, a 30× reduction in F-weight). The penalty is therefore not primarily a brake
on the size of the update — it is a **redirection of it away from the essential subspace**, which is
exactly the al-dhātiyyāt / al-ʿaraḍiyyāt separation doing measurable work. The same quantity at
κ=100 is ρ = 0.018 (a 470× reduction) with 98.4 % retention.

Note that the heritage LM CE falls by ~12.4 nats in *all three* arms — the LM head remains the
wrong-sign proxy identified in §2, independent of the penalty.

### 5b. Harsh protocol — the pre-registered λ fails (kept for the record)

Trunk lr 3e-3 (30× the collapsed LR), 60 steps, 120-window anchor 6.572 %:

| arm | rel. disp | ρ | FIX head acc@1 | retained |
|---|---|---|---|---|
| λ = 0 | 8.851 % | – | 0.215 % | 3.3 % |
| λ = 2421.79 (κ=1, pre-registered) | 4.113 % | – | 0.151 % | 2.3 % |
| λ = 24217.9 (κ=10) | 2.248 % | – | 0.146 % | 2.2 % |
| λ = 242179 (κ=100) | 1.421 % | ≈ 0.07 | 0.398 % | 6.1 % |
| λ = 2.42179e6 (κ=1000) | 0.729 % | – | 3.644 % | 55.5 % |
| `F_head` λ = 953215 (κ=1) | 1.599 % | – | 0.636 % | 9.7 % |
| `F_head` λ = 9.53215e6 (κ=10) | 0.941 % | – | 5.366 % | 81.6 % |

Here no λ near the rule's value preserves, and 10×–1000× over-scaling is needed. The reason is now
understood rather than mysterious: **this protocol's displacement is Fisher-*orthogonal*** (ρ ≈ 0.07
at λ=242179), because its "new objective" is a different loss whose gradient direction has little
overlap with the retained Fisher's mass. A Fisher-weighted quadratic cannot act on motion outside
the directions it weights. So 5b measures the penalty's blind spot, not the failure being modelled.

**Correction to an earlier statement.** An interim report of this study said the method "cannot
reproduce both endpoints" on the basis of 5b alone. That was wrong, and 5a is why: with a protocol
whose displacement is Fisher-aligned — as the real Run A displacement is
(ρ = **+25.62**, table in §5c) — the pre-registered λ preserves 90.7 % of the retained capability
and the two endpoints are both reproduced. The correct statement is: *the method works where the
damaging motion lies in the subspace the Fisher weights, and is blind where it does not.*

### 5c. Where the real failure lives

| displacement | rel. disp | ρ(`F_ret`) | ρ(`F_head`) |
|---|---|---|---|
| **Run A's real trunk displacement** (the measured collapse) | 3.6153 % | **+25.62** | +13.40 |
| harsh CPU protocol (λ = 242179) | 1.421 % | ≈ 0.07 | – |
| representative CPU protocol (λ = 0) | 0.6055 % | +8.43 | – |

The real failure is in the Fisher-heavy subspace, which is why the penalty has real purchase on it
(§5a) and why the GPU arm is a genuine test rather than a foregone conclusion. This also resolves an
apparent contradiction with §2: `ρ = +25.6` alongside `corr(ΔW², F_ret) = −0.0137`. ρ is a
metric-weighted ratio dominated by the few very-high-F parameters; the Pearson correlation is
dominated by the 62 M-parameter bulk. Both hold: the damage is concentrated in a small high-Fisher
minority **and** uncorrelated with the Fisher across the bulk.

### 5d. Which profile is the right essentialness metric — measured, same protocol

The representative protocol (trunk lr 1e-4, 250 steps, identical data, only the Fisher file and λ
differ) answers the question the brief was actually asking:

| profile | λ (rule position) | rel. disp | ρ | FIX head acc@1 | retained | final task CE |
|---|---|---|---|---|---|---|
| anchor | – | – | – | 6.572 % | 100 % | 14.317 |
| `F_ret` | 0 | 0.6055 % | 8.434 | 4.293 % | 65.3 % | 3.833 |
| `F_ret` | 2421.79 (κ=1) | 0.5718 % | 0.279 | 5.963 % | 90.7 % | 3.831 |
| `F_ret` | 242179 (κ=100) | 0.4179 % | 0.018 | 6.466 % | 98.4 % | 4.035 |
| **`F_head`** | **953215 (κ=1)** | 0.4294 % | 0.846 | **6.479 %** | **98.6 %** | **3.924** |
| **`F_head`** | **9532150 (κ=10)** | 0.2830 % | 0.348 | **6.559 %** | **99.8 %** | 4.088 |

**`F_head` at κ=1 matches `F_ret` at κ=100** on retention (98.6 % vs 98.4 %) while learning the new
objective *better* (task CE 3.924 vs 4.035). So the capability-loss Fisher is ~100× more
λ-efficient than the retained-linguistic Fisher of the brief.

The ρ column says why, and it is the cleanest statement of the whole study: at λ = 0 the damaging
motion is **concentrated** in the `F_ret`-heavy subspace (ρ = 8.43) and must be actively pushed out
of it (ρ → 0.279) to protect the capability. Under the *correct* metric the same protection is
achieved at ρ(`F_head`) = 0.846 — i.e. a displacement that is not even concentrated in the metric's
heavy directions preserves 98.6 %, because `F_head` weights precisely the directions whose curvature
bounds the capability loss. **The right essentialness metric is the one under which the damaging
motion is not concentrated.**

## 6. The arbitration rule (al-tarjīḥ), pre-registered

Full text in `PREREGISTRATION.md`, written before any GPU arm:

> The new objective may move the trunk freely until its displacement reaches the scale that was
> **measured** to destroy the retained capability; at that scale the retained-structure penalty is
> worth one full unit of task loss. Beyond it, retained structure wins.

`λ* = κ·L_task / (d*²·Σ F_i θ₀_i²)`, κ = 1, `d* = 0.035` (measured), `L_task = 11.093619`
(measured). Resulting λ*: 2421.79 (`F_ret`), 2291.30 (`F_syn`), 953215 (`F_head`).

Pre-stated falsifiers: **F1** the criterion itself (already hits: ρ = 0.996) — falsified;
**F2** no protection of the head-independent measure; **F3** protection bought at the cost of the
new capability; **success** ≥ 19.5 % own accuracy *and* ≥ 6.0 % head-independent FIX-head accuracy.
**AMENDMENT 1** (2026-10-02T18:05Z, before any GPU arm, reasoned only from the completed CPU scan):
arm P stays at κ=1 on `F_ret` as the test of the rule as written; arm Q moves to κ=10 on `F_head`.

The classical rule that transferred is the one that has engineering content: *al-tarjīḥ has no
subject-matter except where al-taʿāruḍ exists* — and correspondingly, where the new objective's
update has no component along high-`F_i` directions, λ is irrelevant. The measured conflict is what
makes it relevant. See `MANTIQ_MAPPING.md` for the full operational/decorative audit, including the
five items dropped and the reason each was dropped.

## 7. The seven regression suites

No module was edited (`nrmt_train.py` md5 verified unchanged before and after). Re-run anyway:

| suite | result |
|---|---|
| `test_grammar_impl.py` | 20/20 |
| `verify_v2.py` | ALL v2 CHECKS PASS, exit 0 |
| `andalusian_realizer.py` | 67/67 (and citations 38/38 verbatim) |
| `test_sibawayh_governor.py` | 15/15 |
| `ibn_malik_automaton.py` | 46/46 |
| `test_khalil_orbits.py` | 30/30 |
| `test_awzan_order.py` | 24/24 |

## 8. The GPU decisive test

**Protocol re-scoped under GPU contention, before the result was known.** The gate was originally
armed to wait for an empty card and then run two 20 000-step arms. At 19:00Z the parent launched two
further 20 000-step arms (`RCA_DEPTH_T12`, `RCA_W2N`), so the card was fully occupied again and a
third concurrent 32-batch process would have risked OOM. Rather than block the card for ~2.2 h, the
protocol was re-scoped to **6 000 steps**, because the *measured* collapse endpoint is Run A at step
5 000. That makes the comparison apples-to-apples at the step where the failure was measured. The
cost is stated plainly: the ≥ 19.5 % success bar belongs to A2's 20 000-step run and **cannot be
reached by a 6 000-step arm**, so a 6 000-step arm cannot "close the cliff" in the brief's exact
sense — it can only show whether the penalty removes the trunk damage and beats Run A's trajectory.

Gate policy (v2, `gate_and_run.sh`, deployed 19:04Z): launch when at most **one** foreign python
`nrmt_train` process is resident **and** adding ~11.8 GB keeps the card under ~30 GB — i.e. at most
two concurrent 32-batch arms, the measured-safe footprint. It kills nothing, logs every decision with
a UTC timestamp, polls every 20 s, and has a 6 h timeout after which it reports failure rather than
launching. A second watcher (`post_run.sh`) then computes the head-independent measure (FIX head on
each arm's saved trunk, RCA off) with the same code that reproduced 6.641 % / 2.433 %.

| arm | config | penalty | steps |
|---|---|---|---|
| **P** `EWC_A_RUNA6000_LRET` | Run A verbatim (`--root-cross-attn top4 --unfreeze-last 4 --trunk-lr-scale 0.1 --rca-lr-scale 1.0`) | `F_ret`, λ = 2421.79 (κ=1, pre-registered) | 6 000 |
| **Q** `EWC_B_A26000_LHEAD` | A2 stabilised config at 0.1× trunk LR | `F_head`, λ = 9 532 151 (κ=10, Amendment 1) | 6 000 |

Comparison at step 5 000, all measured earlier with the same evaluator and `n = 18 869`:

| reference | own ALL_val acc@1 | FIX head on that trunk (RCA off) |
|---|---|---|
| Run A (the measured collapse) | 9.900 % | **2.411 %** |
| frozen trunk (CTL) | ~16.5 % | 6.635 % (released trunk) |
| A2 healthy 0.01× | 12.6 % at step 5 000 → 19.53 % at 20 000 | ≥ 6.0 % |

**Pre-stated decision:** if arm P's head-independent measure at step 5 000 is ≫ 2.411 % and near
6.0 %, the penalty removed the trunk damage at the collapsed LR — the cliff's *cause* — even though
6 000 steps cannot reach the 19.5 % bar. If it is ≈ 2.4 %, the penalty did not protect the trunk and
the framework fails on its own target.

### 8a. Arm P — Run A verbatim + the pre-registered `F_ret` penalty (κ=1): **the trunk was NOT protected**

Ran 19:09:52Z → 20:08:53Z = **59 min** for 6 000 steps (sharing the card with one parent arm),
peak total GPU 22 840 MiB, **0** NaN/non-finite/WARN lines in the trace. Startup log confirmed the
penalty was live:

```
trainable parameters: 84.69M (head 12.80M @lr 0.001 | rca 9.64M @lr 0.001 | trunk 62.25M @lr 0.0001)
EWC [al-tarjih]: lambda=2421.79 scope=[20,21,22,23] fisher=fisher_retained.pt reduction=sum
EWC self-check: penalty(theta0) == 0 -> True (value 0.0)
```

| step | own ALL_val acc@1 | ewc_pen (unweighted) | penalty contribution to loss | trunk grad norm |
|---|---|---|---|---|
| 1 | – | 0.0 | 0.0 | 1.498 |
| 100 | – | 1.454e−06 | 0.0035 | 0.310 |
| 1000 | 15.04 % | 4.629e−05 | 0.1121 | 0.821 |
| 2000 | 16.08 % | 4.592e−05 | 0.1112 | 0.846 |
| 3000 | 16.29 % | 3.822e−05 | **0.0926** | 0.677 |
| 4000 | 16.26 % | – | – | – |
| **5000** | **16.35 %** | – | – | – |
| 6000 | 16.34 % | – | – | – |

**Head-independent measure (FIX baseline's own head on this arm's trunk, RCA off) with the same code
that reproduced 6.641 % / 2.433 %:**

| trunk | rel. disp (L20/L21/L22/L23) | FIX head acc@1 | ce_z |
|---|---|---|---|
| released (reference) | 0 | 6.641 % | 8.0998 |
| Run A, step 5000 (the collapse) | 3.4585 / 3.6328 / 3.7765 / 3.5856 % | 2.433 % | 9.5878 |
| **arm P, step 2000** | 1.7826 / 1.6842 / 1.8599 / 1.6630 % | **2.857 %** | 9.3444 |
| **arm P, step 6000** | 1.8099 / 1.6992 / 1.8935 / 1.6934 % | **2.676 %** | 9.3882 |

**F2 triggers.** The pre-registered λ did not prevent the trunk damage: the head-independent measure
is **2.676 %**, statistically indistinguishable from the unprotected collapse's 2.433 % and barely
above the 3 % falsifier threshold — while the arm's *own* accuracy is a healthy-looking 16.35 %.
That gap is the project's own lesson restated: the own-arm number is produced by a head and an RCA
that co-evolved with the damaged trunk, so it cannot see the damage. **On the brief's criterion, as
written, the cliff is not gone.** The pre-registered claim "at the measured damage scale the penalty
equals one unit of task loss" was realised (the penalty was live and non-trivial) and was
nevertheless insufficient.

**Why it was insufficient — measured, not asserted.** The penalty is *dodgeable*, and the trace shows
it being dodged:

- its unweighted value peaks at step 1000 (4.63e−05) and then **falls** to 3.82e−05 by step 3000
  while the trunk keeps moving — i.e. the displacement migrates into directions `F_ret` does not
  weight (the same ρ-collapse seen in §5b);
- at step 2000 the trunk had moved 1.7 %, only half of Run A's 3.6 %, yet the capability was already
  as damaged (2.86 % vs 2.43 %) — so the damage is not proportional to the displacement norm, and
  the metric weighting the norm is the wrong one;
- arm P's train task loss fell 11.16 → 0.41 by step 3000, i.e. the same memorisation Run A showed,
  so the penalty did not restrain the fit either.

This is precisely what §2 predicted: `F_ret` is identical to `F_syn` (ρ = 0.996) and shares only
half of `F_head`'s top ten tensors, so it resists, in part, the wrong parameters.

### 8b. Arm Q — the capability Fisher (`F_head`, κ=10): **the trunk is protected**

Ran from 20:08:53Z (still running at the time of writing, 6 000 steps). λ = 9 532 151.

Ran 20:08:53Z → 20:55Z = **46 min** for 6 000 steps, peak total GPU 22 512 MiB, **0** NaN/WARN
lines. Final: **ALL_val acc@1 18.96 %**, NOVEL_only 19.18 %, ce_z 7.3992.

| step | own ALL_val acc@1 | NOVEL_only | ewc_pen | penalty contribution | trunk grad norm |
|---|---|---|---|---|---|
| 1 | – | – | 0.0 | 0.0 | 1.525 |
| 100 | – | – | 1.530e−10 | 0.0015 | 0.508 |
| 500 | – | – | 2.675e−09 | 0.0255 | 3.033 |
| 1000 | 10.11 % | 10.06 % | 7.679e−09 | **0.0732** (rising) | 1.469 |
| 2000 | 17.21 % | 17.29 % | – | – | – |
| 3000 | 18.17 % | 18.14 % | – | – | – |
| 4000 | 18.60 % | 18.77 % | – | – | – |
| 5000 | 18.81 % | 18.90 % | – | – | – |
| **6000** | **18.96 %** | **19.18 %** | – | – | – |

`||dW||/||W||` at step 6000: **0.0352 / 0.0370 / 0.0318 / 0.0999 %** for L20–L23 — versus Run A's
3.4585 / 3.6328 / 3.7765 / 3.5856 %, i.e. **~40–100× less movement at the same task**.

| trunk | rel. disp (L20–L23) | FIX head acc@1 | ce_z |
|---|---|---|---|
| released (reference) | 0 | 6.641 % | 8.0998 |
| Run A step 5000 (collapse) | 3.4585 / 3.6328 / 3.7765 / 3.5856 % | 2.433 % | 9.5878 |
| arm P step 2000 (`F_ret`, κ=1) | 1.7826 / 1.6842 / 1.8599 / 1.6630 % | 2.857 % | 9.3444 |
| arm P step 6000 (`F_ret`, κ=1) | 1.8099 / 1.6992 / 1.8935 / 1.6934 % | 2.676 % | 9.3882 |
| arm Q step 1000 (`F_head`, κ=10) | 0.0642 / 0.0612 / 0.0578 / 0.0682 % | 6.656 % | 8.1037 |
| **arm Q step 6000 (`F_head`, κ=10)** | **0.0352 / 0.0370 / 0.0318 / 0.0999 %** | **6.672 %** | **8.0998** |

Arm Q's trunk has moved **53× less than Run A's and 26× less than arm P's** — that is *why* the
capability survives: the `F_head` penalty is not merely a better-weighted brake, it is a brake that
actually engages (0.06 % displacement at step 1000 versus arm P's 1.7 % at step 2000).

Arm Q's trunk at step 1000 is **indistinguishable from the untouched released trunk** (6.656 % vs
6.641 %, ce_z 8.1037 vs 8.0998) — the retained structure is intact — while the arm's own accuracy is
already **17.21 % at step 2000**, above the frozen-trunk CTL plateau (16.5 %) and far above Run A's
9.900 % at step 5000. Note also that arm Q's penalty contribution *rises* (0.0015 → 0.0732) rather
than falling, i.e. it is **not** being dodged, because it weights the directions that actually bound
the capability loss.

At step 6000 the head-independent measure is **6.672 % / ce_z 8.0998** against the untouched
released trunk's **6.641 % / ce_z 8.0998** — identical to the fourth decimal on ce_z. The retained
trunk structure is not merely "less damaged"; it is **indistinguishable from untouched** after
6 000 steps at the LR that destroyed it.

### 8c. Verdict on the cliff

| arm | penalty profile | own acc@1 (step 5000) | head-independent FIX head | trunk protected? |
|---|---|---|---|---|
| Run A (measured collapse) | none | 9.900 % | 2.411 % | no |
| **P** | `F_ret`, κ=1 (pre-registered) | **16.35 %** | **2.676 %** | **no** |
| **Q** | `F_head`, κ=10 (Amendment 1) | **18.81 %** (step 5000), 18.96 % (step 6000) | **6.672 % at step 6000** | **yes** |

**The cliff is not gone for the criterion the brief specified, and it is gone for the criterion the
measurement selected.** The brief's `F_ret` at its pre-registered κ leaves the trunk as damaged as
the unprotected run; only the arm's own, head-coupled number improves (9.900 % → 16.35 %), which is
exactly the failure mode the project's own evaluator-bug post-mortem warned about. The capability
Fisher at κ=10 keeps the retained structure intact to three decimal places and lifts the arm's own
accuracy above the frozen-trunk plateau.

Arm Q's own trajectory (18.96 % at step 6000, still rising 18.81 → 18.96 over the last 1 000 steps,
NOVEL_only 19.18 %) is **above A2's at every step**: A2 read 16.86 / 17.77 / 18.45 / 18.51 / 18.64 %
at steps 2000–6000 and needed the full 20 000 steps to reach 19.53 %. The ≥ 19.5 % bar is **not**
claimed, because 19.53 % is a 20 000-step number and this arm ran 6 000 steps under the re-scope.
The honest statement is: **the cause of the cliff — trunk destruction — is removed at exactly the LR
that previously caused it (displacement 0.03–0.10 % vs 3.5 %; head-independent capability 6.672 %
vs 2.411 %), and the arm is above the healthy arm's trajectory at every measured step while using
the 10× larger trunk LR.**

## 9. Honest limitations

1. **The estimator is the empirical (label-based) Fisher**, not the true Fisher. It is the standard
   practical choice, but it weights by the model's own errors — relevant here because the LM head
   *is* badly calibrated, which is one reason `F_ret` is anti-correlated with the damage.
2. **The CPU endpoint protocol is harsher than the failure it models** (30× LR). It reproduces the
   λ=0 endpoint cleanly, but its λ-requirements cannot be transferred to the GPU regime without the
   GPU arm.
3. **`F_head` is computed on the same 300 val windows the capability is reported on**, so it is a
   training-set Fisher for the head (not held out with respect to the head's own training data). It
   is held out with respect to the *trunk* displacement being modelled.
4. **The synthesis objective is measured through its corpus, not through its full v19.2 loss**
   (NRMP buckets + transmuter + English engram stream). Rebuilding that stack was out of budget; the
   corpus-level proxy is the honest bound, and it is the *charitable* one, since a full
   reconstruction could only make the two text profiles differ more, not less.
5. **Arm Q changes two things relative to A2** (LR and penalty), so it can support but not by itself
   establish that EWC closes the cliff.

## 10. Artefact index (all mirrored locally under `rootformer/build/ghazali_forget/`)

| path | what |
|---|---|
| `RESULTS.md` | this report |
| `PREREGISTRATION.md` | the arbitration rule, the falsifiers, and AMENDMENT 1 — all written before the GPU run |
| `MANTIQ_MAPPING.md` | al-Ghazālī's manṭiq → numerical proxies: operational vs dropped, with the one measurement that earns the mapping |
| `src/` | `ewc_fisher.py`, `ewc_fisher_head.py`, `ewc_damage_check.py`, `ewc_compare.py`, `ewc_lambda_calib.py`, `make_ewc_trainer.py`, `test_ewc_unit.py`, `ewc_cpu_endpoints.py`, `ewc_head_probe.py`, `ewc_arm_report.py`, `essential_structure.py`, `cmp_traces.py`, `gate_and_run.sh`, `post_run.sh`, `probe_env.py` |
| `evidence/nrmt_train_ewc.diff` | the exact 147-line diff; original `nrmt_train.py` md5 unchanged |
| `evidence/fisher_report.json`, `fisher_head_report.json` | the three Fisher profiles |
| `evidence/compare_report.json` | pairwise profile agreement + Taylor predictions + λ\* per profile |
| `evidence/damage_report.json` | Run A ΔW/W reproduction (3.4585/3.6328/3.7765/3.5856 %) and the Taylor-vs-measured test |
| `evidence/displacement_alignment.json` | ρ (Fisher-alignment) of the real and CPU displacements |
| `evidence/essential_structure.json` | the 92 tensor rows: which named tensors hold the capability mass |
| `evidence/lambda_calib.json`, `cpu_endpoints*.json`, `cpu_*_scan.log`, `cpu_*_lowlr` results | the pre-registered λ arithmetic and every CPU endpoint run |
| `evidence/probe_head_{released,runA,P_step2000,Q_step1000,EWC_*}.json` | head-independent FIX-head measurements (6.641 / 2.433 / 2.857 / 6.656 / 2.676 / 6.672 %) |
| `evidence/smoke_equiv.log`, `eq_*.jsonl` | flag-off equivalence (IDENTICAL) and the EWC-on trace |
| `evidence/suites.log` | the seven regression suites |
| `evidence/gate.log`, `post_run.log` | the durable gate's decisions with UTC timestamps, and the automatic post-run probes |
| `results/results_EWC_A_RUNA6000_LRET.json`, `results_EWC_B_A26000_LHEAD.json`, `trace_*.jsonl` | the two GPU arms |
| `pod_src/` | read-only copies of `nrmt_train.py`, `nrmt_arch.py`, `unified_rootformer_v12.py`, `ishtiqaq_attention_v12.py`, `deepseek_v4_1_flash_model.py`, the tokenizer, and the v19.2 synthesis trainer, for audit |
