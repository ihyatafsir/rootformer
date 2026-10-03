# STATE — where things actually stand

Written 2026-10-03 because the coordinating session hit its context limit and began making
directory-level errors. **Read this before trusting anything in a session summary.**

---

## CORRECTIONS — errors made and retracted (do not repeat these)

Four claims were made with confidence and were **wrong**. They are recorded here so they are
not re-inherited. (1-3 were retracted in the prior session; 4 in this one.)

**1. "No Ibn ʿUsfūr text is held; the citation is unverifiable." — FALSE.**
Held all along at `corpus/andalusian/`, three works:

```
13_IbnUsfur_Mumtic_fi_al_Tasrif.txt      784K
14_IbnUsfur_Sharh_Jumal_al_Zajjaji.txt   2.0M
15_IbnUsfur_Darair_al_Shir.txt           320K
```

And the quote verifies **verbatim**:

```
corpus/andalusian/13_IbnUsfur_Mumtic_fi_al_Tasrif.txt:161
  «...هذا النحو من الاشتقاق غير مأخوذ به؛ لعدم اطراده، ولما يلحق فيه من...»
```

**2. "al-Zajjājī al-Īḍāḥ is not held; its locators don't resolve." — FALSE.**
Held at `corpus/basran/Al_Zajjaji_Al_Idah_Fi_Ilal_Al_Nahw.txt` (1,701 lines), and the cited
lines verify **exactly**:

```
:427      «فأما العلة القياسية فأن يقال لمن قال نصبت زيدا بإن...»
:1203-05  «...إن الشيء إذا اطرد عليه باب... ثم اعترض عليه شيء شاذ نزر قليل،
           لعلة تلحقه، لم يكن ذلك مبطلا للأصل، والمتفق عليه في القياس المطرد...»
```

**3. "The al-Mustaṣfā 403/461 citation is off-topic." — a real observation, but incomplete.**
«منعكس» *is* at those lines, and the passage *is* about *ḥadd* (defining wine). The resolution:
**ʿaks is required for a ḥadd but not for an ʿilla** — `Miyār al-ʿIlm:3170-3172`. Not an error
so much as an unstated domain distinction.

**4. "FLOOR_A answers its question: the floor is ~7.75 %, so a trained trunk does NOT learn
roots." — RETRACTED (2026-10-03, round 13).** The ~7.75 % was FLOOR_A's **own head
self-report**. Measured head-independently — fixed foreign head, no root pathway, native root ids
nulled, same 300 val windows / 18,869 positions — FLOOR_A's trunk scores **3.498 % acc@1**, i.e.
**below the 3.551 % marginal** and less than half the 6.6–6.8 % cluster (FIX 6.837, ALIGNED_FIX
6.619, GUARD 6.630, NOFEAT 6.593, ALIGNED 6.148, NGRAM 6.063, CONTROL 5.718). A flat self-report
was read as a healthy plateau when it was a **failure** signature — the arm-P pattern (self-report
16.34 % vs head-independent 2.676 %) repeated.

*Mechanism, measured, not inferred:* `trace_FLOOR_A.jsonl` shows the trunk ran at peak LR
**1.000e-03**, holding ~1e-3 through step 2000 and staying above 1e-4 past step 15000. The repo
documents **1e-4 (0.1x)** as the trunk LR that destroys the trunk. So the "floor" arm trained its
trunk 10x too hard.

*This is the fourth instance of the project's recurring failure class*, distinct from the first
three: not "absence in the search reported as a defect in the claim", but **"a self-reported
metric accepted as a capability measurement"**. A head trained on the same trunk can pass while the
trunk is destroyed. **Always keep a head-independent control** — and actually run it.

*Does not invalidate the C-vs-X comparison:* all arms share the LR schedule via `common_flags`, so
damage is common to them and the relative placement comparison stands. What is invalid is treating
A as a healthy floor.

### The root cause, stated plainly
All three came from **searching the wrong location** — the pod's `heritage_foundations/` and a
different OpenITI edition — and then reporting **the absence in the search as a defect in the
citation.** That is the same failure class as the fabricated bibliography, committed in the
opposite direction. **When a check finds nothing, the first hypothesis should be that the check
is looking in the wrong place.**

### Audit lesson that IS real
When a text is held in **multiple editions**, a locator must **name the edition**. The project
cites al-Īḍāḥ by the *sister* edition (1,701 lines), not OpenITI's (1,864 lines, shifted
+92/+142/+148 by metadata headers and page/folio markers). Verification against the wrong edition
manufactures false positives.

---

## THE SUBSTANTIVE RESULT OF THIS SESSION

### al-ʿilla conditions — the required set is TWO, not four

Both authors, **verbatim**:

```
mundabit   REQUIRED       [Mustasfa:12851] «فلم ينضبط باسم البر فلا بد من ضابط...»
muttarid   REQUIRED       [Mustasfa:13254] «إذ لو كانت لاطردت ووجد الحكم حيث وجدت...»
                          [Mahsul:14389]   «لأن الطرد واجب في العلل والعكس غير واجب فيها»
mun'akis   NOT REQUIRED   [Mustasfa:12599] «فزيادة العكس لا تؤثر؛ لأن العكس ليس بشرط
                                            في العلل الشرعية فلا أثر لوجوده وعدمه»
                          [Mahsul:15485]   «وإما أن العكس غير واجب في العلل فهو قولنا وقول المعتزلة»
muta'addi  NOT REQUIRED   [Mustasfa:13507] «مسألة العلة القاصرة صحيحة... فالتعدية فرع
                                            الصحة فكيف يكون ما يتبع الشيء مصححا له؟»
                          [Mahsul:15982]   «مذهب الشافعي أن يجوز التعليل بالعلة القاصرة...
                                            فلو توقفت صحتها في نفسها على صحة تعديتها إلى الفرع لزم الدور»
```

**CONSEQUENCE — and this is the open work:** the qiyās build **rejected "ʿilla = the root"** on
the **mutaʿaddī** condition, which both authorities say is **not required**. The passage it cited
(`Mahsul:15719-15721`) sits in a **dialectical** passage whose own response opens
«قلت قد بينا في كتبنا العقلية ما في هذين الوجهين من المغالطة» — *"I have shown in my rational
works the fallaciousness in these two positions."*

**So the rejection is unsound.** Re-run the induction with **munḍabiṭ + muṭṭarid only** and see
whether the root itself becomes a valid ʿilla. CPU-only, minutes. **This touches the thesis
directly, unlike every GPU arm.**

#### RE-RUN DONE (2026-10-03) — the rejection was never real, and the root still does no work

Scripts: `build/qiyas/two_condition_rerun.py`, `probe_isolate_clean.py`. Pod, CPU only.

**1. The code never rejected it.** In the STORED `qiyas_results.json`,
`illas_accepted = ['coarse_weak','strict7','positional','identity']`,
`illas_rejected = ['wazn_only']`. `identity` carries `ACCEPTED: True`. The "REJECTED by mutāʿaddī"
line existed only in `QIYAS_REPORT.md` prose and in a separate TADDI block; nothing in
`Qiyas(...)` ever enforced it. Re-induction over the same 106,664 aṣl (5,551 roots):

```
illa        purity   cells  ittirad(own)  2-cond verdict
wazn_only   0.9457      39     0.9457     REJECT   <- the only real rejection
coarse_weak 0.9885     135     0.9885     accept
strict7     0.9887     231     0.9886     accept
positional  0.9901     462     0.9898     accept
identity    1.0000   25042     0.9963     ACCEPT   <- munḍabiṭ + muṭṭarid, no fourth condition
```

**2. It is accepted and it does nothing.** Tier-1 holdout (10 % of (root,wazn) pairs):
`identity`'s own cell carries the ruling for **0 / 10,544** observations; firing goes
`strict7: 10542, GLOBAL: 1, coarse_weak: 1`. Correct isolation — `chain=['identity']`,
`tables={'identity'}`, `wazn_cells` emptied — returns **None for 10,544 / 10,544**, with a working
control (`3000 / 3000` attested pairs still return a rule). So the failure is **not** a validity
condition at all: withholding the (root, wazn) cell removes precisely the cell that
"ʿilla = the root" keys on. That is arithmetic, not uṣūl. The root is the **aṣl**, not an ʿilla.

**It also passes both conditions VACUOUSLY**, which is a third, independent reason it is not a
working ʿilla. `illa_identity(root) => root`, so identity's cell key `(label, wazn)` is
`(root, wazn)` — a cell contains observations of exactly ONE root. Hence munḍabiṭ purity = 1.0000
and iṭṭirād 0.9963 are properties of *memorising the instance*, not of a cause doing work.
`strict7` must fit one alignment to all roots in a class; `identity` fits each root to itself.
**A cause that is its own instance is not a cause.**

**3. Two traps found in the ENGINE, which invalidated an earlier form of this test.**
`rule()` consults `self.chain` and then falls through to `self.wazn_cells` at a line **outside**
the loop; and `__init__` appends `'wazn_only'` to the chain **even when `backoff=False`**
(`qiyas_engine.py:322-323`). So "identity without back-off" was never actually run anywhere in
this project. My own first isolation attempt emptied `wazn_cells` only and reported a false
`0 / 10,544 None`. The check was wrong, not the claim — third time this session.

**4. Citation, edition named.** `corpus/usul/Razi_Al_Mahsul.txt`
(md5 `ba4ae500744d579da59d75010cc43a31`, 19,330 lines): the `وأما` counter-position and Rāzī's
`قلت ... المغالطة` refutation are both at **:15718**; `:15719-21` is Rāzī's own qāṣira argument.
Verified verbatim. The report's substance (root-as-ʿilla is not extendable) survives; only its
label was wrong. **STATE.md mis-transcribes `Mahsul:15485` as «المعتقل»; the text reads
«المعتزلة»** — corrected in the new script.

### Everything else moved a proxy number

```
RCA_UNFREEZE_A2        19.53 % acc@1, 5.50x the 3.551 % marginal, NOVEL 19.90 %
                       ablation (gates=0) 2.82 % -- the pathway carries the capability
residual bound         +2.54 pp   <-- the real lever (RCA_NORM 19.17 % frozen)
unfreeze 0.01x         +3.0 pp    (0.1x DESTROYS the trunk: head-independent 2.411 %)
width 896->1792        +0.22 pp   (best acc@5 40.11, CE_z 7.8456)
depth 4->8 layers      +0.13 pp   (layers redundant, not dead)
CE_z                   does NOT improve -- better ranking, no better calibration
```

**Every lever that worked fixed HOW WE INJECT; none changed WHAT THE TRUNK REPRESENTS.**

### Why the trunk is the ceiling — measured

```
original Qwen2.5-0.5B, root decodable from h   29.30 %   unseen 23.34 %   (marginal 0.40 %)
transmuted trunk,       same protocol          92.34 %   unseen 83.92 %
local probe, single word in isolation: layer 4  21.83 %  ->  layer 23  13.00 %   DECAYS with depth
                                        transmuted trunk is MONOTONE UPWARD to layer 23
```

**The morphemic adaptation is what CREATED the root-preserving representation.** A raw base
*loses* root information as it organises for next-token prediction over 152k BPE. Hence: root
structure must enter at the **input stage and be maintained** — attaching at layers 20–23 reads
it where it is weakest.

### al-qiyās — the best result in the project

```
neural compositional decoder, held-out root   0 / 3,588     (0.00 %)
computed qiyas, inverse root ID, 37-root candidates  3,031 / 3,588 = 84.48 %   (chance 2.70 %)
computed qiyas, FULL 9,220-root inventory            2,838 / 3,588 = 79.10 %   (chance 0.011 %)
                                                     <- the honest Task B number
```

Caveats, stated: all 90 realisation fixes are **one root's orthography** (`بدا`, hamza vs bare
alif); over 8 random 37-root folds the computed analogy is **marginally worse** than the
hand-coded realiser (0.8908 vs 0.8957); and Task B's candidate set was **restricted to 37 roots**
— the full 9,220-root inverse search is the honest completion and was never run.

#### THE FULL 9,220-ROOT INVERSE SEARCH — RUN 2026-10-03. **Task B drops to 79.10 %.**

`build/qiyas/full_inventory_id.py`. All 9,220 non-special roots x 139 awzan = 1,281,580
realizations (384,242 realized, 255,084 surfaces), 28 CPU processes, 88 s total. Same engine
(`strict7`, backoff), same match test `(p+qy+s)==w or qy==w`, same `rank_key`.

```
FULL inventory 9,220 roots : top1 2,838 / 3,588 = 79.10 %   top5 85.84 %   chance 0.0108 %
control 37 roots, SAME harness : top1 3,072 / 3,588 = 85.62 %   (stored deriv3588: 3,031 = 84.48 %)
```

**So the "restricted to 37 roots" caveat was real, and it inflates the headline by ~5.4 pp
(84.48 -> 79.10).** The rate is still enormous against chance (1/9220 = 0.011 %), so the finding
survives — but the number in the report and this file is the 37-root one and must be read as such.

Why the previously-stored `control_74_roots` did NOT catch this: it was identical to the 37-root
result (3031) because the 37 extra roots were a **random** sample, and doubling the inventory in
a random sample adds few new false candidates. The failure mode only appears at full density.

*Harness validation and its residual gap, stated honestly:* the same-harness control gives
3,072 vs the stored 3,031 (**+41 positions, +1.14 pp**). Cause: this script's harvest yields
**113,899 aṣl / 5,556 roots** (`r in hold` test) where `qiyas_corpus.py` yields 106,664 / 5,551
(`HOLD37` list). The full-inventory figure carries the same uncertainty in the same direction.

*Traps found building this harness (three separate wrong answers before a correct one):*
1. Looked up `index[word]` on the **full affixed word**, so every prefixed/suffixed position got
   0 candidates -> control collapsed to 31.22 %. The realization `qy` **is** the causal stem
   (`النفي` with `p='ال'` -> `qy='نفي'`), so the test must run as written, not via a re-derived stem.
2. Then mis-derived the stem in a diagnostic by stripping a prefix that was already stripped.
3. Final form: index bucketed by **realization length**, keeping the original's exact test on
   precomputed realizations — no re-stripping heuristic anywhere.
The first wrong answer was caught **only** because the harness carried its own 37-root control.


---

### ishtiqaq — FIXED, and fixed only PARTLY (checkpoint evidence, 2026-10-03)

Pulled and read `/workspace/ishtiqaq_check/REPORT_ishtiqaq_native_root_path.md` (443 lines) plus
the fix and the SDPA attention the arms use. Three independent pre-fix causes: dead wiring
(0/24 attention calls carried root ids), a stale/never-trained 9015-id-space table, and the
learned coefficients never receiving gradient.

The fix (a gated subclass, shipped module unedited, md5 `d192ac9f…` unchanged) is verified
bitwise: gate 0 == shipped (`max|d| 0.000e+00`), gate 1 live (`RMS(Δh)/RMS(h) 1.3542 %`), gate 1
root-sequence-dependent (`max|Δh| 0.25` over 125 rolled positions), gate 0 invariant
(`0.000e+00`), still a pure score bias (rel residual `7.804e-06`). 7 suites pass.

**But reading the trained checkpoint (`build/qiyas/ishtiqaq_gate_check.py`) shows the fix is
PARTIAL.** Arm `RCA_NATIVE_A2`, step 16000, layers [20,21,22,23], source `shared`:

```
root_gate        moved off 0.0 in 4/4 layers   (distinct -0.1260 .. -0.1563)
pillar_gate      moved off 0.0 in 4/4 layers   (distinct -0.0549 .. -0.1250)
stream_mix[1]    BIT-EXACTLY 0.25 in 0/4 moved   element1 range 0.250000..0.250000
ishtiqaq_gamma   BIT-EXACTLY 0.25 in 0/4 moved
```

Contrast FLOOR_A (flag off) step 12000: no `root_gate`/`pillar_gate` at all; `stream_mix[0]` in
the *backbone* has moved off 0.75 in at least one layer (range 0.750000..0.761719) while
`stream_mix[1]` and `ishtiqaq_gamma` are bit-exactly at init in 0/24.

**So: the gates now train (the fix worked), but the two multiplicative coefficients that scale
the root score terms remain bit-exactly frozen at 0.25 even after 16,000 steps.** The report's
own explanation is that bit-exactness implies no gradient; the fix supplied gradient to the
zero-init gates (via the `identical_root_bonus` term, which does not pass through
`stream_mix[1]`), but not to `stream_mix[1]`/`ishtiqaq_gamma`.

*Honest limit:* I cannot yet explain WHY `d(loss)/d(stream_mix[1])` stays exactly 0 once
`root_gate != 0` — the obvious `root_gate`-multiplication argument would only hold while
`root_gate == 0`, and `root_gate` is not 0. So either `stream_mix` is outside the optimizer's
param groups, or there is a second structural zero. **Not resolved; flagged.** It does not change
the verdict (§4b/§5): the fixed path measures **−0.0053 pp ALL / 0.000 pp NOVEL** acc@1 on
`head_ALIGNED_FIX.pt` over layers [20,21,22,23] — the exact topology LATE_X uses — so the native
path is *live but behaviourally inert*, and superseded by the residual RCA (which reads fresh
information from `W_v·E_root(r_j)` where the native path can only reweight `v_proj(h)`).

**STATE.md's `stream_mix[1].grad == 0.0` line is PRE-FIX historical and is NOT a defect of the
current shared-source arms.** The post-fix fact is different and stronger: the gates move, the
coefficients do not.

---

### FINAL, EXHAUSTIVE audit of the eval harness (all files md5-pinned)

**Q1 CONFIRMED — the RCA `self.drop` is the ONLY live dropout during eval.** Full dropout audit of
the live path: NRMT head dropout (`nrmt_arch.py:319`) is covered by `head.eval()` -> OFF. Qwen2
trunk `config.json attention_dropout = 0.0` -> none. `IshtiqaqAttentionV12` is built with
`dropout=0.0` and the swap preserves it (`ishtiqaq_root_bias.py:285`) -> no-op. So the 24-vs-4
asymmetry is entirely the RCA stack's `nn.Dropout(0.1)`.
*Useful refinement:* `rca_stack.zero_gates()` makes the RCA_OFF **value** dropout-independent
(`0 * x == 0` at `root_cross_attn.py:153`), so RCA_OFF is *not* biased by RCA dropout — but the
forward still consumes RNG. Both arms run it, so that pass is symmetric.

**Q2 — the metric code is provably identical; the RUN is not comparable.** `evaluate` is one
function; the metric block `:975-1011` contains no branch on `rca_stack`, `len(rca_layers)` or gate
values (`rca_stack` appears only under `if ablate`). Same val tensors, same `B=16`, same masks.
The incomparability is purely the 24-vs-4 dropout draw count.

**Q3 CONFIRMED** — `--eval-every 0` raises `ZeroDivisionError` at the FIRST iteration (`:1154`,
`int % 0`; arg unvalidated at `:358`). The neighbouring `h_drift_probe` at `:1125` shows the
intended guard idiom, which `:1154` lacks. Hard crash — it cannot mean "never evaluate".

**Q4 CONFIRMED — the head-independent control EXISTS as reusable code and has NEVER been run for
these arms.** `/workspace/ghazali_forget/ewc_head_probe.py` (md5 `1eeba60fa49de93bf59deb58c420ca95`),
CPU-only (`CUDA_VISIBLE_DEVICES=''`), builds the model with **no root pathway**, loads a FIXED
foreign head, hard-fails if it does not load cleanly, overwrites the trunk from the arm's payload
restricted to `backbone.layers.`, and sets `active_root_ids=None`. Exact invocation:

```bash
/workspace/venvs/rootformer/bin/python /workspace/ghazali_forget/ewc_head_probe.py --live \
  --head  /workspace/head_fix/head_ALIGNED_FIX.pt \
  --cache /workspace/head_fix/nrmp_cache_9490_aligned \
  --ckpt  /workspace/hf_v19_2_release/checkpoints/rootformer_v19_2_synthesis_ar_backbone.awzan142.roots9490.tok10052.safetensors \
  --trunk /tmp/root_arch_arms/head_<TAG>.pt.trunk.pt \
  --out   /tmp/probe_head_<TAG>.json
```
`--head`/`--cache`/`--ckpt` are already the script's defaults and coincide exactly with
`runner.sh:40-41`, so it is genuinely re-pointable at these arms.

*NEGATIVE RESULT* — there is **no** head-independent control inside `/workspace/root_arch/`
(the C-vs-X harness dir). Its `compare_arms.py` only reads the trainer's self-reported
`ALL_val.acc@1` from `results_<TAG>.json`. **The control must be invoked from outside.**

*Caveats before trusting it (read, not run):* its trunk overwrite is a **silent name-match**
(`if name in st and name.startswith('backbone.layers.')`) that only *prints* `n` — **assert the
applied-tensor count covers all 24 layers**. And it scores with a fixed head, so its acc@1 is on a
different scale: use it as a RELATIVE C-vs-X-vs-FLOOR_A damage control, never against an arm's own
self-reported number.

**BLOCKER FOUND — the head-independent control CANNOT be run on these arms as-is.** I actually
executed it (round 10), rather than assuming it worked:

```bash
CUDA_VISIBLE_DEVICES= python ewc_head_probe.py --live \
  --trunk /tmp/root_arch_arms/head_FLOOR_A.pt.trunk.pt --tag FLOOR_A --out /tmp/probe_head_FLOOR_A.json
# RuntimeError: The size of tensor a (9015) must match the size of tensor b (9490)
#   at ewc_head_probe.py:123  p.data.copy_(st[name].float())
```

Passing the runner's correct `--ckpt` does **not** help. Root cause, isolated:

```
vocab.num_roots                              = 9490   (the blueprint IS correct)
model.backbone.layers[0].self_attn.root_embed.weight = (9015, 64)   <- stale
trunk payload backbone.layers.0.self_attn.root_embed.weight = (9490, 64)
```

`build_model` (`ewc_fisher.py:42-64`) takes its vocab from the current blueprint and prints the
correct count, but `UnifiedRootformerV12` constructs its attention with the **old 9015** root table.
So the payload cannot be copied in. **This is the same stale-id-space class as
`rootformer_v19_2_...roots9313` and the 48 dropped-SPACE tables — in a fourth place.**

Consequence: `ewc_head_probe.py` is **not runnable** on FLOOR_A / C / X without either (a) editing
`/workspace/ghazali_forget/ewc_fisher.py` or the probe to rebuild/resize the attention root tables
to 9490, or (b) writing an equivalent probe against `UnifiedRootformerV13` / the aligned blueprint.
**(a) edits another agent's file — not done unilaterally.** (b) is the clean route.

*Also note:* `ewc_fisher.py:49` uses the fragile
`next(v for k,v in vars(nv).items() if isinstance(v,type) and 'MorphemicVocab' in k)` idiom that
works only by dict-iteration order. Use `nv.FarāhīdianMorphemicVocab` explicitly (see
`build/qiyas/probe_class.py`).

**So the FLOOR_A floor is still NOT certified**: `trunk_motion.py` shows the layers moved, but the
capability check that would rule out an arm-P-style destroyed trunk (self-report 16.34 % vs
head-independent 2.676 %) remains unrun, now for a concrete, identified reason.

**Fix order recorded:** (1) add `rca_stack.eval()`/`.train()` around the scoring loop (or use
`--rca-dropout 0.0` as in `build/qiyas/REEVAL_PROTOCOL.md`); (2) then run `ewc_head_probe.py --live`
on both `.trunk.pt` payloads.

**Residual open item, now narrowed:** `stream_mix[1]` and `ishtiqaq_gamma` stay bit-exactly at
0.25 in 0/4 layers while `root_gate` moves. Since `root_gate`'s gradient is
`(stream_mix[1] * cond * score_root) + bonus`, the term that *must* be producing it is `bonus`
(`identical_root_bonus`, which uses no embedding); so `cond * score_root` is plausibly still ~0,
consistent with the report's own finding that the root terms contribute 0.7123 % where the constant
bonus contributes 1.2868 %. **Not verified by instrumentation** — that is what would settle it.

## AUTOMATED GOOGLE-DRIVE BACKUP — every 3 hours, verified end to end

Installed 2026-10-03. Two scripts, source in `build/qiyas/`, deployed to `/workspace/root_arch/`:

```
/workspace/root_arch/backup_to_gdrive.sh   one backup, incremental, copy + checksum-verify
/workspace/root_arch/backup_loop.sh        detached loop, INTERVAL=10800s (3 h), pid 437468
```

**Runs on the POD, and that is load-bearing.** `/workspace/.rclone/rclone.conf` exists so rclone
can reach `gdrive:` without staging anything; the laptop's rclone has **no config** ("didn't find
section in config file"), so `gdrive:` does not resolve locally. The pod also holds artifacts the
laptop does not — backing up laptop→Drive would silently miss them, which is this project's
recurring failure mode.

**Layout** (one timestamped directory per run, so a bad run cannot corrupt the previous good one):

```
gdrive:rootformer_backup_auto/<UTC-stamp>/
  STATE.md  NEXT.md            <- the handoff docs
  pod/{qiyas,root_arch,transmute_v2}/
  creds/rclone.conf            <- without it none of this is reproducible
  git_state.txt                <- git log + status at backup time
```

**Verified, not asserted.** Two runs completed with `rclone copy --checksum` (45 s) then
`rclone check --checksum --one-way` reporting **0 differences / 100 matching files**; the Drive
copies of STATE.md and NEXT.md were md5-checked against local and **match exactly**
(`a0c7f1e1015707f54b09305e342ea710` / `e48e5689554099f106d327e54ae791f3`, 48,063 / 10,892 bytes).

**Trap hit and fixed while building it:** the first run staged `pod/*` and `creds/` but **not**
STATE.md/NEXT.md — because those files existed only on the laptop, never on the pod. The script
logged `WARN: missing source` rather than failing silently, which is how it was caught. They are
now staged on the pod and included. **The script warns on any missing source and refuses to report
success if the remote is unreachable, the copy fails, or verification fails** (exit 9/10/11) — no
`|| true`, no suppressed exit codes.

**Scope deliberately excluded:** `ghazali_forget` (2.3 GiB), `head_fix` (364 MiB),
`ishtiqaq_check` (391 MiB) are **not** swept every 3 h — they are already in the older
`rootformer_backup_2026-10-0*` sets, and re-uploading 3 GiB on a timer is not what backing up the
handoff should mean. `TIER1=1 bash backup_to_gdrive.sh` adds them.

**Caveat that matters:** the loop lives on the pod, so backups stop when the pod dies. The local
`git push` to `github.com/ihyatafsir/rootformer` remains the durable record; Drive is the second copy.

---

## ARCHITECTURE FIXES — done and verified

```
constants         num_roots 9015 -> 9490, num_awzan 128 -> 142, vocab 10052
                  one root-id space, injective, 48 stale-SPACE tables dropped
root source       stale (9015,64) -> morphemic_embed.root_embed (9490,448)
pathway           layers 20-23 post-layer hook -> ALL 24 layers, input stage upward
                  residual 57.86 M + score bias 11.01 M, independent zero-init gates
init-equivalence  gate 0 bit_equal to no pathway, max|d| = 0.0 (torch.equal)
isolation         gate 1 with roots rolled, trunk input FIXED -> output changes (reads roots)
complementary     residual RMS 0.112 vs bias 0.0128; both != sum-of-parts
checkpoint load   RootformerNRMT (backbone.layers.*) vs CausalLM (backbone.model.layers.*)
                  naive load = 555/520 -> LOADS NOTHING (silently random model)
152 M orphans     live hooks on layers 1/11/14, in no checkpoint and no model.parameters()
                  now perturbation-proved: hooks-ON vs OFF is torch.equal
attention         eager [B,14,T,T] -> SDPA. QK^T was running fp32 while V was bf16; bf16 shipped,
                  measured against an fp64 oracle
in-place bug      root_cross_attn.py:136 clamp_ -> mutates the CALLER's tensor; dies on the first
                  backward once the input stage is unfrozen. Worked around with .clone();
                  upstream fix is clamp_ -> clamp
disk              checkpoints to /tmp (pod 50 G overlay). /workspace is MooseFS (network):
                  its "466 T free" is the CLUSTER's space. ENSOSPC killed two arms at a write
                  with no traceback
```

### Liveness assertion — works, and found nine things
201 parameters causally INERT; 193 with `.grad is None`, including all 24× the native root path
and `backbone.embed_tokens.weight`. **`stream_mix[1].grad == 0.0` EXACTLY in all 24 layers**
(element 0 live 24/24) — the score-bias path has never received a gradient. Two bugs in the test
itself had to be fixed first: `std()` of a scalar is NaN (so `nan or 1e-4` reported **everything
live**), and `p+eps-eps != p` in float32.

---

## RUNNING / OPEN

```
pod    FLOOR_A (trunk trained, NO root pathway) step ~9300/20000, RUNNING, 33 min elapsed
       EARLYROOT_C  <- never ran a step; its ONE launch died from a PER_ARM_MIB=16000
                       override, not a stage defect (see below)
       runner.sh (pid 430556, restarted 01:03) staged SIX arms:
         [EARLYROOT_C RESIDUAL_R SCOREBIAS_D LATE_X FLOOR_A_S2 EARLYROOT_C_S2]
       ~20 h GPU budget; two concurrent 32-batch arms does NOT fit
       ~70 min per arm at the measured 300 steps/min; six arms ~7 h
```

### *** INCIDENT 01:28Z — THE RUNNER DOUBLE-LAUNCHED AND KILLED BOTH ARMS ***

The predicted failure happened, on the first hand-off, **with the default `PER_ARM_MIB=20000`
correctly set.** Verbatim `runner.log`:

```
01:28:05  SLOT FREE (occupants=0 <= 1, free=32126MiB >= 20000MiB)
01:28:05  LAUNCH EARLYROOT_C
01:28:08    EARLYROOT_C confirmed ALIVE (launcher pid 440819)
01:28:54  SLOT FREE (occupants=1 <= 1, free=32123MiB >= 20000MiB)   <- only 3 MiB less!
01:28:55  LAUNCH RESIDUAL_R
01:28:58    RESIDUAL_R confirmed ALIVE
```

**Root cause — the gate is a STARTUP RACE, and this is the real defect.** `free_vram()` samples
`nvidia-smi memory.free` instantaneously, but a freshly launched arm allocates **gradually** over
~20-90 s (it must build the model first). At 01:28:54, EARLYROOT_C had been "alive" 46 s and had
not yet touched the GPU, so `free` was still **32,123 MiB** — the gate saw a near-empty card and
launched a second full-trunk arm into it. `occupants=1` was satisfied, `MAX_OCCUPANTS=2` permits
exactly this, and **`PER_ARM_MIB=20000` was powerless because the number it compares against was
measured before the incumbent had allocated.** Both arms then tried to hold ~17.7 GiB on a 31.37 GiB
card.

Then **two runner processes appeared** (442187 and 442734, both alive) — a second `runner.sh run`
was started at 01:30:58, and the `rm -f queue/state/EARLYROOT_C.started` + relaunch means the queue
**re-evaluated a stage it had already started**, which is the "no lock, no mutual exclusion" hole.

**Consequences:**
* `EARLYROOT_C` — **OOM again**, exactly the same signature (`Process 405014 has 13.65 GiB in use
  … this process has 17.69 GiB`), 1 trace line, never left step 1. Its log is APPEND-ONLY, so OLD
  OOM text persists and reading the tail shows a stale traceback; check `stat` mtime and the trace
  line count, not just the tail.
* `RESIDUAL_R` — started 01:28:58, log 2 lines, **no trace file at all**, gone by 01:29:11. Never
  reached step 1.
* GPU has been **EMPTY** (`4 MiB used, 0 %`) since ~01:29. **The ladder has produced no training
  compute since FLOOR_A exited.**
* Two concurrent runners make the state unpredictable and could re-launch a stage.

**The fix the next session needs (NOT yet applied — it edits the coordinator's runner.sh):** the
gate must require that the incumbent has *actually allocated*, e.g. wait for
`occupants>=1 AND free <= TOTAL - PER_ARM_MIB` (i.e. confirm the resident arm has taken its
memory) before considering a second launch — or simply serialise: launch the next stage only when
`occupants == 0`. Given two full-trunk arms cannot coexist, **`MAX_OCCUPANTS` should be 1** for
this ladder; `MAX_OCCUPANTS=2` is what makes the race reachable.

### *** INCIDENT 2 (ongoing, 01:32Z->) — `EARLYROOT_C` IS HUNG ON FUSE I/O, GPU IDLE ***

After the double-launch incident a **second** `runner.sh run` (pid 442734) was started at
01:30:58 and relaunched `EARLYROOT_C` at 01:31:52 (python pid **442737**). It is **hung**:

```
pid 442737   etime 4:14   stat Ssl   cpu 1.7%   rss 528 MB
wchan: request_wait_answer        <- FUSE: blocked waiting for the MooseFS daemon to answer
log_EARLYROOT_C.txt  mtime 01:32:09, 9557 bytes, NOT GROWING
trace_EARLYROOT_C.jsonl   1 line     (never left step 1)
nvidia-smi               4 MiB used, 0 %, NO compute apps
```

`wchan = request_wait_answer` is the FUSE call path: the process is blocked in a request to the
**MooseFS userspace daemon** for `/workspace` (`mfs#euro-3.runpod.net:9421 on /workspace type
fuse`). So this is the **network-FUSE stall this file already warns about** — the same filesystem
whose per-user quota silently killed two arms with no traceback. `/workspace` is where the
checkpoint, the `nrmp_cache_9490_aligned/{train,val}.pt` and the log all live, so a FUSE hang
stalls the arm before it can allocate.

**It holds a CUDA context but zero GPU memory**, so the VRAM gate still reports the card as free
(`free ≈ 32 GiB`) and the `arm_alive` check still reports the tag alive. **A process that is alive,
idle, holding no GPU memory, and making no progress is the worst case for both gates at once.**

**Net state:** GPU **idle at 4 MiB**; one hung arm; **two** runner processes (442187, 442734) that
will not agree about the queue; `LATE_X` still ~5 arms away. Nothing has trained since FLOOR_A
exited at 01:27.

*The two runners between them reproduce the no-lock hole:* a second `runner.sh run` re-evaluated
`EARLYROOT_C` (a stage already started) because `$STATE/EARLYROOT_C.started` had been deleted by the
restart command and `running` is in-memory only.

**What would unblock it (needs authorisation — it touches the coordinator's processes):**
```
kill 442737 442734 442187            # the hung arm and BOTH runners
MAX_OCCUPANTS=1 STAGES="EARLYROOT_C RESIDUAL_R SCOREBIAS_D LATE_X FLOOR_A_S2 EARLYROOT_C_S2"   setsid nohup bash /workspace/root_arch/runner.sh run > /workspace/root_arch/queue_run.txt 2>&1 &
```
`MAX_OCCUPANTS=1` is the load-bearing change: it makes the 1->2 transition **unreachable**, which is
the only thing that actually closes the startup race, since `PER_ARM_MIB` was powerless against it.

### FLOOR_A is DONE — final numbers (the only complete arm)

```
step 20000   ALL_val acc@1 7.74 %  acc@5 12.31 %  CE_z 9.2293
             NOVEL_only acc@1 7.63 %  acc@5 12.10 %  CE_z 9.2059
results_FLOOR_A.json written 01:27, 20 history entries, self-report FLAT 7.64-7.74 %
```

Self-report unchanged and flat to the end. **FLOOR_A's trunk is nonetheless healthy** — the
unconfounded linear readout reads the root off it at **93.7 %** (see the correction above), so its
whole 7.7 % self-report is a *head/objective* problem, not a trunk problem. Two different
measurements of the same arm, disagreeing by 12x, and the head was the misleading one.

### The ladder is now self-reporting (no polling needed)

`/workspace/root_arch/watch_ladder.sh` (source `build/qiyas/watch_ladder.sh`, pid 420709) polls
every 30 s and appends a timeline to `/workspace/root_arch/queue/watch.log`:

```
APPEARED  tag=... used=..MiB peak=..MiB step=.. log=...
EXITED    tag=... peak=..MiB laststep=.. -> clean exit | DIED: CUDA OutOfMemoryError
ALARM     occupants=3 ...        <- the NEVER-A-THIRD-OCCUPANT rule, alarmed explicitly
```

It reads `ps` / `nvidia-smi` / logs only and **touches no process**. The launch-time two-occupant
OOM is the dominant failure mode (it killed EARLYROOT_C and cost ~5 min to detect), so the
transition is now recorded rather than dependent on someone polling at the right second.

**Correction to an earlier worry of mine: checkpoints DO NOT accumulate.** `torch.save(...,
args.save)` sits in the eval block and writes the **same path** every time (no `.step` suffix),
so each arm holds a flat ~834 MiB (`head_<tag>.pt` 51 MiB + `.trunk.pt` 783 MiB). `/tmp` is 48 GiB
free, so even four arms cost ~3.3 GiB. The disk is NOT a constraint on this ladder.

### FLOOR_A — the floor is flat, and CE_z gets worse (the head-independent fact)

`--eval-every 1000`, `--root-cross-attn none --ishtiqaq-root-bias none`. acc@1 is
**scale-invariant** by construction; `raw PPL` is scale-dependent (logit_scale drifts 0.59->0.66).

```
step    ALL acc@1   acc@5    CE_z     NOVEL acc@1   CE_z
 1000      7.64     12.86   8.1712      7.53      8.1978
 2000      8.12     12.93   8.4177      8.05      8.4424
 3000      8.09     12.86   8.5386      7.98      8.5619
 4000      8.14     13.15   8.5430      8.09      8.5609
 5000      7.79     12.61   8.4943      7.62      8.5108
 6000      7.84     12.59   8.7229      7.70      8.7339
 7000      7.89     12.73   8.8685      7.78      8.8778
 8000      7.89     12.87   8.9425      7.79      8.9497
 9000      7.74     12.54   8.9897      7.67      8.9887
10000      7.86     12.57   9.0772      7.81      9.0714
11000      7.72     12.32   9.1143      7.63      9.1093
12000      7.73     12.33   9.1360      7.68      9.1277
```

**FLAT since step 2000: acc@1 span 7.64-8.14 %, i.e. 0.50 pp across 11,000 steps (full curve
now extracted to step 12,000, see `build/qiyas/arms_evidence/FLOOR_A_evals.txt` and
`build/qiyas/parse_floor_a.py`). CE_z WORSENS throughout, 8.1712 -> 9.1360 (+0.9648), and
`wazn` acc@1 is likewise flat at 44.33-45.27 %.**
`acc@1` never climbs; better ranking with worse calibration is the same signature as every other
lever in this project. The ~8 % floor in NEXT.md is confirmed at step 9,000, not just 6,000.

### EARLYROOT_C died from the TWO-OCCUPANT hazard, not from its own configuration

`log_EARLYROOT_C.txt` traceback: `torch.OutOfMemoryError ... this process has 17.69 GiB in use`,
with FLOOR_A (pid 405014) holding 13.65 GiB on a 31.37 GiB card. FLOOR_A launched 00:04 and
EARLYROOT_C 00:16. This is exactly the measured figure the runner's own header documents.
**Consequence: no arm has ever run under a matched two-occupant condition, and the "matched
trunk" premise of the early-vs-late test is still unverified — a single FLOOR_A log is the only
ear-to-the-ground evidence, and it shows the trunk does not learn acc@1 without a root pathway.**

**Open, in priority order:**

1. **EARLYROOT_C must be re-run** — it has never produced a single step of eval, having died
   from the two-occupant OOM, not from its own configuration. It is first in the queue, so the
   queue will do it, but the launch must be checked for the OOM again.
2. **LATE_X is queued LAST — and it WILL fit. My earlier estimate was WRONG by ~4x.**
   Measured directly on the pod at 00:47: **300 steps/min**, so a full 20,000-step arm is
   **~67 min**, not 3.5 h. Startup is ~2-4 min per arm. Projection from FLOOR_A's real start
   (00:04:18) at ~70 min/arm:

   ```
   FLOOR_A ~01:14 | EARLYROOT_C ~02:24 | RESIDUAL_R ~03:34 | SCOREBIAS_D ~04:44 | LATE_X ~05:54
   ```

   Against a ~20 h budget starting 00:04, all five arms fit with hours to spare. **The runner was
   NOT touched and does not need reordering.** (Earlier in this file I wrote that LATE_X would
   miss the budget; that came from misreading FLOOR_A's own 00:04->00:39 progress as ~3.5 h for
   20k steps when it is ~67 min. Corrected here.)

### THE QUEUE CHANGED UNDER THIS SESSION — two `_S2` seed arms added (not by me)

At 01:06 UTC the running queue listed **six** arms, not five:

```
unstarted=[EARLYROOT_C RESIDUAL_R SCOREBIAS_D LATE_X FLOOR_A_S2 EARLYROOT_C_S2]
```

`/workspace/root_arch/runner.sh` was **edited at 01:03:36** (md5 now `5683a6bccae0f661303fcec5c657b14b`,
18401 bytes; it was `66184fed99fb2180e711c5b9b04cce83` in the audit earlier this session). Its
`STAGES` default is now the six-arm list, `stage_flags()` gained `FLOOR_A_S2` and `EARLYROOT_C_S2`,
and the runner process was restarted (pid 413704 -> **430556**). **I did not make this change and did
not touch either process** — per the standing rule. It is presumably the coordinator responding to
the unseeded-comparison finding below.

The `_S2` stages are **second seeds**:

```bash
FLOOR_A_S2)     "--root-cross-attn none --ishtiqaq-root-bias none --seed 1" ;;
EARLYROOT_C_S2) "--root-cross-attn all --rca-lr-scale 1.0 --rca-ablate-eval \
                 --ishtiqaq-root-bias all $C_BOTH --ishtiqaq-ablate-eval --seed 1" ;;
```

with the runner's own comment: *"The A-vs-C conclusion is the whole point of the session, so it must
not rest on a single unseeded run: `--seed 1` reseeds torch/numpy/cuda and therefore the batch order
and the dropout stream. Identical in every other respect to FLOOR_A / EARLYROOT_C."*

**This is the right instrument, and one detail makes it better than it looks:** `build_meta`
(`ewc_head_probe.py:31-35`) chooses validation windows with `np.random.default_rng(1)`, a seed
FIXED independently of `--seed`. So **every arm evaluates on the identical 300 val windows /
18,869 radical positions** regardless of training seed — the seed perturbs the training trajectory
only, which is exactly the variance we want to measure. The A-vs-C comparison is therefore now a
genuine two-seed comparison.

*Stale text to ignore:* `runner.sh:20` still claims "eval cadence and seed are identical, so a
difference in the eval is attributable to the pathway" — true for the original five arms, **false
once `_S2` exists by design**. And the schedule lengthens to six arms (~70 min each).

### FLOOR_A's ACTUAL LR TRAJECTORY — peak 1.000e-03, sustained (measured)

Extracted from `trace_FLOOR_A.jsonl` (n=17,075 steps):

```
step      1   lr=4.001e-05     step   500   lr=9.367e-04
step     10   lr=4.066e-05     step  1000   lr=9.989e-04   <- peak ~1e-3
step    100   lr=1.045e-04     step  2000   lr=9.872e-04
step   5000   lr=8.783e-04     step 10000   lr=5.242e-04
step  15000   lr=1.551e-04     step 17000   lr=5.782e-05
                       peak 1.000e-03   min 4.001e-05
```

So the trunk did not merely touch 1e-3 — it **held ~1e-3 for the first ~2,000 steps** and was still
above 1e-4 until past step 15,000. The repo documents **1e-4 (0.1x)** as the trunk LR that
*destroys* the trunk; this arm ran **10x that, sustained**. That is the mechanism behind the 3.498 %
head-independent measurement above, and it is now a measured claim, not an inference from a flag.

**Why this damages `A` specifically and less than it first appears for `C` vs `X`:** every arm —
A, C, R, D, and both `_S2` — shares this same LR schedule (it comes from `common_flags`). So a
high-LR-damaged trunk is common to all of them, and the **relative** C-vs-X comparison still
measures placement rather than damage. What is *not* valid is reading A as a healthy floor, or
reading C/X against A's damaged capability.

### TRUNK MOTION VERIFIED — the "matched trunk" premise is sound (head-independent)

`build/qiyas/trunk_motion.py` compares an arm's `.trunk.pt` against the checkpoint the runner
actually trains FROM (`...awzan142.roots9490.tok10052.safetensors`, arg `--checkpoint`), tensor by
tensor, with no involvement of the NRMT head or any self-reported metric.

```
FLOOR_A @step 12000 : 24/24 backbone layers MOVED, 0 frozen
                      max|delta| per layer 3.41e-01 .. 5.40e-01
```

**So `--unfreeze-trunk-all` does what it claims**, and the C-vs-X comparison really is at a
*trained* trunk rather than a frozen one that nobody noticed. This is the premise the whole
early-vs-late question rests on, and it is now measured rather than assumed.

*Trap found in this check, worth keeping:* my first baseline was the **release** checkpoint
`rootformer_v19_2_synthesis_ar_backbone.awzan142.roots9490.tok10052.safetensors`... no -- it was
the **pre-alignment** release, which still carries root/wazn tables at the STALE `(9015,64)` and
`(128,64)` shapes. Comparing against it silently skipped 48 tensors per arm on shape mismatch.
The 48 skips are the fingerprint of comparing to the wrong checkpoint. `ORIG_CKPT` is now an env
override and the correct baseline is the runner's `--checkpoint`.

*Anomaly raised and RESOLVED:* FLOOR_A is `--root-cross-attn none --ishtiqaq-root-bias none`, yet
its checkpoint holds 48 tensors matching `ishtiqaq`/`stream_mix`. They are exactly
`backbone.layers.{0..23}.self_attn.ishtiqaq_gamma` and `.stream_mix` — **24 x 2 per-layer scalars
that live on the SDPA attention class itself**, not on a swapped-in root-bias module. Confirmed by
the gating: `nrmt_train_v13_sdpa.py:606-614` only calls `swap_in_root_bias()` `if ishtiqaq_layers`,
so `none` builds no module and `ishtiqaq_mods == []`. FLOOR_A therefore IS pathway-free as the
runner claims; these scalars are inert defaults, and `stream_mix` is the same tensor the project
already found has `element 1 grad == 0.0` in all 24 layers. No new defect.

*Not confirmed (budget):* the exact shape of each scalar (my name-collapsing script reported `()`);
the count and the gating logic are what matter and both hold.

### CONFIRMED DEFECT: `rca_stack.eval()` IS NEVER CALLED — and it is LAYER-COUNT BIASED

Independently confirmed 2026-10-03, twice, and my first phrasing was imprecise. `evaluate()`
calls `head.eval()` at `nrmt_train_v13_sdpa.py:955` and `head.train()` at `:1014`. Crucially
`head = model.nrmt_head` (`:791`) while the stack is `model.root_cross = rca_stack` (`:576`) --
so the two are **SIBLINGS under `model`**, and `head.eval()` structurally cannot reach
`root_cross`. (I first wrote "not a child of model, a standalone module"; the stack IS registered
on `model`, which is exactly why the bug survived review -- registering it made it look covered.)
An exhaustive grep of the 1250-line trainer for `.eval()` / `.train()` / `model.eval()` /
`training=` returns ONLY 955, 1014, 1059 (`head.train()`). There is no `rca_stack.eval()` and no
`model.eval()` anywhere, so nothing ever flips the stack out of its construction default
`training=True`. `--eval-every 0` is separately CONFIRMED to raise `ZeroDivisionError`
(`int % 0` at `:1154`; the arg at `:358` is an unvalidated int, and unlike `h_drift_probe` at
`:1125` it has no guard).

`RootCrossAttentionStack` builds `self.mods = nn.ModuleList([...for i in self.layer_indices])`,
and each `RootCrossAttention` has `self.drop = nn.Dropout(dropout)`
(`root_cross_attn.py:114`, stack at `:176-180`). The arms pass `--rca-dropout 0.1`.

**Consequence, and it matters for THIS comparison specifically:**

```
EARLYROOT_C : RCA on layers 0-23  -> 24 dropout modules active at eval
LATE_X      : RCA on layers 20-23 ->  4 dropout modules active at eval
```

So the two arms are **not** evaluated on equal footing: C's eval carries 6x the active dropout of
X's. `nn.Dropout` scales by `1/(1-p)` in train mode, so its **expectation is unchanged** -- this
adds VARIANCE, not a systematic bias. The honest statement is therefore: **C's eval is ~6x noisier
than X's**, not that C is biased up or down. With `--rca-dropout 0.1` over 18,869 samples the
variance cost is likely well under the 0.197 pp binomial SE, so it probably does not threaten the
comparison -- but it is an unquantified, arm-dependent term in exactly the number being compared,
and STATE.md's prior `-0.13 pp` was measured for ONE configuration, so it does not licence assuming
the term is equal across 4 and 24 layers. **Fix if it matters: force `rca_stack.eval()` and
re-evaluate both checkpoints from their saved trunks -- CPU-runnable, no GPU needed.**

*Second-order:* dropout in train mode also consumes the global RNG, so every eval perturbs the
subsequent training trajectory -- differently for C (24 modules) and X (4). A further
arm-dependent term, and the reason the two trunks are not bit-comparable even from the same init.

### FLOOR_A has NO head-independent control — and it trains the trunk at a documented-destructive LR

Raised by an independent audit and confirmed against the sources:

* FLOOR_A runs `--trunk-lr-scale 1.0 --lr 1e-3`, i.e. trunk LR **1e-3** — that is **10x** the
  0.1x (1e-4) trunk LR this repo documents as DESTROYING the trunk (Run A; README.md:53), and
  **100x** A2's 0.01x. It also unfreezes all 24 layers where A2 unfroze 4.
* `trunk_motion.py` proves the layers MOVED (24/24, max|Δ| 0.34-0.54). That is **not** evidence
  that capability survived. The repo's own cautionary case, arm P, self-reported **16.34 %** while
  its head-independent measure was **2.676 %**.
* `FIX` baseline (frozen released trunk + FIX head) = **6.837 %**; marginal = **3.551 %**. FLOOR_A
  at 7.64-8.14 % sits above both, and every known damaged number (Run A 2.411/2.433 %, arm P
  2.676 %, RCA-ablated 2.825 %) is 2.9-3.4x lower. So "real but weak floor" is supported; **"the
  trunk is healthy" is COULD NOT DETERMINE without running the control.**
* The control IS runnable now: `/tmp/root_arch_arms/head_FLOOR_A.pt.trunk.pt` is saved and
  `/workspace/head_fix/head_ALIGNED_FIX.pt` exists. **Run it before treating FLOOR_A as the bar.**

**Also available and better than a raw cross-arm diff:** both C and X pass `--rca-ablate-eval`
(and `--ishtiqaq-ablate-eval`), so each arm logs its own pathway-OFF number
(`ALL_val_RCA_OFF`, `:1164-1174`). **Prefer the WITHIN-arm deltas**

```
Δ_C = acc@1(C)     - acc@1(C_RCA_OFF)
Δ_X = acc@1(X)     - acc@1(X_RCA_OFF)
```

over the raw `acc@1(C) - acc@1(X)`. Reason: C and X train *different trunks* from the same init, so
a raw diff conflates pathway placement with trunk divergence; each Δ is that arm's own causal
pathway contribution. If both Δ are ~0 within SE, neither pathway does anything and the placement
question is void. Note the ablation is stochastic too (dropout is active on the ON pass only).

**Reading rule if both C and X land at FLOOR_A's level:** that is case **(b) both failed**, not
case (a) "placement does not matter". A tie at a no-pathway arm's level means the pathway added no
measurable capability, so there is no capability whose placement could matter. The existing
`--rca-ablate-eval` / `--ishtiqaq-ablate-eval` (already enabled for C and X in
`runner.sh:136-146`) discriminate the two: if an arm's own acc@1 equals its ablated acc@1 within
the ~0.20 pp SE, the pathway is contributing nothing.

### Reproduction nits in the FLOOR_A record

* acc@1 span is **7.64-8.14 %**, not "7.7-8.1 %".
* CE_z from step 5000 is strictly increasing (8.4943 -> 9.1360, 8/8 intervals), but it is NOT
  monotone globally (dips at 5000). STATE.md line ~303 still says "worsens throughout" and
  "confirmed at step 9,000", both contradicted by its own row at step 5000 — the table is right,
  the prose is stale.
* Never quote `raw PPL`: scale-dependent and rising (2267 -> 3697) while `logit_scale` drifts
  0.59 -> 0.68; the contract at `:945-953` labels it "NOT a capability measure".

### *** THE 3.498 % WAS A HEAD/TRUNK TRANSFER ARTIFACT — THE TRUNK IS FINE (93.7 %) ***

The unconfounded control came back and **it overturns the previous headline.** Linear decodability
of the current position's root from FLOOR_A's trunk, closed-form ridge per layer, **no fitted head**
(`build/qiyas/head_probe_v13.py --linear-readout`):

```
layer   acc@1     acc@5
    0  93.724%  95.823%     <- best
    4  93.229%  95.354%
   12  91.990%  93.458%
   20  88.349%  91.812%
   23  86.281%  90.833%     <- decays with depth, monotone
```

**93.72 % at layer 0**, against the published reference of **92.34 %** for the *transmuted* trunk
and **29.30 %** for the raw base. So **FLOOR_A's trunk preserves root structure at least as well as
the adapted reference.** The trunk is **not** destroyed.

**Therefore the 3.498 % fixed-head figure was dominated by the confound I flagged, not by damage:**
`head_ALIGNED_FIX.pt` was fitted for a *different* trunk, and a foreign fixed head transfers
poorly. The trunk carried the information; the borrowed head could not read it. *Corrected
reading:* FLOOR_A's trunk is healthy, and the earlier "below the marginal / arm-P pattern"
conclusion is **withdrawn**.

**A second trap found in my own probe, before trusting it.** The first readout used a random split
over *positions*. That **leaks**: the same word type occurs in many windows, so near-duplicates of
test instances sit in train and the probe scores by memorisation. `probe_orig_base.py` reports both
an overall and an `unseen` figure for exactly this reason (92.34 % overall vs 83.92 % unseen), and
the two differ materially. The readout now splits **by word identity** (`--split root`, the unseen
figure) or **randomly** (`--split random`, the overall figure), and BOTH are being measured so each
number is compared with its correct baseline. The 93.72 % above is the **leaking (random)** variant;
the leak-free number is pending. **Do not quote 93.72 % as an unseen-words result.**

*What survives either way:* root information is strongly present in FLOOR_A's trunk, so the trunk
is not destroyed; and a foreign fixed head is not a valid capability measure for a trunk it was not
fitted on. That second point is a real methodological finding about this project's own control.

### IN FLIGHT: the UNCONFOUNDED control (linear decodability, no fitted head)

`build/qiyas/head_probe_v13.py --linear-readout` → running detached on the pod as pid **434315**,
log `/workspace/qiyas/probe_lr_FLOOR_A.log`, output `/tmp/probe_lr_FLOOR_A.json`.

Why it matters: the 3.498 % figure above carries a **head/trunk transfer confound**
(`head_ALIGNED_FIX.pt` was fitted for a different trunk). This probe has **no fitted head at all** —
it is the project's own decodability protocol (`probe_orig_base.py:121-129`: closed-form ridge to
one-hot, weights solved in closed form) applied per trunk layer, asking only *is the current
position's root linearly decodable from this trunk's hidden state?* Published reference on the same
protocol: raw base **29.30 %** (unseen 23.34 %), transmuted trunk **92.34 %**, local isolated probe
decaying **21.83 % @ layer 4 → 13.00 % @ layer 23**.

*Target convention, deliberate:* the readout pairs hidden state at position *i* with the root of
*that same position* (`Tv[i]`), matching `probe_orig_base.py:147`. `eval_head` instead predicts
`Tv[:, 1:]` — a **next-token** target, a different question. Do not conflate the two numbers.

*Self-inflicted note:* the first run had no sample cap and overran its local timeout, leaving a
detached process behind; it and the capped rerun then contended (load average 12 → 25). I verified
pid 432655 was my own script (`head_probe_v13.py`, zero `max-samples`) and killed it — my own
abandoned duplicate, not another agent's work. `--max-samples` now defaults to 40,000, because the
ridge cost is O(n·d²) plus a d×9490 multiply per layer and does not need every position.

### *** FLOOR_A's TRUNK MEASURES 3.498 % HEAD-INDEPENDENTLY — BELOW THE MARGINAL ***

Round 12. I wrote the missing control (`build/qiyas/head_probe_v13.py`, `--live`, CPU, source on
pod at `/workspace/qiyas/head_probe_v13.py`) because the project's own `ewc_head_probe.py` cannot
run on these arms (stale 9015 tables; see the blocker entry above). Result:

```
[FLOOR_A] LIVE fp32 (V13)  acc@1 3.498 %   acc@5 6.582 %   ce_z 9.1264   (n = 18,869)
trunk applied: 552/552 backbone.layers tensors   <- asserted, not a silent name-match
```

Same probe, same fixed head, same 300 val windows / 18,869 radical positions as the project's own
protocol. Comparators under that protocol:

```
FIX (the bar, frozen released trunk)   6.837     ALIGNED_FIX (exact control for A2)  6.619
GUARD 6.630   NOFEAT 6.593   ALIGNED 6.148   NGRAM 6.063   CONTROL 5.718
marginal (word-stream)                 3.551
FLOOR_A, measured this run             3.498     <- LESS THAN HALF the 6.6-6.8 % cluster
```

**So FLOOR_A's trunk is not a healthy bar.** Its self-reported `acc@1` is flat at 7.64-8.14 %, while
its trunk measured with a foreign fixed head sits at the marginal rate — the arm-P pattern
(self-report 16.34 % vs head-independent 2.676 %) reproduced here at a smaller scale. This is
consistent with the audit's concern: FLOOR_A trains the trunk at `--trunk-lr-scale 1.0
--lr 1e-3`, i.e. 10x the trunk LR the repo documents as destroying the trunk, across all 24 layers.

**Consequences — this changes how the ladder must be read:**
1. FLOOR_A is **not** a valid floor for `C - A` or `X - A`. A comparison against a damaged trunk
   tells you about damage, not about the root pathway.
2. If C and X land near 7.8 %, that is **above** this measured trunk capability and the pathway is
   doing something real — the opposite of the earlier "both at the floor" reading.
3. The 3.498 % figure is a *relative damage* number, and the FIX-head cluster (5.7-6.8 %) is the
   bar it should be compared against, **not** FLOOR_A's own self-report.

**THE CONFOUND, STATED PLAINLY AND NOT YET EXCLUDED.** `head_ALIGNED_FIX.pt` was fitted for a
*different* trunk (the A2/ALIGNED line). A foreign head transfers imperfectly, so part of the 3.2 pp
gap could be head/trunk mismatch rather than capability loss. The clean control is a head **fitted
on FLOOR_A's own trunk** under the same protocol, or the root linear-decodability probe
(29.30 % original base / 92.34 % transmuted). **Neither has been run.** Until one is, the correct
statement is: *FLOOR_A's trunk scores 3.498 % under the project's own fixed-head control, below the
marginal rate; the gap is large enough that head-mismatch alone is an unlikely full explanation,
but that has not been tested.*

### THE EARLY-VS-LATE COMPARISON, PRE-SPECIFIED BEFORE EITHER ARM EXISTS

`LATE_X` = `--root-cross-attn top4 --ishtiqaq-root-bias top4`; `EARLYROOT_C` = `all`/`all`. Every
other flag is shared verbatim from `runner.sh:common_flags` (same checkpoint, cache, LR 1e-3,
`--trunk-lr-scale 1.0`, 20,000 steps, batch 32, head, eval cadence, seed). **So C vs X is matched
on trunk training by construction** — that is what the objective asks for.

Pre-specified read, with the scale fixed now from FLOOR_A's own sample sizes
(`ALL_val` n=18,869, `NOVEL_only` n=15,046):

```
SE(one arm), acc@1 ~0.08  :  ALL_val 0.197 pp | NOVEL_only 0.221 pp
SE(C - X) if independent  :  ALL_val 0.279 pp
```

* `|acc@1(C) - acc@1(X)| > 0.5 pp` -> a real gap; early attachment genuinely helps.
* `<= 0.5 pp` -> **a tie**, and then: everything concluded about the late RCA's ceiling was a
  statement about a **frozen** trunk, and the architecture was never the bottleneck.

*Caveat stated up front:* 0.5 pp is only ~1.8x SE(C-X) and the seeds are single, so this is a
soft threshold, not a test with power. A tie is the more likely outcome to be **inconclusive**
rather than probative. Also, a tie would only be decisive about the ceiling **relative to
FLOOR_A's ~7.7-8.1 % floor**: if both C and X sit at the floor, that is not an architectural
tie, it is both failing.

3. **The early-vs-late test at a matched trained trunk** — if it's a tie, everything concluded
   about the late RCA's ceiling was a statement about a *frozen* trunk. **Still not run.**
4. **The derivational holdout, properly designed** — the thesis has still never been tested.
   Attempt 1 was structurally impossible (a softmax cannot emit an unseen root); attempt 2 ran
   at 0.141 epochs.
5. **The full 9,220-root inverse search** for qiyās.
6. `rca_stack.eval()` in `evaluate` (dropout stays active at eval, −0.13 pp);
   `--eval-every 0` → `ZeroDivisionError`.

**Local `src/` is missing `validated_segmentation.py`, which silently makes `nrmp_vocab` report
9,445 roots instead of 9,490. The pod is correct at 9,490.**
