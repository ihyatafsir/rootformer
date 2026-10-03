# STATE — where things actually stand

Written 2026-10-03 because the coordinating session hit its context limit and began making
directory-level errors. **Read this before trusting anything in a session summary.**

---

## CORRECTIONS — errors made and retracted (do not repeat these)

Three claims were made with confidence and were **wrong**. They are recorded here so they are
not re-inherited.

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
       EARLYROOT_C  <- DIED at step 1 by CUDA OOM, NOT a stage failure (see below)
       runner.sh (pid 413704) staged [EARLYROOT_C RESIDUAL_R SCOREBIAS_D LATE_X], WAITING
       ~20 h GPU budget; two concurrent 32-batch arms does NOT fit
```

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
```

**FLAT since step 2000: 7.74-8.14 % across 7,000 steps. CE_z MONOTONE WORSE, 8.1712 -> 8.9897.**
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
2. **LATE_X is queued LAST and will not run inside the ~20 h budget.** The runner started
   00:28 UTC with 4 stages; FLOOR_A alone runs ~3.5 h more, then each full-trunk arm ~4-5 h.
   `FLOOR_A -> EARLYROOT_C -> RESIDUAL_R -> SCOREBIAS_D -> LATE_X` puts the early-vs-late test at
   ~20+ h, i.e. past the budget. The runner was **NOT touched**. **Reordering it is a human
   decision** — it is another agent's process, and the hard rules forbid killing one. `LATE_X` =
   `--root-cross-attn top4 --ishtiqaq-root-bias top4`; every other flag identical to C.
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
