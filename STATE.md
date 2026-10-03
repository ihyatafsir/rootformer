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
computed qiyas, inverse root ID from surface  3,031 / 3,588 = 84.48 %   (chance 2.70 %)
```

Caveats, stated: all 90 realisation fixes are **one root's orthography** (`بدا`, hamza vs bare
alif); over 8 random 37-root folds the computed analogy is **marginally worse** than the
hand-coded realiser (0.8908 vs 0.8957); and Task B's candidate set was **restricted to 37 roots**
— the full 9,220-root inverse search is the honest completion and was never run.

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
pod    FLOOR_A (trunk trained, NO root pathway) ~8 % at step 2k/20k
       EARLYROOT_C (all 24 layers, both mechanisms)
       runner.sh queue staged: RESIDUAL_R, SCOREBIAS, early-vs-late, scale, derivational holdout
       ~20 h GPU budget; two concurrent 32-batch arms is the measured-safe ceiling
```

**Open, in priority order:**

1. **The qiyās re-run with TWO conditions** — CPU, minutes. May restore the root-as-ʿilla.
2. **The early-vs-late test at a matched trained trunk** — if it's a tie, everything concluded
   about the late RCA's ceiling was a statement about a *frozen* trunk.
3. **The derivational holdout, properly designed** — the thesis has still never been tested.
   Attempt 1 was structurally impossible (a softmax cannot emit an unseen root); attempt 2 ran
   at 0.141 epochs.
4. **The full 9,220-root inverse search** for qiyās.
5. `rca_stack.eval()` in `evaluate` (dropout stays active at eval, −0.13 pp);
   `--eval-every 0` → `ZeroDivisionError`.

**Local `src/` is missing `validated_segmentation.py`, which silently makes `nrmp_vocab` report
9,445 roots instead of 9,490. The pod is correct at 9,490.**
