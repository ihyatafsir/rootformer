# al-qiyās as a computed rule — and the derivational holdout at 0 / 3,588

**Session:** delegated subagent of `session-a4f1047e-2a01-48e1-8224-820d03d1563c`
**Pod:** `g6exduq0bd17z8` @ `213.173.104.76:46758` · Python `/workspace/venvs/rootformer/bin/python`
**Date:** 2026-10-02 · **All work CPU-only.** `nvidia-smi` was checked first: GPU at 100 %,
two other agents' arms running (`RCA_DEPTH_T8` PID 101203, `RCA_NORM` PID 106246). Neither was
touched; no GPU job was launched by this task. Pod load stayed ≈15 of 48 cores.
**Nothing was edited in place.** All new code lives in `/workspace/qiyas/`, mirrored to
`rootformer/build/qiyas/`. The shipped modules are byte-identical to their post-`lisan_root_fix`
state (`nrmp_vocab.py` md5 `0ee0ab6301d7f26eddc7bd079f7fede7`, blueprint
`5bd3e828b41fb1a6a6039d47c2f359fc`, both mtime 2026-10-01), and the qiyās pathway is installed
at **runtime** on the instance, so no certified file could move.

---

## 0. Verdict, in one paragraph

**Yes — a computed analogy emits and realises forms of roots the learned decoder could not, and
it does so on the identical 3,588 positions where that decoder returned 0 / 3,588.** Qiyās,
with its rules induced *without* the 37 held-out roots, realises the correct surface for
**3,023 / 3,588 = 84.25 %** of the gold positions (95.39 % on the 3,169 punctuation-free ones)
and recovers the held-out root from the surface alone at **84.48 % top-1 / 84.50 % top-5**
(95.68 % on the clean subset), against chance of 2.70 % / 13.51 %. The neural compositional
decoder's held-out-root accuracy was **0.00 % top-1 and top-5, joint and content-only, at 6 k
and 24 k steps**. That said, the margin over the *existing hand-coded* realiser on those
positions is **+90 positions and 0 breaks — and all 90 are one root's orthographic convention**
(`بدا`: the corpus writes `أبدا` / `مبتدا` where the hand-coded rule inserts a hamza). On
randomly chosen 37-root folds the computed analogy is on average marginally **worse** than the
hand-coded realiser (0.8908 vs 0.8957). So the honest form of the answer is: **generalisation to
unseen roots here is a property of a rule over determinate root classes, not of a learned
representation — and it is real, measurable, and small.**

---

## 1. The procedure, and which qiyās form it implements

### 1.1 Anatomy (`qiyas_engine.py`)

```
aṣl    the attested case     an attested (root, wazn) -> surface realisation
farʿ   the new case          a (root, wazn) whose realisation is not attested
ʿilla  the effective cause   a DETERMINATE PROPERTY of the root shared by aṣl and farʿ
ḥukm   the ruling            the root->surface ALIGNMENT (not a string: a template with slots)
```

The ḥukm is represented as an **alignment** over the root's radicals:

| token | meaning |
|---|---|
| `^i` | the *i*-th radical slot of the root (0-based) |
| `c`  | the literal character *c*, an afformative the wazn supplies |

Radical slots must increase but need not be contiguous and need not exhaust the root — dropping a
radical is exactly what iʿlāl does. Worked examples:

| root | wazn | surface | alignment | classical name of the operation |
|---|---|---|---|---|
| كتب | فَاعِل | كاتب | `^0ا^1^2` | plain substitution |
| قول | فَعَلَ | قال | `^0ا^2` | iʿlāl by elision of the hollow radical |
| كرر | فَعَلَ | كرّ | `^0^1` | idghām of the geminate |
| شفي | فِعَال | شفاء | `^0^1اء` | madd on the nāqiṣ |
| دعا | فِعْلَى | دعوى | `^0^1وى` | the wāwī nāqiṣ |
| رضي | فِعْلَى | رضى | `^0^1ى` | the yāʾī nāqiṣ |

The alignment of a surface to a root is computed by **longest-common-subsequence DP** over the
radicals, i.e. it uses a *maximum-cardinality* order-preserving matching. This matters: the naïve
left-to-right greedy aligner mis-anchors the imperfective (in `يشفي` it matches the prefix `ي`
to the root `شفي`'s *final* radical, yielding a 1-radical alignment). LCS gets `ي^0^1^2`.
`min_arity = 2`: a surface that realises fewer than two radicals of its root is not admitted as a
realisation of that root at all.

The **ḥukm of a cell** `(ʿilla-label, wazn)` is the **modal alignment** over the attested aṣl in
that cell. Induction is counting. Nothing is fitted by gradient descent, and every one of the
shipped tokenizer's ~40 hand-coded wazn branches is *re-derived* by this count.

### 1.2 Which qiyās form, and why

**Implemented: qiyās al-ʿilla** — extension by a shared effective cause.

**Not implemented: qiyās al-shabah** (extension by resemblance) — it would let an unmotivated
similarity stand in for a cause, and in this setting it is unfalsifiable: any root resembles any
other in some respect, so *every* farʿ would be admissible and the procedure would have no test to
fail. **Not implemented: qiyās al-dalāla** (extension by textual indication) — that needs a text,
not a rule, and would amount to re-reading the corpus.

The reason is not aesthetic. The claim under test is that *a shared root representation
generalises to unseen derivational forms*. That is a claim about an effective cause — the property
of the root that carries the ruling across. Only qiyās al-ʿilla makes that cause explicit and
therefore measurable. The classical literature keeps the distinction: al-Rāzī lists the six
rational *masālik al-ʿilla* and separates them —
«المناسب والمؤثر والشبه والدوران والطرد والسبر»
(`/workspace/scholastic_sanitized/Razi_Al_Mahsul.txt:17327–17328`) — so *al-shabah* is a
recognised but distinct mode, and this work deliberately takes the *ʿilla* mode.

### 1.3 The ʿilla functions

Each is a **total, single-valued** function of the root string, so *munḍabiṭ* in the narrow sense
determinacy is 1.000 by construction for every one of them (measured; see §3).

| name | labels | what it says |
|---|---|---|
| `identity` | the root itself | the brief's literal scheme: aṣl and farʿ share the root |
| `wazn_only` | `*` (constant) | **negative control**: no cause at all, the context-free template |
| `coarse_weak` | SAHIH, WEAK, MUDAAF, MAHMUZ | the over-general cause that lumps ajwaf + nāqiṣ + lafīf + mithāl |
| `strict7` | SAHIH, MITHAL, AJWAF, NAQIS, LAFĪF, MUDAAF, MAHMŪZ | the classical division the iʿlāl / idghām / hamza rules are stated per |
| `positional` | `strict7` refined by *which* weak radical and its identity (AJWAF_و / AJWAF_ي, NAQIS_و / NAQIS_ي, MUDAAF_12 / MUDAAF_23, …) | the finest division |

A **back-off lattice** joins them (`positional → strict7 → coarse_weak → wazn_only → GLOBAL`).
Each level induces **its own** table — a root string is not a class name, so looking a coarser
label up in a finer table silently misses and drops every farʿ to the context-free rule. That was
a real bug in the first implementation: `strict7`, `positional` and `coarse_weak` returned
*identical* farʿ numbers because all three collapsed to the global rule. The fire level is
reported for every prediction (`strict7:NAQIS`, `GLOBAL`, …), which is what makes §§3–4
measurable.

---

## 2. Correctness: brute force, unit tests, and flag-off inertness

`test_qiyas.py` — **73 / 73 checks**, identical locally and on the pod
(`PASS 73 FAIL 0 · ALL CHECKS PASSED`). Where a result can be checked by exhaustive search it is:

* **The induced ḥukm is the true argmax.** For a small cell, `brute_force_cell()` enumerates the
  *entire* bounded alignment space (every order-preserving subset of radicals of size ≥ 2,
  interleaved with 0–2 literals from an alphabet) and scores every candidate by exact matches.
  The induced modal alignment equals the brute-force argmax, on a regular cell *and* on a cell
  with a planted irregular member (`brute force picks the majority alignment: 6`). The search
  space is not trivial (> 50 candidates for a 3-letter root over a 3-letter alphabet).
* **The aligner is arity-maximal.** Over 300 random (root, surface) pairs, the number of radicals
  the LCS aligner uses equals the brute-force maximum, computed by explicit recursion.
* **`instantiate`/`align` are mutually consistent**: for every generated (alignment, root) pair,
  re-aligning the produced surface reproduces that surface exactly.
* **iṭṭirād is measured leave-one-out**, not in-sample: each observation's own counts are
  decremented at every level of the lattice for the duration of its test, so a rule cannot be
  credited for memorising the example it was counted from.
* **The wiring contract is tested.** Flag OFF returns the original value and still calls the
  original; flag ON with a `NullEngine` is inert; flag ON in `replace` mode bypasses the original
  (0 calls with logging off) and changes the output; `uninstall()` restores the exact prior state.

---

## 3. The uṣūlī validity conditions — sources, measurements, and what was rejected

### 3.1 The conditions, with primary-text citations

All four are quoted from texts **actually held on the pod**. Note the paths: al-Rāzī and
al-Ghazālī are in `/workspace/scholastic_sanitized/`, **not** in `/workspace/heritage_foundations/`.

**munḍabiṭ — precise / well-defined.**
> «المسألة الثانية الوصف الحقيقي إذا كان ظاهرا **مضبوطا** جاز التعليل به»
> — `/workspace/scholastic_sanitized/Razi_Al_Mahsul.txt:15721–15722`

> «يجوز التعليل بالأوصاف العرفية … ولكن **بشرطين** أحدهما أن يكون **مضبوطا متميزا عن غيره**»
> — `Razi_Al_Mahsul.txt:15907–15909`

> «الحكمة غير مضبوطة فلا يجوز ربط الأحكام بها» — `Razi_Al_Mahsul.txt:14922`

*Tested as:* (a) determinacy — the ʿilla is a total single-valued function of the root; (b) cell
purity — support of the modal alignment over the cell's observations.

**muṭṭarid — consistent, holds without exception.**
> «والمتفق عليه في **القياس المطرد**» … «إن الشيء إذا **اطرد** عليه باب، فصح في القياس وقام في
> المعقول، ثم اعترض عليه شيء شاذ نزر قليل، **لعلة تلحقه**، لم يكن ذلك مبطلا للأصل»
> — `/workspace/heritage_foundations/Al_Zajjaji_Al_Idah_Fi_Ilal_Al_Nahw.txt:1203–1205`

> «وذلك يقع خروجا عن **القياس المطرد**» — `Al_Zajjaji_Al_Idah_…:1377`

> «| باب القول على **الاطراد والشذوذ** … الكلام في الاطراد والشذوذ على **أربعة أضرب**: مطرد في
> القياس والاستعمال جميعا وهذا هو الغاية المطلوبة … ومطرد في القياس شاذ في الاستعمال»
> — `/workspace/heritage_foundations/Ibn_Jinni_Al_Khasais.txt:1371, 1394–1397`

> «**الفصل الثامن في الطرد** … فهذا هو المراد من **الاطراد والجريان**» and the opponent's sharper
> formulation: «**الاطراد عبارة عن كون الوصف بحيث لا يوجد إلا ويوجد معه الحكم**»
> — `Razi_Al_Mahsul.txt:15132–15143`

> «لأن **الطرد واجب في العلل والعكس غير واجب** فيها» — `Razi_Al_Mahsul.txt:14389`

*Tested as:* the fraction of attested (root, wazn) observations the ʿilla's **own** modal rule
reproduces exactly, under leave-one-out. al-Zajjājī's licence for "a rare shādhdh exception with
its own ʿilla does not invalidate the aṣl" is what makes a *threshold* (rather than 1.000) the
classically correct reading.

**munʿakis — co-extensive / present wherever the ruling applies and absent where it does not.**
> «وقال قوم الدوران أقوى وعبروا عن ذلك بأن **العلة المطردة المنعكسة** أقوى مما لا يكون كذلك»
> — `Razi_Al_Mahsul.txt:17330`

> «وكان **مطردا منعكسا** في طريق الحمل» — `/workspace/scholastic_sanitized/Ghazali_Miyar_al_Ilm.txt:3046`
> «… فقد ذكرت ما هو ذاتي **ومطرد ومنعكس**» — `Ghazali_Al_Mustasfa.txt:403`
> «هو **مطرد منعكس** فما الحد عندك؟» — `Ghazali_Al_Mustasfa.txt:461`

**This condition is contested, and the report must say so.** al-Rāzī *reports* it as the position
of «قوم» and immediately disputes it as the ground of causation: he argues the attribute affects
the ruling by its *munāsaba*, not by its *dawārān* (same page-range, `Razi_Al_Mahsul.txt:17330–17335`).
al-Ghazālī uses muṭṭarid-and-munʿakis of a **ḥadd** (definition), not of an ʿilla.
*Tested as:* for every cell with ≥ 30 observations, the cell's own ḥukm applied **inside** the
class minus its accuracy applied **outside** the class (same wazn) — the discrimination *lift*.

**mutāʿaddī — extendable: it must exist outside the locus of the aṣl.** (The fourth condition,
which the brief's three-condition list omits and which turns out to be the decisive one.)

> «وأما إن كانت العلة **متعدية** لم يصح أن يكون **محل الحكم علة للحكم** لأن **العلة المتعدية هى
> التي توجد في غير مورد النص** وخصوصية مورد النص يستحيل حصولها في غيره لأن الشئ لا يكون نفس غيره»
> — `Razi_Al_Mahsul.txt:15719–15721`

*Tested as:* on a held-out set of (root, wazn) **pairs**, the share of rulings carried by the
primary ʿilla's **own** cell with no back-off. This is the criterion that rejects ʿilla = the root.

### 3.2 Measured rates

Induction set: **106,664 admissible aṣl** (the held-out corpus's own forms that the shipped
pipeline reproduces exactly), 5,551 distinct roots, **0 unalignable**.

| ʿilla | cells | munḍabiṭ purity | impure cells | muṭṭarid (own rule) | exceptions | munʿakis in / out / **lift** | mutāʿaddī | verdict |
|---|---|---|---|---|---|---|---|---|
| `wazn_only` | 39 | **0.9457** ✗ | 0.821 | **0.9457** ✗ | 5,795 | – / – / **0.0000** ✗ | 0.0000 | **REJECTED** |
| `coarse_weak` | 135 | 0.9885 | 0.452 | 0.9885 | 1,226 | 0.9957 / 0.8748 / +0.1209 | 1.0000 | accepted |
| **`strict7`** | 231 | 0.9887 | 0.342 | 0.9886 | 1,212 | 0.9957 / 0.9205 / +0.0752 | **1.0000** | **ACCEPTED — used** |
| `positional` | 462 | **0.9901** | **0.156** | **0.9898** | **1,084** | 0.9962 / 0.9323 / +0.0639 | 0.9985 | accepted |
| `identity` | 25,042 | 1.0000 | 0.000 | 0.9963 | 397 | 1.0000 / 0.8703 / +0.1297 | **0.0000** ✗ | **REJECTED by mutāʿaddī** |

Thresholds were declared before the numbers were seen: purity ≥ 0.98, iṭṭirād ≥ 0.98, lift > 0.02.
munḍabiṭ and muṭṭarid are judged on the ʿilla's **own** rule (no fall-back crutches); the
with-back-off rates are reported alongside and are not substituted for them.

**Rejected.**

1. **`wazn_only` — rejected by three conditions at once.** It is the context-free template: 82 %
   of its cells are impure (42 alignment patterns across 39 cells), iṭṭirād is 0.9457 (5,795
   exceptions), and its munʿakis lift is **exactly 0 by construction** (a constant cause has no
   outside, so `in` and `out` are the same set). This is the negative control behaving as it must:
   a "cause" that is present everywhere is present where the ruling does *not* apply.
2. **`identity` (ʿilla = the root) — rejected by mutāʿaddī, and only by mutāʿaddī.** On the
   attested set it is the *best* ʿilla: purity 1.000, iṭṭirād 0.9963, lift +0.1297. But hold out
   (root, wazn) **pairs** and induce from the rest — then the identity cell is empty by
   definition, and the primary-ʿilla firing rate is **0.0000**: not one ruling is carried by the
   root itself; every one is carried by the back-off class (realisation still 0.9975, so the
   system works — but the ʿilla proposed in the brief does no work). The same collapse shows in
   the derivational holdout: `identity` *without* back-off puts **2,259 / 2,259 = 100 %** of its
   farʿ predictions at `GLOBAL` and scores **85.01 %** — *worse* than the shipped realiser's
   88.31 %. This is al-Rāzī's point at `al-Maḥṣūl:15719–15721` stated as a measurement: the ʿilla
   must exist outside the locus of the aṣl, and a root is its own locus.
   The brief's literal scheme is therefore **not extendable**, and the ʿilla that is used instead
   is the root's **phonological class** — a property the farʿ can share.
3. **`coarse_weak` fails in the unit tests but survives on the corpus.** On the constructed
   classical set it fails iṭṭirād on `فَعَلَ` (ajwaf elides its middle radical, nāqiṣ and mithāl
   realise all three: 0.7500 vs strict7's 1.0000) — the exact Ibn-ʿUsfūr-shaped objection to an
   over-general ʿilla. On the real corpus the margin narrows to 0.9885 vs 0.9886 and it passes the
   threshold. Reported as measured; it is **not** claimed to be rejected on the corpus.

**A finding the four tests did not anticipate: they do not agree on a winner.** Refining the ʿilla
improves munḍabiṭ (0.9885 → 0.9901) and muṭṭarid (0.9885 → 0.9898) and *degrades* munʿakis lift
(+0.1209 → +0.0639): broader cells are more discriminating on the lift measure and less precise on
the purity measure. And operationally the refinement is **inert** — on the 2,395 farʿ types
`strict7`, `positional` and `coarse_weak` produce *identical* predictions (89.65 %; provenance
2,259 / 2,257+2 / 2,259 respectively). So `positional` is **operational for the validity
measurement and decorative for the output**, and `strict7` is the one shipped.

---

## 4. Wiring, and the ablation that proves it acts

`qiyas_wire.py` installs the procedure on the **instance** (`tok.realize_root_and_wazn` is
replaced by a dispatching closure that keeps the original bound method). Three reasons this is not
a dodge but the correct instrument here: flag OFF is *provably* inert (the wrapper calls the
original and returns its value unchanged — verified on all 188,722 held-out types, **0
mismatches**); `uninstall()` restores the exact prior behaviour (108,682 ✓); and every call is
counted, so "it acts" is measured rather than asserted.

| mode | flag OFF | flag ON | Δ types | newly fixed | newly broken | hook `n_replaced` |
|---|---|---|---|---|---|---|
| `fallback` (only where the shipped realiser returned the bare root or a control token) | 108,682 | 108,639 | **−43** | 40 | 83 | 123 |
| **`replace`** (qiyās decides wherever it has an analogue) | 108,682 | **108,811** | **+129** | 391 | 262 | 675 |

Flag OFF reproduces the recorded shipped-default type-level round trip **exactly**: 108,682 /
188,722 = **57.5884 %**, 0 mismatches against `du_after.json`. The pathway therefore acts: with the
flag on, 675 stems change and the round-trip total moves by +129 types.

*Honest note:* `fallback` mode is **net negative** (−43). Restricting qiyās to only the cases the
hand-coded realiser visibly failed does not help, because among those cases the induced modal rule
is wrong more often than right. The gain only appears when qiyās is allowed to decide outright.

---

## 5. The decisive test: the derivational holdout against 0 / 3,588

### 5.1 Exactly reproducing the 3,588 positions

`streams.pt` stores only tuple ids, so the surface words behind the 3,588 scored positions are in
no artifact. `recover_deriv_gold.py` reconstructs them from the split definition
(`scratch_lm_comp.py:101–300`) and **verifies the reconstruction independently at three points**:

| quantity | reconstructed | recorded |
|---|---|---|
| sentences of the 16 held-out works containing a held-out root | **3,627** | 3,627 (`RESULTS_COMP.md`) |
| held-out-root occurrences in them | **3,987** | 3,987 (`audit_comp.json`) |
| sentence-initial occurrences (not targets: `r_tg = R[1:]`) | **399** | 3,987 − 3,588 |
| **target positions** | **3,588** | 3,588 |
| per-root histogram over the 37 roots | **exact match** | `holdout_records` in `comp24k_eval.json` |

The histogram check is the load-bearing one: the counts `كره 243, صبح 178, شخص 166, دفع 166, …`
match **all 37 roots exactly**. The gold surfaces are the real corpus words.

### 5.2 Task A — realisation, given the ʿilla (root and wazn)

Rules induced from **104,549 aṣl whose root is not one of the 37**. Shipped baseline computed by
calling the realiser live (cross-checked: **2,944 / 2,944 = 1.0000** agreement with the recorded
per-type flags on the overlapping words).

| | positions | rate |
|---|---|---|
| **neural compositional decoder** (root prediction, top-1 **and** top-5, 6 k and 24 k) | **0 / 3,588** | **0.00 %** |
| shipped hand-coded realiser | 2,933 / 3,588 | 81.74 % |
| **computed qiyās** | **3,023 / 3,588** | **84.25 %** |
| computed qiyās, punctuation-free subset | 3,023 / 3,169 | **95.39 %** (shipped 92.55 %) |

Δ **+90 positions, 90 fixes, 0 breaks**, 101 positions with no analogue in the lattice; provenance
`strict7` for 3,487 / 3,588 (97.2 %) — i.e. the class ʿilla itself carries the ruling almost
everywhere, which is the mutāʿaddī property measured directly.

**What the +90 actually is, stated plainly.** All 90 are the same phenomenon and all 90 belong to
**one root**, `بدا`:

```
root بدا · أَفْعَلَ    corpus word أبدا     shipped rule أبدأ     qiyas أبدا
root بدا · مَفْعَل     corpus word مبتدا    shipped rule مبتدأ    qiyas مبتدا
root بدا · مُفْتَعِل   corpus word المبتدا  shipped rule المبتدأ  qiyas المبتدا
```

The hand-coded branch writes a hamza on the nāqiṣ root's final alif; the corpus (undiacritised,
sanitised) writes a bare alif; the induced alignment reproduces the corpus convention because that
is what the aṣl attest. This is a real improvement on this corpus and this metric — and it is an
**orthographic convention recovered from data, not a morphological insight**. The remaining 464
failures are also not qiyās failures: they are dominated by tokenisation artefacts in the gold set
(419 / 3,588 gold words contain a non-Arabic-letter character, 337 of them the Arabic comma `،`,
which the sentence splitter does not break on — so punctuation sits inside the stem and *neither*
realiser can reproduce the "word"; e.g. both produce `شخص` for the gold `شخص،`).

### 5.3 Task B — inverse qiyās: recover the root from the surface

Given only the word, candidates are the 37 held-out roots × 139 awzan; a candidate is admissible
iff qiyās's realisation equals the word; roots are ranked by the ʿilla level that fired, then cell
purity, then support, then aṣl frequency.

| | top-1 | top-5 |
|---|---|---|
| **neural decoder (same 3,588 gold roots)** | **0 / 3,588 = 0.00 %** | **0 / 3,588 = 0.00 %** |
| chance over 37 candidates | 2.70 % | 13.51 % |
| **computed qiyās** | **3,031 / 3,588 = 84.48 %** | **3,032 / 3,588 = 84.50 %** |
| computed qiyās, punctuation-free subset | 3,032 / 3,169 = **95.68 %** | 95.68 % |
| control: 74 candidates (37 held-out + 37 random content roots) | 84.48 % (chance 1.35 %) | 84.50 % (chance 6.76 %) |

The rule is nearly deterministic: mean **1.16 admissible candidate instances** per position, 556
positions with no admissible candidate at all; top-1 ≈ top-5 because the surface either pins the
root or excludes it. The 74-candidate control gives **identical** accuracy, so the ranking is not
an artefact of the candidate-set size.

*Protocol caveat, because it matters:* the gold prefix/suffix segmentation is supplied to Task B.
Prefixes and suffixes are orthographic clitics and do not name the root — only the root is hidden —
but this is not a fully blind identification test. The candidate set is also restricted to the
held-out roots (plus the control), not the full 9,220-root inventory; the restricted set is exactly
the set the decoder failed on, and the control shows the ranking is stable as the set grows, but a
full-inventory inverse search is the right next experiment and was not run.

### 5.4 The type-level derivational holdout, and the random folds

The same root holdout applied to the 188,722-type held-out lexicon (qiyās induced without the 37):

| subset | types | shipped default | qiyās (`replace`) | Δ |
|---|---|---|---|---|
| types whose root is one of the 37 | 2,395 | 88.3090 % | **89.6451 %** | **+32** |
| all other types | 186,327 | 57.1935 % | 57.2456 % | +97 |
| **all types** | **188,722** | **57.5884 %** | **57.6568 %** | **+129** |

Per-ʿilla on the 2,395 farʿ types (ranked by the engine's own provenance):

| ʿilla | farʿ types | baseline | qiyās | fixes | breaks | no analogue | provenance |
|---|---|---|---|---|---|---|---|
| `strict7` | 2,395 | 88.31 % | **89.65 %** | 32 | **0** | 136 | strict7 2,259 |
| `positional` | 2,395 | 88.31 % | 89.65 % | 32 | 0 | 136 | positional 2,257, strict7 2 |
| `coarse_weak` | 2,395 | 88.31 % | 89.65 % | 32 | 0 | 136 | coarse_weak 2,259 |
| `identity` (with back-off) | 2,395 | 88.31 % | 89.65 % | 32 | 0 | 136 | **strict7 2,259 — identity fired 0×** |
| `identity` (no back-off) | 2,395 | 88.31 % | **85.01 %** | 53 | **132** | 136 | **GLOBAL 2,259 (100 %)** |

**And the counter-evidence, which belongs in the same table.** Over 8 random 37-root folds:

| | mean | range |
|---|---|---|
| shipped default | 0.8957 | [0.8729, 0.9131] |
| computed qiyās | **0.8908** | [0.8511, 0.9131] |
| qiyās fixes / breaks per fold | 0.5 / 1.5 | — |

**On randomly chosen root folds the computed analogy is on average marginally worse than the
hand-coded realiser** (fold 0: 0.876 → 0.851). It is better specifically on the 37 roots that the
neural experiment held out, which were selected as the *rarest* (least sentence-containing) roots.
This is the honest boundary of the claim, and it is the reason the verdict in §0 is "real,
measurable, and small" rather than "better".

---

## 6. Effect on type-level round-trip (57.59 %)

| | value |
|---|---|
| shipped default (flag OFF, reproduced exactly on 188,722 types) | **57.5884 %** |
| + computed qiyās, `fallback` | 57.5656 % (−43 types) |
| **+ computed qiyās, `replace`** | **57.6568 % (+129 types, +0.0684 pp)** |
| restricted to the 2,395 held-out-root types | 88.3090 % → **89.6451 % (+32 types, +1.336 pp)** |
| token-level context (for scale; not re-measured here) | 82.11 % |

The brief's "57.59 %" is confirmed as the shipped default: `lisan_root_fix/SUMMARY.md:73` records
57.5884 % distinct-form round trip after the root fix (and `:71` records 82.1065 % token round
trip). The 57.5926 figure that appears in `eval_heldout_results.json:13073` is a *per-file*
morphemic type rate for `Farabi_Fusus_al_Hikam.txt`, not the corpus-wide round trip — the
corpus-wide mode-1 value in that artifact is 55.9611 % (pre-fix). Both are noted so the number
cannot be mis-cited later.

---

## 7. Operational versus decorative

**Operational — each one changes a number, a verdict, or an output.**

| concept | what it does here |
|---|---|
| the **ʿilla itself** (root class) | carries 97.2 % of the 3,588 rulings at its own level; the whole farʿ result depends on it |
| **ḥukm as an alignment** | the mechanism; re-derives every hard-coded wazn branch by counting |
| **munḍabiṭ** (purity) | rejects `wazn_only`; separates `strict7` from `coarse_weak` (0.9887 vs 0.9885) |
| **muṭṭarid** (iṭṭirād) | rejects `wazn_only`; the criterion that rejects the over-general ʿilla in the unit tests (0.7500 vs 1.0000) |
| **mutāʿaddī** | the decisive condition: rejects ʿilla = the root (0.0000 firing) and explains the 85.01 % vs 88.31 % regression |
| **the back-off lattice** | 94–97 % of farʿ reach a rule at all (136 of 2,395 do not) |
| **qiyās al-ʿilla** as the chosen mode | makes the cause explicit; the reason §5 is measurable at all |

**Decorative in this system — measured, sourced, and quantitatively real, but not load-bearing.**

| concept | why it is decorative *here* |
|---|---|
| **munʿakis** (lift) | every verdict it could give is already given by munḍabiṭ + muṭṭarid: `wazn_only` is rejected by both before the lift is consulted. It still *discriminates* (0.0000 for `wazn_only`, +0.1209 for `coarse_weak`), and al-Rāzī himself reports it as contested («وقال قوم…», `al-Maḥṣūl:17330`) and holds that **ʿaks is not required** (`:14389`). Reported, not relied on. |
| the `strict7 → positional` refinement | improves purity 0.9885 → 0.9901 and iṭṭirād 0.9885 → 0.9898, and changes **zero** holdout predictions (2,259 vs 2,257+2 identical outputs) |
| **qiyās al-shabah / al-dalāla** | deliberately not implemented — al-shabah would be unfalsifiable here, al-dalāla would be re-reading the corpus |
| the `fallback` mode of the wire | net **−43** types; it demonstrates the hook acts but is not the configuration that helps |

**Rejected and not shipped:** `wazn_only` (by munḍabiṭ, muṭṭarid, munʿakis) and `identity`-as-ʿilla
(by mutāʿaddī). `coarse_weak` was rejected in the unit tests (iṭṭirād 0.7500) but passes on the
corpus and is reported as such, not claimed as rejected.

---

## 8. What is sourced, and what is not

**Sourced, with `file:line`, from texts actually held.**

| claim | citation |
|---|---|
| the analogical cause and the aṣl→farʿ transfer by a shared cause | `heritage_foundations/Al_Zajjaji_Al_Idah_Fi_Ilal_Al_Nahw.txt:427` |
| iṭṭirād as the criterion of sound qiyās; a rare shādhdh exception with its own ʿilla does not invalidate the aṣl | `Al_Zajjaji_Al_Idah_Fi_Ilal_Al_Nahw.txt:1203–1205`, `:1377` |
| the fourfold iṭṭirād / shudhūdh division | `heritage_foundations/Ibn_Jinni_Al_Khasais.txt:1371, 1394–1397` |
| munḍabiṭ as a condition of ʿilliyya | `scholastic_sanitized/Razi_Al_Mahsul.txt:15721–15722`, `:15907–15909`, `:14922` |
| iṭṭirād as the operational definition "no ʿilla without the ruling" | `Razi_Al_Mahsul.txt:15132–15143` |
| ṭard required, ʿaks not | `Razi_Al_Mahsul.txt:14389` |
| munʿakis / dawārān, reported as contested | `Razi_Al_Mahsul.txt:17330` |
| muṭṭarid + munʿakis of a ḥadd | `Ghazali_Miyar_al_Ilm.txt:3046`, `Ghazali_Al_Mustasfa.txt:403, 461` |
| the six masālik al-ʿilla incl. *al-shabah* and *al-dawārān* | `Razi_Al_Mahsul.txt:17327–17328` |
| the extendable (mutāʿaddiya) ʿilla as the one that exists outside the locus of the text | `Razi_Al_Mahsul.txt:15719–15721` |
| Ibn Jinnī's *al-ishtiqāq al-akbar*, the object of the iṭṭirād objection | `Ibn_Jinni_Al_Khasais.txt:7625, 7639–7641` |
| the analogy does not hold uniformly, and the exceptions must be named | `heritage_foundations/Ibn_Faris_Mujam_Maqayis_Al_Lughah.txt:918–923` (editor's introduction, quoting *al-Ṣāḥibī* p. 33) |

**NOT sourced — flagged, not dressed up.**

* **Ibn ʿUsfūr's «غير مأخوذ به؛ لعدم اطراده» is not verifiable against the held corpus.** No text
  of Ibn ʿUsfūr is held. The only occurrence of the string «ابن عصفور» in the entire
  `heritage_foundations/` corpus is an editor's page reference at
  `Ibn_Jinni_Al_Khasais.txt:19037` (`ابن عصفور ج1 ص189`) — a bibliography line, not the dictum.
  It is quoted in the task brief and is reported here as **cited from the brief, not confirmed
  from the corpus**.
* al-Suyūṭī's *al-Muzhir* is not held; the dictum is usually traced there.
* The three-way naming *qiyās al-ʿilla / al-shabah / al-dalāla* is uṣūlī terminology; what the held
  corpus supplies is al-Zajjājī's «العلة القياسية» (`:427`) and al-Rāzī's six *masālik* including
  *al-shabah* (`:17327–17328`). The naming is standard, but the specific three-way label is **not
  quoted from a held text**.
* al-Rāzī's *al-Maḥṣūl* and al-Ghazālī's *al-Mustaṣfā* / *Miʿyār al-ʿIlm* are in
  `/workspace/scholastic_sanitized/`, **not** in `/workspace/heritage_foundations/`. Cited with
  their true paths above.
* `munḍabiṭ` appears in al-Rāzī as **مضبوط**, not as the later nominalised **منضبط**; a grep for
  the exact form «منضبط» across the uṣūlī and heritage texts returns nothing. The concept is
  sourced; the nominal form is not.

---

## 9. Honest limits

1. **The 3,588 task is not the decoder's task.** The decoder inferred the root from context; qiyās
   is given the root (Task A) or the whole surface (Task B). What is comparable is the property
   under test, and the verb "emit": the decoder *could not emit* the root at all (its held-out
   rows were trained only as negatives — 22.45 bits, 9.2 bits worse than uniform over 9,490 types,
   `RESULTS_COMP.md §7`); the computed procedure emits it and realises it correctly for 84–95 % of
   the same positions. The comparison is of capacities, not of identical inputs, and is labelled
   as such throughout.
2. **The margin over the hand-coded realiser is one root's orthography.** +90 / 0 on the 3,588,
   all `بدا`. On random root folds the computed analogy is on average *slightly worse*. Anyone
   reading "+84.25 % vs 0 %" as "qiyās replaces the realiser" has read it wrong; the correct
   reading is "the rule is *inducible* from attested forms and generalises across roots".
3. **Task B's candidate set is restricted** to the 37 held-out roots (and the 74-root control), and
   the gold prefix/suffix split is supplied. A full 9,220-root inverse search was not run.
4. **The gold set inherits tokenisation artefacts.** 419 / 3,588 gold words carry punctuation; the
   punctuation-free subset numbers (95.39 % / 95.68 %) are given alongside so the artefact can be
   seen and discounted.
5. **The aṣl are the shipped pipeline's own successful analyses.** An attested form was admitted as
   aṣl only where `decode(encode(w)) == w`, so the induction set is not independent gold; it is
   the corpus plus the analyser. The holdout is on **roots**, which is the axis the neural
   experiment held out, and it is enforced strictly — but a hand-annotated gold morphology set
   would be a stronger basis.
6. **There is a known circularity in proving ʿilliyya by iṭṭirād, and al-Rāzī states it**:
   «…فإذا أثبتم حصول الحكم في الفرع بكون ذلك الوصف علة وبينتم عليته بكونه مطردا لزم الدور وهو باطل»
   (`al-Maḥṣūl:15143–15145`). The iṭṭirād measured here is a *descriptive* consistency rate over a
   fixed attested set, not a proof of causation, and is offered as the former.
7. **The seven frozen suites were re-run to confirm the shipped modules are still green** (§10);
   this work edited none of them.

---

## 10. Artifacts

Pod `/workspace/qiyas/` ⇄ local `rootformer/build/qiyas/`:

| file | what it is |
|---|---|
| `qiyas_engine.py` | the procedure: ʿilla functions, LCS alignment, induction, the four conditions, brute force |
| `qiyas_wire.py` | the runtime hook (modes `replace` / `fallback` / `overlay`), flag, counters, `uninstall` |
| `test_qiyas.py` | 73 / 73 checks incl. brute-force validation and flag-off inertness |
| `qiyas_corpus.py` | harvesting, the ʿilla condition measurements, the root holdout, random folds, the ablation |
| `recover_deriv_gold.py` | exact reconstruction + verification of the 3,588 gold positions |
| `qiyas_deriv3588.py` | Tasks A and B on the exact 3,588 |
| `clean3588.py`, `analyse.py`, `diag3588.py` | punctuation-free subsets, the fourth condition, failure attribution |
| `qiyas_results.json` | all corpus numbers, per-ʿilla, folds, hook counters, changed-type logs |
| `deriv3588_results.json` | Tasks A and B, examples fixed / broken, controls |
| `deriv_gold.json` | the 3,588 verified gold positions (word, root, wazn, affixes, index) |
| `corpus.log`, `gold.log`, `d3588.log`, `analyse.log` | full run logs |

Deterministic and CPU-only; no GPU job was launched and no other agent's process was signalled.

### 10.1 The seven frozen suites, re-run after all of this work

No module was edited, so the suites were not *required* — they were run to demonstrate that the
shipped repository is still green, through the same in-place runner the `lisan_root_fix` work used
(`/workspace/lisan_root_fix/run_suites.sh QIYAS_POST`, `ROOTFORMER_VALIDATED_SEG=1`, CPU, `nice -n 10`):

| suite | result |
|---|---|
| `test_grammar_impl.py` | **20 / 20**, 0 FAIL |
| `verify_v2.py` | **ALL v2 CHECKS PASS** |
| `andalusian_realizer.py` | **67 / 67**, 0 FAIL |
| `test_sibawayh_governor.py` | **15 / 15** |
| `ibn_malik_automaton.py` | **46 / 46** |
| `test_khalil_orbits.py` | **30 / 30**, 0 FAIL |
| `test_awzan_order.py` | **24 / 24**, 0 FAIL |

Module identity, unchanged throughout: `nrmp_vocab.py` md5 `0ee0ab6301d7f26eddc7bd079f7fede7`,
blueprint md5 `5bd3e828b41fb1a6a6039d47c2f359fc`, `models/morphemic_tokenizer_v12_arabic.py` md5
`a1185eafa89b20a98ade3ae58aa12ffb`, `andalusian_realizer.py` md5 `80eec93a2217e482548201961ca295fc`;
all mtimes are 2026-10-01, i.e. before this session began. Full log: `pod_out/seven_suites.log`.
