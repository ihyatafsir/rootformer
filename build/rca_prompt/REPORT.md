# Qualitative prompt read — `head_RCA_UNFREEZE_A2` vs its own RCA gate-ablation

**Artifacts:** `/workspace/root_attn/head_RCA_UNFREEZE_A2.pt` + `head_RCA_UNFREEZE_A2.pt.trunk.pt`
on the shipped `rootformer_v19_2_synthesis_ar_backbone.awzan142.roots9490.tok10052.safetensors`,
over the aligned cache `/workspace/head_fix/nrmp_cache_9490_aligned`.
**Harness:** `/workspace/rca_prompt/harness.py` (mirrored to
`rootformer/build/rca_prompt/harness.py`), plus `roundtrip_refine.py`.
**All outputs:** `/workspace/rca_prompt/fp32/` (mirrored locally to
`rootformer/build/rca_prompt/fp32/`).
**Inference only.** No module, checkpoint or cache was written; the two resident training arms
were left untouched.

---

## 0. The harness reproduces the published model — after one repair

`repro_eval.py` (the existing independent path) reports **11.14 % / 1.44 %** for this arm, not
19.53 % / 2.83 %. The cause is a dropped trained parameter, not a modelling difference:
`head_RCA_UNFREEZE_A2.pt` contains **`feat_gate = 0.21979446709156036`**, a scalar that
multiplies the head's linear root-history/operator/morph conditioning branch
(`NRMTHead._linear_features`, `out = out * self.feat_gate`). Both `repro_eval.py` and the first
version of this harness build `RootformerNRMT(...)` without `feat_gate=`, so the scalar is
silently discarded and the conditioning branch is **4.5× too strong** (branch RMS
`37.196/√896 = 1.243` instead of `0.273`). Rebuilding the head with `feat_gate=True` restores
the published numbers exactly:

| condition | acc@1 | acc@5 | CE_z | n | source |
|---|---|---|---|---|---|
| **RCA ON — reproduced here** | **19.582 %** | **36.632 %** | **8.2190** | 18869 | this harness |
| RCA ON — trainer log (published) | 19.529 % | 36.414 % | 8.1519 | 18869 | `results_RCA_UNFREEZE_A2.json` |
| RCA ON — `repro_eval.py` (head mis-built) | 11.140 % | 19.370 % | 7.8727 | 18869 | `repro_A2_FINAL.json` |
| **RCA OFF — reproduced here** | **2.830 %** | **5.268 %** | **9.6430** | 18869 | this harness |
| RCA OFF — trainer log (published) | 2.825 % | 5.273 % | 9.6430 | 18869 | trainer `ALL_val_RCA_OFF` |
| RCA OFF — `repro_eval.py` (head mis-built) | 1.436 % | 3.789 % | 10.0047 | 18869 | `repro_A2_GATEOFF.json` |
| majority-class marginal (this pass) | 3.567 % | — | — | 18869 | most frequent *true* root |

`NOVEL_only` (n = 15046): ON 19.932 % / 37.558 %, OFF 2.672 % / 4.991 %.
The residual 0.05 pp on acc@1 is explained by the artifact: the trunk `.pt` was saved via
`p.detach().to(torch.bfloat16)`, so the RCA module's float32 training weights are bf16-rounded.

**Load-path facts (logged in `run.log`):** gates loaded
`[-1.8359, 1.8594, 1.8906, -1.9297]` (trainer final: `[-1.8327, 1.8558, 1.8905, -1.9293]`),
layers `[20, 21, 22, 23]`, `rca_out_norm=True`, `observe=r_{<=t}`, head 19/19 tensors,
0 missing / 0 unexpected. The 300 val windows and the ʿāmil (operator) stream are built with the
trainer's own code (`starts_for(., 300, seed=1)`, `SibawayhGovernor`); the novely mask reproduces
`all=18869 novel=15046` exactly.

**The held-out text is provably the text the model was scored on.** The val stream was rebuilt
from the corpus with the project's own split (`nrmp_train.CORPORA`, seed 1337, 47 files → 3 val
files: `02_Suhayli_Nataij_al_Fikr.txt`, `Ghazali_Ihya_Ulum_al_Din.txt`,
`Mustafa_Sabri_Mawqif_al_Aql_vol2.txt`) and compared element-wise with the shipped cache:
all four streams **byte-identical** (`rebuild_check.json`), 837 575 word events, whose surface
strings are therefore aligned to the scored positions.

**Ablation path:** `RootCrossAttentionStack.zero_gates()` / `restore_gates()` — the same code the
trainer's `--rca-ablate-eval` uses. Nothing was reimplemented.

---

## 1. Sampling (stated, uncurated)

* 30 of the trainer's own **300 held-out val windows**, drawn uniformly at random without
  replacement, `numpy.random.default_rng(20261002).choice(300, 30, replace=False)`
  → window indices `[5, 15, 28, 30, 37, 44, 46, 58, 59, 74, 77, 92, 118, 156, 158, 165, 168, 190,
  193, 204, 211, 215, 257, 260, 263, 268, 269, 274, 285, 298]`.
* Each prompt feeds the model the **full 128-word window** (exactly the tensors the aggregate was
  computed on) and reads the prediction at **head position 64**, i.e. for window word 65. Context
  shown = the 24 words before the target.
* **No curation.** 11 of the 30 targets are particle/control roots (`<P:..>`, `<PARTICLE>`) which
  the project's own metric excludes; they are kept and flagged. 28/30 contexts are novel 12-grams.
* Corpora caveat, stated because it matters for judging "truth": the sampled files are
  OCR-damaged heritage scans (`دلبل على وحود الأنبياء`, `~~` page markers, `"` / `)` glued to
  tokens). Some "true next words" are themselves garbled (`دايل`, `ست`, `~~إصلاح`). This is the
  model's actual training and evaluation diet.
* Decoding: `nrmp_vocab.MorphemicVocab.decode_word(prefix, root, wazn, suffix)` — the project's
  realiser — applied identically to the model's tuple and to the gold tuple. Morph heads are
  read free-running (`cond_roots=None` → the model's own argmax root conditions wazn/prefix/
  suffix), exactly as the trainer's `evaluate()` does.

---

## 2. Q1 — does RCA-on produce context-sensitive predictions?

**Yes, unambiguously — and so does the ablation, so distinctness alone is not the discriminator.**

| | RCA ON | RCA OFF |
|---|---|---|
| distinct top-1 roots over 18 869 radical positions | **1 501** | 3 049 |
| share of the single most frequent predicted root (`اله`) | **3.77 %** | 1.73 % |
| distinct full emitted tuples | **5 707** | 7 749 |
| share of the single most frequent full tuple | **2.54 %** (`أله`) | 1.38 % (`قال`) |
| adjacent-argmax change rate, in-window | **86.14 %** | 99.55 % |
| adjacent-argmax change rate, radical pairs only | **90.81 %** | 99.62 % |
| distinct top-1 surfaces over the 30 prompts | **29 / 30** | 30 / 30 |
| prompts where both conditions agree on the root | 4 / 30 | — |

Reference point for the collapse question: the earlier from-scratch model emitted **one tuple
80.14 % of the time**. Here the largest single tuple is **2.54 %** — a 31× lower concentration,
and it renders as a real word (`أله`), not `''`. On the 30-prompt read the most common RCA-on
surface occurs **twice**.

Caveat, stated plainly: RCA-OFF is *more* distinct than RCA-ON (3 049 roots at 99.6 % adjacent
change) while being 7× less accurate. Distinctness rules out a constant emitter; it does not by
itself indicate information. The accuracy of ON (19.58 %) is what separates them, and it is what
the gate ablation destroys (2.83 %).

---

## 3. Q2 — are the predicted roots plausible, or high-frequency filler?

**Neither filler nor echo; they are context-conditioned mid-frequency roots whose accuracy decays
with the predicted root's rarity.**

Not filler — the model does not lean on the head of the frequency distribution:
only **5.18 %** of its predictions are in the 10 most frequent roots, 31.2 % in the top 100;
median predicted root rank **200 / 9490**, mean **371**. (The most frequent *true* root has
median rank 259, so the truth is mid-frequency too.)

It is not echoing the visible root either:

| reference predictor (condition-independent, same positions) | acc@1 |
|---|---|
| copy the current input root `r_t` (the model sees it) | **0.58 %** |
| copy `r_{t-1}` | 1.37 % |
| copy `r_{t-2}` | 1.09 % |
| always the most frequent root (marginal) | 3.567 % |
| **RCA ON** | **19.58 %** |
| RCA OFF | 2.83 % |

RCA-ON predicts `r_t` itself only **0.95 %** of the time. The pathway is 34× the copy-current
baseline, so it is doing genuine next-root prediction, not pass-through.

**Frequency rank vs correctness (RCA ON, radical positions):**

| predicted-root train-frequency rank | share of predictions | acc@1 |
|---|---|---|
| 1–10 | 5.18 % | **40.39 %** |
| 11–100 | 25.98 % | **19.50 %** |
| 101–1000 | 62.09 % | **18.55 %** |
| 1001–10000 | 6.75 % | **13.43 %** |

Mean rank of a **correct** prediction 294.7 (median 175) vs **wrong** 389.1 (median 208) — lower
rank is associated with correctness, but only weakly; the model is not simply betting on
frequency (a frequency-only bettor would earn 3.567 %, and would be 100 % in the top bucket).

**Radical relationship to the truth:** the prediction shares all three radicals with the true
next root (root identity, up to repeated-radical roots) **16.8 %** of the time, shares exactly
two **3.7 %**, shares ≤1 radical **76.6 %**. So roughly one prediction in five is
morphologically related to the truth; the other four in five are unrelated roots — not rare
noise, just wrong.

**RCA OFF frequency profile, by contrast:** median predicted rank **840**, 47.3 % of predictions
in rank 1001–10000, accuracy 20.66 % / 5.69 % / 2.09 % / 1.04 % across the four buckets. When it
does land (median rank of its rare correct predictions: 24) it is because it happened onto a
frequent root. Its radical overlap with the truth: 2.6 % same, 93.1 % share ≤1 radical.

---

## 4. Q3 — what does RCA-OFF actually emit?

**Not a constant, and not `<PARTICLE>`. The qualitative form of "below the marginal" is a
diffuse, rare-root emitter.**

Top emitted roots over 18 869 radical positions (`RCA_OFF_emission_profile`):

| root | count | share | modal tuple for this root renders as |
|---|---|---|---|
| `اله` | 326 | 1.73 % | `اله` |
| `قال` | 313 | 1.66 % | `قال` |
| `سلم` | 200 | 1.06 % | `وسلم` |
| `كون` | 184 | 0.98 % | `كون` |
| `عبد` | 158 | 0.84 % | `عبد` |
| `علم` | 152 | 0.81 % | `علم` |
| `وقل` | 141 | 0.75 % | `وقل` |
| `حدد` | 122 | 0.65 % | `حدد` |
| `بنن` | 119 | 0.63 % | `بنن` |
| `حمد` | 114 | 0.60 % | `حمد` |
| `وجب` | 113 | 0.60 % | `وجب` |
| `وما` | 112 | 0.59 % | `وما` |

Control-token counts in the ablation's 18 869 emissions: `<PARTICLE>` **0**, `<UNK>` **0**,
`<PAD>` **0**, `<BOS>` **0**, `<EOS>` **0**; empty surface **0**; modal tuple `(1, 6397, 1, 1)`
= `قال` at **1.38 %**. 3 049 distinct roots, 7 749 distinct tuples.

So the ablation does *not* collapse; it emits a near-uniform dribble over the rare tail
(47 % of predictions in frequency ranks 1001–10000, 99.55 % adjacent change) and lands below the
marginal because the tail is where the truth rarely is. "Below the marginal" here means
**worse than a constant prior by being diffusely wrong**, not "degenerate to one token".
(The earlier from-scratch collapse — one tuple 80 % of the time, rendering as the empty string —
is a *different* failure mode, and this arm does not have it in either condition.)

---

## 5. Q4 — well-formedness, control-token leakage, and the tokenizer defect

**The model's own output never leaks a control token and is always renderable:**

| | RCA ON | RCA OFF |
|---|---|---|
| decoded emissions classified `well_formed` | **18 869 / 18 869** | **18 869 / 18 869** |
| literal `<...>` in the decoded surface | **0.00 %** | **0.00 %** |
| empty decoded surface | **0.00 %** | **0.00 %** |
| control root (`<PAD>/<BOS>/<EOS>/<UNK>/<PARTICLE>`) | **0.00 %** | **0.00 %** |

**But the defect is on the truth/input side, and it is large.** Over the whole held-out val
stream (837 574 word events, `roundtrip_refine.json`):

| | count | share |
|---|---|---|
| target root is control/particle (excluded from the metric) | 423 276 | **50.54 %** |
| — `<PARTICLE>` (unrepresentable: realiser yields `''`) | 170 808 | **20.40 %** |
| — `<P:..>` particle (realisable, e.g. `في`, `هذه`) | 245 515 | 29.32 % |
| — `<UNK>` | 6 953 | 0.83 % |
| target root is radical (metric domain) | 414 298 | 49.46 % |
| of those, gold tuple fails the exact round trip | 49 902 | **12.04 %** |
| — failure is punctuation-only (e.g. `كلامنا،` → `كلامنا`) | 5 108 | 1.23 % |
| — **genuine form failure** (e.g. `الله` → `أله`, `نستديم` → `استديم`, `عوامنا` → root `عمن`) | 44 794 | **10.81 %** |

So: half the metric's own data is control/particle material, a fifth of all held-out targets are
forms the tuple cannot represent at all, and ~11 % of the radical targets do not re-realise the
corpus word. **None of that leaks into what this arm emits** — the old failure (a literal control
tag appearing in decoded text) is absent from all 37 738 emissions measured here. The 4000-position
spot check inside the harness gave 17.9 % round-trip failure; the full-stream figure above
(12.04 %) is the authoritative one — the spot check happened to land on a punctuation-heavy
stretch of the stream.

---

## 6. Is RCA-ON the *same collapse with a better score*? — No.

| signature | from-scratch model (earlier finding) | A2 RCA ON | A2 RCA OFF |
|---|---|---|---|
| modal emitted tuple share | **80.14 %** | 2.54 % | 1.38 % |
| modal tuple decoded | **`''` (empty string)** | `أله` | `قال` |
| distinct emitted tuples | 1 (80 % of positions) | 5 707 | 7 749 |
| held-out root acc@1 | **0.00 %** | 19.58 % (19.93 % novel) | 2.83 % |
| copy-current-root echo | — | 0.95 % | 1.19 % |
| control-token leak | tag emitted into text | **0.00 %** | **0.00 %** |
| adjacent-argmax change | ~0 % | 86.14 % | 99.55 % |

The A2 model is **qualitatively different** from the collapsed from-scratch model. It is not a
constant emitter, it does not emit the empty string, it does not leak control tokens, its
predictions change with context and with position, it does not echo the visible root, and its
accuracy is 34× the copy-current baseline and 5.5× the majority-class marginal. The gate ablation
confirms the pathway is what carries this: zeroing four scalars drops acc@1 19.58 → 2.83 %, and
what remains is diffuse rare-root guessing, not a constant.

---

## 7. Honest limits and the unflattering part

* **19.58 % acc@1 means four of five next roots are wrong**, and being right about the root is not
  being right about the word: of the two root-correct prompts (P28, P29), P28 predicted
  root `صوت` for truth `صوتكم` but emitted the surface `صوت` — wrong suffix. The model is a weak
  root LM, not a usable next-word predictor.
* The 30-prompt read is **under-powered and slightly unlucky, and I am not using it as the
  accuracy claim**: 2 root-correct of the 19 radical prompts vs an expectation of 3.05 at the
  position-64 rate (16.03 % on the full 300 windows); the 0/30 for RCA-OFF is fully consistent
  with 2.83 % (P = 0.42). All strong claims above rest on the full 18 869-position pass, not on 30
  prompts. What the 30 prompts uniquely show is the *shape* of the output: readable, varied,
  contextually plausible Arabic root forms, both conditions, no collapse and no leaks.
* One fixed read position (window word 65) was used for all 30 prompts; accuracy varies by
  position (16.03 % at position 64 vs 19.58 % overall), so per-prompt hit rates are not the
  aggregate.
* The corpora are OCR-damaged; some gold labels are corrupt, which depresses *both* conditions
  and cannot be separated here.
* Three held-out files (one of them the OCR-heavy `Mustafa_Sabri` volume) carry the whole
  qualitative read; the model's own file-level val split, not a chosen set, but a narrow one.
* The eval is teacher-forced on the **input** side (gold prefix/root/wazn/suffix streams are fed);
  this is the same protocol as the published 19.53 % and is not a free-running generation test.

---

## Appendix A — the 30 raw prompt blocks (verbatim from `fp32/prompts.md`)


- sampling: 30 of the trainer's 300 held-out val windows, drawn uniformly at random without replacement, seed **20261002** (`numpy default_rng(20261002).choice(300, 30, replace=False)`).

- every prompt feeds the model the full 128-word held-out window (identical windows to the ones the 19.53 % aggregate was computed on); the prediction is read at window position 64, i.e. for window word 65.

- the 24 words before the target are shown as the "context"; `…` marks the fact that the model saw further words to the left.

- surfaces are produced by `nrmp_vocab.MorphemicVocab.decode_word` (the project's own realiser), applied identically to the model's tuple and to the gold tuple.

- 19/30 prompts have a target that the project's own metric counts as a radical (non-special) position; the other 11 have a particle/control target root, kept in because the sample is uncurated.


## P01 — 02_Suhayli_Nataij_al_Fikr.txt (sentence 1582), window 19264, target stream idx 19329
*target: **not** counted by the metric (particle/control root); 12-gram context novel: True*

**Context (last 24 of the 128 window words; … = more words to the left):**

… الأحوال، لا يخص مضيا من استقبال # ومثله: (وما كان ربك مهلك القرى) # ثم قال عز وجل: (وما كنا مهلكي القرى) ، فالحظ

**True next word:** `هذه`  (root `<P:هذه>` = `35`, freq rank 84 of 9490, seen 11957x in train)  — gold tuple `(1, 35, 1, 1)` realises as `هذه`

**RCA ON  (gates loaded) — top-1:** root `سبح` (id 3660, freq rank 505) → tuple `(1, 3660, 1, 1)` → `سبح`   ❌ wrong root

  top-5: 1. `سبح` (root `سبح`) | 2. `علق` (root `علق`) | 3. `ويل` (root `ويل`) | 4. `صيغ` (root `صيغ`) | 5. `ورد` (root `ورد`)

**RCA OFF (gates forced to 0) — top-1:** root `لسن` (id 7440, freq rank 516) → tuple `(1, 7440, 1, 1)` → `لسن`   ❌ wrong root

  top-5: 1. `لسن` (root `لسن`) | 2. `تيس` (root `تيس`) | 3. `اله` (root `اله`) | 4. `ورد` (root `ورد`) | 5. `سوا` (root `سوا`)

---

## P02 — 02_Suhayli_Nataij_al_Fikr.txt (sentence 4154), window 49216, target stream idx 49281
*target: radical, counted by the metric; 12-gram context novel: True*

**Context (last 24 of the 128 window words; … = more words to the left):**

… هذه المسألة أن يسأل عن المعنى الذي من أجله قال: # (ولتصنع على عيني) بحرف " على " # وقال في موضع آخر: (تجري

**True next word:** `بأعيننا)`  (root `عين` = `5819`, freq rank 66 of 9490, seen 14554x in train)  — gold tuple `(8, 5819, 8, 11)` realises as `بأعيننا`

**RCA ON  (gates loaded) — top-1:** root `دمع` (id 2756, freq rank 1300) → tuple `(1, 2756, 1, 1)` → `دمع`   ❌ wrong root

  top-5: 1. `دمع` (root `دمع`) | 2. `أمر` (root `أمر`) | 3. `لفظ` (root `لفظ`) | 4. `نيب` (root `نيب`) | 5. `سرر` (root `سرر`)

**RCA OFF (gates forced to 0) — top-1:** root `دمع` (id 2756, freq rank 1300) → tuple `(1, 2756, 1, 1)` → `دمع`   ❌ wrong root

  top-5: 1. `دمع` (root `دمع`) | 2. `نيب` (root `نيب`) | 3. `ندي` (root `ندي`) | 4. `شبه` (root `شبه`) | 5. `عقر` (root `عقر`)

---

## P03 — Mustafa_Sabri_Mawqif_al_Aql_vol2.txt (sentence 623), window 85120, target stream idx 85185
*target: radical, counted by the metric; 12-gram context novel: True*

**Context (last 24 of the 128 window words; … = more words to the left):**

… دلبل على وحود الأنبياء غير ماجاء وإن قال إن نبوءتهم ثبتت بالممجزات الى ظورت على أيدمهم» قلنا لا معجزة عندك والمقل لا يقءلها ولا

**True next word:** `دايل`  (root `ديل` = `2869`, freq rank 3373 of 9490, seen 71x in train)  — gold tuple `(1, 2869, 42, 1)` realises as `دايل`

**RCA ON  (gates loaded) — top-1:** root `قال` (id 6397, freq rank 5) → tuple `(1, 6397, 1, 1)` → `قال`   ❌ wrong root

  top-5: 1. `قال` (root `قال`) | 2. `دلل` (root `دلل`) | 3. `كون` (root `كون`) | 4. `وقل` (root `وقل`) | 5. `ابن` (root `ابن`)

**RCA OFF (gates forced to 0) — top-1:** root `كون` (id 7309, freq rank 16) → tuple `(1, 7309, 1, 1)` → `كون`   ❌ wrong root

  top-5: 1. `كون` (root `كون`) | 2. `سيم` (root `سيم`) | 3. `مرد` (root `مرد`) | 4. `حب` (root `حبب`) | 5. `خلف` (root `خلف`)

---

## P04 — Mustafa_Sabri_Mawqif_al_Aql_vol2.txt (sentence 1874), window 99328, target stream idx 99393
*target: radical, counted by the metric; 12-gram context novel: True*

**Context (last 24 of the 128 window words; … = more words to the left):**

… الثالطة من مقالة كاتب نحو النو راو قد أضيفة التشيلة ف تلك المقالات من عحائب الأحوال أن تلك الغلطة الفظيمة الشائعة بمصر الإسلامية» لم

**True next word:** `ست`  (root `ستت` = `3688`, freq rank 697 of 9490, seen 1161x in train)  — gold tuple `(1, 3688, 45, 1)` realises as `ست`

**RCA ON  (gates loaded) — top-1:** root `نشر` (id 8136, freq rank 901) → tuple `(1, 8136, 1, 1)` → `نشر`   ❌ wrong root

  top-5: 1. `نشر` (root `نشر`) | 2. `يبعث` (root `بعث`) | 3. `يلتدن` (root `لدن`) | 4. `صرع` (root `صرع`) | 5. `شعر` (root `شعر`)

**RCA OFF (gates forced to 0) — top-1:** root `شذذ` (id 4184, freq rank 896) → tuple `(1, 4184, 120, 1)` → `شذذ`   ❌ wrong root

  top-5: 1. `شذذ` (root `شذذ`) | 2. `تسبن` (root `سبن`) | 3. `يشكل` (root `شكل`) | 4. `وما` (root `وما`) | 5. `وسف` (root `وسف`)

---

## P05 — Mustafa_Sabri_Mawqif_al_Aql_vol2.txt (sentence 4124), window 124800, target stream idx 124865
*target: radical, counted by the metric; 12-gram context novel: True*

**Context (last 24 of the 128 window words; … = more words to the left):**

… تعالى اللامتناهية كأاسبق ذكره الجزء الأول وأسيفو افى الكلام إسهاب يمل قارى هذا الزمان للكنى أذ كر له طريقا موجزة منهلة التناول فأقول: لو

**True next word:** `وُجد`  (root `وجد` = `8807`, freq rank 46 of 9490, seen 18268x in train)  — gold tuple `(1, 8807, 45, 1)` realises as `وجد`

**RCA ON  (gates loaded) — top-1:** root `قدر` (id 6471, freq rank 65) → tuple `(1, 6471, 1, 1)` → `قدر`   ❌ wrong root

  top-5: 1. `قدر` (root `قدر`) | 2. `نقص` (root `نقص`) | 3. `اه` (root `اهه`) | 4. `خرج` (root `خرج`) | 5. `رجل` (root `رجل`)

**RCA OFF (gates forced to 0) — top-1:** root `قال` (id 6397, freq rank 5) → tuple `(1, 6397, 45, 1)` → `قال`   ❌ wrong root

  top-5: 1. `قال` (root `قال`) | 2. `اني` (root `اني`) | 3. `دعت` (root `دعت`) | 4. `كده` (root `كده`) | 5. `رجل` (root `رجل`)

---

## P06 — Mustafa_Sabri_Mawqif_al_Aql_vol2.txt (sentence 5247), window 137408, target stream idx 137473
*target: radical, counted by the metric; 12-gram context novel: True*

**Context (last 24 of the 128 window words; … = more words to the left):**

… فى عل الله والذى هو امام المجدوس الهارحى بالئسية إلينا ولما كانت الصفة أقرب إلى الموصوف من الأثر إلى المؤائر اعتير استيقان و ود

**True next word:** `الله`  (root `اله` = `283`, freq rank 7 of 9490, seen 63768x in train)  — gold tuple `(1, 283, 45, 1)` realises as `أله`

**RCA ON  (gates loaded) — top-1:** root `شبث` (id 4094, freq rank 3347) → tuple `(1, 4094, 1, 1)` → `شبث`   ❌ wrong root

  top-5: 1. `شبث` (root `شبث`) | 2. `النسس` (root `نسس`) | 3. `كتب` (root `كتب`) | 4. `مرخ` (root `مرخ`) | 5. `اذا` (root `اذا`)

**RCA OFF (gates forced to 0) — top-1:** root `شبث` (id 4094, freq rank 3347) → tuple `(1, 4094, 1, 1)` → `شبث`   ❌ wrong root

  top-5: 1. `شبث` (root `شبث`) | 2. `ورب` (root `ورب`) | 3. `مرخ` (root `مرخ`) | 4. `علم` (root `علم`) | 5. `اذا` (root `اذا`)

---

## P07 — Mustafa_Sabri_Mawqif_al_Aql_vol2.txt (sentence 6004), window 146176, target stream idx 146241
*target: **not** counted by the metric (particle/control root); 12-gram context novel: True*

**Context (last 24 of the 128 window words; … = more words to the left):**

… تعريض بأ اماديين للروح فما نه من غير الماديات؛ وهما حدر باللغت أنه اختار الدليل المقلى فى بات وجو د اروح وهو شمو ركل

**True next word:** `ذى`  (root `<PARTICLE>` = `4`, freq rank 1 of 9490, seen 1571716x in train)  — gold tuple `(1, 4, 1, 1)` realises as ``

**RCA ON  (gates loaded) — top-1:** root `صمعد` (id 4671, freq rank 7036) → tuple `(1, 4671, 1, 1)` → `صمعد`   ❌ wrong root

  top-5: 1. `صمعد` (root `صمعد`) | 2. `بلغ` (root `بلغ`) | 3. `كلم` (root `كلم`) | 4. `اب` (root `ابب`) | 5. `قطا` (root `قطا`)

**RCA OFF (gates forced to 0) — top-1:** root `صمعد` (id 4671, freq rank 7036) → tuple `(1, 4671, 1, 1)` → `صمعد`   ❌ wrong root

  top-5: 1. `صمعد` (root `صمعد`) | 2. `اله` (root `اله`) | 3. `الحلل` (root `حلل`) | 4. `اللزم` (root `لزم`) | 5. `صبح` (root `صبح`)

---

## P08 — Mustafa_Sabri_Mawqif_al_Aql_vol2.txt (sentence 8951), window 179584, target stream idx 179649
*target: **not** counted by the metric (particle/control root); 12-gram context novel: True*

**Context (last 24 of the 128 window words; … = more words to the left):**

… الفاسفية فماب على الفلسفة ااثبتة عدم مراعتها لمعلومة اللامتناهى التى هى أثم المماومات المثبتة أى عاب على هذه الفلسفة عدم تقديرها له ذه الملومة

**True next word:** `وعى`  (root `<PARTICLE>` = `4`, freq rank 1 of 9490, seen 1571716x in train)  — gold tuple `(1, 4, 1, 1)` realises as ``

**RCA ON  (gates loaded) — top-1:** root `ندد` (id 9409, freq rank 1405) → tuple `(1, 9409, 1, 1)` → `ندد`   ❌ wrong root

  top-5: 1. `ندد` (root `ندد`) | 2. `ارف` (root `ارف`) | 3. `قلم` (root `قلم`) | 4. `حمز` (root `حمز`) | 5. `فرد` (root `فرد`)

**RCA OFF (gates forced to 0) — top-1:** root `ارف` (id 168, freq rank 4538) → tuple `(5, 168, 74, 20)` → `والاروفة`   ❌ wrong root

  top-5: 1. `والاروفة` (root `ارف`) | 2. `حمز` (root `حمز`) | 3. `ندد` (root `ندد`) | 4. `وغر` (root `وغر`) | 5. `شكل` (root `شكل`)

---

## P09 — Mustafa_Sabri_Mawqif_al_Aql_vol2.txt (sentence 9295), window 183616, target stream idx 183681
*target: radical, counted by the metric; 12-gram context novel: True*

**Context (last 24 of the 128 window words; … = more words to the left):**

… ما يدهش ولا بحصى من عحائب عام الحياة يرونها منهية إلى سران عظمين أحدها عقل الإننان بكل ما فيه من قوة واستعداد المجيرة بكل

**True next word:** `قواها`  (root `قها` = `6881`, freq rank 3023 of 9490, seen 90x in train)  — gold tuple `(1, 6881, 133, 1)` realises as `قواها`

**RCA ON  (gates loaded) — top-1:** root `شكل` (id 4344, freq rank 461) → tuple `(1, 4344, 45, 1)` → `شكل`   ❌ wrong root

  top-5: 1. `شكل` (root `شكل`) | 2. `دكن` (root `دكن`) | 3. `ملائث` (root `ملث`) | 4. `كمت` (root `كمت`) | 5. `متعبد` (root `عبد`)

**RCA OFF (gates forced to 0) — top-1:** root `دكن` (id 2701, freq rank 2955) → tuple `(1, 2701, 1, 1)` → `دكن`   ❌ wrong root

  top-5: 1. `دكن` (root `دكن`) | 2. `ملث` (root `ملث`) | 3. `كمت` (root `كمت`) | 4. `حيل` (root `حيل`) | 5. `مدرات` (root `مدر`)

---

## P10 — Ghazali_Ihya_Ulum_al_Din.txt (sentence 1934), window 215360, target stream idx 215425
*target: radical, counted by the metric; 12-gram context novel: True*

**Context (last 24 of the 128 window words; … = more words to the left):**

… به طلب الأسانيد العالية ~~أو طلب الحديث الذي لا يحتاج إليه في طلب الآخرة وقال عيسى عليه السلام كيف ~~يكون من أهل العلم من

**True next word:** `مسيره`  (root `مسر` = `7762`, freq rank 1204 of 9490, seen 490x in train)  — gold tuple `(1, 7762, 55, 3)` realises as `مسيره`

**RCA ON  (gates loaded) — top-1:** root `دنا` (id 2772, freq rank 233) → tuple `(1, 2772, 45, 1)` → `دنا`   ❌ wrong root

  top-5: 1. `دنا` (root `دنا`) | 2. `كلمه` (root `كلم`) | 3. `طرق` (root `طرق`) | 4. `الفجر` (root `فجر`) | 5. `درك` (root `درك`)

**RCA OFF (gates forced to 0) — top-1:** root `وحد` (id 8819, freq rank 37) → tuple `(3, 8819, 1, 1)` → `الوحد`   ❌ wrong root

  top-5: 1. `الوحد` (root `وحد`) | 2. `الغد` (root `غدد`) | 3. `صدق` (root `صدق`) | 4. `ترك` (root `ترك`) | 5. `ضرر` (root `ضرر`)

---

## P11 — Ghazali_Ihya_Ulum_al_Din.txt (sentence 3215), window 231232, target stream idx 231297
*target: **not** counted by the metric (particle/control root); 12-gram context novel: True*

**Context (last 24 of the 128 window words; … = more words to the left):**

… يحتمل ~~والهداية التي تنفع الناس تمكث ~~وغيرهما وهو بدعة إذ لم ينقل ذلك بطريق الرواية وإجراؤه على الظاهر غير ~~محال فيجب إجراؤه على الظاهر

**True next word:** `~~والذوق`  (root `<PARTICLE>` = `4`, freq rank 1 of 9490, seen 1571716x in train)  — gold tuple `(1, 4, 1, 1)` realises as ``

**RCA ON  (gates loaded) — top-1:** root `ثني` (id 1140, freq rank 243) → tuple `(1, 1140, 1, 1)` → `ثني`   ❌ wrong root

  top-5: 1. `ثني` (root `ثني`) | 2. `كمل` (root `كمل`) | 3. `زيل` (root `زيل`) | 4. `كني` (root `كني`) | 5. `ملق` (root `ملق`)

**RCA OFF (gates forced to 0) — top-1:** root `زيل` (id 3642, freq rank 1751) → tuple `(1, 3642, 1, 1)` → `زيل`   ❌ wrong root

  top-5: 1. `زيل` (root `زيل`) | 2. `ونفه` (root `نفه`) | 3. `بتت` (root `بتت`) | 4. `استغن` (root `سغن`) | 5. `كني` (root `كني`)

---

## P12 — Ghazali_Ihya_Ulum_al_Din.txt (sentence 6795), window 275648, target stream idx 275713
*target: radical, counted by the metric; 12-gram context novel: True*

**Context (last 24 of the 128 window words; … = more words to the left):**

… يا ملائكتي إلى عبدي ترك شهوته ولذته وطعامه وشرابه من أجلي (10) ~~وقيل في قوله تعالى {فلا تعلم نفس ما أخفي لهم من قرة

**True next word:** `أعين`  (root `عين` = `5819`, freq rank 66 of 9490, seen 14554x in train)  — gold tuple `(1, 5819, 8, 1)` realises as `أعين`

**RCA ON  (gates loaded) — top-1:** root `نفس` (id 8232, freq rank 59) → tuple `(1, 8232, 45, 1)` → `نفس`   ❌ wrong root

  top-5: 1. `نفس` (root `نفس`) | 2. `عين` (root `عين`) | 3. `بون` (root `بون`) | 4. `أعمال` (root `عمل`) | 5. `ارق` (root `ارق`)

**RCA OFF (gates forced to 0) — top-1:** root `بون` (id 802, freq rank 3192) → tuple `(1, 802, 45, 1)` → `بون`   ❌ wrong root

  top-5: 1. `بون` (root `بون`) | 2. `ارق` (root `ارق`) | 3. `شام` (root `شام`) | 4. `وخفق` (root `خفق`) | 5. `فرعه` (root `فرع`)

---

## P13 — Ghazali_Ihya_Ulum_al_Din.txt (sentence 12609), window 348480, target stream idx 348545
*target: **not** counted by the metric (particle/control root); 12-gram context novel: True*

**Context (last 24 of the 128 window words; … = more words to the left):**

… السوق لآخرته فيلازم المسجد ويواظب ~~إلا الصبيان وأهل الذمة لأنهم كانوا في المساجد بعد ~~آخره ذكر الله وخير كفر الله عنهما ما بينهما من

**True next word:** `سيئ`  (root `<PARTICLE>` = `4`, freq rank 1 of 9490, seen 1571716x in train)  — gold tuple `(1, 4, 1, 1)` realises as ``

**RCA ON  (gates loaded) — top-1:** root `قول` (id 6917, freq rank 23) → tuple `(1, 6917, 1, 1)` → `قول`   ❌ wrong root

  top-5: 1. `قول` (root `قول`) | 2. `فوت` (root `فوت`) | 3. `ترك` (root `ترك`) | 4. `جمع` (root `جمع`) | 5. `وقت` (root `وقت`)

**RCA OFF (gates forced to 0) — top-1:** root `وذل` (id 8863, freq rank 140) → tuple `(1, 8863, 45, 1)` → `وذل`   ❌ wrong root

  top-5: 1. `وذل` (root `وذل`) | 2. `علل` (root `علل`) | 3. `اتا` (root `اتا`) | 4. `بصر` (root `بصر`) | 5. `منع` (root `منع`)

---

## P14 — Ghazali_Ihya_Ulum_al_Din.txt (sentence 22609), window 472128, target stream idx 472193
*target: radical, counted by the metric; 12-gram context novel: True*

**Context (last 24 of the 128 window words; … = more words to the left):**

… الله الذي ~~خلقك فقلت كيف أذكره قال قل بقلبك عند تقلبك في ثيابك ثلاث مرات من غير أن ~~تحرك به لسانك الله معي الله

**True next word:** `ناظر`  (root `نظر` = `8183`, freq rank 113 of 9490, seen 9358x in train)  — gold tuple `(1, 8183, 42, 1)` realises as `ناظر`

**RCA ON  (gates loaded) — top-1:** root `زلل` (id 3535, freq rank 679) → tuple `(1, 3535, 1, 1)` → `زلل`   ❌ wrong root

  top-5: 1. `زلل` (root `زلل`) | 2. `فقل` (root `فقل`) | 3. `شهد` (root `شهد`) | 4. `ذلل` (root `ذلل`) | 5. `قلل` (root `قلل`)

**RCA OFF (gates forced to 0) — top-1:** root `مقت` (id 7845, freq rank 1832) → tuple `(1, 7845, 1, 1)` → `مقت`   ❌ wrong root

  top-5: 1. `مقت` (root `مقت`) | 2. `خلف` (root `خلف`) | 3. `جمع` (root `جمع`) | 4. `وفي` (root `وفي`) | 5. `حقق` (root `حقق`)

---

## P15 — Ghazali_Ihya_Ulum_al_Din.txt (sentence 22943), window 476544, target stream idx 476609
*target: radical, counted by the metric; 12-gram context novel: True*

**Context (last 24 of the 128 window words; … = more words to the left):**

… كان أعذب للتلاوة وأدوم للقيام وأقل للمنام وقال ~~أبو بكر بن عبد الله المزني ثلاثة يحبهم الله تعالى رجل قليل النوم قليل ~~الأكل قليل

**True next word:** `الراحة`  (root `رحح` = `3066`, freq rank 1338 of 9490, seen 413x in train)  — gold tuple `(3, 3066, 42, 20)` realises as `الراحة`

**RCA ON  (gates loaded) — top-1:** root `فقل` (id 6276, freq rank 75) → tuple `(1, 6276, 1, 1)` → `فقل`   ❌ wrong root

  top-5: 1. `فقل` (root `فقل`) | 2. `حال` (root `حال`) | 3. `اللبث` (root `لبث`) | 4. `جبا` (root `جبا`) | 5. `اللذع` (root `لذع`)

**RCA OFF (gates forced to 0) — top-1:** root `كثر` (id 6977, freq rank 69) → tuple `(1, 6977, 55, 1)` → `كثير`   ❌ wrong root

  top-5: 1. `كثير` (root `كثر`) | 2. `الخرب` (root `خرب`) | 3. `جبا` (root `جبا`) | 4. `الخير` (root `خير`) | 5. `القفش` (root `قفش`)

---

## P16 — Ghazali_Ihya_Ulum_al_Din.txt (sentence 24218), window 493376, target stream idx 493441
*target: **not** counted by the metric (particle/control root); 12-gram context novel: True*

**Context (last 24 of the 128 window words; … = more words to the left):**

… يقال يؤتى بالفاحش المتفحش يوم القيامة في صورة كلب أو في ~~جوف كلب وقال الأحنف بن قيس ألا أخبركم بأدوإ الداء اللسان البذي والخلق

**True next word:** `~~بالعبارات`  (root `<PARTICLE>` = `4`, freq rank 1 of 9490, seen 1571716x in train)  — gold tuple `(1, 4, 1, 1)` realises as ``

**RCA ON  (gates loaded) — top-1:** root `عدد` (id 5318, freq rank 143) → tuple `(1, 5318, 1, 1)` → `عدد`   ❌ wrong root

  top-5: 1. `عدد` (root `عدد`) | 2. `فيل` (root `فيل`) | 3. `وذم` (root `وذم`) | 4. `وتت` (root `وتت`) | 5. `غطم` (root `غطم`)

**RCA OFF (gates forced to 0) — top-1:** root `غفل` (id 5947, freq rank 716) → tuple `(1, 5947, 1, 1)` → `غفل`   ❌ wrong root

  top-5: 1. `غفل` (root `غفل`) | 2. `حمل` (root `حمل`) | 3. `وذم` (root `وذم`) | 4. `عدد` (root `عدد`) | 5. `غطم` (root `غطم`)

---

## P17 — Ghazali_Ihya_Ulum_al_Din.txt (sentence 25013), window 503872, target stream idx 503937
*target: **not** counted by the metric (particle/control root); 12-gram context novel: True*

**Context (last 24 of the 128 window words; … = more words to the left):**

… الناس لكثرة الاعتياد تساهلوا في أمر ~~الغيبة ولم يكترثوا بتناول أعراض الخلق ومهما خطر لك خاطر بسوء على مسلم ~~فينبغي أن تزيد في مراعاته

**True next word:** `وتدعو`  (root `<PARTICLE>` = `4`, freq rank 1 of 9490, seen 1571716x in train)  — gold tuple `(1, 4, 1, 1)` realises as ``

**RCA ON  (gates loaded) — top-1:** root `غيظ` (id 6037, freq rank 1598) → tuple `(1, 6037, 1, 1)` → `غيظ`   ❌ wrong root

  top-5: 1. `غيظ` (root `غيظ`) | 2. `حزن` (root `حزن`) | 3. `نتت` (root `نتت`) | 4. `نظر` (root `نظر`) | 5. `ولي` (root `ولي`)

**RCA OFF (gates forced to 0) — top-1:** root `ويل` (id 9107, freq rank 1633) → tuple `(1, 9107, 1, 1)` → `ويل`   ❌ wrong root

  top-5: 1. `ويل` (root `ويل`) | 2. `نسب` (root `نسب`) | 3. `حذف` (root `حذف`) | 4. `ولي` (root `ولي`) | 5. `ركب` (root `ركب`)

---

## P18 — Ghazali_Ihya_Ulum_al_Din.txt (sentence 29730), window 564224, target stream idx 564289
*target: radical, counted by the metric; 12-gram context novel: True*

**Context (last 24 of the 128 window words; … = more words to the left):**

… مهما مضى ~~ركن من أركانها على هذا الوجه لأنا نكتفي بالنية السابقة عند الإحرام بشرط ~~أن لا يطرأ عليها ما يغلبها ويغمرها ويحتمل أن

**True next word:** `يقال`  (root `قال` = `6397`, freq rank 5 of 9490, seen 79815x in train)  — gold tuple `(1, 6397, 118, 1)` realises as `يقال`

**RCA ON  (gates loaded) — top-1:** root `عقد` (id 5577, freq rank 271) → tuple `(1, 5577, 1, 1)` → `عقد`   ❌ wrong root

  top-5: 1. `عقد` (root `عقد`) | 2. `يفسد` (root `فسد`) | 3. `يغلب` (root `غلب`) | 4. `كلم` (root `كلم`) | 5. `وفي` (root `وفي`)

**RCA OFF (gates forced to 0) — top-1:** root `فهم` (id 6360, freq rank 232) → tuple `(1, 6360, 118, 1)` → `يفهم`   ❌ wrong root

  top-5: 1. `يفهم` (root `فهم`) | 2. `كون` (root `كون`) | 3. `وقل` (root `وقل`) | 4. `ريد` (root `ريد`) | 5. `يعقب` (root `عقب`)

---

## P19 — Ghazali_Ihya_Ulum_al_Din.txt (sentence 30356), window 572032, target stream idx 572097
*target: radical, counted by the metric; 12-gram context novel: True*

**Context (last 24 of the 128 window words; … = more words to the left):**

… أما إنا ~~على ذلك لانتهم نصيحتك فأقصر عليك من لسانك قال فدفعه الله عني ~~يتبعونه فوقف فقال هل لكم من حاجة أو تسألون عن

**True next word:** `شيء`  (root `شيء` = `4477`, freq rank 47 of 9490, seen 18217x in train)  — gold tuple `(1, 4477, 45, 1)` realises as `شيء`

**RCA ON  (gates loaded) — top-1:** root `وصل` (id 8938, freq rank 355) → tuple `(1, 8938, 1, 1)` → `وصل`   ❌ wrong root

  top-5: 1. `وصل` (root `وصل`) | 2. `دنا` (root `دنا`) | 3. `قوم` (root `قوم`) | 4. `بيت` (root `بيت`) | 5. `عبد` (root `عبد`)

**RCA OFF (gates forced to 0) — top-1:** root `حوض` (id 2015, freq rank 1276) → tuple `(1, 2015, 1, 1)` → `حوض`   ❌ wrong root

  top-5: 1. `حوض` (root `حوض`) | 2. `أصل` (root `أصل`) | 3. `ابن` (root `ابن`) | 4. `كرم` (root `كرم`) | 5. `ديارهم` (root `دير`)

---

## P20 — Ghazali_Ihya_Ulum_al_Din.txt (sentence 31668), window 588800, target stream idx 588865
*target: radical, counted by the metric; 12-gram context novel: True*

**Context (last 24 of the 128 window words; … = more words to the left):**

… بعمله أو يدل به ولا يخاف على ~~نفسه فإذن هذا هو العلاج القامع لمادة العجب من القلب ~~ينظر إلى الكفار والفساق وقد سلبوا نعمة

**True next word:** `الإيمان`  (root `يمن` = `9113`, freq rank 219 of 9490, seen 5303x in train)  — gold tuple `(3, 9113, 13, 1)` realises as `الإيمان`

**RCA ON  (gates loaded) — top-1:** root `ربت` (id 3009, freq rank 1961) → tuple `(1, 3009, 1, 1)` → `ربت`   ❌ wrong root

  top-5: 1. `ربت` (root `ربت`) | 2. `تمن` (root `تمن`) | 3. `ويمتطقان` (root `مطق`) | 4. `محجة` (root `محج`) | 5. `يتجناا` (root `جنا`)

**RCA OFF (gates forced to 0) — top-1:** root `تمن` (id 962, freq rank 2581) → tuple `(1, 962, 1, 1)` → `تمن`   ❌ wrong root

  top-5: 1. `تمن` (root `تمن`) | 2. `ربت` (root `ربت`) | 3. `يمتطقان` (root `مطق`) | 4. `محجة` (root `محج`) | 5. `نقت` (root `نقت`)

---

## P21 — Ghazali_Ihya_Ulum_al_Din.txt (sentence 33149), window 607552, target stream idx 607617
*target: radical, counted by the metric; 12-gram context novel: True*

**Context (last 24 of the 128 window words; … = more words to the left):**

… وقال الذي لمس الناب ليس كما يقول بل هو صلب لا ~~لين فيه وأملس لا خشونة فيه وليس في غلظ الأسطوانة أصلا بل هو

**True next word:** `مثل`  (root `مثل` = `7649`, freq rank 45 of 9490, seen 18352x in train)  — gold tuple `(1, 7649, 45, 1)` realises as `مثل`

**RCA ON  (gates loaded) — top-1:** root `جلد` (id 1435, freq rank 538) → tuple `(1, 1435, 45, 1)` → `جلد`   ❌ wrong root

  top-5: 1. `جلد` (root `جلد`) | 2. `معمد` (root `عمد`) | 3. `صدق` (root `صدق`) | 4. `مثل` (root `مثل`) | 5. `ظاهر` (root `ظهر`)

**RCA OFF (gates forced to 0) — top-1:** root `يبر` (id 9428, freq rank 3954) → tuple `(1, 9428, 74, 1)` → `يبور`   ❌ wrong root

  top-5: 1. `يبور` (root `يبر`) | 2. `حرف` (root `حرف`) | 3. `قيد` (root `قيد`) | 4. `دنفش` (root `دنفش`) | 5. `بالغ` (root `بلغ`)

---

## P22 — Ghazali_Ihya_Ulum_al_Din.txt (sentence 33681), window 614528, target stream idx 614593
*target: **not** counted by the metric (particle/control root); 12-gram context novel: True*

**Context (last 24 of the 128 window words; … = more words to the left):**

… إلى أن هذه ~~الأمور من الكبائر وقال الشافعي رضي الله عنه إذا شرب الحنفي النبيذ حددته ~~ولم أرد شهادته فقد جعله كبيرة بإيجاب الحد

**True next word:** `ولم`  (root `<P:ولم>` = `9306`, freq rank 163 of 9490, seen 6980x in train)  — gold tuple `(1, 9306, 1, 1)` realises as `ولم`

**RCA ON  (gates loaded) — top-1:** root `شبه` (id 4114, freq rank 169) → tuple `(1, 4114, 45, 1)` → `شبه`   ❌ wrong root

  top-5: 1. `شبه` (root `شبه`) | 2. `عرف` (root `عرف`) | 3. `خير` (root `خير`) | 4. `ورائدة` (root `ورد`) | 5. `فوجد` (root `وجد`)

**RCA OFF (gates forced to 0) — top-1:** root `همل` (id 8662, freq rank 810) → tuple `(4, 8662, 1, 1)` → `وهمل`   ❌ wrong root

  top-5: 1. `وهمل` (root `همل`) | 2. `خير` (root `خير`) | 3. `كذا` (root `كذا`) | 4. `لعم` (root `لعم`) | 5. `عضب` (root `عضب`)

---

## P23 — Ghazali_Ihya_Ulum_al_Din.txt (sentence 41211), window 712512, target stream idx 712577
*target: radical, counted by the metric; 12-gram context novel: True*

**Context (last 24 of the 128 window words; … = more words to the left):**

… ولكنه ينبه في الجملة على كيفية مصير ~~الكثرة في حكم المشاهدة واحدا ويستبين بهذا الكلام ترك الإنكار والجحود ~~لمقام لم تبلغه وتؤمن به إيمان

**True next word:** `تصديق`  (root `صدق` = `4539`, freq rank 158 of 9490, seen 7109x in train)  — gold tuple `(1, 4539, 37, 1)` realises as `تصديق`

**RCA ON  (gates loaded) — top-1:** root `صدف` (id 4538, freq rank 1889) → tuple `(1, 4538, 1, 1)` → `صدف`   ❌ wrong root

  top-5: 1. `صدف` (root `صدف`) | 2. `قام` (root `قام`) | 3. `نتق` (root `نتق`) | 4. `تقق` (root `تقق`) | 5. `كفر` (root `كفر`)

**RCA OFF (gates forced to 0) — top-1:** root `كفر` (id 7169, freq rank 226) → tuple `(1, 7169, 1, 1)` → `كفر`   ❌ wrong root

  top-5: 1. `كفر` (root `كفر`) | 2. `يزن` (root `يزن`) | 3. `صدف` (root `صدف`) | 4. `وثقق` (root `ثقق`) | 5. `سعع` (root `سعع`)

---

## P24 — Ghazali_Ihya_Ulum_al_Din.txt (sentence 41702), window 718976, target stream idx 719041
*target: **not** counted by the metric (particle/control root); 12-gram context novel: False*

**Context (last 24 of the 128 window words; … = more words to the left):**

… فيكون باعثا له على بذل كل ما يقدر ~~يهمه أمره ولا يبالي به ظفر خصمه أو لم يظفر هلك به حقه أو لم يهلك

**True next word:** `فإن`  (root `<P:فإن>` = `9217`, freq rank 56 of 9490, seen 16610x in train)  — gold tuple `(1, 9217, 1, 1)` realises as `فإن`

**RCA ON  (gates loaded) — top-1:** root `فوت` (id 6363, freq rank 745) → tuple `(1, 6363, 1, 1)` → `فوت`   ❌ wrong root

  top-5: 1. `فوت` (root `فوت`) | 2. `هلك` (root `هلك`) | 3. `فكن` (root `فكن`) | 4. `هزل` (root `هزل`) | 5. `هان` (root `هان`)

**RCA OFF (gates forced to 0) — top-1:** root `نطع` (id 8176, freq rank 2709) → tuple `(1, 8176, 1, 1)` → `نطع`   ❌ wrong root

  top-5: 1. `نطع` (root `نطع`) | 2. `هزل` (root `هزل`) | 3. `فكن` (root `فكن`) | 4. `هلك` (root `هلك`) | 5. `بلهس` (root `لهس`)

---

## P25 — Ghazali_Ihya_Ulum_al_Din.txt (sentence 41938), window 722112, target stream idx 722177
*target: **not** counted by the metric (particle/control root); 12-gram context novel: True*

**Context (last 24 of the 128 window words; … = more words to the left):**

… قدرتك وربما ~~يطرأ عليك في الحال ما يزيل عقلك ويبطل قوة حركتك وكيف تعول على حضور ~~الطعام وربما يسلط الله تعالى ~~احتمل أمثال ذلك

**True next word:** `ولم`  (root `<P:ولم>` = `9306`, freq rank 163 of 9490, seen 6980x in train)  — gold tuple `(1, 9306, 1, 1)` realises as `ولم`

**RCA ON  (gates loaded) — top-1:** root `طرق` (id 5026, freq rank 194) → tuple `(1, 5026, 1, 1)` → `طرق`   ❌ wrong root

  top-5: 1. `طرق` (root `طرق`) | 2. `حرف` (root `حرف`) | 3. `عله` (root `عله`) | 4. `كذب` (root `كذب`) | 5. `يقن` (root `يقن`)

**RCA OFF (gates forced to 0) — top-1:** root `حرف` (id 1734, freq rank 253) → tuple `(1, 1734, 1, 1)` → `حرف`   ❌ wrong root

  top-5: 1. `حرف` (root `حرف`) | 2. `ذهن` (root `ذهن`) | 3. `وكف` (root `وكف`) | 4. `كذاب` (root `كذب`) | 5. `شفع` (root `شفع`)

---

## P26 — Ghazali_Ihya_Ulum_al_Din.txt (sentence 43205), window 739008, target stream idx 739073
*target: **not** counted by the metric (particle/control root); 12-gram context novel: True*

**Context (last 24 of the 128 window words; … = more words to the left):**

… معرفة الله تعالى وكذلك ما يقاربه ~~ويختص به فشرفه على قدر تعلقه به ~~أحدها علمهم بالله وملائكته وكتبه ورسله وشرائع أنبيائه والثاني قدرتهم على

**True next word:** `~~إصلاح`  (root `<PARTICLE>` = `4`, freq rank 1 of 9490, seen 1571716x in train)  — gold tuple `(1, 4, 1, 1)` realises as ``

**RCA ON  (gates loaded) — top-1:** root `طرق` (id 5026, freq rank 194) → tuple `(1, 5026, 1, 1)` → `طرق`   ❌ wrong root

  top-5: 1. `طرق` (root `طرق`) | 2. `هلل` (root `هلل`) | 3. `نفس` (root `نفس`) | 4. `قلل` (root `قلل`) | 5. `ذرر` (root `ذرر`)

**RCA OFF (gates forced to 0) — top-1:** root `عيب` (id 5803, freq rank 880) → tuple `(1, 5803, 1, 1)` → `عيب`   ❌ wrong root

  top-5: 1. `عيب` (root `عيب`) | 2. `قبح` (root `قبح`) | 3. `سند` (root `سند`) | 4. `ظهره` (root `ظهر`) | 5. `كتب` (root `كتب`)

---

## P27 — Ghazali_Ihya_Ulum_al_Din.txt (sentence 43492), window 742656, target stream idx 742721
*target: radical, counted by the metric; 12-gram context novel: True*

**Context (last 24 of the 128 window words; … = more words to the left):**

… موافقة للمتخيلة وإنما الافتراق ~~بمزيد الوضوح والكشف فإن صورة المرئي صارت بالرؤية أتم انكشافا ووضوحا وهو ~~كشخص يرى في وقت الإسفار قبل انتشار ضوء

**True next word:** `النهار`  (root `نهر` = `8318`, freq rank 307 of 9490, seen 3655x in train)  — gold tuple `(3, 8318, 84, 1)` realises as `النهار`

**RCA ON  (gates loaded) — top-1:** root `خيل` (id 2426, freq rank 448) → tuple `(1, 2426, 1, 1)` → `خيل`   ❌ wrong root

  top-5: 1. `خيل` (root `خيل`) | 2. `درك` (root `درك`) | 3. `واسم` (root `وسم`) | 4. `صدر` (root `صدر`) | 5. `إسحاق` (root `سحق`)

**RCA OFF (gates forced to 0) — top-1:** root `وجه` (id 8816, freq rank 127) → tuple `(1, 8816, 1, 1)` → `وجه`   ❌ wrong root

  top-5: 1. `وجه` (root `وجه`) | 2. `رجع` (root `رجع`) | 3. `ريد` (root `ريد`) | 4. `خرج` (root `خرج`) | 5. `عبر` (root `عبر`)

---

## P28 — Ghazali_Ihya_Ulum_al_Din.txt (sentence 43958), window 748480, target stream idx 748545
*target: radical, counted by the metric; 12-gram context novel: True*

**Context (last 24 of the 128 window words; … = more words to the left):**

… وألقوا أسماعهم نحو ~~قوله وألقوا أبصارهم إلى الأرض فقال داود إني رسول الله إليكم يقرئكم ~~السلام ويقول لكم ألا تسألون حاجة ألا تنادوني أسمع

**True next word:** `صوتكم`  (root `صوت` = `4723`, freq rank 424 of 9490, seen 2463x in train)  — gold tuple `(1, 4723, 45, 9)` realises as `صوتكم`

**RCA ON  (gates loaded) — top-1:** root `صوت` (id 4723, freq rank 424) → tuple `(1, 4723, 1, 1)` → `صوت`   ✅ correct root

  top-5: 1. `صوت` (root `صوت`) | 2. `عمر` (root `عمر`) | 3. `كلم` (root `كلم`) | 4. `قال` (root `قال`) | 5. `غيث` (root `غيث`)

**RCA OFF (gates forced to 0) — top-1:** root `قذع` (id 6483, freq rank 3803) → tuple `(1, 6483, 1, 1)` → `قذع`   ❌ wrong root

  top-5: 1. `قذع` (root `قذع`) | 2. `غيث` (root `غيث`) | 3. `قلد` (root `قلد`) | 4. `ولخ` (root `ولخ`) | 5. `غوي` (root `غوي`)

---

## P29 — Ghazali_Ihya_Ulum_al_Din.txt (sentence 46812), window 784832, target stream idx 784897
*target: radical, counted by the metric; 12-gram context novel: False*

**Context (last 24 of the 128 window words; … = more words to the left):**

… والأحمق من أتبع نفسههواها وتمنى على الله الأماني ~~فانظرى لنفسك فما أمرك بمهم لغيرك ولا تضيعى أوقاتك فالأنفاس معدودة فإذا ~~مضى منك نفس فقد

**True next word:** `ذهب`  (root `ذهب` = `2965`, freq rank 128 of 9490, seen 8393x in train)  — gold tuple `(1, 2965, 45, 1)` realises as `ذهب`

**RCA ON  (gates loaded) — top-1:** root `ذهب` (id 2965, freq rank 128) → tuple `(1, 2965, 1, 1)` → `ذهب`   ✅ correct root

  top-5: 1. `ذهب` (root `ذهب`) | 2. `وجب` (root `وجب`) | 3. `صحح` (root `صحح`) | 4. `قال` (root `قال`) | 5. `فعل` (root `فعل`)

**RCA OFF (gates forced to 0) — top-1:** root `وجب` (id 8804, freq rank 70) → tuple `(1, 8804, 1, 1)` → `وجب`   ❌ wrong root

  top-5: 1. `وجب` (root `وجب`) | 2. `كذب` (root `كذب`) | 3. `دبب` (root `دبب`) | 4. `سبط` (root `سبط`) | 5. `ظفر` (root `ظفر`)

---

## P30 — Ghazali_Ihya_Ulum_al_Din.txt (sentence 49543), window 820160, target stream idx 820225
*target: radical, counted by the metric; 12-gram context novel: True*

**Context (last 24 of the 128 window words; … = more words to the left):**

… من أحب قال بلى قلت يا رسول الله فإني أحبك وأحب هؤلاء الفقراء ~~فقال صلى الله عليه وسلم صب على يده فإنه منهم وقال

**True next word:** `الجنيد`  (root `جند` = `1512`, freq rank 897 of 9490, seen 772x in train)  — gold tuple `(3, 1512, 55, 1)` realises as `الجنيد`

**RCA ON  (gates loaded) — top-1:** root `فقل` (id 6276, freq rank 75) → tuple `(1, 6276, 84, 1)` → `فقال`   ❌ wrong root

  top-5: 1. `فقال` (root `فقل`) | 2. `الجند` (root `جند`) | 3. `مهار` (root `مهر`) | 4. `رجل` (root `رجل`) | 5. `قسم` (root `قسم`)

**RCA OFF (gates forced to 0) — top-1:** root `فقل` (id 6276, freq rank 75) → tuple `(1, 6276, 45, 1)` → `فقل`   ❌ wrong root

  top-5: 1. `فقل` (root `فقل`) | 2. `صاحب` (root `صحب`) | 3. `الخفش` (root `خفش`) | 4. `أله` (root `اله`) | 5. `صهب` (root `صهب`)

---
