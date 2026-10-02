# Sībawayh's ʿāmil + Al-Khalīl K1 — wiring the written algorithms into the decode path

**Second pass.** The Sībawayh ʿāmil/persistence/valency wiring (pass 1) plus the Al-Khalīl K1
bare-alif fix (pass 2, this round).

**Audit (`test_grammar_impl.py`): 9/20 → 13/20 → 15/20.**
* AL-KHALIL section: 1/3 → **3/3**
* SIBAWAYH section: 4/9 → **8/9** (only the awzān-layout check remains — see §8)
* `verify_v2.py`: **ALL v2 CHECKS PASS** (unchanged throughout)
* `test_sibawayh_governor.py`: **15/15 PASS**
* `verify_k1.py`: **8/8 PASS**

Remaining 5 FAILs after this round: `imperfect-awzan range` (the re-index job, §8),
Marātib al-Maʿārif, al-Taqālīb ×2, al-Shāṭibī's wāw — none of them in this half.

## 1. Al-Khalīl K1 (this round)

`SibawayhNRMPGovernance.khalil_forbidden_roots` no longer forbids bare-alif roots; it is
**delegated to the corrected `AlKhalilV2.mask()`** (`classical_governance_v2`), which compiles
the rules Kitāb al-ʿAyn states, and is augmented with the control/special roots (PAD, BOS, UNK,
`<PARTICLE>`, `end`, `start` — EOS deliberately not, or the decoder could never close a sentence).
An ENGINEERING fallback (`_compile_khalil_mask`) carries the same rules if that module moves.

**Before/after on the two AL-KHALIL checks** (`verify_k1.py`, same vocabulary, pre-change class
loaded from `sibawayh_governance_engine.py.bak`):

| | BEFORE | AFTER |
|---|---|---|
| forbidden roots in total | 311 | 36 |
| …of which REAL bare-alif lexicon roots | **249** | **0** |
| `[FAIL/PASS] no legitimate hamza-initial root is excluded` | FAIL — 249 (ابا، ابب، ابت، ابخ، ابد…) | **PASS — 0** |
| `[FAIL/PASS] spot-check اصل/ارض/اخذ/احد/اسس/افل/اثر` | FAIL — all 7 killed | **PASS — 0 killed** |
| `[PASS] C1==C2 roots are excluded` | 28 | **28** (symmetric difference 0) |
| `end`/`start` leaked control strings | excluded | excluded |
| PAD/BOS/UNK/`<PARTICLE>` | excluded | excluded |
| EOS | not excluded | not excluded |
| `جقق` (qāf+kāf / jīm+qāf violation, Kitāb al-ʿAyn) | excluded | excluded |

The 36 that remain forbidden: 28 C1==C2 (ببب، ببر، تتا…), `end`, `start`, `جقق`, and the
PAD/BOS/UNK/`<PARTICLE>` ids.  `AlKhalilV2` reports **276 bare-alif roots now allowed**;
the lexicon still lists 32 of its own entries as forbidden, and every one of those is a
C1==C2 junk entry or a control string in the lexicon file, not a real root.

Citations for the rules the delegation implements (all previously verified verbatim in
`Al_Khalil_Al_Ayn.txt` and quoted in `classical_governance_v2.py`):
«القاف والكاف لا يجتمعان في كلمة واحدة …»; «الجيم مع القاف لا يأتلف إلا بفصل لازم»;
«الهمزة والغين لا تجتمعان في بناء كلمةٍ واحدةٍ»; «الضاد والصاد لا يأتلفان في كلمةٍ واحدةٍ
أصليّة الحروف».  The bare-alif rule that was removed is the one with **no** textual support:
it was an artefact of the blueprint normalising hamza to bare alif (أصل → اصل).

## 2. THE RE-INDEX BLAST RADIUS (why `range(114,130)` is wrong, and everything it touches)

**The root cause is an inventory change, not a code change.**  The checkpoint-matching vocabulary
at `/workspace/rootformer_v12/v18_next_root_morph/data/nrmp_vocab.json` has **130** awzān, and its
ids **114–129 are exactly the 16 muḍāriʿ paradigms** — which is why the audit's
`range(114,130)` was correct when written.  The current blueprint has **142**; **12 nominal
patterns were inserted at sorted positions**, so 124 of the 130 ids moved and the muḍāriʿ block
shifted **+12** (114–129 → 126–141).  Registered: 12 inserted, 6 unmoved.

The 12 inserted patterns (new ids): `أَفَاعِيل` 6, `أَفْعِل` 11, `فَعَائِل` 46, `فَعَائِيل` 47,
`فَعَالِل` 48, `فَعَالِيل` 49, `فَعَاوِيل` 50, `فَوَاعِل` 72, `فُعَّل` 82, `فُعُل` 83, `فِعَّال` 96,
`فِعْلَى` 100.  Two of them (`فَعَائِل`, `فَوَاعِل`) are exactly what `patch_awzan_labels.py`
(Oct 1 12:08) created by splitting the old `مَفَاعِل` label; the other ten came with the same
regeneration (the blueprint was rewritten Oct 1 12:12).

**What `range(114,130)` selects today — 12 of 16 wrong:**
`114 مُسْتَفْعَلَة، 115 مُسْتَفْعِل، 116 مُفَاعَلَة، 117 مُفَعْلَل، 118 مُفْعْلِل، 119 مُفْتَعَل،
120 مُفْتَعَلَة، 121 مُفْتَعِل، 122 مُفْعَل، 123 مُفْعِل، 124 مُنْفَعِل، 125 مِفْعَال` are
**nominals mislabelled as imperfect verbs**; only 126–129 are muḍāriʿ, and **130–141 (12
muḍāriʿ) are missed**.  The muḍāriʿ should be **114–129** (verified: deleting the 12 inserted
patterns reproduces the 130-awzān order **exactly**, and the muḍāriʿ land back on
`[114…129]`).  Equivalently, on the 142-awzān order they are `[126…141]`.

**Worse: the hardcoded `past_awzan_ids` list is 28/28 wrong.**
`{8,12,14,15,16,19,22,23,24,25,26,27,29,31,33,35,38,39,40,45,49,50,54,61,75,76,82,83}` was
authored against the 130-vocab.  Under the current order **every one of the 28 entries points at
a different pattern** — e.g. id 8 now selects `أَفْعَل` (elative) instead of `أَفْعَلَ`
(form-IV past); id 61 selects `فَعِلَ` instead of `فَعْلَلَ`; id 82 selects `فُعَّل` instead of
`فُعْلُل`.  This is live in `classical_governance_v2.SibawayhV2`, in
`sibawayh_governance_engine.SibawayhNRMPGovernance.past_awzan_ids`, and in
`sibawayh_governor.past_ids`.  (Consequence today: the HARF_JAZM mask still admits exactly the
16 muḍāriʿ, because that mask is driven by the *derived* imperfect set, but the
`all_verbal_awzan_ids` union wrongly includes ~28 nominal patterns — which wrongly denies the
feminine suffix (ة) to those patterns in `apply_suffix_exclusion_mask`.)

**Every place that would need re-indexing / re-derivation:**

*Code — id lists*
1. `word_class.py:102` `self.imperfect_wz = set(range(114, 130))` (used at :130) — **left alone
   as instructed**; wrong by 12 of 16 today.
2. `classical_governance_v2.py:168-171` `SibawayhV2.imperfect` (range) **and** `SibawayhV2.past`
   (28 hardcoded ids) — both stale; `past` 28/28 wrong.
3. `sibawayh_governance_engine.py:99,100` — `imperfect` is now **derived** (R3) here; the
   ENGINEERING fallback `range(114, min(130, …))` and `past_awzan_ids` remain stale.
4. `sibawayh_governor.py:507` fallback range and `past_ids` — same.
5. `per_stream_eval.py:128` `V = {'r': 9114, 'w': 130, …}` — stale vocabulary size constant.
6. `test_grammar_impl.py:96` — the check itself, `range(114, 130)`; correct iff the 130-layout
   is restored.
7. Pattern-based, therefore **unaffected**: `constituent_stack.wazn_role` (matches wazn *strings*
   like `مَفْعُول`, `فَاعِل`), `nrmt_arch.build_operator_table` (keyed on roots), the Sībawayh
   masks' *derived* sets in `sibawayh_governance_engine`/`sibawayh_governor`.

*Data — where the order actually lives*
8. `models/morphemic_tokenizer_v12_arabic.py` — `awzan_set` (line 255/279) filled from
   `AWZAN_TEMPLATES` (line 171) and `QUADRILITERAL_AWZAN` (line 236); the order is not authored
   anywhere, it is `sorted(awzan_set)` in `nrmp_vocab.py:58`.  **This is the place to fix**: either
   restore the 12 templates' absence (reproduces the 130 order bit-for-bit, keeping the
   checkpoints valid) or freeze an explicit ordered `awzan` list in the blueprint and have
   `MorphemicVocab` honour it instead of sorting.
9. `data/rootformer_v12_arabic_blueprint.json` (live, 142), `…blueprint.ORIGINAL.json`,
   `…blueprint_137awzan.json` (the 137-awzān intermediate) — each carries its own layout.
10. `/workspace/rootformer_v12/v18_next_root_morph/data/nrmp_vocab.json` — the 130-awzān
    reference order (keep as the gold mapping).
11. Cached streams that store wazn **ids** and are stale the moment the order changes:
    `/workspace/nrmp_cache/{train,val}.pt` (Sep 30 18:52 = 130-order),
    `/workspace/sf_data_gov/{train,val,test_gen,test_deriv}.pt` + `meta.json` (Oct 1 10:07 =
    130-order, i.e. also older than the 12:12 blueprint).
12. Derived/report JSONs that would silently keep old ids: `constituent_stack.json`,
    `word_class_coverage.json`, `gold_roundtrip.json`, `nrmp_results.json`, `eval_*.json`.

*Checkpoints*
13. `checkpoints/rootformer_v19_2_synthesis_ar_backbone.safetensors` — `wazn_embed`/`wazn_head`
    with 130 rows (Sep 30 13:41).  A **re-order** invalidates them unless the two wazn tensors are
    permuted by the same bijection (`new_id → old_id`); **deleting the 12 inserted patterns needs
    no permutation at all**, because the remaining rows keep their exact old indices.  That is the
    cheapest way back to a working checkpoint + a passing audit check.

## 3. Blocker for TRAINING (untouched, as instructed)

`checkpoints/rootformer_v19_2_synthesis_ar_backbone.safetensors` (Sep 30 13:41) does **not** match
`data/rootformer_v12_arabic_blueprint.json` (regenerated Oct 1 12:12): `backbone.embed_tokens`
9856 → 9868 and `morphemic_embed.wazn_embed` / `nrmp_head.wazn_head` 130 → 142.  Loading raises a
size mismatch **before any ʿāmil code runs**, so every checkpoint in this release is stale against
the current blueprint; `nrmp_run.py --generate`, `nrmt_train.py` and both `*_governed_eval.py`
cannot load as shipped.  Left exactly as found — the four findings are recorded here for whoever
owns the vocabulary/checkpoint regeneration.  (Note the interaction with §2: restoring the 130
order fixes the `wazn_*` half of this mismatch; the `embed_tokens` 9856→9868 and the root-slot
changes are separate.)

## 4. The API (`sibawayh_governor.py`)

```python
class SibawayhGovernor:
    def __init__(self, vocab, constituent_stack=None, blueprint_path=None)
    def analyze(self, words) -> list        # (category, case, role, reason); persistence applied;
                                            # accepts surface words OR (prefix, root, wazn, suffix)
    def transition_allowed(self, prev_class, next_class, coordinator_between=False) -> bool
    def operator_of(self, word, next_word=None) -> str      # resolves إن/أن by the next class
    def reset()/reset_chain(), observe_word/tuple/sentence, state_for_next(), states(),
        states_of_tuples(), cases(), allowed_next_classes(), mask_wazn_logits(),
        op_ids_for_ids(), op_ids_for_tuples()
```

## 5. Where it is actually called (decision points)

| file | decision point | change |
|---|---|---|
| `sibawayh_governance_engine.py` | `get_operator_state` | delegates to the persistent chain; `reset_chain()`; stateless fallback kept, INNA before JAZM |
| `sibawayh_governance_engine.py` | `imperfect_awzan_ids` | **derived** from R3 + the muḍāriʿ ending (`[126..141]`), not `range(114,130)` |
| `sibawayh_governance_engine.py` | `khalil_forbidden_roots` | delegated to `AlKhalilV2.mask()` (K1) + control roots |
| `rootformer_v18_nrmp_model.py` | `generate_words` | `state_for_next()` per step; chain seeded from the prompt and advanced after each word; `mask_wazn_logits` (transition table) at the wazn decision; `ConstituentStack`/`governor.analyze` on prompt and hypothesis, returned as `prompt_governance`/`governance`/`valency_stack_used` |
| `nrmt_train.py` | operator feature stream | `op_ids_of` walks each window through `op_ids_for_ids` (persistent), with fallback |
| `andalusian_grammatical_algorithms.py` | `PERMISSIBLE_TRANSITIONS[2]` | admits Fiʿl→Fiʿl; `PERMISSIBLE_TRANSITIONS_NO_COORDINATOR`, `transition_allowed`, `has_coordinator_prefix`, coordinator-aware `filter_logits_by_pos` |
| `nrmp_generate.py` | POS gate | `transition_allowed(prev_pos, cat, has_coordinator_prefix(prefix))` |
| `rootformer_v20_nrmp_model.py`, `nrmp_generate.py`, `nrmt/nrmp_governed_eval.py`, `governed_lm.py` | every other `get_operator_state` caller | `reset_chain()` at a prompt/sentence boundary |

**Files NOT touched** (other agents'): `ibn_malik_automaton.py`, `andalusian_realizer.py`,
`word_class.py`, `constituent_stack.py`, `basran_syntactic_engine.py`,
`classical_governance_v2.py`, `test_grammar_impl.py`, `verify_v2.py`, checkpoints, blueprint.

## 6. The rules and their sources (verbatim)

* **R1 the three classes** — al-Kitab 1/11: «فالكلم: اسم، وفعل، وحرف جاء لمعنى ليس باسم ولا فعل»
* **R2 an operator governs only its own class** — al-Kitab 3/8: «واعلم أن حروف الجزم لا تجزم إلا
  الأفعال، ولا يكون الجزم إلا في هذه الأفعال المضارعة للأسماء، كما أن الجر لا يكون إلا في
  الأسماء»; and 2/422–3/5: «لأن اللام وحتى إنما يعملان في الأسماء فيجران، وليستا من الحروف التي
  تضاف إلى الأفعال»
* **R3 the muḍāriʿ is marked by the four ziyādāt** — al-Kitab 1/13: «أوائلها الزوائد الأربع:
  الهمزة، والتاء، والياء، والنون. وذلك قولك: أفعل أنا، وتفعل أنت أو هي، ويفعل هو، ونفعل نحن»
* **R4 government persists across the intervening noun** — al-Kitab 1/421: «فأما النعت الذى جرى
  على المنعوت فقولك: مررت برجل ظريف قبل، فصار النعت مجرورا مثل المنعوت لأنهما كالاسم الواحد»;
  «فإن أطلت النعت فقلت: مررت برجل عاقل كريم مسلم، فأجره على أوله»
* **R5 inqiṭāʿ al-ʿamal** — the constituent boundary of 1/421 + the class restriction of 3/8
* **R6 «إن» vs «أن» by the next word's class** — al-Kitab 3/119: «وأما إن فإنما هي بمنزلة الفعل لا
  يعمل فيها ما يعمل في أن، كما لا يعمل في الفعل ما يعمل في الأسماء، ولا تكون إن إلا مبتدأة، وذلك
  قولك: إن زيدا منطلق، وإنك ذاهب»; «وأما أن فهي اسم وما عملت فيه صلة لها»; conditional إنْ 3/56
  «ومن غيرهما: إن، وإذ ما»
* **R7 two operators do not govern one operand** — Abu Hayyan: «ولا يجتمع عاملان على معمول واحد
  إلا في التقدير»; Ibn 'Usfur: «لئلا يؤدي إلى أن يعمل عاملان في معمول واحد»; Sibawayh's nearest,
  1/73: «لا يعمل في اسم واحد نصب ورفع … وإنما كان الذى يليه أولى لقرب جواره»
* **R8 the coordinator exception** — Alfiyyah, bāb al-ʿaṭf: «وعطفك الفعل على الفعل يصح»;
  «فاعطف بواو سابقا أو لاحقا … في الحكم أو مصاحبا موافقا»; «وانقل بها للثان حكم الأول»
* **R9 naʿt agreement** — al-Kitab 1/421-423; Alfiyyah, bāb al-naʿt: «فأولينه من وفاق الأول …
  ما من وفاق الأول النعت ولي»
* **K1 the Kitāb al-ʿAyn letter pairs** — «القاف والكاف لا يجتمعان في كلمة واحدة …»; «الجيم مع
  القاف لا يأتلف إلا بفصل لازم»; «الهمزة والغين لا تجتمعان في بناء كلمةٍ واحدةٍ»; «الضاد والصاد
  لا يأتلفان في كلمةٍ واحدةٍ أصليّة الحروف» (quoted in `classical_governance_v2.py`, whose
  `AlKhalilV2.mask()` is what this release now delegates to)

### NOT FOUND (not stretched into a citation)
* «ولا يعمل عاملان في معمول واحد» as a **Sibawayh** sentence — 0 hits in al-Kitab; it is Abu
  Hayyan's / Ibn 'Usfur's wording (cited above).
* «حتى يقطع عمله» / «العامل يعمل فيما يليه» — 0 hits in al-Kitab (the wording in
  `constituent_stack.py`'s docstring is not Sibawayh's).
* Any textual basis for the removed **bare-alif** exclusion — none; it was a blueprint artefact.

### ENGINEERING (labelled in the code)
E1 the class→class transition table (an encoding of R1 for a POS automaton; the coordinator rule
in it is R8).  E2 the māḍī is not identifiable from the wazn alone in this blueprint.  E3 clitic
stripping accepts only known-operator cores.  E4 bare-alef إن/أن collapse tie-break.  E5
`_compile_khalil_mask` is the fallback copy of `AlKhalilV2.mask()`.

## 7. Files

**Created** (pod `/workspace/hf_v19_2_release/`): `sibawayh_governor.py` (994 l),
`test_sibawayh_governor.py` (121 l), `verify_k1.py` (105 l), `awzan_blast.py` (70 l),
`smoke_decode.py` (49 l), `SIBAWAYH_GOVERNOR_REPORT.md`, `SIBAWAYH_GOVERNOR.diff`.

**Modified** (`.bak` next to each, md5-verified equal to the pre-change file):

| file | diff lines |
|---|---|
| `sibawayh_governance_engine.py` | 138 |
| `rootformer_v18_nrmp_model.py` | 106 |
| `nrmt_train.py` | 29 |
| `andalusian_grammatical_algorithms.py` | 57 |
| `nrmp_generate.py` | 13 |
| `rootformer_v20_nrmp_model.py` | 5 |
| `nrmt_governed_eval.py` | 4 |
| `nrmp_governed_eval.py` | 4 |
| `governed_lm.py` | 5 |

## 8. Evidence

* `smoke_decode.py` — real tensors through `generate_words`: governor wired True,
  constituent_stack wired True, 6 words generated, `valency_stack_used: True`, the persistent
  `state_for_next()` called once per step, per-word `governance`, `prompt_governance` with jarr on
  word 2.  (Drops the 3 shape-mismatched tensors — see §3.)
* `verify_k1.py` — the §1 table, loading the pre-change class from the `.bak`.
* `awzan_blast.py` — the §2 mapping: insertions, the muḍāriʿ move, 28/28 `past` mislabels.
