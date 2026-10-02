# Algorithm implementation ledger — what "implement them all" means

Measured on `/workspace/hf_v19_2_release`, not remembered. "Called from" means an actual call site,
not an import. The test of *implemented* is not that a class exists — it is that the shipped path
invokes it and the audit passes.

## The gap, in one line

```
AlKhalilV2  -> called from governed_lm.py, nrmt_train.py, nrmp_eval_clean.py, khalil_fidelity.py
SibawayhV2  -> NOWHERE
IbnMalikV2  -> NOWHERE
ShatibiWawV2-> NOWHERE
IbnMadaV2   -> NOWHERE
IrtishafRealizer / IrabRealizer   -> NOWHERE
IktifaSaturation / JussiveApocope -> NOWHERE
SibawayhConstituentStack          -> NOWHERE
KhalilPermutationOrbits           -> NOWHERE
ConstituentStack -> only from khalil_students_andalusian_master_engine.py (itself uncalled)
WordClassifier   -> only from constituent_stack.py
```

The shipped audit agrees and says so: **9/20**, with *"SibawayhConstituentStack exists … but is not
called from the NRMP path"*, *"state is recomputed from t-1 only"*, *"roots are independent one-hot
ids"*. So the algorithms are written and the pipeline is not using them.

## The full inventory

Ordered by dependency — each row's output is the next row's input.

| # | algorithm | school | source it must satisfy | code | status |
|---|---|---|---|---|---|
| 1 | phonotactics: letter incompatibility, C₁=C₂, makhraj | al-Khalīl | *al-ʿAyn*: «الهمزة والغين لا تجتمعان…», «القاف والكاف لا يجتمعان…» | `AlKhalilV2.mask()` | **WIRED** (nrmt_train) |
| 2 | al-taqālīb + مستعمل/مهمل | al-Khalīl | *al-ʿAyn* intro: «يُكتب مُستعملها ويُلغى مُهملها» | `khalil_attest_v4.py` | **WIRED** (tokenizer) |
| 3 | root calculation by restoration | al-Khalīl | Ibn Jinnī, *Sirr*: «فترد الواو التي هي عين الفعل من كنت» | `khalil_root_calculator.py` | wired into validator, **mode 2 only** |
| 4 | al-taṣrīf: qalb, ḥadhf, idghām, ibdāl | Andalusian | *al-Mumtiʿ*: «الواو المكسورة بمنزلة الياء… سيود», «وتحذف العين…» | `tasrif_engine.py` | **WIRED** (validator) |

| 6 | **الكثرة** (usage frequency) | al-Shāṭibī, Ibn Jinnī | «الثابت في الأصول أن الكثرة دليل الأصالة»; «والحمل على الأكثر واجب» | `kathra.py` (12.4M-token table) | **IMPLEMENTED, wired into mode 2** |
| 5 | the decision procedure | Ibn ʿUsfūr | *al-Mumtiʿ*'s list of أدلة | `ValidatedSegmentation._rank` | **COMPLETE** — kathra now supplies the missing criterion, and `كنت→كون` + `يشذ→شذذ` both resolve |
| 7 | al-ʿāmil and its persistence | Sībawayh | *al-Kitāb*: «الجر لا يكون إلا في الأسماء»; inqiṭāʿ al-ʿamal | `SibawayhV2` | **NOWHERE** |
| 8 | the constituent / valency stack | Sībawayh | the ʿāmil's reach through the clause | `SibawayhConstituentStack`, `constituent_stack.py` | **NOWHERE** |
| 9 | POS automaton + coordinator exception | Ibn Mālik | *Alfiyyah*: Fiʿl→Fiʿl allowed with و/ف | `IbnMalikV2` | **NOWHERE** |
| 10 | marātib al-maʿārif (definiteness ranks) | Ibn Mālik | *Alfiyyah* v. 55: «كهم وذي… وهند وابني والغلام والذي» | `IbnMalikV2.DEFINITENESS` | **NOWHERE** |
| 11 | wāw function classification | al-Shātibī | ATF / ḤĀL / ISTIʾNĀF / QASAM | `ShatibiWawV2` | **NOWHERE** |
| 12 | iʿrāb realisation + diptote | Andalusian | mamnūʿ min al-ṣarf: two single causes | `irab_realizer.py` | **NOWHERE** |
| 13 | al-iktifāʾ (clitic saturation) | Sībawayh's students | the slot closes by clitic | `iktifa_apocope.py` | **NOWHERE** |
| 14 | jussive apocope in surface realisation | Andalusian | *Lāmiyyat al-Afʿāl* | `iktifa_apocope.py` | **NOWHERE** |
| 15 | permutation-orbit parameter sharing | al-Khalīl | التقليب as a parameter-sharing principle | `KhalilPermutationOrbits` | **NOWHERE** |
| 16 | word class (ism/fiʿl/ḥarf + subclass) | Sībawayh | three classes; 46 categories | `word_class.py` | only via `constituent_stack` |
| 17 | loanword exemption for the dhalq test | al-Khalīl | «إلا أن تكون الكلمة معربة من كلام العجم» | `loanword_rule.py` | **NOWHERE** |
| 18 | wazn-conditioned iʿlāl | Andalusian | *al-Mumtiʿ*: which weak letter, decided by الاشتقاق | `ilal_conditioned.py` | **NOWHERE** |

## What "implemented" requires, per item

Not "the class exists". For each: **called from the shipped path**, **verified against the audit**,
and **carrying its citation**. The 11 audit failures name the specific work:

| audit failure | the item that fixes it |
|---|---|
| `SibawayhConstituentStack` not called from the NRMP path | 8 |
| government state recomputed from t−1 only | 7 |
| `T[2]={1,3}` blocks Fiʿl→Fiʿl unconditionally | 9 |
| definiteness hierarchy not implemented | 10 |
| roots are independent one-hot ids — no orbit sharing | 15 |
| wāw collapses to ISTINAF only | 11 |
| imperfect-awzan range wrong | awzān remap |
| «إن» → HARF_JAZM instead of INNA | 7 |
| 249 real roots wrongly forbidden | 1 |
| «في الأجسام الشفافة» government lost | 7 |
| particle root index 4 not masked in generation | decode mask |

## Order of work

1. **الكثرة** (#6) — it is the missing half of the decision procedure and blocks mode 2.
2. **Sībawayh's ʿāmil + the stack** (#7, #8) — the largest gap, and the audit's top complaint.
3. **Ibn Mālik's automaton + marātib** (#9, #10).
4. **The realisation layer** (#11–#14) — iʿrāb, iktifāʾ, apocope; these produce surface forms.
5. **Orbits** (#15) — a modelling change, affects the architecture rather than the decoder.
6. **The remainder** (#16–#18).

Items 1–5 are what the audit measures; they are also the ones with dead code already written,
so each is a wiring job plus a sourced test rather than a from-scratch implementation.

---

## Progress

**#6 الكثرة — done.** `models/kathra.py`, counted from a real 12,403,007-token corpus (51 files).
Table 4.92 MB, 294,926 forms. Decisive evidence:

```
كنت   كون/كان = 64,151   vs  كنن/كن = 726   -> كون
يشذ   شذذ/يشذ = 81       vs  شوذ/يشوذ = 0    -> شذذ
```

Neither binary ordering could get both; the frequency criterion gets both. Now position 4 of the
rank, after the generation gate, per al-Shāṭibī «والحمل على الأكثر متعين».

Honest limits, recorded by the implementer as E1–E7 in the module: what exactly is counted, the
normalisation, tokenisation, the hapax cut, the corpus choice, and the opt-in decade bucket are all
ENGINEERING. On a hand-classified sample of the changes kathra causes, 30 were genuine fixes and 21
were new errors — frequency is noisy at small counts (66 vs 64 decided one case). The decade bucket
(`ROOTFORMER_KATHRA_BUCKET=10`, the default) keeps 29 of the 30 fixes and drops 11 of the 21 errors.

Mode 1 (the shipped default) is unaffected: 1,103 of 20,000 roots, 5.51%, unchanged, because mode 1
never re-assigns a root and never consults the rank.

---

## BLOCKER FOUND — the awzān inventory was re-sorted after the checkpoints

Surfaced by the Sibawayh wiring; it blocks training and it is not a code bug.

```
checkpoint  rootformer_v19_2_synthesis_ar_backbone.safetensors   Sep 30 13:41
blueprint   rootformer_v12_arabic_blueprint.json                 Oct 1  12:12
            embed_tokens  9856 -> 9868
            wazn_embed / wazn_head  130 -> 142
```

**Root cause.** The id order is not authored anywhere: `nrmp_vocab.py` does `sorted(awzan_set)`.
Twelve nominal patterns were added on Oct 1, and because the list is SORTED, **124 of the 130 ids
moved**. The muḍāriʿ block slid from 114–129 to 126–141. Consequences, all measured:

- The audit's `range(114,130)` selects **12 nominals mislabelled as imperfect verbs** and misses 12
  real muḍāriʿ.
- `past_awzan_ids` — 28 hardcoded ids — is **28/28 wrong**. Live in `SibawayhV2.past`,
  `sibawayh_governance_engine.past_awzan_ids`, `sibawayh_governor.past_ids`. Effect:
  `all_verbal_awzan_ids` wrongly contains ~28 nominal patterns, so `apply_suffix_exclusion_mask`
  denies the feminine ة to them.
- `word_class.py:102` (`range(114,130)`) — feeds the constituent stack's 88.5 % coverage figure.
- Every id-bearing artifact predates Oct 1: `nrmp_cache/*.pt`, `sf_data_gov/*.pt`, all checkpoints.

**The fix is not "re-derive the lists".** Sorted insertion is the defect, so re-deriving would only
re-arm it. The order must become **APPEND-ONLY**: keep the original 130 patterns at their original
ids and append the 12 new ones as ids 130–141.

That choice is strictly better than both alternatives:

| | 130-only (delete the 12) | **append-only (recommended)** | keep the sorted 142 |
|---|---|---|---|
| muḍāriʿ back on 114–129 | ✓ | ✓ | ✗ |
| all 28 hardcoded ids valid again | ✓ | ✓ | ✗ |
| the 12 patterns kept | ✗ | ✓ | ✓ |
| wazn tensor handling | loads as-is | **extend by 12 rows** (zero-init) | permute 130 rows |

Appending needs **no permutation** — the existing rows keep their indices, so the checkpoint is
extended rather than reindexed, which is a standard, safe operation. `فَعَائِل`/`فَوَاعِل` survive,
which matters: those are the two labels `patch_awzan_labels.py` created deliberately for the sound
plural (حقائق → فَعَائِل).

**Still separate:** `embed_tokens` 9856→9868 and the root-slot changes are a second, independent
divergence. Both must be settled before any checkpoint loads, and therefore before training.

**Sequence from here:** (1) make the awzān order append-only, (2) extend the wazn tensors, (3) fix
the `embed_tokens` divergence, (4) rebuild `nrmp_cache`, (5) build the gold set, (6) train.

---

## AUDIT PROGRESS: 9/20 -> 16/20

| pass | audit | what closed |
|---|---|---|
| start | 9/20 | — |
| Sībawayh wiring | 13/20 | ʿāmil persistence, constituent stack, coordinator transition |
| al-Khalīl K1 | 15/20 | 249 real bare-alif roots restored (311 → 36 forbidden; C1=C2 intact 28/28) |
| Ibn Mālik | **16/20** | marātib al-maʿārif (the six ranks, with comparison) |

**Remaining 4:** the awzān layout (below), al-Taqālīb ×2 (#15, orbit parameter sharing), and the
al-Shāṭibī wāw classifier (#11, agent running).

### What Ibn Mālik's pass added, and what it admitted

`ibn_malik_automaton.py`, with **21 verbatim citations** — including Ibn Mālik's own «قام وقعد زيد»
(*Sharh al-Kāfiya*), Sībawayh's attested «ونخلع ونترك» (al-Kitāb l.944), al-Shāṭibī on the
definiteness comparison («لا ينعت بكل معرفة … لا بما هو فوق رتبته»), al-Suhaylī, and Ibn Maḍāʾ.
The decoder now reaches it: `nrmp_generate.py` → `IbnMalikPOSAutomaton.transition_allowed` → the
cited automaton. Delegation proved behaviour-preserving over all 5×5×2 cells.

Honest labels the implementer attached, rather than dressing them as fidelity:

- **Harf→Harf forbidden is ENGINEERING** — NOT FOUND in any source, and «إن في الدار» is a
  counter-example. Kept only to preserve the existing decoder table.
- The **Fiʿl→Fiʿl prohibition** without a coordinator is **SOURCED-COMPOSITE** — a composite
  inference from «العطف بحرف» + the two-operators-one-operand qāʿidah, not a single verse.
- **Ibn Mālik's six Tashīl ranks are collapsed into one PRONOUN rank.** The source splits
  mutakallim > mukhāṭab > ʿalam > ghāʾib. Recorded as a known simplification, not hidden.
- The exact wording «لا يعمل عاملان في معمول واحد» is **NOT in al-Kitāb** (it is the later
  grammarians'); Sībawayh's own l.936/938 statements are used instead.

### Two integration tasks handed back

1. `sibawayh_governor.py`'s `R9_naat_wifaq` uses an ad-hoc definiteness test, self-labelled
   ENGINEERING. The cited drop-in is `IbnMalikAutomaton.naat_ok()` (al-Shāṭibī). Needs wiring.
2. The definiteness comparison currently has **no live decode-time consumer** other than R9 —
   the automaton exposes it, the decode loop does not call it.

---

## AUDIT: 20/20 — the algorithm phase is complete

```
RESULT: 20/20 checks pass, 0 FAIL
```

| pass | audit | closed |
|---|---|---|
| start | 9/20 | — |
| Sībawayh | 13/20 | ʿāmil persistence, constituent stack, coordinator transition |
| al-Khalīl K1 | 15/20 | 311 → 36 forbidden roots; C₁=C₂ intact 28/28 |
| Ibn Mālik | 16/20 | marātib al-maʿārif |
| wāw + awzān range | 18/20 | wāw functions; awzān ids append-only |
| **al-Taqālīb** | **20/20** | orbit parameter tying, 967 orbits |

### But the audit's evidence per check is uneven — record this

Two checks are weaker than their PASS suggests, and one was actively broken:

- **`permutation-equivalent roots share embedding parameters` was hard-coded `False`.** It read
  `emb = dict(nv.__dict__)` and never used it — it measured nothing and *could never pass*. It is
  now a real measurement with a negative control (a مهمل permutation must stay untied). Implication:
  every earlier audit reading (9, 13, 15, 16) was taken with one check dead.
- **`definiteness hierarchy` is an attribute-existence probe**, not a behavioural test:
  `hasattr(IbnMalikPOSAutomaton, 'RANK') or 'maratib' in ...`. The hierarchy *is* implemented and
  tested in `ibn_malik_automaton.py` (46/46, including the `naat_ok` direction from al-Shāṭibī), but
  the audit does not demonstrate it.
- Its detail string still prints "not implemented" on PASS, so the output reads as a contradiction.

**So: 20/20 is the true number and I reproduced it, but it is not 20 equally strong proofs.** The
next honest step would be to strengthen those two checks to behavioural tests before treating the
audit as a guarantee.

### Orbit implementation — what it does and does not claim

967 tieable orbits over **attested permutations only**; 425 مهمل inventory roots excluded;
4,305 pairs (was 7,727). Real counts: `orbit_of('علم')=['علم','عمل','لمع','معل']` — the chapter's
four مستعملات; `excluded('عبد')['muhmal']=['عدب']`. Tying is shape-preserving (must run after
`load_state_dict`); true storage sharing would break every checkpoint, so it is opt-in and wired
nowhere.

The **al-ishtiqāq al-akbar rejection** is carried in the module header, the class docstring,
`nrmt_arch`, and the audit comment: only parameter tying is implemented, and no shared-meaning
claim. Ibn ʿUsfūr: «الصحيح أن هذا النحو من الاشتقاق غير مأخوذ به؛ لعدم اطراده».

Open and stated: al-ʿAyn's ع ل م chapter leaves لعم/ملع with **no verdict** (no فقط, no مهمل
group), so they are excluded for want of positive attestation — not because he condemned them;
2,168 roots likewise have no verdict; member ordering beyond the representative and `fold_hamza`
are ENGINEERING.

---

## CORRECTION: the awzān diagnosis above was wrong, and the news is better

The Sībawayh agent (and I, repeating it) said the blueprint had been re-sorted and 124 ids moved.
**That was wrong.** Verified by the awzān agent:

- The blueprint was **never mis-ordered**. `data/rootformer_v12_arabic_blueprint.json` already had
  the 12 new patterns **appended at the end** of `classical_awzan` (blueprint ids 353–364).
- The renumbering happened in **exactly one place**: `nrmp_vocab.py:58`,
  `sorted(list(self.base_tok.awzan_set))`.
- So **no blueprint rewrite and no embedding permutation were needed.**

**The fix:** `nrmp_vocab.py` now calls an authored `canonical_awzan_order()` with literal lists
(`AWZAN_CANONICAL_125`, `AWZAN_APPENDED_12`), which APPENDS new patterns, tolerates-and-reports
patterns missing from the end, and **RAISES on a hole in the middle** — a future addition can no
longer renumber anything. Verified `awzan_list[:130]` is byte-identical to the archived 130-awzān
vocab; the 16 muḍāriʿ are ids 114–129 by name; all 28 `past_awzan_ids` are correct by name
(pre-fix: 28/28 wrong — e.g. id 45 was فَاعِلَة, now فَعَلَ). Pre-fix the verbal set wrongly
contained 16 nominal patterns, denying them the feminine ة; post-fix, zero.

### The two consequences that matter

**1. The training cache is STILL VALID — no rebuild needed.**

```
/workspace/nrmp_cache/{train,val}.pt  (Sep 30 18:52) store wazn ids 1..120, root ids 3..9113
```

Because the fix restores ids 0–129 exactly, that cache remains correct. My earlier claim that it
"must be rebuilt" was an artefact of the wrong diagnosis. A rebuild IS still wanted eventually —
the cache never contains the 12 new patterns — but it is **no longer a blocker**, and it was never
corrupted.

**2. A loadable checkpoint now exists.**

`checkpoints/rootformer_v19_2_synthesis_ar_backbone.awzan142.safetensors` (793 MB, new file — the
original is untouched at md5 `07a3250b…`):

| tensor | change |
|---|---|
| `morphemic_embed.wazn_embed` [130,224] | append 12 zero rows |
| `nrmp_head.wazn_head` [130,896] | append 12 zero rows |
| `backbone.embed_tokens` [9856,896] | **INSERT** 12 zero rows at index 353 — *not* an append, so the trained token→row association is preserved |

565 other tensors bit-identical; the original raises a size mismatch against the current vocab
(control), the extended one strict-loads into the sub-modules.

### The single action left before training

`nrmt_train.py` still **defaults to the original checkpoint**, so a run today would silently fail
or mis-load. Either run it with
`--checkpoint checkpoints/rootformer_v19_2_synthesis_ar_backbone.awzan142.safetensors`, or make
that the default (the original is md5-recorded and can be restored). The 793 MB artifact was
deliberately not overwritten while other agents were working in the release.

### Flagged, not fixed (honest, all pre-existing)

- **Analyzer templates lag the inventory.** Only فَعَائِل and فَوَاعِل have `AWZAN_TEMPLATES`
  shapes (حقائق، دقائق، رسائل → id 130; حوادث → id 133). قناديل and أساليب still decompose to
  `<PARTICLE>`/`<NONE>`, so ids 131, 132, 134–136, 138–141 may be **unreachable from surface
  text** — the new patterns exist in the vocabulary but nothing can produce them yet.
- **Blueprint ↔ Qwen row alignment.** Because 12 tokens were inserted mid-vocabulary, blueprint ids
  ≥ 353 no longer match Qwen2.5-0.5B rows, yet `transmute_from_clean_qwen2_5` slices Qwen's matrix
  by index — so any *future* re-transmutation from clean Qwen would misalign the root block.
- **`unified_rootformer_v12.py:113`: `root_part = token_id - 349` is stale** — roots now start at
  365.
- `nrmp_head.*` is not consumed by `RootformerNRMT` (8 unexpected keys); it belongs to
  `RootformerV18_NRMP`. Structurally verified, but no end-to-end NRMP forward pass was run.
- Zero-init `wazn_head` rows 130–141 give all-zero logits: harmless when warm-started, but an
  untrained id would argmax to index 0.

---

## Ṣīghat muntaha — the best-sourced fix of the session

I had written `_SIGHAT_MUNTAHA = ('مفاعل','مفاعيل','فعالل','فعاليل')`. After the vocabulary gained
the plural block at ids 130–136, words of those patterns took jarr **kasrah** instead of the diptote
**fatha**: حقائق → «حققِ».

**The fix was not made by analogy.** The sources state the criterion, and it is a SHAPE test, not a
list of two names:

- Ibn Mālik, *Sharh al-Kāfiya*: «والمراد بالشبه: أن يكون أوله مفتوحا، وثالثة ألفا بعدها حرفان، أو
  ثلاثة أوسطها ساكن. **فيدخل في ذلك ما أوله ميم أو غيرها من الحروف**.»
- al-Shāṭibī, *al-Maqāṣid al-Shāfiyya*: «وإنما يريد ما كان على هذا الشكل من الجموع **مطلقا**،
  **فيدخل تحته (فواعل، وفعائل وفعالل، وفياعل) وكذلك إذا دخلتها الياء قبل الآخر** … الحكم في
  الجميع سواء.»
- al-Shāṭibī's ḍābiṭ: «كل جمع ثالث حروفه ألف ثابتة، **وبعدها حرفان، أو ثلاثة أحرف أوسطها ياء**.»
- Ibn Mālik, *Alfiyyah*: «**ولسراويل بهذا الجمع** شبه اقتضى عموم المنع» — سراويل is the فَعَاوِيل
  shape, tied to this plural by the Alfiyyah itself.

All 7 patterns verified against the ḍābiṭ by tail-shape (2 letters, or 3 with a medial yāʾ sākin).
**NOT FOUND: none** — no held source restricts the category to the two named shapes. Five citations
and five `RULE_SOURCES` entries added so the extension stays runtime-auditable.

Effect: حقائق/رسائل/صحائف/دوائر and حوادث/حقائيق/أساليب/دواوين now take fatha; مصانعَ، دنانيرَ،
حمراءَ، حبلى، فتىِ، عصاِ، قراءِ، الغلامِ unchanged. Through the live vocab, ids 130 (640 corpus
words) and 133 (692) are the ones the tokenizer actually emits.

### Two things found and NOT fixed (reported)

1. **The other half of «في حققِ» is a surface gap**, not the diptote:
   `models/morphemic_tokenizer_v12_arabic.py:587 realize_root_and_wazn` has no branch for the
   appended block and falls through to `return r`, so `decode_word` renders حقائق as the bare root
   «حقق». The realiser fix supplies the correct ENDING; the surface needs a tokenizer template.
   *Being fixed by a follow-up agent, with a corpus regression measurement.*
2. **Nisba-yāʾ exception.** With فَوَاعِل now recognised, Ibn Mālik's own حَوَارِيّ is treated as a
   diptote where he says it is munsarif («وإنما لم يعتد بياء نحو حواري … شبيهة بياء النسب»). No
   yāʾ-nisba exclusion exists in the realiser (it never existed for مفاعيل either). Flagged, not
   changed. Live impact is small — the tokenizer decodes حواري's surface to «حري» anyway, and
   جواري/حوالي map to فَوْعَل (unaffected).
