# AL-TAQĀLĪB (التقليبات) — al-Khalīl's permutation orbits as a parameter-tying mechanism

Release: `/workspace/hf_v19_2_release` · Python: `/workspace/venvs/rootformer/bin/python`
Author: AL-TAQALIB (root-embedding side) subagent · Status: **both assigned audit failures closed; audit 20/20**

---

## 1. Deliverable and API

**New canonical module** `khalil_orbits.py` (release root; mirror `rootformer/build/khalil_orbits.py`)
**Re-exported** where the audit imports it: `models/khalil_combinatorics.py`
(`from models.khalil_combinatorics import KhalilPermutationOrbits` now succeeds).

```python
from khalil_orbits import KhalilPermutationOrbits, tie_root_embedding_tables

orbits = KhalilPermutationOrbits(vocab)          # vocab = nrmp_vocab.MorphemicVocab, or iterable/dict of roots
orbits = KhalilPermutationOrbits(vocab, lengths=(2, 3, 4, 5),   # al-Khalīl's 2/6/24/120 faces
                                 membership='attested')          # 'attested' | 'inventory_not_muhmal'

orbits.orbit_of('علم')      # ['علم', 'عمل', 'لمع', 'معل']  — the tieable family (al-ʿAyn's مستعملات)
orbits.partners('علم')      # family minus the root
orbits.excluded('عبد')      # {'muhmal': ['عدب'], 'unattested': [...], 'not_in_inventory': ['دبع']}
orbits.is_attested(r) / orbits.is_muhmal(r) / orbits.permutations_of(r) / orbits.orbit_index()

orbits.tie_embeddings(emb.weight)          # ACTUAL parameter sharing; shape-preserving (checkpoint-safe)
orbits.tie_gradients(emb.weight)           # after backward(): one update per tied family
orbits.loss(emb.weight)                    # within-orbit dispersion; 0 when tied; differentiable
orbits.pairs()                             # id pairs for nrmt_arch.orbit_consistency_loss
tie_root_embedding_tables(model_or_state, orbits)   # ties the 24 per-layer tables the checkpoints carry
orbits.stats() / orbits.report()           # measured counts (see §4)
```

* `membership='attested'` (default, faithful): a permutation is a family member **only if**
  `data/khalil_attest_v4.json` records it as مستعمل. Positive attestation only — nothing al-ʿAyn
  merely omits is treated as condemned, and nothing it condemns is ever admitted.
* `membership='inventory_not_muhmal'`: in the shipped inventory **and not** marked مهمل. Ties
  strictly more; measured in §4; **not** the default.
* `min_orbit_size=2`, `representative='farahidi'|'lowest_id'`, `fold_hamza=True` (see §8),
  graceful fallback when the attest record is absent (warns, switches to the loose policy).

Integration (opt-in, no default behaviour changed):
* `nrmt_arch.RootformerNRMT._build_orbit_pairs` now sources its pairs from
  `KhalilPermutationOrbits` (attested-only); legacy builder retained as a guarded fallback.
* `nrmt_arch.RootformerNRMT.apply_orbit_tying(method='mean')` — new, **must be called after
  `load_state_dict`**; not called by anything automatically.

---

## 2. Citations (verbatim) and the mandatory caveat

**al-Khalīl, *Kitāb al-ʿAyn*, introduction** — `corpus/basran/Al_Khalil_Al_Ayn.txt:164`:

> «قال الليث: قال الخليل: اعلم أن الكلمة الثنائيَّةَ تَتَصَرَّف على وَجْهَيْن نحو: قَدْ، دَقْ، شَدْ،
> دَشُ «1» والكلمةُ الثلاثَّيُة «2» تتصرَّفُ على ستة أوجُه، وتُسمَّى مَسدُوسة «3» ... والكلمة
> الرباعية تتصرَّف على أربعة وعشرين وجها ... فَتصيرَ أربعة وعشرين وَجْهاً، **يُكَتَب مُسْتَعْمَلها.
> ويُلغى مُهْمَلها** ... والكلمة الخماسية تتصرّف على مائة وعشرين وجها ... **يُسْتَعْمَل أقَلُّه
> ويُلغى أكثره**»

(The brief renders the clause «يُكتب مُستعملها ويُلغى مُهملها»; the corpus edition reads
«يُكَتَب مُسْتَعْمَلها. ويُلغى مُهْمَلها» — same words, different vowelling/brace. Quoted as the corpus has it.)

**al-Khalīl's own chapter for ع ل م** — `Al_Khalil_Al_Ayn.txt:5817`:

> «باب العين واللاّم والميم معهما ع ل م، ع م ل، م ع ل، ل م ع مستعملات»

**al-Khalīl's explicit مهمل chapter for ع د ب** — `Al_Khalil_Al_Ayn.txt:4760`:

> «باب العين والدال والباء معهما ع ب د- د ع ب- ب ع د- ب د ع مستعملات ع د ب- د ب ع مهملان»

**Ibn Jinnī, *al-Khaṣāʾiṣ*, «باب في الاشتقاق الأكبر»** — `corpus/basran/Ibn_Jinni_Al_Khasais.txt:7639-7641`:

> «وأما الاشتقاق الأكبر فهو أن تأخذ أصلا من الأصول الثلاثية3، فتعقد عليه وعلى **تقاليبه4 الستة**
> **معنى واحدا**, تجتمع التراكيب الستة وما يتصرف من كل واحد منها عليه»

### THE REJECTION (this caveat is in the code, in the audit, and here)

**Ibn ʿUsfūr, *al-Mumtiʿ fī al-Taṣrīf*** — `corpus/andalusian/13_IbnUsfur_Mumtic_fi_al_Tasrif.txt:157-161`:

> «أما الاشتقاق الأكبر هو عقد تقاليب الكلمة كلها على المعنى واحد، نحو ما ذهب إليه [أبو الفتح] بن جني
> من عقد تقاليب "القول" الستة على منى الخفة. **ولم يقل به أحد من النحويين إلا أبا الفتح.** وحكى هو
> عن أبي علي أنه كان يأنس به في بعض الأماكن. **والصحيح أن هذا النحو من الاشتقاق غير مأخوذ به؛ لعدم
> اطراده**، ولما يلحق فيه من التكلف لمن رامه.»

**Consequently: what is implemented is parameter tying, and ONLY parameter tying.** The *semantic*
claim of al-ishtiqāq al-akbar — that the six permutations share one MEANING — is rejected by the
tradition (`لعدم اطراده`) and is **not** implemented, optimised, or implied. Nothing in
`khalil_orbits.py` asserts shared meaning; the mechanism is licensed solely by al-Khalīl's own
مستعمل/مهمل filter («يُكَتَب مُسْتَعْمَلها. ويُلغى مُهْمَلها»). The caveat appears verbatim in:
`khalil_orbits.py` module header + class docstring, `models/khalil_combinatorics.py` re-export
comment, `nrmt_arch.py` docstring item 6 + `apply_orbit_tying` docstring, and the audit check.

---

## 3. Audit before/after (measured)

| run | result | AL-TAQALIB checks |
|---|---|---|
| before (my first run of `test_grammar_impl.py`) | **17/20** | `permutation-equivalent roots share embedding parameters` **FAIL**; `KhalilPermutationOrbits class exists` **FAIL** |
| after (my changes) | **20/20** | both **PASS** |

Two notes on the numbers, stated plainly:

1. The brief said the audit reads 16/20. When I started it already read **17/20**: a concurrent
   agent had just fixed the Ibn Mālik coordinator check. The third remaining failure when I
   started (`imperfect-awzan range (114..129) …`) belongs to the awzān-id work and **was fixed by
   that other agent during my run** (`-- e.g. يَفْعَلُ`), so the audit reached 20/20 with my two
   checks being the only ones I changed. I did not touch `nrmp_vocab.py`, the tokenizer, or the
   blueprint.
2. **The first AL-TAQALIB check was hard-coded `False`** in `test_grammar_impl.py`
   (`check('permutation-equivalent roots share embedding parameters', False, '…')`; the preceding
   `emb = dict(nv.__dict__)` was unused). It could not have passed under any implementation. I
   replaced it with a real measurement (ENGINEERING fix, reported here rather than silently):
   it now builds the orbits, asserts علم/عمل are one family, ties a random
   `num_roots × 16` table, asserts the two rows become **identical**, and asserts the مهمل
   permutation `عدب` (bāb ع د ب) is **not** tied — a negative control. `test_grammar_impl.py.bak.taqalib`
   holds the original.
3. Unrelated audit-integrity observation (not mine, not changed): the check
   `definiteness hierarchy (Maratib al-Maʿārif) implemented` reports PASS while its own detail
   says *“not implemented (pronoun > proper > demonstrative > definite > indefinite)”* — the
   predicate is `… or 'maratib' in dir(IbnMalikPOSAutomaton).__str__().lower()`, which is
   satisfied for the wrong reason.

---

## 4. Orbit counts over the real inventory (measured, `rootformer_v12_arabic_blueprint.json`,
9,114 rows; `data/khalil_attest_v4.json`, 4,381 attested / 2,750 مهمل, 1,428 chapters)

**Default — `lengths=(3,)`, `membership='attested'` (S³, what the audit names):**

| quantity | value |
|---|---|
| 3-radical inventory roots considered | 6,285 (2,829 non-radical/short/long rows skipped) |
| permutation keys (radical multisets) | 2,562 |
| **tieable orbits** (≥ 2 attested members) | **967** |
| **orbit sizes** (faces) | 2 → 353 · 3 → 238 · 4 → 206 · 5 → 116 · 6 → 54 |
| tied rows / faces | 3,155 rows / 3,148 faces (7 rows are duplicate hamza spellings: بدء/بدأ/بدا, جزء/جزأ/جزا, أمر/امر) |
| rows collapsed by tying | 2,188 |
| isolated roots (family of one) | 537 |
| al-ʿAyn **مهمل** inventory roots kept out | **425** |
| unattested inventory roots kept out | 2,168 |
| internal id pairs (`orbit_consistency_loss` input) | **4,305** (was 7,727) |

**Alternative — `membership='inventory_not_muhmal'` (measured, not default):** 1,378 orbits,
faces {2:513, 3:261, 4:286, 5:177, 6:141}, 4,696 tied rows, 6,956 pairs, the same 425 مهمل roots
still excluded, 1,164 isolated.

**Other radical counts (al-Khalīl's 24- and 120-face sentences), `attested` policy:** 0 tieable
orbits at length 2, 4 and 5 — the record holds only **29 quadriliteral attestations** and no
pentliteral ones, so no length-4/5 key reaches two attested members. Under
`inventory_not_muhmal`, `lengths=(2,3,4,5)` gives 1,947 orbits (1,378 of length 3 + **560 of
length 4** + **9 of length 5**), 8,591 pairs, faces up to 8 — i.e. `n! > 3` is handled, but the
faithful default is exercised at S³ only. This is a data limitation, not a modelling choice, and
it is reported rather than papered over.

**Worked cases:** `orbit_of('علم') = ['علم','عمل','لمع','معل']` (4 of the 6 faces — exactly the
four the chapter lists as مستعملات); `orbit_of('عبد') = ['عبد','دعب','بعد','بدع']` with
`excluded('عبد')['muhmal'] = ['عدب']` (دبع is not in the inventory).

---

## 5. Tests

`test_khalil_orbits.py` (new, release root): **30/30 PASS**. Key lines:

```
[PASS] KhalilPermutationOrbits importable from models.khalil_combinatorics
[PASS] علم and عمل are in one orbit  -- orbit(علم)=['علم', 'عمل', 'لمع', 'معل']
[PASS] after tie_embeddings the two rows are IDENTICAL (parameter sharing)  -- tied 967 orbits
       / 3155 rows, 2188 collapsed, method=mean
[PASS] al-ʿAyn bāb ع د ب marks عدب (and دبع) مهمل  -- «... ع د ب- د ب ع مهملان»
[PASS] عدب is NOT in orbit(عبد) / after tying, عدب keeps its own row
[PASS] orbit(علم) has exactly the four faces al-ʿAyn lists as مستعملات
[PASS] the unlisted faces لعم and ملع are not tied in  -- excluded={'muhmal': [], 'unattested': ['لعم','ملع']}
[PASS] record counts are the documented 4,381 attested / 2,750 مهمل
[PASS] every tied member is positively attested / no مهمل root is ever a tied member
[PASS] loss > 0 untied and ~0 tied (0.65 → 4.5e-17) / loss is differentiable
[PASS] tying leaves state_dict keys and shape unchanged (shape-preserving)
[PASS] a tied state_dict still loads into a fresh embedding
[PASS] OrbitTiedEmbedding (storage sharing) has a DIFFERENT shape -> checkpoint-breaking
[PASS] shipped checkpoint root tables are tied or REPORTED, never resized
       -- tied=0 skipped=24 of 24 tables; rows=9015 != vocab num_roots=9114
RESULT: 30/30 checks pass, 0 FAIL
```

Run: `cd /workspace/hf_v19_2_release && /workspace/venvs/rootformer/bin/python test_khalil_orbits.py`

---

## 6. Files created / modified, with exact diffs

**Created** (no `.bak` needed; content is the artifact):
| file | remote | local mirror |
|---|---|---|
| `khalil_orbits.py` | `/workspace/hf_v19_2_release/khalil_orbits.py` | `rootformer/build/khalil_orbits.py` |
| `test_khalil_orbits.py` | `/workspace/hf_v19_2_release/test_khalil_orbits.py` | `rootformer/build/test_khalil_orbits.py` |
| `AL_TAQALIB_ORBITS.diff` | `/workspace/hf_v19_2_release/AL_TAQALIB_ORBITS.diff` | `rootformer/build/AL_TAQALIB_ORBITS.diff` |

**Modified** (each with a `.bak.taqalib` beside it in the release, and mirrored locally):
| file | change | `.bak` md5 |
|---|---|---|
| `models/khalil_combinatorics.py` | append re-export block (`KhalilPermutationOrbits`, `OrbitTiedEmbedding`, `normalize_root`); engine above untouched | `ba86bd656c423b70165e0c5cf2ca51c4` |
| `nrmt_arch.py` | item-6 docstring; `_build_orbit_pairs` → attested-only orbits + guarded legacy fallback; new `apply_orbit_tying()` | `3d8f71a386f28d273533e7f0607eaffd` |
| `test_grammar_impl.py` | the hard-coded AL-TAQALIB `False` replaced by a real, negative-controlled check | `d39fea6a68d27d0d7baa1f4152e1b500` |

`AL_TAQALIB_ORBITS.diff` (186 lines) is the exact `diff -u <bak> <file>` for all three
(`models/khalil_combinatorics.py` +41/−0, `nrmt_arch.py` +51/−1, `test_grammar_impl.py` +36/−6).

**Structural safety evidence for `nrmt_arch.py`:** `NRMTHead(d_model=8, …).state_dict().keys()`
is **identical** before/after (18 keys, no `orbit*` key — `orbit_pairs` is a plain list
attribute, never a buffer); the only new attribute on `RootformerNRMT` is `apply_orbit_tying`.
`orbit_consistency_loss` keeps its signature. Consumers: `nrmt_train.py` (prints the pair count
and adds `lambda_orbit * l_orb`) now sees 4,305 pairs instead of 7,727 — intended, and the only
downstream behavioural change.

---

## 7. Checkpoint conflict — stated, not forced

* **`tie_embeddings` / `tie_root_embedding_tables` are shape-preserving** (rows overwritten in
  place). They invalidate nothing, but must run **after** `load_state_dict`; the reverse order
  silently undoes the tying. Nothing calls them automatically.
* **True storage sharing would break loading.** `OrbitTiedEmbedding` has `weight` of
  `n_orbits × d` plus an `orbit_index` buffer. Every released checkpoint stores root tables with
  the full vocabulary's row count, so it **cannot** load into that; the class is provided,
  documented as checkpoint-breaking, and wired nowhere.
* **The tables the released checkpoints actually carry are not the NRMT table.**
  `checkpoints/rootformer_v19_2_synthesis_ar_backbone.safetensors` has 24 tensors named
  `backbone.layers.<N>.self_attn.root_embed.weight`, shape **`[9015, 64]`** — while the current
  blueprint/vocabulary reports **9,114** roots. That is the known blueprint/checkpoint mismatch
  being handled by another agent; `tie_root_embedding_tables` therefore **skips** mismatched
  tables and reports them (`rows=9015 != vocab num_roots=9114 -- left untouched`) instead of
  resizing. Any id-indexed tying is only meaningful once the id remap is settled, so **I did not
  force it**, and I did not attempt to reconcile the 99-row difference.
* `models/unified_rootformer_v12.py` and `rootformer_v18_nrmp_model.py` were **not modified**:
  neither is where the shipped per-layer root tables are built (`grep root_embed
  models/unified_rootformer_v12.py` → nothing; the tables come from the IshtiqāqAttention
  module), and the audit needs no change there. No new default behaviour means no new
  checkpoint-loading risk. If tying is wanted in training, use `apply_orbit_tying()` (NRMT head)
  or `tie_root_embedding_tables(model, orbits)` (backbone) after loading.

---

## 8. Things I could NOT ground in a source / open items

1. **al-ʿAyn does not condemn لعم / ملع in the ع ل م chapter.** The chapter lists four
   مستعملات, but it has no فقط and no explicit مهمل group, so `khalil_attest_v4.json`'s
   deliberately conservative complement rule leaves لعم/ملع with **no verdict**. The default
   `membership='attested'` therefore excludes them for *want of positive attestation*, not
   because al-ʿAyn called them مهمل — which is why `orbit_of('علم')` has 4 members rather than 6.
   I did not weaken the record to make the story tidier. (Where al-ʿAyn **is** explicit — ع د ب —
   the exclusion is enforced: عدب is kept out.)
2. **The negative record is conservative by design.** Any root that al-ʿAyn never discussed is
   absent from both sets (2,168 such inventory roots at length 3). Those are excluded under the
   default policy and included under the loose one; both counts are reported.
3. **Ordering fidelity is partial.** The representative (first member) matches the chapter
   headword in the two chapters checked (ع ل م → علم, ع د ب → عبد), but the ordering of the
   *remaining* members is NOT claimed to reproduce al-ʿAyn's chapter order (his ع د ب order is
   عبد، دعب، بعد، بدع vs our عبد، دعب، بدع، بعد). Labelled ENGINEERING in the code.
4. **Hamza folding (`fold_hamza=True`) is ENGINEERING.** The lexicon writes the hamza radical as
   bare alif and `khalil_attest_v4.py` canonicalises the same way, so folding is required for the
   keys to meet — but it merges ء/ا spellings of one face (2 such merges, producing the two
   7-row "orbits" and one 8-row orbit in the loose policy). Face counts (≤ 6 for S³) are reported
   separately from row counts for exactly this reason.
5. **Duplicate implementations left alone (outside my assigned files).**
   `rootformer_v20_nrmp_model.py:73-97` carries the same inventory-only `_build_orbit_pairs` /
   `orbit_consistency_loss` pair-building logic and therefore still uses 7,727 non-attested pairs;
   `khalil_students_andalusian_master_engine.py:47` carries a second, weaker
   `KhalilPermutationOrbits` (all inventory permutations, no attestation, no tying API).
   Recommended follow-up for whoever owns those files: delegate both to `khalil_orbits`.
6. **`lengths=(4,)` / `(5,)` under the faithful policy yield no orbits** in this data (29
   quadriliteral attestations, 0 pentliteral): al-Khalīl's 24- and 120-face sentences are cited
   and supported by the code, but cannot be exercised without more attestation data.
