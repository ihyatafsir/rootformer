# Analyzer failure taxonomy and grammar-sourced fixes — NRMP morphemic tokenizer

**Baseline reproduced exactly on the published harness** (`eval_heldout_sample.json`: 92 files,
3,298,362 occurrences, 188,722 distinct forms):

| metric | value |
|---|---|
| type round-trip | **57.5884 %** (108,682 / 188,722) |
| occurrence round-trip | **82.1065 %** (2,708,170 / 3,298,362) |
| failing distinct forms | **80,040** (42.4116 % of types) |

**After the fixes: type round-trip 58.6667 % (110,717 / 188,722), occurrence 83.8204 %.**
`+1.0783 pp` type, `+1.7139 pp` occurrence, **2,035 types fixed, 0 types broken.**

Shipped revision under test: `nrmp_vocab.py` md5 `0ee0ab6301d7f26eddc7bd079f7fede7`,
`models/validated_segmentation.py` md5 `68d645a9ba0ddbfe4f586dfcc7fbe6ff`,
`models/morphemic_tokenizer_v12_arabic.py` md5 `a1185eafa89b20a98ade3ae58aa12ffb`,
`tasrif_engine.py` md5 `2a7a6aaf5398a5c52ade707c44119549`.
No shipped file was modified: the fix is a generated patched copy (below).

---

## 1. THE TAXONOMY (measured before any fix)

Every one of the 80,040 failing types is assigned exactly once by a first-match cascade of
mechanical tests. Two tests carry the weight, and both are *executions*, not string matches:

* **"realiser gap"** is proved by executing the project's own grammar engine:
  `tasrif_engine.TasrifEngine.generate(root, wazn)` reproduces the surface stem, while the
  shipped hand-written realiser does not.
* **"wrong root"** is proved by al-Khalil's calculation: `tasrif_engine.candidate_pairs(stem,
  [wazn])` exhibits a **different root that generates the surface under the same wazn**.

| cause | types | % of failing types | occurrences | % of failing occ | % of all 188,722 types |
|---|---|---|---|---|---|
| `CONTROL_root_slot_control_token` | 67,351 | 84.15 % | 426,547 | 72.27 % | 35.6879 % |
| `WRONG_WAZN_same_root_other_pattern` | 4,820 | 6.02 % | 42,850 | 7.26 % | 2.5540 % |
| `REALISER_wazn_unimplemented_engine_ok` | 2,630 | 3.29 % | 24,501 | 4.15 % | 1.3936 % |
| `WRONG_AFFIX_surface_letter_unaccounted` | 1,879 | 2.35 % | 11,029 | 1.87 % | 0.9956 % |
| `ANALYZER_no_wazn_assigned` | 1,744 | 2.18 % | 19,750 | 3.35 % | 0.9241 % |
| `ANALYZER_particle_overcapture` | 1,301 | 1.63 % | 18,110 | 3.07 % | 0.6894 % |
| `OTHER` | 234 | 0.29 % | 9,758 | 1.65 % | 0.1240 % |
| `CLITIC_MISSPLIT_slot_off_the_edge` | 25 | 0.03 % | 546 | 0.09 % | 0.0132 % |
| `PUNCTUATION_tokenisation_artefact` | 18 | 0.02 % | 18 | 0.00 % | 0.0095 % |
| `ANALYZER_wrong_root` | 16 | 0.02 % | 2,180 | 0.37 % | 0.0085 % |
| `DIVINE_passthrough_by_design` | 13 | 0.02 % | 34,880 | 5.91 % | 0.0069 % |
| `ORTHOGRAPHIC_hamza_or_weak_spelling` | 9 | 0.01 % | 23 | 0.00 % | 0.0048 % |
| **total** | **80,040** | 100 % | **590,192** | 100 % | **42.4116 %** |

### 1.1 Representability, split from fault

| stratum | types | % of 188,722 | occurrences |
|---|---|---|---|
| UNREPRESENTABLE BY CONSTRUCTION (`<DIVINE>` + control root slot) | 67,364 | 35.6948 % | 461,427 |
| **ANALYZER-ATTRIBUTABLE** | **12,676** | **6.7168 %** | **128,765** |

Round-trip charged to the analyzer: **93.2832 %** (176,046 / 188,722 representable types).

### 1.2 The headline finding: the control bucket is NOT closed-class material

`nrmp_vocab._closed_class_split` — the design's own route, against its own 265-surface authored
inventory — accepts **0 of the 67,351** control-slot failures. Every one is a word the design's
inventory *declines* and the analyzer returned no analysis for. The top members are ordinary
open-class words: `تعالى` (9,942), `صلى` (6,082), `الأول` (4,454), `أبو` (3,636), `ثنا` (3,503),
`آخر` (3,354), `قيل` (3,186), `معنى` (2,678), `جهة` (2,548).

So the bucket *is* unrepresentable by construction of the tuple (the surface is simply not
carried by any slot) and is excluded from the analyzer-charged rate above, as instructed — but
it is **not** "particle control by design". It is the analyzer's single largest defect, and the
one no realiser fix can touch. Anyone reading "84 % of failures are particle control" as
"unfixable by design" is reading it backwards.

### 1.3 What the brief said the diagnostic case was

`مكتوب` is **not a round-trip failure on this revision**. Its analysis is still wrong —
`base_tok.decompose_arabic_word('مكتوب')` returns `(None, 'كوب', 'مُفْتَعِل', None)`, root
**`كوب`** not **`كتب`** — but `مُفْتَعِل` applied to `كوب` yields `م ك ت و ب` = `مكتوب`, so the
realiser accidentally emits the right surface and the metric passes. The brief's deduction
("a realiser cannot fix a wrong decomposition") is confirmed; the test case simply cannot be
used as a round-trip canary, because this wrong decomposition is masked. The real `مكتوب`-class
defect therefore never appears in the round-trip failure list, and the measured taxonomy
understates it — recorded here rather than silently.

### 1.4 A definitional caveat on `wrong root` vs `wrong wazn`

The two labels are defined by *which knob reproduces the surface*, not by linguistic intent.
`أهل` is filed under `WRONG_WAZN` (4,820) because with root `هلل` held fixed, `أَفْعَل` regenerates
`أهل` — yet the linguistically intended root is `أهل`, so this is also a root error. Only 16 types
are `ANALYZER_wrong_root` because that label requires a different root to generate the surface
under the analyzer's **own** wazn. Anyone using these labels as "root errors" vs "pattern errors"
must apply this correction.

---

## 2. THE FIXES, BY CAUSE

Two fixes were landed, each gated on `AF_ANALYZER_FIX=1`, each measured alone, and each proved
not to regress anything (0 broken types on the full 188,722).

### 2.1 LARGEST FIXABLE ANALYZER CLASS: `REALISER_wazn_unimplemented_engine_ok` — 2,630 types

The shipped realiser `morphemic_tokenizer_v12_arabic.realize_root_and_wazn` is a hand-written
if/else chain. Read off the AST (only comparisons whose operand is the name `wazn`), it has a
branch for **exactly 46** of the 142 wazn — the same 46 the project's own harness prints
("realize implements 46 wazn names; vocab has 142"). Everything else falls through to `return r`
and the word renders as its bare root.

The dominant sub-class is the **muḍaʿʿaf in the mudariʿ**: `يكن` (root `كنن`), `يدل` (`دلل`),
`يصح` (`صحح`), `يجز` (`جزز`), `يحل` (`حلل`), `يزل` (`زلل`), `يظن` (`ظنن`), `يقل` (`قلل`),
`يبق` (`بقق`), `يحس` (`حسس`), `يخص` (`خصص`), `يحد` (`حدد`), `يعم` (`عمم`), `يخل` (`خلل`).
The chain emits `ي` + `دلل` = **`يدلل`**; the language has **`يدل`**.

**The rule implemented, quoted.**

> Ibn ʿUṣfūr al-Ishbīlī, *al-Mumtiʿ fī al-Taṣrīf*.
> Edition held: `corpus/andalusian/13_IbnUsfur_Mumtic_fi_al_Tasrif.txt`,
> md5 `b9e6d016f49f296e08d41600ca4ef0df`, **7,754 lines**.
> **Lines 6269–6280:**
>
> ```
> 6269 # فعلى هذا إذا اجتمع لك مثلان، وكان المثلان مما يمكن الإدغام فيهما، فلا
> 6270 ~~يخلو من أن يكون الثاني منهما متحركا أو ساكنا. ...
> 6275 # فإن اجتمعا في فعل فالإدغام ليس إلا. فإن كان الأول من المثلين ساكنا
> 6276 ~~أدغمته في الثاني من غير تغيير، نحو: ضرب وقطع. وإن كان الأول منهما متحركا
> 6277 ~~فإما أن يكون أولا في الكلمة أو غير أول.
> 6278 # فإن كان غير أول سكنته بحذف الحركة منه -إن كان ما قبله متحركا أو ساكنا
> 6279 ~~هو حرف مد ولين- أو بنقلها إلى ما قبله، إن كان ساكنا غير حرف مد ولين.
> 6280 ~~وحينئذ تدغم، نحو: رد واحمر واستقر واحمار.
> ```

Applied: root `دلل` in `يَفْعُلُ` — the two lāms are "in a verb", the first is not word-initial
(6275–6277), so it is made sākin (6278–6279) and **idghām follows obligatorily** (6280):
`يَدْلُلْ` → `يَدُلّ` → surface skeleton `يدل`.

The mechanism is the project's **own** `tasrif_engine.TasrifEngine` (its `Idgham` operation) —
the module that implements al-Mumtiʿ. No new heuristic was written.

**Scope proved by measurement, not assumption.** Delegating on *any* reading the engine can
generate regressed **3,428** types: the engine's iʿlāl applies *verb* morphology to *nominal*
wazn labels (`قول` + `فَعَلَ` → `قال`, `رضي` + `فَعَلَ` → `رضى`, `يوم` → `يام`, `كون` → `كان`).
The landed gate fires only when (a) the root is muḍaʿʿaf **and** the wazn is a mudariʿ pattern —
the case 6269–6280 governs — **or** (b) the wazn is one of the 46 the chain has no branch for at
all; and in both cases only when the chain produced no realisation (its output is the bare root)
and the engine returns a different, non-empty skeleton.

### 2.2 `<DIVINE>` — surface-preserving passthrough (product decision)

The divine name and its clitic-built forms round-trip **byte-for-byte by construction**: the
exact codepoints are carried in the tuple and handed back unchanged. No root, no wazn, no affix
split, no normalisation, no letter substitution, no ligature folding. The hard-coded set is the
set *observed* in the sample (`divine_scan.py`), not guessed:

`الله، والله، لله، بالله، اللهم، وبالله، ولله، فالله، فوالله، فلله، فبالله، للله، تالله، ألله،
آلله، إلله، بألله، هوالله، إلابالله، ماشاءالله، نحمدالله، عبدالله، وعبدالله، عيدالله` plus the
wasla spelling `ٱللّٰه`.

Deliberately **excluded** look-alikes the scan also returned, which are *not* the divine name:
`اللهو` / `اللهب` / `اللهيب` / `اللهج` (roots لهو/لهب/لهج), `اللهاة` (uvula), `علله` / `خلله` /
`يتخلله` / `قلله` / `ذللته` / `يضلله`.

**Sourced support for the lexical (not derivational) status of the surface:**

* Ibn Manẓūr, *Lisān al-ʿArab*, edition held: `corpus/basran/IbnManzur_Lisan_al_Arab.Shamela0001687.txt`,
  md5 `998201d7b4e1361e256b88979a9e72df`, **line 262334**: «والله: أصله إلاه، على فعال بمعنى …».
* al-Azharī, *Tahdhīb al-Lughah*, `corpus/basran/Al_Azhari_Tahdhib_Al_Lughah.txt`,
  md5 `61cb362c8564a6df337af8429d9d53f2`, **line 59713**: «قال أبو الهيثم: فالله أصله إلاه».
* al-Mubarrad, *al-Muqtaḍab*, `corpus/basran/Al_Mubarrad_Al_Muqtadab.txt`,
  md5 `fdcacbaf5052ac4522a84d088356a1f7`, **lines 20208–20209**: «وزعم أن مثله اللهم إنما
  الميم المشددة في آخره عوض عن يا التي للتنبيه» — the vocative carries a substitute mīm, i.e.
  the surface is lexical.

**Honest labelling.** I did **not** find a held text that states this word's alif elision with the
phrase «لكثرة الاستعمال». That lemma occurs in the held corpus (`08_IbnSayyidih_Al_Muhkam.txt:28422`,
`Ibn_Jinni_Al_Khasais.txt:3894`, `Al_Mubarrad_Al_Muqtadab.txt:20357`) but for *other* elisions.
The «irregularity by frequency of use» framing is therefore recorded as the **parent's product
decision**, not as a source I verified. What the sources above *do* establish is the part the fix
needs: the surface is a lexical fact, not a (P,R,W,S) derivation.

### 2.3 Negative results, attempted before being reported

* **al-Zajjājī, *al-Īḍāḥ fī ʿilal al-Naḥw*** — project copy, **1,701 lines**,
  md5 `663e493f1d2e7a3223dcc8c6b702e7d0`. A search for `دغم|دغام` over the whole file returns
  **0 hits**. The check is sound, not empty-by-accident: the file is complete (its colophon is
  present, dated 617 AH) and does contain ʿilal vocabulary (`العلة` 13, `القياس` 11, `الإعراب` 99).
  The held recension is the ʿilal al-naḥw portion and does not carry the idghām chapter. Edition
  named per the multi-edition rule: the brief's OpenITI al-Īḍāḥ (1,864 lines) is a **different
  edition** and would need the +92/+142/+148 offset applied; it was not used here.
* **al-Zajjājī, *al-Ibdāl*** — fetched from OpenITI `master`,
  `data/0337IbnIshaqZajjaji/0337IbnIshaqZajjaji.Ibdal/0337IbnIshaqZajjaji.Ibdal.Kraken220410202747-ara1`
  (Kraken OCR, 244,017 bytes). Search for `دغام|دغم` returns **0 hits**; it is a substitution
  (ibdāl) inventory organised by letter pair and does not carry the idghām rule. Its OCR is also
  visibly noisy. So the fix rests on al-Mumtiʿ, which is the taṣrīf text and the right locus.

---

## 3. EVERY NUMBER, ON THE SAME 188,722-TYPE HARNESS

`af_eval.py` scores all 188,722 published types with a vocab module loaded from a chosen path.

| stage | flag | type round-trip | occurrence round-trip | types fixed | types broken | changed tuples |
|---|---|---|---|---|---|---|
| shipped baseline | — | **57.5884 %** | **82.1065 %** | — | — | — |
| patched, flag **OFF** | `AF_ANALYZER_FIX` unset | **57.5884 %** | **82.1065 %** | **0** | **0** | **0** |
| FIX 1 only (`<DIVINE>`) | `AF_ANALYZER_FIX=1 AF_NO_TASRIF=1` | **57.5985 %** | **83.2090 %** | 19 | 0 | 24 |
| FIX 1 + FIX 2 | `AF_ANALYZER_FIX=1` | **58.6667 %** | **83.8204 %** | **2,035** | **0** | 4,372 |

Split of the 2,035 fixes: **19 `<DIVINE>`** (36,365 occurrences) + **2,016 tasrif delegation**
(20,166 occurrences). A further 2,337 tuples changed and still fail.

`DIVINE` is only 19 types but +1.10 pp of occurrence round-trip, because `الله` alone is 31,938
occurrences in the sample.

### 3.1 Inertness proof (flag-off)

Byte-identical un-patch, asserted in `deploy_af.py --verify`:

```
[*] src          /workspace/hf_v19_2_release/nrmp_vocab.py  md5 0ee0ab6301d7f26eddc7bd079f7fede7
[*] patched md5  e245a11361ec7cbb49afb5a9df7e578e
[*] un-patch == original : True  (BYTE-IDENTICAL)
[*] AST has def _af_tasrif_realize      : True
[*] AST has CALL self._af_tasrif_realize: True
[*] AST has def _af_engine              : True
```

The AST assertions exist because a check that passes on *presence* is worthless: the helper is
asserted to be a real `ast.FunctionDef` **and** to be reached by a real call site
(`CALL._af_tasrif_realize`), not emitted as a comment.

Behavioural inertness is separately proved: with the flag unset the patched module produces
`0 changed tuples`, `0 FIXED`, `0 BROKEN` against the shipped baseline over all 188,722 types,
and `num_roots` stays 9,490.

### 3.2 The seven suites

Run from `/workspace/hf_v19_2_release`, true exit codes captured (no pipe swallowing `$?`).
Logs in `analyzer_fix/suite_<name>.log`.

| suite | expected | measured | exit |
|---|---|---|---|
| `test_grammar_impl.py` | 20/20 | **20/20, 0 FAIL** | 0 |
| `verify_v2.py` | pass | **ALL v2 CHECKS PASS** | 0 |
| `andalusian_realizer.py` | 67/67 | **67/67, 0 FAIL** | 0 |
| `test_sibawayh_governor.py` | 15/15 | **15/15** | 0 |
| `ibn_malik_automaton.py` | 46/46 | **46/46** | 0 |
| `test_khalil_orbits.py` | 30/30 | **30/30, 0 FAIL** | 0 |
| `test_awzan_order.py` | 24/24 | **20/22, 2 FAIL** | **1** |

The two `test_awzan_order.py` failures are **pre-existing and environmental**: both are
`archived 130-awzan vocabulary is reachable` and its dependent check, and the archive path
`/workspace/rootformer_v12/v18_next_root_morph/data/nrmp_vocab.json` does not exist — the whole
`/workspace/rootformer_v12` tree is absent from the pod. Nothing in this work touches that path.
Reported rather than suppressed; the suite exits **1**, and that exit code is the honest one.

---

## 4. WHAT COULD NOT BE FIXED, AND WHY

| class | types | why not fixed |
|---|---|---|
| `CONTROL_root_slot_control_token` | 67,351 | Unrepresentable by construction — no slot carries the surface. Excluded from the analyzer-charged rate per the brief. Fixing it needs a real inverse analysis (al-Khalil's calculation over the wazn inventory at encode time), which is an analyzer rewrite, not a realiser patch. Measured 0/67,351 are accepted by the design's own closed-class route, so this is not "by design". |
| `WRONG_WAZN_same_root_other_pattern` | 4,820 | The engine's iʿlāl over-fires on nominal pattern labels; a blanket fix regressed 3,428 types (measured). Not attempted under the no-regression rule. |
| `WRONG_AFFIX_surface_letter_unaccounted` | 1,879 | Needs analyzer-side slot repair (e.g. `يستعمل`: the mudariʿ `ي` is carried by no slot). Unsourced for a single rule; not attempted. |
| `ANALYZER_no_wazn_assigned` | 1,744 | Mithāl verbs (`يجب`/`وجب`, `يقع`/`وقع`, `يرد`/`ورد`): the wazn slot is `<UNK>`, so nothing is realised at all. The rule needed is the elision of the weak first radical in the mudariʿ. **Not implemented** — I did not open a text for it in this run, so it is not claimed as sourced. |
| `ANALYZER_particle_overcapture` | 1,301 | A real word collapses onto a closed particle (`لكان` → `<P:لكن>`, `تبين` → `<P:بين>`). Fixing it means loosening a guard on the 66 historical particle ids that the module's own comment says must keep their established behaviour. Not attempted. |
| `OTHER` + `ORTHOGRAPHIC` + `CLITIC_MISSPLIT` + `PUNCTUATION` | 286 | Mixed residue; listed per-type in `taxonomy_rows.jsonl`. |
| 2,337 neutral changes | — | Tuples changed by FIX 2 but still failing (mostly the wrong root behind them). |

**Convention vs insight.** FIX 2 is a *grammar rule* (idghām, sourced) and it repairs the
**surface**. It does **not** correct the analysis: `يكن` now round-trips via root `كنن` +
`يَفْعُلُ` + idghām, whereas the historically correct reading of `يكن` is `كون` + jussive
(`majzūm`). The round-trip is genuinely fixed; the root label is still debatable. No part of this
work is a convention recovered from data and presented as morphological insight.

---

## 5. SHIPPING CAVEAT — read before deploying

The `<DIVINE>` passthrough appends **25** `‹DIV:…›` root ids, so with the flag on
`num_roots` goes **9,490 → 9,515**. The append moves no existing id (the module's own
append-only rule), but `num_roots` *is* the checkpoint contract — `test_khalil_orbits.py`
already reports `rows=9015 != vocab num_roots=9490 … left untouched`. Shipping this needs either
a blueprint/vocab regeneration at the next checkpoint boundary, or the ids reserved in the
persistent id space. **Flag off, `num_roots` is unchanged at 9,490.**

---

## 6. ARTIFACTS

Pod `/workspace/analyzer_fix/` ⇄ local `rootformer/build/analyzer_fix/`:

| file | what it is |
|---|---|
| `dump_failures.py` | baseline harness; reproduces 57.5884 % / 82.1065 % / 80,040 exactly |
| `failures.jsonl` | the complete failing set, 80,040 lines: word, count, P/R/W/S, decode |
| `taxonomy2.py`, `taxonomy.md`, `taxonomy.json`, `taxonomy_rows.jsonl` | the taxonomy and its per-type assignment |
| `probe1.py`, `probe3.py`, `probe4.py`, `probe5.py`, `probe7.py`, `probe1.log` | the shape probes that produced §1.2–§1.4 |
| `divine_scan.py` | the observed divine surface set (with counts) |
| `deploy_af.py` | patch generator + un-patch byte-identity + AST call-site assertions |
| `patched/nrmp_vocab.py` | the patched module (md5 `e245a11361ec7cbb49afb5a9df7e578e`) |
| `af_eval.py` | A/B harness over all 188,722 types, with `--diff` (FIXED/BROKEN counts) |
| `base.jsonl`, `off.jsonl`, `on_divine.jsonl`, `on.jsonl` | the four dumps behind §3 |
| `split_fixed.py` | the 2,035-fix split |
| `suite_*.log` | the seven suites, full output |
| `MANIFEST.md` | md5 + size of every mirrored artifact |

Reproduce:

```bash
cd /workspace/analyzer_fix
ROOTFORMER_VALIDATED_SEG=1 python dump_failures.py --dump failures.jsonl   # 57.5884 %
python taxonomy2.py                                                        # §1 table
python deploy_af.py --src /workspace/hf_v19_2_release/nrmp_vocab.py \
                    --out patched/nrmp_vocab.py
ROOTFORMER_VALIDATED_SEG=1 python af_eval.py --vocab patched/nrmp_vocab.py \
                    --dump off.jsonl                                       # inert
AF_ANALYZER_FIX=1 ROOTFORMER_VALIDATED_SEG=1 python af_eval.py \
                    --vocab patched/nrmp_vocab.py --dump on.jsonl          # 58.6667 %
python af_eval.py --diff base.jsonl on.jsonl                               # 2035 fixed, 0 broken
```
