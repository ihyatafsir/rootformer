# Round-trip fix — NRMP morphemic tokenizer (Bug 1 control tokens, Bug 2 al-qalb)

**Owner of the change:** subagent `session-a4f1047e` (per-bug round-trip repair).
**Single file modified:** `/workspace/hf_v19_2_release/nrmp_vocab.py`
`7a3f3a57f27a3a206d3e9deca7de7c3e` → `3df966a63c1dcee2ea53f3741a3b1eb0`
Backup kept: `/workspace/hf_v19_2_release/nrmp_vocab.py.bak.rtfix-20261001-191450` (pristine).

No other release file was touched. Diff: `nrmp_vocab_roundtrip_fix.diff` (318 lines).

---

## 1. Headline result (mode 1, the shipped default)

Sample unchanged: 3,298,362 forms / 188,722 distinct, 92 held-out files (cached scan reused).

| metric | before | after | Δ |
|---|---|---|---|
| **round-trip, token level** | 73.8610 % | **80.7170 %** | **+6.8560 pp** |
| **round-trip, distinct forms** | 55.9611 % | **56.4423 %** | **+0.4812 pp** |
| failing distinct forms | 83,111 | 82,203 | **−908** |
| UNK (any slot) | 4.1936 % | 1.2229 % | −2.9707 pp |
| morphemic coverage | 48.4880 % | 48.4296 % | −0.0584 pp |
| root attestation (al-ʿAyn) | 69.0459 % | 69.8522 % | +0.8063 pp |
| root muhmal (condemned) | 0.7619 % | 0.7794 % | +0.0175 pp |
| collapse to `<PARTICLE>` | 46.0563 % | 50.1547 % | +4.0984 pp (reclassification, see §6) |
| forms given a real root | 1,661,045 | 1,623,862 | −37,183 (see §6) |

**226,134 occurrences newly pass. 908 distinct forms newly pass. 0 newly broken** — verified
over the union of all keys, not just the intersection.

All three modes improve; the fix is mode-independent (it lives in the shared `encode_word` /
`decode_word` path, not in `validated_segmentation`):

| mode | token before → after | type before → after |
|---|---|---|
| 0 greedy | 73.2445 % → 80.6393 % (+7.39 pp) | 56.0941 % → 56.4290 % (+0.33 pp) |
| 1 default | 73.8610 % → 80.7170 % (+6.86 pp) | 55.9611 % → 56.4423 % (+0.48 pp) |
| 2 root-reseg | 71.1940 % → 78.0985 % (+6.90 pp) | 54.3196 % → 54.8140 % (+0.49 pp) |

Mode 2 still must not ship (78.10 % < 80.72 %); unchanged conclusion.

---

## 2. Per-bug / per-mechanism attribution

Measured by deploying three intermediate builds and re-running the harness on the identical
cached sample, then pairing each distinct form's fate across builds.

| build | what it adds | token rt | type rt | newly fixed | newly broken |
|---|---|---|---|---|---|
| baseline | shipped default | 73.8610 % | 55.9611 % | — | — |
| **B** | **Bug 2 only**: al-qalb generation + hard control-token invariant | 74.6625 % | 55.9659 % | 26,437 occ / 9 forms | 0 |
| **C** | **Bug 1**: closed-class inventory + 3 encoder recovery paths + root/wazn collision guard | 79.8398 % | 56.3983 % | cumulative 197,202 occ / 825 forms | 0 |
| **D** | **Bug 1 cont.**: في/فى and ي/ى orthographic closure | 80.7170 % | 56.4423 % | cumulative 226,134 occ / 908 forms | 0 |

Paired transitions over the same 188,722 forms:

```
baseline -> B        ok->ok 105611   ok->FAIL      0   FAIL->ok      9   FAIL->FAIL 83102   NET +26437
B        -> C        ok->ok 105620   ok->FAIL      0   FAIL->ok    816   FAIL->FAIL 82286   NET +170765
C        -> D        ok->ok 106436   ok->FAIL      0   FAIL->ok     83   FAIL->FAIL 82203   NET +28932
```

Expressing the two requested bugs separately:

* **Bug 1 (control tokens as literal text)** = C+D less the al-qalb part =
  **+6.0545 pp token / +0.4764 pp type** = **199,697 occurrences / 899 distinct forms**.
  The literal-tag invariant on its own is worth **0 pp** — see §4.
* **Bug 2 (al-qalb contraction)** = **+0.8015 pp token / +0.0048 pp type** =
  **26,437 occurrences / only 9 distinct forms** (عليه، إليه، عليها، إليها، علينا، فعليه،
  وعليه، وعلينا، فعلينا). Negligible at type level; the #2 failure by corpus volume.
* Check: 199,697 + 26,437 = 226,134 occ = +6.8560 pp ✓; 899 + 9 = 908 forms ✓.

---

## 3. What was actually wrong

### 3.1 Bug 1 is not one bug, and it is not decode-side

665,930 occurrences (20.19 % of the corpus) decoded with a literal control token. Breakdown:

| mechanism | occurrences | distinct | fix |
|---|---|---|---|
| particle **is** in the project's authored closed inventory but has no `<P:…>` id | 75,603 | 314 | append the missing closed surfaces to the root vocab |
| validator's closed-class rule **never reached** — the greedy analyser returned no root, and `decompose()` bails out before `_closed_split` runs | 547,717 | 70,109 | run the same decomposition from `nrmp_vocab` when the analyser yields nothing |
| analyser's **slots** wrong: closed-class host in PREFIX, clitic in ROOT (`له` → ل+ه) | 16,654 | 9 | re-route to `<P:ل>`/`<P:له>` + suffix |
| single letters / stray markers (`و ب ا ج ل ه ص د ع ح`) | 25,955 | 45 | recovered only where the letter is an authored word (`و`); rest not recoverable |

Root cause of the largest slice — `models/validated_segmentation.py::decompose()`:

```python
bp, br, bw, bs = base
if not br or (br and bw is None and bp is None and bs is None):
    return base          # <-- returns BEFORE _closed_split() is ever called
...
closed = self._closed_split(clean)
```

The closed-class rule therefore never sees exactly the words it exists to catch. `ولو`, `بأن`,
`وإذا`, `وأما`, `الى` are all decomposable by `_closed_split` but were discarded as a bare
`<PARTICLE>`. **The validator file was not modified** — the recovery is implemented inside
`nrmp_vocab.encode_word` (new helper `_closed_class_split`), partly to avoid colliding with the
other agents and partly because `nrmp_vocab` is where the surface is actually lost.

### 3.2 Bug 2 — al-qalb, generation direction

Implemented as the inverse of the existing analysis direction in
`models/validated_segmentation.py::_closed_split` (`core[:-1] + 'ى' in self._closed`).
The alif is **turned into yāʾ**, not deleted, and only for the particles the sources name —
Ibn ʿUsfūr «عليه وإليه ولديه», al-Zajjajī «لدى زيد ولديك», Abū Ḥayyān «عليك ولديك»:

```python
AL_QALB_PARTICLES = frozenset({'على', 'إلى', 'الى', 'لدى', 'علي', 'إلي', 'لدي'})
```

Applied only when the suffix is a pronominal clitic (Sibawayh's own enumeration). **Not**
applied to `ما` (ماه), `لا`, `هذا`, `كذا`, `متى` (متاك) — the task's warning is honoured
explicitly. `الى` is included because it is the bare-alif spelling of `إلى` (recovering
`اليه` 439, `اليها`, `الينا`, `اليهم`).

### 3.3 A latent pre-existing bug exposed, then contained

`encode_word` had an unguarded check on the analyser's **root**:

```python
particle_cand = f'<P:{r}>'
if particle_cand in self.root2id:
    r_id = self.root2id[particle_cand]     # throws the wazn away
```

It only looked safe because no `COMMON_PARTICLES` surface happened to equal a root the analyser
returns for another word. Expanding the inventory exposed the collision: `خلاف` analyses as root
`خلف` + `فِعَال`, and `خلف` is also a closed adverb, so the particle branch discarded the wazn
and decoded `خلاف` → `خلف` (947 occ), plus `اختلاف` 665, `بخلاف` 947, `مختلفة` 578, `يختلف` 322,
`مخالفة` 239, `الخلاف` 442, and `معناه` → `عنه` 598 — **13,779 occurrences / 960 forms newly
broken** in the first draft.

Guard added: *a root with a wazn is a root, not a particle*, applied **only to the ids this
patch appends**, so the 66 historical particle ids keep their established behaviour
byte-for-byte. After the guard: **0 newly broken**.

---

## 4. The decode-side invariant, stated plainly

`decode_word` no longer has any path that can write a control tag into text:

```python
if root_str in ['<UNK>', '<PARTICLE>', '<PAD>']:
    return p_text + s_text          # was: p_text + root_str + s_text
```

Verified over all 188,722 distinct forms: **0 decoded strings contain `<UNK>`, `<PARTICLE>`,
`<PAD>`, `<BOS>`, `<EOS>` or `<NONE>`** (was 665,930 occurrences).

Honest reading: the invariant is a hard requirement and it is met, but on its own it is worth
**0 pp** of round trip. For a word that reaches `<PARTICLE>`/`<UNK>` the surface was destroyed
at *encode* time and is simply not in the 4-tuple; no decoder can restore it. The fix therefore
had to make the encoder stop destroying it.

This is the main correction to the stated hypothesis: Bug 1's *symptom* is decode-side, but its
*volume* is encode-side information loss.

---

## 5. What could NOT be recovered, and why

Residual after the fix: **636,024 occurrences (19.283 %), 82,203 distinct forms.**

| bucket | occurrences | types | why not recoverable |
|---|---|---|---|
| unanalysed, no reading at all (`<PARTICLE>`) | 450,733 (13.665 %) | 69,509 | analyser coverage hole; not closed-class, so registering them would be corpus memorisation. تعالى 9,942 · صلى 6,082 · وسلم 5,680 · الأول 4,454 · أبو 3,636 · ثنا 3,503 · آخر 3,354 · قيل 3,186 · معنى 2,678 · جهة 2,548 · شيئا 2,256 · الآخر 2,214 |
| root+wazn regenerates the wrong surface | 114,449 (3.470 %) | 11,339 | realisation/morphology bugs: أبي→بيي 5,477 · يكن→كنن 3,998 · يجب→وجب 2,832 · أهل→هلل 2,745 · النبي→البيي 2,369 · الأرض→الرضض 1,847 · يدل→دلل 1,823 · أبيه→بييه 1,418 · يقع→وقع 1,204 · يصح→صحح 1,096 |
| **`الله` — excluded by owner instruction** | **31,938 (0.968 %)** | 1 | reported as outstanding, not touched |
| root slot `<UNK>`, single letters | 20,215 (0.613 %) | 44 | `ب ا ج ل ه ص د ع ح م آ ال` — editorial markers / letter names, not words |
| closed-class mis-split (pre-existing latent bug, deliberately not fixed) | 18,689 (0.567 %) | 1,310 | the analyser returns a `COMMON_PARTICLES` surface as the root of a **real** word: لكان→لكن 1,094 · تبين→بين 913 · بيان→بين 885 · يقبل→قبل 678 · يبين→بين 518 · البيان→البين 409 · قبول→قبل 340 · لكونه→لكنه 270. Same collision as §3.3 but on the 66 historical ids; fixing it changes established behaviour, so it is reported, not touched |

Also unrecoverable by rule, with counts: `منا` (312 — `من`+`نا` loses a letter to idghām),
`وسلم` (5,680 — و+سلم, `سلم` is not closed-class), `إلف`, and the bare-preposition-alone forms
the project deliberately refuses (`ب` 2,531, `ل` 1,382 …).

**Deliberately not done — the tautology trap.** A self-registering atomic surface token
(`<P:whatever the encoder is shown>`) would push round trip toward ~100 % immediately, and it is
the mechanism the task's hint points at. It was rejected because the held-out sample would then
be measuring memorisation of itself — the exact circularity `eval_heldout.py` was written to
prevent. The inventory actually shipped is authored from grammar the project already carries
(`CLOSED_PARTICLES`, `COMPOUND_PARTICLES`, `SUPPLEMENT_CLOSED`, the blueprint's Sibawayhian
particle partition, the 4×12 preposition×clitic grid, plus one documented orthographic closure)
and never read off a corpus. A **non-circular** alternative does exist: build the surface lexicon
offline from the **training-side** `kathra_counts.json`, which excludes all four held-out corpora
(measured overlap 94–98 % at form level under hamza folding, so exact-surface recovery would be
lower). That is a vocabulary-design decision for the parent, not a bug fix.

---

## 6. Cost of the fix

* **Vocabulary grows 9,114 → 9,313 roots (+199, +2.2 %)**, appended **after** `raw_roots`.
  Appending is the only insertion that cannot move an existing root id; every one of the 9,114
  pre-existing ids is unchanged by construction. Verified deterministic across cwd/module
  resolution (`num_roots = 9313`, `appended = 199` in three separate working directories).
* **Trained checkpoints hold 9,015-row / 9,114-id root tables**, so the 199 new ids have no
  trained embedding and a trained model cannot *predict* them. The tokenizer round trip improves
  regardless (it is tokenizer-only); realising the gain end-to-end needs the root embedding/head
  grown and retrained, or the new surfaces routed to a separate output. Stale hardcodes to
  update: `per_stream_eval.py:130` (`'r': 9114`), `neural_transmuter_head.py:73`
  (`num_roots: int = 9114`), `rootformer_analyzer_v2.py:8` (comment). `verify_hardcoded_sites.py`
  still passes 10/10 because it only asserts `V['w'] == num_awzan`, not `V['r']`.
* **Morphological reclassification, not a trade in round trip.** 37,183 occurrences / 189 forms
  moved `REAL_ROOT → PARTICLE`; **all 37,183 still round-trip**. Inspection shows they are
  *spurious* roots the findings doc already identified — بها 4,588, لما 4,290, لها 4,049,
  دون 2,325, علي←root علل 1,884, اذا 1,808, واما 1,619, الا 1,586, لنا 1,057, كذا 989, حين 746.
  This is why `root_forms` falls and root attestation *rises* (+0.81 pp); `morphemic_rate` is
  flat (−0.058 pp ≈ 1,927 occurrences).
* **`collapse to <PARTICLE>` +4.10 pp is not comparable to the baseline row.** The harness
  defines `ROOT_PARTICLE` as `<P:…>` **or** `<PARTICLE>`, so words that were `<UNK>`-root are now
  counted as particles (UNK −2.97 pp) and the row moves by construction. It is not a quality
  loss; the same rows are broken out honestly in §5.
* **Bug 2 has almost no type-level effect** (9 forms). Its value is 26,437 occurrences of the
  commonest contamination words; anyone optimising for the type-level number should not expect
  this bug to move it.

---

## 7. Seven suites — no regressions

Run from `/workspace/hf_v19_2_release` against both the pristine and the final patched release.

| suite | pristine | patched (final) |
|---|---|---|
| `test_grammar_impl.py` | 20/20 | **20/20** |
| `verify_v2.py` | ALL PASS (rc 0) | **ALL PASS (rc 0)** |
| `andalusian_realizer.py` | 67/67 | **67/67** |
| `test_sibawayh_governor.py` | 15/15 | **15/15** |
| `ibn_malik_automaton.py` | 46/46 | **46/46** |
| `test_khalil_orbits.py` | 30/30 | **30/30** |
| `test_awzan_order.py` | 24/24 | **24/24** |
| `verify_hardcoded_sites.py` (extra) | 10/10 | **10/10** |

`test_khalil_orbits.py` changes only its report line (`rows=9015 != vocab num_roots=9114` →
`9313`); it reports the mismatch and never resizes, so it still passes.

---

## 8. Concurrency

* `nrmp_vocab.py` md5 was re-checked before **every** write and matched the value last read.
  **No collision occurred** on this file, and no other agent's backup appeared for it
  (`ls` shows only `nrmp_vocab.py.bak.awzan-appendonly` from Sep 30 and mine).
* During the pristine-vs-patched suite comparison the release held the pristine file for ≈3 min
  and was then put back. Final state is the patched file, md5
  `3df966a63c1dcee2ea53f3741a3b1eb0`.
* The measurement harness was **copied, not modified**: `/workspace/rt_fix/eval_heldout_rtfix.py`
  writes to `/workspace/rt_fix/results/…`, so `/workspace/eval_heldout_results.json` and
  `EVAL_HELDOUT_REPORT.md` are untouched. Pristine results preserved at
  `/workspace/rt_fix/results/eval_heldout_results.PRISTINE.json`.
* Other agents were active throughout (`nrmp_train.py`, `nrmt_train.py`,
  `farahidian_transmutation_engine.py`, many files at 17:14). None of them touched
  `nrmp_vocab.py`. Note `nrmp_train.py`/`nrmt_train.py` consume `vocab.num_roots`; the +199 rows
  will change their head shape — see §6.

---

## 9. Is the 26 % a small number of mechanical bugs, or something deeper?

**Both, and the split is now measurable.**

* It is **not** diffuse morphological inadequacy. **6.86 pp of the 26.14 pp was mechanical and is
  now fixed**: a closed-class vocabulary that only covered 66 of ~237 authored surfaces; a
  closed-class rule that was structurally unreachable because the analyser bailed out first; an
  al-qalb rule implemented in one direction only; an orthographic variant (`فى`) with no id.
  Those are exactly the bugs the hypothesis predicted, and they came with **0** offsetting
  breakages once the root/particle collision was guarded.
* But the largest single slice — **13.665 pp of the corpus, 450,733 occurrences, 69,509 distinct
  forms — is the analyser having no reading for the word at all**, and that is deeper: encode-time
  information loss no decoder can undo. The framing "both DECODE-side" is wrong for this slice;
  the decode bug was only its visible symptom.
* A further **3.470 pp** is realisation error (right root, wrong surface), **0.567 pp** is the same
  root/particle collision still live on the 66 historical particle ids, and **0.968 pp** is `الله`,
  excluded by instruction.

Summary: **the 26 % is roughly a quarter mechanical vocabulary/rule bugs (now fixed, +6.86 pp)
and three quarters analyser coverage and realisation depth (not fixed here).** Round trip cannot
approach ~100 % without substantially better morphology or an atomic surface lexicon — and the
latter trades morphology for memorisation.
