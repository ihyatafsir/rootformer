# Fidelity audit — `lisan_3pillar_master_roots.jsonl` (9,015 root records)

**Artefact:** `/workspace/rootformer_v12/v18_next_root_morph/data/lisan_3pillar_master_roots.jsonl`
**Standard applied:** a claim counts as grounded only if the source text can be produced verbatim, with a file and a line.
**Date of run:** see `coverage_census.json` / `grounding_global.json` timestamps. All scripts live in `/workspace/raghib_audit/`.

---

## 0. Summary of findings

| question | answer |
|---|---|
| How many of the 9,015 roots does al-Rāghib actually treat? | **1,295 (14.4 %)**; 7,720 (85.6 %) have no al-Mufradāt entry |
| Of roots where he *does* have an entry, is the lexicon faithful? | **82.5 % AGREES, 14.3 % PARTIAL, 3.2 % WRONG** (n=63) — but it paraphrases: only 46.8 % carry a ≥25-char verbatim run of him |
| Of roots where he does not, can the claims be checked? | 27.3 % of the sample is checkable against **no** held source; 36.3 % of Asās citations and 52.0 % of "Aṣl al-Wadʿ" citations point at roots the named source never treats |
| Are the translations faithful? | The `translation` fields **yes** (10/10 sampled); the `anchors` fields are **not translations** but unverified synthesized attributions |
| Is the Arabic real classical text? | **Yes** — 41.8 % carry a ≥25-char verbatim run in a held lexicon (76.0 % at ≥15 chars) |
| Is it attributed to the right book? | **No.** Named: Maqāyīs (53 % of records, supplies 0.8 %), Asās (claimed 34 %, supplies 3.3 %). Actual top sources: Tahdhīb al-Lugha (17.2 %) and al-Ṣiḥāḥ (7.2 %) — **never named at all** |
| Verdict | **Not authority.** Usable only as a hint/index. The Arabic is often recoverable; the citations and English glosses are not trustworthy. **Rebuild from the primary texts.** |

---

## 0b. What the artefact actually is

Six fields per root: `root`, `ontological_core`, `haqiqah_literal`, `majaz_scholastic`,
`derivations`, `canonical_definition{arabic,english}`. It is **English prose that names classical
authorities**, not a transcription. Provenance note: its `ontological_core` text is byte-identical
to the `core` field of `data/farahidian_3pillar_lexicon_fast.json` (also 9,015 keys) — the same
generated pipeline under a different file name.

---

## 1. Coverage census (exhaustive — every one of 9,015 roots, not a sample)

`Raghib_Al_Mufradat.txt` was parsed into **1,573 entry headers**.
Validation that the parse is complete: all 28 `||| كتاب X` chapter boundaries are present; probe
roots (`بغت`, `قول`, `علم`, `كتب`, `ضرب`, `رحم`, `أله`) all resolve to the correct entry; no
multi-word header was missed.

| | roots | share |
|---|---|---|
| **Has an al-Mufradāt entry** | **1,295** | **14.36 %** |
| — exact root form | 1,274 | 14.13 % |
| — 2-letter surface form of a 3-radical weak root (e.g. `أب` for أبو) | 21 | 0.23 % |
| **Has NO al-Mufradāt entry** | **7,720** | **85.64 %** |

By lexicon root length: 3-letter **1,277 / 6,258 = 20.4 %**; 4-letter **16 / 2,547 = 0.6 %**;
5-letter **0 / 197 = 0.0 %** (2,744 of the 9,015 rows are 4–5 letter strings that are not
al-Rāghib-style triliteral roots at all).

> **This single number bounds everything the lexicon can legitimately claim from al-Rāghib: 14.4 %.**

A looser "drop all weak letters" matching rule was tested and **rejected** — it collapses genuinely
distinct roots (`بوغ`→`بغي`, `جور`→`أجر`). The 1,295 figure uses only exact forms plus the
defensible 2-letter weak-root rule.

---

## 2. Fidelity sample — 260 roots, stratified, seeded

**Seed `20241017`.** Strata = (al-Mufradāt entry? × Asās al-Balāghah entry?), which is the only
frequency-like signal available (no root-frequency table exists on the pod).

| stratum | population | sampled |
|---|---|---|
| MA (Mufradāt + Asās) | 1,155 | 33 |
| Ma (Mufradāt only) | 140 | 30 |
| mA (Asās only) | 2,170 | 63 |
| ma (neither) | 5,550 | 134 |
| **total** | **9,015** | **260** |

**Checkability of the sample.** A second and third named authority turned out to be held, so
checkability is higher than the brief assumed:

| witness | sampled roots checkable | share |
|---|---|---|
| al-Mufradāt (al-Rāghib) | 63 | 24.2 % |
| Maqāyīs al-Lugha (Ibn Fāris) | 116 | 44.6 % |
| Asās al-Balāghah (al-Zamakhsharī) | 96 | 36.9 % |
| **at least one of the three** | **189** | **72.7 %** |
| **none of the three** | **71** | **27.3 %** |

### Verdicts — all 260 cases

| witness | AGREES | PARTIAL | WRONG | UNCHECKABLE |
|---|---|---|---|---|
| al-Mufradāt | 52 (20.0 %) | 9 (3.5 %) | 2 (0.8 %) | 197 (75.8 %) |
| Maqāyīs | 76 (29.2 %) | 28 (10.8 %) | 12 (4.6 %) | 144 (55.4 %) |
| Asās al-Balāghah | 70 (26.9 %) | 25 (9.6 %) | 1 (0.4 %) | 164 (63.1 %) |

### Verdicts on **checkable** cases only (the honest denominator)

| witness | n | AGREES | PARTIAL | WRONG | AGREES+PARTIAL |
|---|---|---|---|---|---|
| **al-Mufradāt** | 63 | **82.5 %** | 14.3 % | **3.2 %** | 96.8 % |
| Maqāyīs | 116 | 65.5 % | 24.1 % | **10.3 %** | 89.7 % |
| Asās al-Balāghah | 96 | 72.9 % | 26.0 % | **1.0 %** | 99.0 % |

**Anti-fabrication control.** Each of the 260 × 3 verdicts carried a mandatory verbatim Arabic
quotation. All **780 quotes were mechanically re-checked as exact substrings of the cited source
(780/780 passed; 0 fabricated quotes)**; one quote was 14 characters instead of the required 15.
Any quotation in this report is therefore text that was actually retrieved.

**Full side-by-side evidence for 28 cases (all 14 WRONG roots, 6 PARTIAL, 8 AGREES, with file +
line for each source passage) is in `EVIDENCE_APPENDIX.md`.**

### Every WRONG root found (14 roots / 15 verdicts)

| root | witness | what the record claims | what the source actually says |
|---|---|---|---|
| `ابز` | Maqāyīs 2002 | "no primary Arabic lexical root `ابز` is recognized in the canonical sources" | `الهمزة والباء والزاء يدل على القلق والسرعة وقلة الاستقرار` — Ibn Fāris recognises it and assigns an aṣl |
| `ازر` | Maqāyīs 3741 | aṣl = the izār garment | `أصل واحد، وهو القوة والشدة` — the reverse |
| `حوب` | Maqāyīs 19452 | aṣl = the camel-driving cry | `أصل واحد يتشعب إلى إثم، أو حاجة أو مسكنة`; the cry is explicitly excluded |
| `سبت` | Maqāyīs 34845 | aṣl = cutting/severing | `أصل واحد يدل على راحة وسكون` — the reverse |
| `قمح` | Maqāyīs 59661 | aṣl = wheat maturation | head-raising at water; `ومما شذ عن هذا الأصل القمح` |
| `ضمي` | **Mufradāt 20558 / Maqāyīs 41391** | gloss for root `ضمي` (= ظلم / metathesis) | both cited entries are for the **different root `ضم`** (`الجمع بين الشيئين`) |
| `درهم` | Mufradāt 12267 | falling-from-extreme-old-age origin | `الدرهم: الفضة المطبوعة المتعامل بها` |
| `عدس` | Maqāyīs 51458 | a treading/كدح aṣl | `ليس فيه من اللغة شىء` beyond lentil + mule-cry |
| `مقر` | Maqāyīs 68943 | neck-crushing | bitterness + steeping salted fish |
| `واب` | Maqāyīs 75405 | shame/shyness | `كلمتان تدل إحداهما على تقعير شىء، والأخرى على غضب` |
| `انك` | Maqāyīs 5051 | hardship/constriction/bending | `ليس فيه أصل، غير أنه قد ذكر الآنك` |
| `حفص` | Maqāyīs 18626 | gathering/casting aṣl | `ليس أصلا، ولا فيه لغة تنقاس` |
| `طبس` | Maqāyīs 43843 | overlaying/fitting origin | `ليس بشىء` |
| `صهر` | Asās | literal = melting fat | Asās's *literal* section is kinship only; melting sits in its **majāz** section |

The failure mode is consistent and diagnosable: where Ibn Fāris wrote `ليس أصلا` ("this is not a
root") the generator produced a confident invented `Aṣl al-Wadʿ` anyway.

---

## 3. The citation question

### Which authorities does the lexicon actually name? (all 9,015 records)

| authority named | records | share | do we hold the text? |
|---|---|---|---|
| **Maqāyīs al-Lugha** (Ibn Fāris) via the phrase "Aṣl al-Wadʿ" | 4,799 | 53.2 % | **YES** — `heritage_foundations/Ibn_Faris_Mujam_Maqayis_Al_Lughah.txt`, 4,604 headers, covers 3,826 roots (42.4 %) |
| **Asās al-Balāghah** (al-Zamakhsharī) | 3,856 | 42.8 % | **ONLY AS A DERIVED CORPUS** — `rootformer/data/rootformer_v7_6_zamakhshari_{train,val}.jsonl`, 7,701 records, 3,721 normalised roots. The Arabic is genuine Asās text (e.g. `بغت`: `بغته الأمر وباغته، وجاءه بغتة، ولا رأى للمبغوت، والمبغوت مبهوت`). **No primary edition (.txt/PDF) exists on the pod or in the local `corpus/`.** |
| **Lisān al-ʿArab** (Ibn Manẓūr) | 1,421 | 15.8 % | **NO** — not held anywhere. (The local `lisan_3pillar_master_roots.jsonl` is unrelated.) |
| **al-Mufradāt** (al-Rāghib) | 1,193 | 13.2 % | **YES** |
| al-Zamakhsharī by name 291 · al-Rāghib by name 288 · Sibawayh 47 · al-Khalīl 9 | | | Kitāb al-ʿAyn and al-Ṣiḥāḥ *are* held but are barely cited |

### Do the named sources say what the fields claim? — exhaustive falsity rates

| the field claims | records making the claim | source has **NO** entry for that root | rate |
|---|---|---|---|
| `haqiqah_literal` → "according to Asās al-Balāghah" | 3,042 | **1,105** | **36.3 %** |
| `ontological_core` → "Aṣl al-Wadʿ" (Ibn Fāris) | 4,797 | **2,493** | **52.0 %** |
| `ontological_core`/`majaz` → "al-Mufradāt" | 1,191 | **1,187** | **99.7 %** |

Naming al-Mufradāt is almost a *tell* that the root is not in al-Mufradāt.

**The generator knew.** The records contain self-invalidating hedges inside the very field that
cites the source: 189 records literally say *"Asās al-Balāghah is not attested"*, and many more say
*"Asās al-Balāghah (as delineated from Lisān al-ʿArab)"* / *"though not attested there"* /
*"No literal physical meaning is attested in Asās al-Balāghah or the provided Lisān al-ʿArab
snippets."* A citation wrapped in "not attested" is still presented under the field name
*according to Asās al-Balāghah*.

### Is the supplied Arabic a quotation? Yes — but from the *wrong* sources

Longest verbatim run of `canonical_definition.arabic` inside **any held lexicon** (16.4 M normalised
characters across al-Mufradāt, al-Dharīʿa, Tafṣīl, Maqāyīs, Kitāb al-ʿAyn, Tahdhīb al-Lugha,
al-Ṣiḥāḥ, Jamharat al-Lugha and the Asās corpus):

| all 9,015 records | ≥40 chars | 25–39 | 15–24 | <15 | too short |
|---|---|---|---|---|---|
| | **1,593 (17.7 %)** | 2,175 (24.1 %) | 3,082 (34.2 %) | 2,108 (23.4 %) | 57 (0.6 %) |

**41.8 % of records carry a ≥25-character verbatim run of real classical lexicographic Arabic, and
76.0 % carry a ≥15-character run.** The supplied Arabic is therefore *not* invention; it is largely
genuine dictionary text. (Spot-checked: the matches are substantive definitions, not formulaic
clichés — only 49 of the ≥25-character group contain a formulaic phrase such as `قال تعالى`.)

**But the citation layer does not identify it.** Which held source actually supplies the verbatim run:

| actual source of the record's Arabic | records with ≥25-char run | is this source ever named in the records? |
|---|---|---|
| **Tahdhīb al-Lugha** (al-Azharī) | **1,548 (17.2 %)** | **NEVER NAMED** |
| **al-Ṣiḥāḥ** (al-Jawharī) | **648 (7.2 %)** | **NEVER NAMED** |
| al-Mufradāt (al-Rāghib) | 611 (6.8 %) | named by 1,193 |
| Kitāb al-ʿAyn (al-Khalīl) | 415 (4.6 %) | named by 9 (by name) |
| Asās al-Balāghah | 297 (3.3 %) | claimed by 3,042 in `haqiqah_literal` |
| Jamharat al-Lugha (Ibn Durayd) | 175 (1.9 %) | never named |
| **Maqāyīs al-Lugha** (Ibn Fāris) | **72 (0.8 %)** | **claimed by 4,799 (53.2 %)** |
| al-Dharīʿa | 2 (0.02 %) | — |

> **Maqāyīs al-Lugha is invoked by 53.2 % of the records but is the verbatim source of only 0.8 % of
> them.** Asās al-Balāghah is claimed as the literal meaning by 3,042 records but supplies the text of
> 297. Conversely the two sources that actually supply most of the Arabic — Tahdhīb al-Lugha and
> al-Ṣiḥāḥ — are never mentioned. The compilation is real; the bibliography is fabricated.

Restricted to al-Rāghib alone, as the brief asked:

| | n | ≥25-char verbatim run vs al-Mufradāt | ≥40-char | mean share of the record's Arabic that is verbatim al-Rāghib |
|---|---|---|---|---|
| root **has** a Mufradāt entry | 1,295 | **606 (46.8 %)** | 383 (29.6 %) | **24.1 %** |
| root has **no** Mufradāt entry | 7,720 | 58 (0.8 %) | 23 (0.3 %) | 14.5 % |

So even for the covered roots the record's Arabic is a **paraphrase/expansion of al-Rāghib, not a
transcription** — on average only a quarter of it is verbatim him.

Counter-example showing the lexicon can be accurate: for `بغت` the record's
`canonical_definition.arabic` reads `البَغْتُ والبَغْتَةُ: الفَجْأَة، وهو أَن يَفْجَأَكَ الشيءُ من حيث لا
تحتسب` while al-Mufradāt line 4974 has `البغت: مفاجأة الشيء من حيث لا يحتسب` — same sense, **not a
quotation**. It is a synthesis, and must never be cited as al-Rāghib.

---

## 4. Do al-Dharīʿa and Tafṣīl al-Nashʾatayn extend coverage? **No.**

Measured over the **7,720** roots that have no al-Mufradāt entry:

| work | occurs as a token anywhere | occurs as a standalone entry |
|---|---|---|
| *al-Dharīʿa ilā Makārim al-Sharīʿa* | 254 (3.3 %) | **0 (0.0 %)** |
| *Tafṣīl al-Nashʾatayn* | 121 (1.6 %) | **1 (0.01 %)** — a false positive (`فصل`) |

Neither is a root-keyed lexicon (they are an ethics treatise and a philosophical-anthropological
treatise). They **cannot close the E1 coverage gap.** By contrast the genuinely root-keyed witnesses do:

| witness | covers of all 9,015 |
|---|---|
| al-Mufradāt | 1,295 (14.4 %) |
| Asās al-Balāghah corpus | 3,325 (36.9 %) |
| Maqāyīs al-Lugha | 3,826 (42.4 %) |
| Raghib ∪ Asās | 3,465 (38.4 %) |

Of the roots missing from al-Mufradāt, Asās supplies **2,170 (28.1 %)** and Maqāyīs a further large
block; the union of all held lexicons reaches roughly **6,478 / 9,015 ≈ 72 %** (heuristic — see
limitations). **Maqāyīs al-Lugha and the Asās corpus, not al-Rāghib's other two works, are the route
to the coverage gap.**

---

## 5. The translated JSONs

Six files, 36–683 items each, keys `chapter_index, title_ar, title_en, anchors, arabic_text,
translation, timestamp, duration_seconds`.

* **`arabic_text`** is the genuine Arabic of the work, with `### |` section markers.
* **`translation`** is a **genuine, close, faithful translation** — 10/10 sampled passages across all
  six works render the Arabic accurately, including Quranic citations and poetry. It adds scholarly
  apparatus (transliteration, bracketed glosses, occasional interpretive "Meaning:" sentences) but
  omits nothing substantive in the passages examined.
* **`title_ar` is a placeholder in 100 % of records** — always `Section N`, never the real Arabic
  title, while `title_en` is substantive.
* **`anchors` is NOT a translation.** It is an LLM-generated per-section digest of the form
  `Root: X * Lisan / Ayn: … * Al-Raghib (Mufradat): … * Al-Zamakhshari (Asas): …` — i.e. it makes
  exactly the class of cross-source attribution this audit is testing, and those attributions are
  unverified.

**Provenance test:** only **1 / 300 (0.3 %)** sampled lexicon records share even an 8-word verbatim
run with the `anchors` layer, so the artefact is **not** copied from these files. A naive
token-vocabulary overlap test gives 0.906 but is meaningless (the anchors vocabulary is 21k distinct
English tokens); the 8-gram test is the real one and it is negative.

**Verdict on this deliverable:** the `translation` fields are a *stronger* source than feared and are
usable; the `anchors` and `title_ar` fields are not translations and the `anchors` attributions should
not be trusted.

---

## 6. Verdict

**(a) In the 14.4 % of roots where al-Rāghib has an entry.**
Semantic fidelity is good: **82.5 % AGREES, 14.3 % PARTIAL, 3.2 % WRONG** (n = 63). But it is a
*paraphrase*: only 46.8 % carry a ≥25-character verbatim run and mean verbatim coverage is 24.1 %.
→ **Usable as a hint / derived summary. NOT usable as authority. Cite `Raghib_Al_Mufradat.txt` at the
line, never this file.**

**(b) In the 85.6 % of roots where he has no entry.**
UNCHECKABLE against al-Rāghib by construction, and the citation attached to them is often false:
36.3 % of "according to Asās al-Balāghah" claims and 52.0 % of "Aṣl al-Wadʿ" claims point at roots
the named source never treats, and 99.7 % of al-Mufradāt name-drops are on uncovered roots. Against
the other witnesses: Maqāyīs **10.3 % WRONG**, Asās **1.0 % WRONG** (among checkable). **27.3 % of
the sample cannot be checked against any held source at all.** The Arabic in these rows is still
frequently real classical text — it simply comes from Tahdhīb al-Lugha, al-Ṣiḥāḥ and others that the
prose never credits.
→ **Not authority, and a weak hint for the English glosses. The Arabic is often recoverable, but
only by re-locating it in the real dictionary.**

**Overall recommendation:**
1. **Do not use this lexicon as a source of authority** and do not present its named citations as
   verified. The citation layer is not merely imprecise — it names the wrong books: Maqāyīs is cited
   by 53 % of records yet supplies 0.8 % of the text, while the two largest real sources (Tahdhīb
   al-Lugha, al-Ṣiḥāḥ) are cited by none. Every attribution in the file must be treated as unverified.
2. It may be kept as a **hint / index**, and specifically its `canonical_definition.arabic` is the
   most valuable field: 41.8 % of records carry a ≥25-character verbatim run of genuine classical
   lexicographic Arabic (76.0 % at ≥15 characters) that can be re-located with a string search in the
   held lexicons. **Re-derive the citation from that search, then discard the original attribution.**
3. **Rebuild the root senses from the primary texts.** The immediately available route: al-Mufradāt
   (1,295 roots), Maqāyīs (3,826), the Asās corpus (3,325), and — since they turn out to be the
   lexicon's real backbone and are held in full — Tahdhīb al-Lugha and al-Ṣiḥāḥ, for a union of
   roughly 72 % of the 9,015 roots.
4. The highest-value, cheapest corrections: (i) run the verbatim-matching procedure in
   `grounding_all.py` over every record and replace each citation with the source that actually
   matches; (ii) strip the source attributions from every field where no match is found; (iii) delete
   or flag the 189+ records that self-admit non-attestation while still being labelled "according to
   Asās al-Balāghah"; (iv) discard the English glosses as authority and regenerate them from the
   located Arabic.

---

## 7. Limitations stated plainly

* The Mufradāt header parse (the core deliverable) was validated; the **Maqāyīs/ʿAyn/Tahdhīb/
  Jamharat/Ṣiḥāḥ parses are heuristic**. Ṣiḥāḥ matched only 16 roots and Jamharat 53, so the
  "union of all held lexicons ≈ 72 %" is approximate (understated, if anything). The Maqāyīs figure
  (42.4 %) is reliable enough to carry the argument; the Ṣiḥāḥ/Jamharat figures should not be quoted.
* The 260 semantic verdicts were produced by LLM adjudicators under a strict rubric requiring a
  verbatim quote. **All 780 quotes were independently re-verified as exact substrings of the cited
  source, so no quotation is fabricated.** The AGREES/PARTIAL/WRONG boundary nevertheless involves
  judgement; the WRONG set was written up case by case in `EVIDENCE_APPENDIX.md` so it can be
  overruled by inspection.
* The Asās al-Balāghah "text" is a derived corpus, not a critical edition. Claims checked against it
  are checked against that corpus, which is itself unaudited against the print edition.
* No root-frequency table exists on the pod, so stratification is by source coverage rather than
  corpus frequency.
* The "actual source" attribution in section 3 assigns each record to the single held lexicon giving
  its longest verbatim run. Where a later dictionary quotes an earlier one (Tahdhīb al-Lugha quotes
  al-Layth and al-Khalīl extensively) the match may credit the quoter rather than the originator.
  That does not affect the two conclusions it supports: the Arabic is genuine classical text, and it
  is not the text of the books the records name.
* All 780 adjudication quotations were verified as exact substrings of the cited source, so no
  quotation here is fabricated; the AGREES/PARTIAL/WRONG boundary is judgement, exercised
  conservatively and written up case by case in `EVIDENCE_APPENDIX.md`.

## Files

| file | contents |
|---|---|
| `coverage_census.json` | per-root Mufradāt match for all 9,015 roots |
| `fidelity_sample_enriched.json` | 260-root sample with source text from 7 lexicons |
| `verdicts_final.json` | 260 verdicts × 3 witnesses with quote + reason |
| `wrong_cases.json` | the 15 WRONG verdicts in full |
| `EVIDENCE_APPENDIX.md` | 28 side-by-side record/source cases with file + line |
| `grounding_global.json` | verbatim-run measurement for every record vs Raghib + Asas |
| `grounding_all_lexicons.json` | verbatim-run measurement vs **all nine** held lexicons, with the matching source per record |
| `other_works.json`, `checkability.json`, `citation_exhaustive.log` | supporting measurements |
