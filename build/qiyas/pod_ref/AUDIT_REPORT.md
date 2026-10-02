# Grammar-Algorithm Citation & Rule Fidelity Audit

**Artifact under audit:** `/workspace/hf_v19_2_release` (this is the canonical set — `/workspace/taxpin/*.py`
are symlinks into it, and `ALGORITHM_IMPLEMENTATION_LEDGER.md` measures against it).
**Pod:** `g6exduq0bd17z8` @ `213.173.104.76:46758`. **Read-only** everywhere except `/workspace/grammar_audit/`.
**Corpus searched:** 60 deduplicated primary-text files, 119.4 M characters, from
`hf_v20_1_release/corpus/{basran,andalusian}`, `heritage_foundations/`, `scholastic_masters/`,
`andalusian_canon_raw/`.
**Machinery:** `extract_citations_v4.py` → `citations4.json`; `verify_citations_v3/v4.py` →
`verification_final.json`; `diagnostics.py`; full table in `citations_full.tsv` (290 rows).

---

## 0. Text availability — the brief is partly wrong, in the project's favour

The brief states the project "does NOT hold" Abū Ḥayyān's *Irtishāf*/*al-Tadhyīl*, al-Shāṭibī's
*al-Muwāfiqāt*, or Ibn Mālik's own works, and asks me to check. **They are all present on the pod** —
the documented path `/workspace/corpus/andalusian/` does not exist, but the texts do:

| text | actual location | held |
|---|---|---|
| Abū Ḥayyān, *Irtishāf al-Ḍarab* | `andalusian_canon_raw/09_AbuHayyan_Irtishaf_al_Darab.txt` (3.4 MB) | **yes** |
| Abū Ḥayyān, *al-Tadhyīl wa-l-Takmīl* | `andalusian_canon_raw/10_AbuHayyan_Tadhyil_al_Tashil.txt` (6.8 MB) | **yes** |
| al-Shāṭibī, *al-Muwāfiqāt* | `andalusian_canon_raw/12_Shatibi_Al_Muwafaqat.txt` (3.9 MB) | **yes** |
| al-Shāṭibī, *Sharḥ al-Alfiyyah* | `andalusian_canon_raw/11_Shatibi_Sharh_Alfiyyah.txt` (9.2 MB) | **yes** |
| Ibn Mālik, *al-Alfiyyah / al-Tashīl / Sharḥ al-Kāfiya / Lāmiyyat al-Afʿāl* | `andalusian_canon_raw/03,04,05,06_*` | **yes** |
| al-Suhaylī, *Natāʾij al-Fikr*; Ibn Maḍāʾ, *al-Radd*; Ibn Sayyidih | `02`, `01`, `07`, `08` | **yes** |

These are genuine OpenITI/Shamela editions (the *Irtishāf* file carries `#META#` records naming the
Khanji 1418/1998 printing). The project's own citation paths (`corpus/andalusian/09_...`) resolve
correctly against `/workspace/hf_v20_1_release/corpus/`.

**What is genuinely not held** (searched the entire filesystem, `find / -iname '*Mumtic*' -o -iname
'*IbnUsfur*' -o -iname '*Jumal*'` → **empty**):

- **Ibn ʿUsfūr, *Sharḥ Jumal al-Zajjājī*** — cited 5× as `corpus/andalusian/14_...`
- **Ibn ʿUsfūr, *al-Mumtiʿ fī al-Taṣrīf*** — cited 3× as `corpus/andalusian/13_...`
- al-Jurjānī, *Dalāʾil al-Iʿjāz*; Ibn Khaldūn, *al-Muqaddimah*; the Mecelle; al-Taftāzānī, *Sharḥ al-ʿAqāʾid*

The `andalusian/` set runs `01`–`12`. Two citation targets numbered `13` and `14` were **planned and
never assembled** — yet they carry precise line locators (13449, 11470, 11682, 13436, 3096, 157–161).

---

## 1. The citations — real count

The brief's "~72" is close. The modules use **four different dict schemas**, which is why a naive scan
misses two-thirds of them:

| schema | module | records |
|---|---|---|
| `{book, file, line, arabic}` | `andalusian_realizer.py` | 43 |
| `{book, file, ar, note}` | `ibn_malik_automaton.py` | 21 |
| `{book, ar}` (Bulaq page locator inside `book`) | `sibawayh_governor.py` | 16 |
| `{work, file, line, text}` | `khalil_orbits.py` | 5 |
| **structured registry total** | | **85** |

Plus **205 inline quotation spans** (`«…»` / `"…"`), of which **199 are citation claims**
(the other 6 are code/test artefacts). **Total: 284 citation claims** across the 15 modules.
`farahidian_syntactic_realizer.py`, `irab_realizer.py` and `tasrif_engine.py` carry Arabic *data*
(awzān, weak-letter sets, test words) but **no attributed quotations at all**.

Per-module claims: `andalusian_realizer` 80, `sibawayh_governor` 52, `ibn_malik_automaton` 50,
`validated_segmentation` 40, `khalil_orbits` 21, `andalusian_grammatical_algorithms` 17,
`khalil_students_andalusian_master_engine` 12, `classical_governance_v2` 8, `ibn_jinni_ishtiqaq` 2,
`constituent_stack` 2.

---

## 2. Mechanical verification

Matching ladder (strongest first): **L0** exact substring of the raw file; **L1** exact substring after
removing only OpenITI markup (`### ||`, `#`, `~~`, `msNNN`, `PageVxxPxxx`) and joining wrapped lines —
this is exactly what the project's own docstring claims it did, so L1 counts as VERBATIM; **L2** + diacritics
removed; **L3** + orthographic normalisation; **L4** `…`/`#` elision fragments present in order;
**L5** ≥60 % token coverage → PARAPHRASE.

### 2.1 Structured registry (85 records)

| class | n | % |
|---|---|---|
| **VERBATIM** (exact after markup strip; 19 also exact in raw bytes) | **60** | 70.6 |
| **NEAR-VERBATIM** (diacritics / orthography / elision) | **15** | 17.6 |
| **PARAPHRASE** | 5 | 5.9 |
| **MISATTRIBUTED** | **0** | 0.0 |
| **ABSENT** | **0** | 0.0 |
| **UNCHECKABLE** (names an unheld Ibn ʿUsfūr file) | 5 | 5.9 |
| *(one further Ibn ʿUsfūr record scored a spurious 0.647 token-overlap PARAPHRASE against al-Muqtadab; it is in substance UNCHECKABLE too)* | | |

**Mechanical VERBATIM pass rate: 60/60 = 100 %.** Every record the pipeline classified VERBATIM was
re-tested by literal substring search and passed. None is a phrase-trap.

**Locator accuracy.** Of the 63 records whose declared path resolves to a held file:

- **56 line pointers exact (88.9 %)**
- **6 off by 1–7 lines**, never more: Alfiyyah ʿaṭf verses cited as 549/550 (actual 546/547);
  *Tadhyīl* quotes cited as 6140 (actual 6133, 6138); al-Shāṭibī 3566-3568 (actual 3571);
  al-Suhaylī 1794 (actual 1793)
- 1 declares no line
- **0 fabricated line pointers**

**Page locators.** The 16 `sibawayh_governor` records cite Bulaq pages instead of paths. **10/10
`PageVxxPyyy` markers exist in `Sibawayh_Al_Kitab.txt`, and every quoted passage sits within ±19
lines of its marker.** These are real.

### 2.2 Inline claims (199)

| class | n | note |
|---|---|---|
| **VERBATIM** | **95** | 69 attributed + 26 real text with no nearby attribution |
| **NEAR-VERBATIM** | 41 | |
| **PARAPHRASE** | 15 | |
| **ABSENT** | 21 | 14 are code/term artefacts; 4 quote works not held; **3 are real** |
| **MISATTRIBUTED** | 17 | **hand-reviewed: 15 are false positives of my name-proximity heuristic** — the module's own attribution is correct (al-Shāṭibī, al-Alfiyyah, al-Kitāb). 2 are genuinely questionable |
| **UNCHECKABLE** | 10 | all Ibn ʿUsfūr |

### 2.3 Combined

| class | n | % of 284 |
|---|---|---|
| VERBATIM | 155 | 54.6 |
| NEAR-VERBATIM | 56 | 19.7 |
| **exact or near** | **211** | **74.3** |
| PARAPHRASE | 20 | 7.0 |
| MISATTRIBUTED | 17 (≈2 real) | 6.0 |
| ABSENT | 21 (≈3 real) | 7.4 |
| UNCHECKABLE | 15 | 5.3 |

---

## 3. Failure modes — and they are **not** the lexicon's

The lexicon's defect was **invented attribution with real Arabic**: a text named 4,799 times that
supplied 0.8 % of the content, texts never named that supplied 17 %, and 189 records self-admitting
"not attested" while still carrying the label. **That pattern does not reproduce here.**

What is actually wrong:

1. **Six records cite Ibn ʿUsfūr with precise-looking locators into files that do not exist.**
   `andalusian/14_IbnUsfur_Sharh_Jumal_al_Zajjaji.txt:13449/11470/11682/13436` and
   `andalusian/13_IbnUsfur_Mumtic_fi_al_Tasrif.txt:3096` / `…:157-161`. The project has never possessed
   these texts. This is the single largest real defect: **5 of 85 registry records (5.9 %) cannot be
   verified in principle**, and several load-bearing rules (jussive "two marks", the diptote
   two-causes definition, the *al-ishtiqāq al-akbar* rejection) rest on them.

2. **One quotation attributed to a held text is genuinely absent.** `khalil_students_andalusian_master_engine.py:515`
   labels «العلم نور يضيء العقل ويهدي إلى الحق» as **al-Ghazālī, *Tahāfut al-Falāsifa***. That book is
   held (`scholastic_masters/Ghazali_Tahafut_al_Falasifa.txt`); the sentence is not in it, nor anywhere
   in the 119 M-character corpus. It is a floating aphorism under a specific book's name — **the one
   clean instance of the lexicon's exact failure mode.**

3. **A chapter title that is not Sībawayh's.** `iktifa_apocope.py:8` grounds the *al-iktifāʾ* half on
   "Sibawayh, al-Kitab: باب ما يكتفى فيه بالشيء عن الشيء". **No such chapter exists in al-Kitab.** The
   nearest is `### | هذا باب ما يكون في اللفظ من الأعراض` (line 314). The words *istiġhnāʾ* / *yastaghnūna*
   do occur (line 315), but about lexical suppletion (يدع vs ودع), not valency slots.

4. **Imprecise line pointers** (6 of 63, all ≤7 lines) and **line-welding**: several quotes join text
   across OpenITI line breaks and markup, so they are not byte-exact substrings of the file even though
   every word is the file's. This is disclosed in the docstring, and it is why 15 registry records land
   in NEAR-VERBATIM rather than VERBATIM.

5. **Two quotes carry the *lead-in* clause invented but the body real** —
   `validated_segmentation.py:395` «ومتى أمكن تناول الكلمة على ظاهرها لم يجز العدول عن ذلك بها»:
   `لم يجز العدول عن ذلك` is in al-Khaṣāʾiṣ, the opening clause is not.

Against that, the **positive** findings are strong and unusual: al-ʿAyn's introduction and both worked
chapters (lines 164, 4760, 5817) are cited exactly, with exact line numbers; al-Kitāb's 10 page
locators are all real; Alfiyyah vv. 54, 55, 542, 569, 662, 670, 672 are all correct; al-Shāṭibī
3563, 48806, 48811; Abū Ḥayyān *Tadhyīl* 6092-6141, 32653; Ibn Jinnī *al-Khaṣāʾiṣ* 7639;
al-Suhaylī 1793 — all real, all correctly attributed.

---

## 4. Rule-level audit

Counts: **FAITHFUL 6 · OVERREACH 1 · UNSOURCED 0 · CONTRADICTS 0 · UNCHECKABLE 2** (sub-layers).

| # | rule | primary passage (verified) | verdict |
|---|---|---|---|
| 1 | al-Taqālīb orbits + مستعمل/مهمل cut | al-ʿAyn **164**, **4760**, **5817** — all VERBATIM, exact lines | **FAITHFUL** |
| 2 | ʿāmil persistence & government | al-Kitāb **10715**, **5261–5269** (PageV01P421), **937–938** (PageV01P073) | **FAITHFUL** (module declares its own gaps) |
| 3 | jussive apocope | al-Kitāb **290**, **22025**, **22047** — all VERBATIM | **FAITHFUL** |
| 4 | al-iktifāʾ (clitic saturation) | al-Kitāb **315** real, but the named chapter is not | **OVERREACH** |
| 5 | marātib al-maʿārif | Alfiyyah **54**, **55**; al-Shāṭibī **3563–3565**, **3571–3573**; Abū Ḥayyān *Tadhyīl* **6092–6095**, **6133–6141** | **FAITHFUL** |
| 6 | ṣīghat muntaha al-jumūʿ | Alfiyyah **662**, **670**, **672**; al-Shāṭibī **48806** | **FAITHFUL** (Andalusian layer) / **UNCHECKABLE** (Ibn ʿUsfūr layer) |
| 7 | al-qalb alif→yāʾ | al-Zajjājī *Ḥurūf al-Maʿānī* **331**; Abū Ḥayyān *Tadhyīl* **32653** | **FAITHFUL** |
| 8 | al-ajwaf vs al-nāqiṣ | Ibn Jinnī *al-Khaṣāʾiṣ*; Ibn Jinnī *Sirr* **3226** | **FAITHFUL** (Ibn Jinnī layer) / **UNCHECKABLE** (Ibn ʿUsfūr's positional restriction) |
| 9 | nisba-yāʾ exception | al-Shāṭibī **48811**, exact line | **FAITHFUL** |

### Worked examples — code beside the text

**1. Orbits and the مستعمل/مهمل cut — FAITHFUL**
`khalil_orbits.py:156–168` cites al-ʿAyn; `orbit_of()` (l.466–481) returns only positively attested permutations.
```
"al_ayn_chapter_3lm": { "file": "corpus/basran/Al_Khalil_Al_Ayn.txt", "line": "5817",
    "text": "باب العين واللاّم والميم معهما ع ل م، ع م ل، م ع ل، ل م ع مستعملات" }
"al_ayn_chapter_3db": { "line": "4760",
    "text": "باب العين والدال والباء معهما ع ب د- د ع ب- ب ع د- ب د ع مستعملات ع د ب- د ب ع مهملان" }
```
`Al_Khalil_Al_Ayn.txt:5817` — identical, byte-for-byte. `:4760` — identical. The implementation's
"4 members for علم, not 6" and its explicit عدب/دبع exclusion match al-Khalīl's own verdicts exactly.

**2. The accept/reject filter — VERBATIM (elided)**
`khalil_orbits.py:156` quotes al-ʿAyn line 164 with `…` elision:
«والكلمة الرباعية تتصرَّف على أربعة وعشرين وجها ... يُكَتَب مُسْتَعْمَلها. ويُلغى مُهْمَلها ... يُسْتَعْمَل أقَلُّه ويُلغى أكثره»
→ found at `Al_Khalil_Al_Ayn.txt:164`, L4 elision(4), all four fragments present in order.
The module even records that the brief's wording «يُكتب مُستعملها» differs from the edition's
«يُكَتَب مُسْتَعْمَلها» — and quotes the edition.

**3. Jussive apocope — FAITHFUL**
`iktifa_apocope.py:105–112`:
```python
if r and len(r) == 3 and r[-1] in WEAK and b[-1] in WEAK:
    return b[:-1]                 # naqis: delete the final weak letter
if r and len(r) == 3 and r[1] in WEAK:
    for i in range(1, len(b)):
        if b[i] in WEAK:
            return b[:i] + b[i + 1:]   # ajwaf: medial elision
return b + SUKUN
```
`Sibawayh_Al_Kitab.txt:290` — «وأعلم أن الآخر إذا كان يسكن في الرفع حذف في الجزم ... وذلك قولك لم يرم ولم يغز ولم يخش»;
`:22025` — «ومثل ذلك: لم يبع ولم يقل»; `:22047` — «...ارمه، ولم يغزه، واخشه، ولم يقضه، ولم يرضه».
All three VERBATIM. Each branch of the code has a matching Sībawayh witness.

**4. al-Iktifāʾ — OVERREACH**
`iktifa_apocope.py:8`:
```
(1) AL-IKTIFA' -- "sufficing". Sibawayh, al-Kitab: باب ما يكتفى فيه بالشيء عن الشيء.
```
Search of `Sibawayh_Al_Kitab.txt` for «يكتفى» returns only line 10188 («يستغنى الكلام ويكتفى») and
line 23020 («وقط، معناها الاكتفاء») — **no such chapter heading exists**. Line 314 reads
`### | هذا باب ما يكون في اللفظ من الأعراض`, and line 315 is the real (VERBATIM, and correctly
registered) passage: «اعلم أنهم مما يحذفون الكلم وإن كان أصله في الكلام غير ذلك، ويحذفون ويعوضون،
ويستغنون بالشيء عن الشيء...» — about **lexical suppletion** (يدع / ودع). The implemented doctrine
(a clitic closes a valency slot so no further maʿmūl is expected) is an engineering reframing, not
what the cited passage says. The rule is real as *istiġhnāʾ*; the name, the chapter title and the
valency-saturation generalisation are not sourced.

**5. ʿĀmil government — FAITHFUL, and the module says what it cannot source**
`sibawayh_governor.py:29–33` cites `al-Kitab 3/8`:
«واعلم أن حروف الجزم لا تجزم إلا الأفعال، ولا يكون الجزم إلا في هذه الأفعال المضارعة للأسماء، كما أن الجر لا يكون إلا في الأسماء»
→ `Sibawayh_Al_Kitab.txt:10713–10715`, page marker `PageV03P008` at line 10709, i.e. **+4 lines**.
Then, unprompted, at `:87–92`:
```
NOT FOUND (stated rather than stretched into a citation):
  * «ولا يعمل عاملان في معمول واحد» as a SIBawayh sentence ...
  * «حتى يقطع عمله» / «العامل يعمل فيما يليه» -- these phrases occur nowhere in al-Kitab
    (searched undiacritized).
```
My independent search confirms both are absent from every held text. **A module that volunteers the
quotations it could not find is not a module that invents quotations.**

**6. marātib al-maʿārif — FAITHFUL, correctly reported**
`ibn_malik_automaton.py:66–71`:
```python
'M_ranks_tashil': {
    'book': "Ibn Mālik, al-Tashīl (as reported verbatim by al-Shāṭibī, Sharh al-Alfiyyah, on l.55)",
    'file': 'corpus/andalusian/11_Shatibi_Sharh_Alfiyyah.txt:3562-3565',
    'ar': 'وقد جعل لها في "التسهيل" ست مراتب، فأعلاها ضمير المتكلم، ثم ضمير المخاطب، ثم العلم، ...'}
```
`11_Shatibi_Sharh_Alfiyyah.txt:3562–3565` — exact. The phrase «ست مراتب» appears **only** in
al-Shāṭibī across all held texts (not in *al-Tashīl*, *Irtishāf*, or *Tadhyīl*). The module states
this explicitly in the `book` field. My initial suspicion of misattribution here was **wrong**; this
is a model of how to file a reported quotation.

**7. Ṣīghat muntaha al-jumūʿ and the nisba-yāʾ exception — FAITHFUL**
`andalusian_realizer.py:245` (key `shatibi_sighat_muntaha_dabit`):
«وهذا التمثيل أشار فيه إلى قيود معتبرة في منع الجمع، وضابطه: كل جمع ثالث حروفه ألف ثابتة، وبعدها حرفان، أو ثلاثة أحرف أوسطها ياء، عار من التأنيث أو ياء النسب.»
→ `11_Shatibi_Sharh_Alfiyyah.txt:48811`, **VERBATIM, exact line**. The exception the code encodes
(`sighat_muntaha()`, l.518) is licensed word-for-word by «عار من التأنيث أو ياء النسب».
Supporting: Alfiyyah 662, 670, 672 all VERBATIM/NEAR at the exact declared lines.

**8. al-qalb alif→yāʾ — FAITHFUL, and the module records its own earlier error**
`validated_segmentation.py:540–545`:
```python
# AL-QALB, NOT AL-HADHF -- the sources say the alif is TURNED INTO ya', not deleted.
#   Ibn 'Usfur (Sharh Jumal al-Zajjaji): «إن العرب قد تقلب الألف ياء مع المضمر في نحو: عليه وإليه ولديه».
#   Al-Zajjaji (Huruf al-Ma'ani, لدى): «ومع المضمر تنقلب ياء تقول لدى زيد ولديك».
#   Abu Hayyan: «وقلبت ألفه ياء لإضافته إلى المضمر، كما قلبوا في عليك ولديك».
# I had written "the alif is dropped" and offered ما + ه -> مه, which no source supports.
```
`Al_Zajjaji_Huruf_Al_Maani.txt:331` — «ومع المضمر تنقلب ياء تقول» ✓ VERBATIM.
`10_AbuHayyan_Tadhyil_al_Tashil.txt:32653` — «وقلبت ألفه ياء لإضافته إلى المضمر، كما قلبوا في عليك ولديك» ✓ VERBATIM.
The Ibn ʿUsfūr clause is **UNCHECKABLE** (text not held). Two of three witnesses real; the module
discloses the third's status and corrects its own prior implementation.

**9. al-ajwaf vs al-nāqiṣ — FAITHFUL, with the limit stated**
`validated_segmentation.py:272–288` cites Ibn Jinnī and Ibn ʿUsfūr:
```
# WHICH weak letter may vanish is not a free choice -- it is the ajwaf / naqis division,
#   Ibn Jinni, al-Khasa'is: «لما سكنت عين فعلت ولامه حذفوا العين البتة فقالوا: قلت وبعت وخفت، ولم يقولوا: قولت ولا بيعت ولا خيفت»
# Two honest limits. (a) The sources state the contrast for THIS position only: the naqis ...
#   نحو: رمت هند» (al-Mumti').  (b) كنت itself is never used as an ajwaf exemplar; the
#   analysis is Ibn Jinni's, in Sirr Sina'at al-I'rab: «وكذا كان القياس أن تقول في كنت: كوني...»
```
`Ibn_Jinni_Al_Khasais.txt` contains «لما سكنت عين فعلت», «حذفوا العين البتة», «قولت ولا بيعت» ✓.
`Ibn_Jinni_Sirr_Sinat_Al_Irab.txt:3226` — «وكذا كان القياس أن تقول في كنت: كوني، تحذف» ✓.
The positional restriction itself is cited to al-Mumtiʿ → **UNCHECKABLE**, and the module says so in
the comment rather than hiding it.

**10. A quotation attributed to a held text that is not there — ABSENT**
`khalil_students_andalusian_master_engine.py:513–517`:
```python
( "Al-Ghazālī (Tahāfut al-Falāsifah)",
  "العلم نور يضيء العقل ويهدي إلى الحق",
  "the knowledge is light illuminates the intellect and guides to truth",
  "Al-Shāṭibī Conjunction Disambiguation (و) + Al-Suhaylī Verbal Realism" ),
```
`Ghazali_Tahafut_al_Falasifa.txt` is held (477 KB). The sentence occurs **nowhere** in it — nor in any
of the 60 corpus files. This is the audit's one unambiguous instance of a real-looking attribution
attached to content the named text does not contain.

**11. A locator pointing into a file that does not exist — UNCHECKABLE**
`khalil_orbits.py:191–200`:
```python
"ibn_usfur_rejection": {
    "work": "Ibn ʿUsfūr, al-Mumtiʿ fī al-Taṣrīf",
    "file": "corpus/andalusian/13_IbnUsfur_Mumtic_fi_al_Tasrif.txt",
    "line": "157-161",
    "text": "ولم يقل به أحد من النحويين إلا أبا الفتح. وحكى هو عن أبي علي أنه كان يأنس به ..."}
```
`find / -iname '*Mumtic*'` → empty. Grep for «ولم يقل به أحد من النحويين» across all 119 M chars →
empty. Meanwhile **al-Khaṣāʾiṣ**, which the project *does* hold and which the module correctly cites at
7639 for the *positive* formulation, has Ibn Jinnī **expounding** *al-ishtiqāq al-akbar*, not rejecting
it. So the module's central "the tradition rejects this, and here is the rejection" caveat — its
loudest fidelity claim — rests on an unverifiable quotation from a text it does not possess.

**12. Imprecise but honest pointers — the Alfiyyah ʿaṭf couplet**
`ibn_malik_automaton.py:110–113` declares `03_IbnMalik_Alfiyyah.txt:549` for
«تال بحرف متبع عطف النسق ... كاخصص بود وثناء من صدق». The verse is at **line 546**; the next
(`:117`) declares 550 for «فالعطف مطلقا بواو ثم فا», actually at **547**. Both quotes are VERBATIM
and the chapter is right; only the numbers drift by 3. At `:630` the same module cites «Alfiyyah l.569»
for «قام وقعد» — that one is **exactly** right (569 = «وعطفك الفعل على الفعل يصح»).

---

## 5. Texts cited but not held

| work | cited as | times |
|---|---|---|
| Ibn ʿUsfūr, *Sharḥ Jumal al-Zajjājī* | `andalusian/14_IbnUsfur_Sharh_Jumal_al_Zajjaji.txt` | 5 registry + 3 inline |
| Ibn ʿUsfūr, *al-Mumtiʿ fī al-Taṣrīf* | `andalusian/13_IbnUsfur_Mumtic_fi_al_Tasrif.txt` | 2 registry + 6 inline |
| al-Jurjānī, *Dalāʾil al-Iʿjāz* | author label only | 1 |
| Ibn Khaldūn, *al-Muqaddimah* | author label only | 1 |
| Mecelle-i Aḥkām-i ʿAdliyye | label only | 1 |
| al-Taftāzānī, *Sharḥ al-ʿAqāʾid* | label only | 1 |

**None of these can have been verified. They must not be presented as if they were.**

---

## 6. Verdict

**The algorithm modules are genuinely grounded — they are not the lexicon.**

The lexicon's signature was invented attribution over real Arabic: a source named 53 % of the time that
supplied 0.8 % of the text, two sources supplying 24 % while never being named, and hundreds of records
wearing labels they contradicted. **None of that reproduces here.** On the contrary:

- 60 of 85 registry records (70.6 %) are exact substrings of the held primary texts after markup
  stripping; 75 (88.2 %) are exact or near-verbatim. Mechanical VERBATIM pass rate **60/60 = 100 %**.
- **0 MISATTRIBUTED and 0 ABSENT among the 63 registry records whose declared file resolves.**
- **0 fabricated line pointers**; 56/63 exact, the other 6 off by ≤7 lines. All 10 Bulaq page
  locators real.
- The corpus is cited where it actually says the thing: al-ʿAyn's introduction and both worked
  chapters; al-Kitāb's jazm, naʿt, inna/anna and jazāʾ chapters; Alfiyyah 54, 55, 542, 569, 662, 670,
  672; al-Shāṭibī 3563, 48806, 48811; Abū Ḥayyān *Tadhyīl* 6092-6141, 32653; Ibn Jinnī *al-Khaṣāʾiṣ*
  7639, *Sirr* 3226; al-Suhaylī 1793; al-Zajjājī *Ḥurūf al-Maʿānī* 331.
- The modules **volunteer their own gaps**: `sibawayh_governor.py:87–92` lists two phrases it searched
  for and could not find; `validated_segmentation.py:272–288` states two limits on its own rule;
  `validated_segmentation.py:545` records that its earlier version asserted something "no source
  supports"; `khalil_orbits.py:51–53` quotes the corpus edition against the brief's own wording.

**The real defects, ranked:**

1. **Two Ibn ʿUsfūr texts are cited with fabricated-looking precision into files the project has never
   held** (6 registry records, 9 inline). The rules resting on them — the jussive "two marks", the
   diptote two-causes definition, the *al-ishtiqāq al-akbar* rejection, the ajwaf/nāqiṣ positional
   restriction — are **UNCHECKABLE and must be relabelled as such**, not carried as verified.
2. **The *al-iktifāʾ* chapter title is not Sībawayh's**, and the doctrine implemented under that name
   generalises his *istiġhnāʾ* passage well past what it says → **OVERREACH**.
3. **One quotation (al-Ghazālī/*Tahāfut*) names a held book that does not contain it** → the single
   genuine ABSENT case, and the one point where the lexicon's failure mode does appear.
4. Minor: 6 line pointers drift ≤7 lines; a few quotes weld lines across markup and are therefore not
   byte-exact.

So: **6 rules FAITHFUL, 1 OVERREACH, 0 UNSOURCED, 0 CONTRADICTS, 2 UNCHECKABLE sub-layers.** The
citations are, on the whole, earned. The gap is not invention — it is **five-to-six items filed with
a confidence (exact file, exact line) that the project's own library cannot support**, which in an
audit that treats "cited" as "verified" is the same practical defect as invention even though the
underlying scholarship is real.

---

### Artifacts
- `citations_full.tsv` — all 290 extracted spans: module, line, class, match level, declared locator, found file:line, line-pointer flag, quote, corpus snippet
- `citations4.json` — raw extraction (85 structured + 205 inline)
- `verification_final.json` — per-record verdicts
- `extract_citations_v4.py`, `verify_citations_v3.py`, `verify_citations_v4.py`, `diagnostics.py`
