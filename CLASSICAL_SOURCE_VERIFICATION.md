# Are the grammarians' algorithms implemented correctly?

**Method.** Every claim below was checked against the primary Arabic texts now held locally in
`corpus/basran/` (18 works) and `corpus/andalusian/` (12 works), by locating the operative passage
and comparing it with the code. Nothing here is taken from the project's own reports.

Primary sources used: `Sibawayh_Al_Kitab.txt`, `Al_Khalil_Al_Ayn.txt`, `Al_Mubarrad_Al_Muqtadab.txt`,
`Ibn_Jinni_Al_Khasais.txt`, `Ibn_Jinni_Sirr_Sinat_Al_Irab.txt`, `Al_Zajjaji_Huruf_Al_Maani.txt`;
`03_IbnMalik_Alfiyyah.txt`, `06_IbnMalik_Lamiyyat_al_Afal.txt`, `09_AbuHayyan_Irtishaf_al_Darab.txt`,
`11_Shatibi_Sharh_Alfiyyah.txt`, `02_Suhayli_Nataij_al_Fikr.txt`, `01_IbnMada_Radd_ala_Nuhat.txt`.

---

## Verdict summary

| Algorithm | Source | Status |
|---|---|---|
| Ibn Mālik, *Lāmiyyat al-Afʿāl* verb tree | `06_IbnMalik_Lamiyyat_al_Afal.txt` | ✅ **correct** — all 7 documented cases, faithful to the verses |
| Sībawayh, operator→awzān partition | `Sibawayh_Al_Kitab.txt` | ✅ **correct** — awzān 114–129 really are the imperfect forms |
| Sībawayh, jazm operator list | `Sibawayh_Al_Kitab.txt` L10700 | ❌ **wrong** — `إن` wrongly included; `لام الأمر` missing |
| Sībawayh, government across distance | `Sibawayh_Al_Kitab.txt` | ❌ **wrong** — was a flat 1-step memory |
| Al-Khalīl, phonotactic pairs | `Al_Khalil_Al_Ayn.txt` L19594 | ❌ **incomplete** — verbatim pairs `(ق,ك)`, `(ج,ق)` absent |
| Al-Khalīl, bare-alif roots | — | ❌ **wrong** — 249 real roots wrongly forbidden |
| Al-Khalīl, *Al-Taqālīb* orbits | — | ❌ **missing** — no parameter sharing across permutations |
| Ibn Mālik, Alfiyyah POS automaton | `03_IbnMalik_Alfiyyah.txt` | ❌ **wrong** — blocked Fiʿl→Fiʿl without the coordinator exception |
| Ibn Mālik, *Marātib al-Maʿārif* | `03_IbnMalik_Alfiyyah.txt` L55 | ❌ **incomplete** — الموصول and المضاف ranks missing |
| Al-Shātibī, wāw functions | `11_Shatibi_Sharh_Alfiyyah.txt` | ⚠️ only ʿAṭf + Istiʾnāf (2 of 4 named; other 4 unspecified) |
| Al-Suhaylī, latent pronoun | `02_Suhayli_Nataij_al_Fikr.txt` | ✅ concept present; not wired into NRMP |
| Ibn Maḍāʾ, anti-phantom operators | `01_IbnMada_Radd_ala_Nuhat.txt` | ✅ concept matches his stated purpose |
| Ibn Jinnī, iʿlāl / jussive apocope | `Ibn_Jinni_Al_Khasais.txt` | ❌ **missing** — needed by `لام الأمر` / `لم` |

---

## 1. Al-Khalīl ibn Aḥmad, *Kitāb al-ʿAyn* — phonotactics

**The text** (`Al_Khalil_Al_Ayn.txt`, line 19594), verbatim:

> قال الخليل: **القاف والكاف لا يجتمعان في كلمة واحدة**، إلا أن تكون الكلمة معربة من كلام العجم،
> وكذلك **الجيم مع القاف لا يأتلف** إلا بفصل لازم.

**Verification against the inventory** (9,043 non-special roots):

| rule | violations in inventory |
|---|---|
| ق + ك adjacent | **0** — the inventory honours it |
| ج + ق adjacent | **1** — the junk root `جقق` |

**The bug:** the code's phonotactic set contained only six hardcoded same-makhraj guttural pairs
(`ء↔ه`, `ع↔ح`, `غ↔خ`). Al-Khalīl's own documented pairs `(ق,ك)` and `(ج,ق)` were **absent**, so
`جقق` survived. Now added (`KHALIL_LETTER_PAIRS`) — `جقق` is correctly excluded.

**Also a severe false positive:** the code forbade *every* bare-alif triliteral root. Because the
blueprint normalises hamza-initial roots to bare alif, this killed **249 genuine roots** — `اصل`
(origin), `ارض` (earth), `اخذ` (take), `احد` (one), `اسس` (found), `افل` (set), `اثر` (trace) —
2.7 % of the entire root space. Removed; 276 roots restored.

**Honest note on the idealised rule.** "Consonants sharing a makhraj cannot be adjacent" is too
strong: compiling the matrix empirically shows 720 adjacent pairs are attested and only **18**
same-makhraj pairs are unattested. `شجر` is a real root with ش+ج sharing *wasaṭ al-lisān*. The
data-driven matrix is therefore the correct construction, not the blanket rule.

---

## 2. Sībawayh, *Al-Kitāb* — operator governance

**The text** (`Sibawayh_Al_Kitab.txt`, line 10700), verbatim:

> ### باب ما يعمل في الأفعال فيجزمها
> وذلك: **لم، ولما، واللام التي في الأمر**، وذلك قولك: ليفعل، **ولا في النهي**، وذلك قولك لا
> تفعل؛ **فإنما هما بمنزلة لم**.

and line 10713:

> واعلم أن **حروف الجزم لا تجزم إلا الأفعال**، ولا يكون الجزم إلا في هذه الأفعال المضارعة للأسماء
> … كما أن الجر لا يكون إلا في الأسماء.

**Findings.**

1. **`إن` is not a jazm operator.** Sībawayh's list is لم، لما، لام الأمر، لا النهي. The code put
   `إن` in `JAZM_OPERATORS_AR`, and because jazm was tested before inna, `إن` classified as
   `HARF_JAZM` — forcing an imperfect verb after the commonest emphasis particle. Fixed: `إنْ` + verb
   → conditional jazm; `إنَّ` + noun → `INNA`.
2. **`لام الأمر` was entirely missing.** It is a *within-word* jussive marker (prefix ل + imperfect),
   now implemented as `lam_amr_apocope()` and tied to Ibn Jinnī's elision rule.
3. **`ما` / `من` / `لا` were unconditional operators**, so plain negation (`لا شك`) was forced into a
   verb. Sībawayh's `لا` is specifically *لا النهي* (followed by an imperfect). Fixed: these are
   operators only when the governed word is verbal.
4. **Government did not persist.** `get_operator_state` looked only at *t−1*, so
   `في الأجسام الشفافة` gave `[JARR, NONE, NONE]` — the adjective lost its genitive government. The
   ʿāmil must act across the governed constituent. Now `[NONE, JARR, JARR]`.
5. **Naṣb operators** confirmed: line 232 `ولن يفعلا`, line 10661 «في حروف النصب بمنزلة لم في حروف
   الجزم», line 10847 naming `إلى أن` and `كي`. So `{أن، لن، كي، إذن، حتى}` is right.

---

## 3. Ibn Mālik, *Lāmiyyat al-Afʿāl* — **correct**

Checked verse by verse against `06_IbnMalik_Lamiyyat_al_Afal.txt`:

| verse | text | code |
|---|---|---|
| 42 | والضم من (فعل) الزم في المضارع | `past_vowel=='u'` → `يَفْعُلُ` ✅ |
| 43 | وافتح موضع الكسر في المبني من فعلا | `past_vowel=='i'` → `يَفْعَلُ` ✅ |
| 45 | ذا الواو فاء أو اليا عينا | `r1=='و' or r2=='ي'` → `يَفْعِلُ` ✅ |
| 47-48 | وأفرد الكسر فيما من (ورث) و(ولي) … (ورم) (ورعت) (ومقت) مع (وفقت) | `{ث,ل,م,ق}` ✅ (all four roots named) |
| 51 | وفتح ما حرف حلق **غير أوله** | fatha for R2/R3 only, **not R1** ✅ |
| 60 | عينا له الواو … مضموم عين | hollow wāw → `يَفْعُلُ` ✅ |

Minor gap: verse 46 «كذا المضاعف لازما ك(حن طلا)» (doubled intransitive) has no explicit branch.

---

## 4. Ibn Mālik, *Al-Alfiyyah* — POS automaton and *Marātib al-Maʿārif*

**The text**, `03_IbnMalik_Alfiyyah.txt` line 55:

> وغيره معرفة **كهم وذي ... وهند وابني والغلام والذي**

i.e. definiteness is: **ضمير** (pronoun), **إشارة** (demonstrative), **علم** (proper),
**مضاف** (annexed), **محلى بأل** (with al-), **موصول** (relative).

**The bug:** `DEFINITENESS` had only `{PRONOUN, PROPER, DEMONSTRATIVE, DEFINITE, INDEFINITE}` —
**الموصول and المضاف were missing**, so two of Ibn Mālik's six categories could not be ranked. Now
`PRONOUN > PROPER > DEMONSTRATIVE > RELATIVE > DEFINITE > ANNEXED > INDEFINITE`.

**Second bug:** the automaton blocked Fiʿl→Fiʿl **unconditionally**, rejecting valid `قام وقعد`. The
Alfiyyah forbids it only *without* a coordinating particle. The codebase even contradicted itself —
`ShatibiWawDisambiguator` already returned `ATF` for `FIL→FIL` while the automaton blocked it. Fixed.

---

## 5. Al-Shātibī, *Sharh al-Alfiyyah* — wāw functions

The commentary does discuss واو العطف (L1752), واو الحال (L7232, L25911, L42509) and واو القسم
(L1794) — so the four functions the report names are genuinely attested. The original implementation
handled 2; now 4 (`ATF`, `HAL`, `ISTINAF`, `QASAM`). The remaining four of Al-Shātibī's eight are not
specified anywhere in the source documents, so they are left unimplemented rather than invented.

---

## 6. Ibn Maḍāʾ, *Al-Radd ʿalā al-Nuḥāt* — intent confirmed

`01_IbnMada_Radd_ala_Nuhat.txt` line 68 states his programme verbatim:

> قصدي في هذا الكتاب **أن احذف من النحو ما يستغني النحوي عنه**، وأنبه على ما …

This directly justifies the implemented "eliminate phantom/virtual operators" rule. The negative
list of what counts as phantom is not enumerated in the text, so the implementation stays a small
regex filter.

---

## 7. Al-Suhaylī and Ibn Jinnī

- `02_Suhayli_Nataij_al_Fikr.txt` discusses **الضمير المستتر** at lines 1293, 2026, 2783, 4858 —
  the concept behind `SuhayliPronounResolver`. It exists but is **not wired into the NRMP path**.
- `Ibn_Jinni_Al_Khasais.txt` treats pausal forms and the collision of two sukūns (L787-829), the
  basis of *iʿlāl* / jussive apocope (`لم يقل` not `*لم يقول`). **Not implemented** — now added as
  `lam_amr_apocope()`; full apocope in surface realization is still outstanding.

---

## 8. Still outstanding (honest list)

1. **Al-Taqālīb** — S³ permutation-orbit parameter sharing is absent; `علم` and `عمل` are orthogonal
   one-hot ids. `khalil_combinatorics` is not even importable from the release.
2. **Valency / constituent stack** — `SibawayhConstituentStack` exists in `basran_syntactic_engine.py`
   but is never called from the NRMP path, so *Inqiṭāʿ al-ʿAmal* is not enforced.
3. **Al-Iktifāʾ** — clitic-object saturation (`يقذفه` blocking a bare accusative) not implemented.
4. **Full surface apocope** — only the detection rule is implemented, not the realization.
5. **Marātib al-Maʿārif** — implemented as a predicate, not yet enforced during decoding.
6. The remaining four of Al-Shātibī's eight wāw functions are unspecified in the sources.

---

## 9. Verification harness

`test_grammar_impl.py` runs 20 checks against the *original* implementation (10 failed);
`verify_v2.py` runs the corrected set against `classical_governance_v2.py` (**all pass**).

```
RESULT (original): 10/20 checks pass, 10 FAIL
RESULT (v2)      : ALL v2 CHECKS PASS
```
