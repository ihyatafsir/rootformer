# Implementation-fidelity ledger

**Principle.** Al-Khalīl, Sībawayh, Ibn Mālik and the Andalusian grammarians are not a hypothesis
awaiting validation — the system has been in continuous scholarly use for twelve centuries. The open
question is whether a *given implementation* is faithful to what they specified. So the instrument is
**agreement with the primary text**, not an accuracy delta. An algorithm can be correctly implemented
and still move accuracy by zero; that is a property of the corpus, not of the implementation.

Every entry below cites the text it was checked against. Sources are in `corpus/basran/` and
`corpus/andalusian/`.

---

## Al-Khalīl, *Kitāb al-ʿAyn* — phonotactics

### Letter-incompatibility rules: **4 specified, 2 were implemented**

| # | rule as stated | line | was | now |
|---|---|---|---|---|
| 1 | «الهمزة والغين لا تجتمعان في بناء كلمةٍ واحدةٍ» | 19418 | **absent** | ✓ |
| 2 | «القاف والكاف لا يجتمعان في كلمة واحدة» | 19594 | ✓ | ✓ |
| 3 | «الجيم مع القاف لا يأتلف إلا بفصل لازم» | 19594 | ✓ | ✓ |
| 4 | «الضاد والصاد لا يأتلفان في كلمةٍ واحدةٍ أصليّة الحروف» | — | **absent** | ✓ |

The two conditions attached to rules 3 and 4 are satisfied by construction in this representation: an
*adjacent* pair in a triliteral root has no separator between them, and all three root letters are the
original radicals by definition.

**Effect on the inventory: zero roots.** `ء+غ` and `ض+ص` are attested **0×** in every ordering across
all 9,114 roots — the inventory was built from attested Arabic and is already Khalīlian. The single
genuine violation the rules catch is `جقق` (jim+qaf) — a junk entry.

This is why the mask does not move accuracy. **The mask is validated by the data rather than
correcting it**, which is a stronger result than a busy mask, but it also means the mask cannot be a
source of information.

### The trap to avoid — over-extension

Of 1,444 possible adjacent letter pairs, **724 are unattested** in the inventory. Masking those would
be *unfaithful*: Al-Khalīl distinguishes «لا يأتلف» (does not harmonise — impossible) from merely
unattested. Half the letter space being unused is the normal consequence of a finite lexicon, not a
prohibition. Only the four rules above license exclusion.

### Other Al-Khalīl corrections

| item | finding |
|---|---|
| C₁ = C₂ | 28 impossible roots — correct |
| bare-alif-initial roots | an earlier version banned **all** of them, wrongly killing **249 real roots** (`اصل`, `ارض`, `اخذ`, `احد`, `اسس` — 2.7 % of the root space). Now allowed: 276 |
| control strings | `end`/`start` leaked into the inventory — excluded |

---

## Sībawayh, *Al-Kitāb* — operators (ʿawāmil)

| item | text | finding |
|---|---|---|
| jazm operators | «لم، ولما، **واللام التي في الأمر**، ولا في النهي» (10700) | `إن` had been wrongly included; **`لام الأمر` was missing entirely** |
| jazm scope | «حروف الجزم لا تجزم إلا الأفعال» (10713) | verbs only — now enforced |
| government persistence | *inqiṭāʿ al-ʿamal* | was a flat 1-step memory: `في الأجسام الشفافة` → `[JARR, NONE, NONE]`. Now `[NONE, JARR, JARR]` |
| ambivalent operators | `أن`/`إن` | disambiguated by the **next word's** POS, not by surface form alone |

**Not implemented:** Sībawayh's valency and constituent stack — how far an operator's government
reaches through a clause, and where it is cut off. This is the largest remaining gap.

---

## Ibn Mālik — *Alfiyyah* and *Lāmiyyat al-Afʿāl*

| item | text | finding |
|---|---|---|
| definiteness ranks | v. 55 «كهم وذي… وهند وابني والغلام والذي» — **six** ranks | had four; الموصول and المضاف were missing |
| POS automaton | — | blocked Fiʿl→Fiʿl unconditionally, so valid `قام وقعد` was rejected — and it **contradicted the wāw classifier in the same codebase** |
| *Lāmiyyat al-Afʿāl* | v. 42, 43, 45, 47-48, 51, 60 | **already correct — verified 7/7** |
| awzān 114–129 | — | genuinely the imperfect forms; verified |

**Not implemented:** *Marātib al-Maʿārif* enforcement at decode time.

---

## Andalusian — taṣrīf (Ibn ʿUsfūr, Abū Ḥayyān, al-Shāṭibī)

### New classical analyzer (this session)

The shipped analyzer delegated to a lookup and discarded any root not literally in the inventory as
`<UNK>`. It applied **no derivational rules at all**. Added, in order:

| rule | basis | share of words resolved |
|---|---|---|
| iʿlāl (weak-letter substitution و/ي/ا) | muʿtall bāb divisions | **2.97 %** |
| mazīd (augmented 4-letter, drop the addition) | Ibn Mālik's students | 1.62 % |
| clitic/suffix stripping | Sībawayh's ḥurūf | ~2.4 % |
| idghām (doubled root C₁C₂→C₁C₂C₂) | Andalusian | 0.47 % |
| ibdāl / hamza (أ إ آ ؤ ئ → ء) | Andalusian | 0.19 % |

**Effect: `<UNK>` 4.29 % → 3.69 %, fixing 244 of 1,718 failures (14.2 %).**

Known weakness: `rule_weak` substitutes weak letters **blindly** and tests against the inventory. The
taṣrīf tradition determines *which* weak letter is correct **from the wazn**. That is the correct
implementation and should roughly double the iʿlāl recovery.

### Al-Shāṭibī, wāw functions

Four of the eight functions are classified (ATF / HAL / ISTINAF / QASAM). The other four are
**not specified** in the sources held here — that is a gap in the corpus, not in the code.

**Not implemented:** *al-Iktifāʾ* (clitic saturation), full jussive apocope in surface realisation.

---

## How each claim is verified

| artefact | what it does |
|---|---|
| `test_grammar_impl.py` | the 20-check audit against the primary texts — **10/20 failed** on the original implementation |
| `verify_v2.py` | the same audit on the corrected implementation — **all pass** |
| `khalil_fidelity.py` | extracts the letter-incompatibility rules from *al-ʿAyn* and measures each against the inventory, including attestation counts |
| `classical_analyzer.py` | measures the derivational ladder's UNK recovery, attributed per rule |

## Open items, ranked

1. **Sībawayh's valency / constituent stack** — largest gap; needed for how far governance reaches.
2. **Wazn-conditioned iʿlāl** — replaces blind substitution; should roughly double iʿlāl recovery.
3. **Word-class inventory** — 44 % of positions collapse into a single `<PARTICLE>` token, discarding
   the ism/fiʿl/ḥarf and ḥarf-subclass distinctions the grammarians catalogue.
4. **al-Iktifāʾ**, jussive apocope realisation, definiteness at decode.
5. Al-Shāṭibī's remaining four wāw functions — requires sources not currently held.
