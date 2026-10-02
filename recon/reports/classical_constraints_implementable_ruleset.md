# Classical Arabic Grammatical Constraints → Implementable Rule Set

**Derived exclusively from five Rootformer audit reports** (no rules invented; every rule is traceable to a line in one of the five sources):

| # | Source report |
| :-- | :--- |
| R1 | [sibawayh_hard_exclusion_and_khalil_fidelity_report.md](sibawayh_hard_exclusion_and_khalil_fidelity_report.md) |
| R2 | [comparative_audit_khalil_students_andalusians_vs_current_algorithms.md](comparative_audit_khalil_students_andalusians_vs_current_algorithms.md) |
| R3 | [andalusian_grammatical_algorithms_and_architectural_code.md](andalusian_grammatical_algorithms_and_architectural_code.md) |
| R4 | [how_ibn_khalil_would_architect_neural_networks.md](how_ibn_khalil_would_architect_neural_networks.md) |
| R5 | [ibn_khalil_next_root_morph_prediction_architecture.md](ibn_khalil_next_root_morph_prediction_architecture.md) |

**Target object being constrained:** a word event factorized as the 4-tuple `(P_t, R_t, W_t, S_t)` = (prefix, triliteral root, wazn/pattern, suffix) — R1 §3.A; R2 §1; R5 §3. Three candidate sets must be built:
`V_Roots` (§7.1), `V_Awzān` (§7.2), `V_Prefix` / `V_Suffix` (§7.3), plus an operator inventory `V_Op` (§7.4).

**Rule-kind legend** (each rule below is tagged with exactly one):
- **HARD MASK** — boolean `M[i] ∈ {0, −∞}`; implementable directly.
- **SCORING BIAS** — differentiable/additive preference; not a hard constraint in the sources.
- **POST-HOC FILTER** — applied to already-generated candidate strings/tokens.
- **VAGUE** — the document states the rule but not precisely enough to implement literally (explicitly flagged, not filled in).

---

## 1. Al-Khalīl ibn Aḥmad al-Farāhīdī — *Kitāb al-ʿAyn* (+ *Madkhal Kitāb al-ʿAyn*)

### K1. Radical inventory size
- **Rule:** The classical Arabic radical-root inventory is **9,114** roots. R2 §1 calls it "9,114 Radical Root Inventory"; R1 §5.1 asks for a "**binary 9,114 × 9,114 root compatibility adjacency tensor**". The deployed codebase vocabulary is **9,856 root vocab (9,114 classical)** (R2 §1).
- **Kind:** HARD MASK (vocabulary whitelist: mask all indices outside `V_Roots`).
- **Data needed:** `V_Roots` (9,856 entries; 9,114 classical).

### K2. Phonotactic incompatibility matrix (*I'tilāf wa Tanāfur al-Ḥurūf*)
- **Rule (R1 §5.1):** "Consonants with identical or adjacent points of articulation (*Makhraj*) cannot combine as adjacent radicals in a genuine root."
- **Rule (R2 §1):** "Phonotactic Incompatibility Matrix ($C_1 \neq C_2$, homorganic gutturals)" — i.e. `C1 ≠ C2` is inviolable, and homorganic gutturals cannot co-occur.
- **Kind:** HARD MASK (this is the canonical hard-mask rule of the corpus).
- **Data needed:**
  - `GUTTURALS = {ء, هـ, ع, ح, غ, خ}` — exact set given in R1 §5.1 as "*Ḥurūf al-Ḥalq*: `ء`, `هـ`, `ع`, `ح`, `غ`, `خ`".
  - `LABIALS = {ف, ب, م}` — "*Ḥurūf al-Shafah*: `ف`, `ب`, `م`". R1 says only "Co-occurrence is strictly regulated" — the specific regulation is **not stated** (VAGUE for labials).
  - A 9,114 × 9,114 binary adjacency tensor (R1 §5.1), which is *requested to be compiled*; it is **not provided** in any of the five documents.

### K3. Impossible-root examples (literal, verbatim)
- **Rule:** Roots of the shape `*عحـ` and `*حخـ` are **Mustaḥīl** (phonotactically impossible) (R1 §5.1). The `C1 == C2` case is hard-excluded in current code (R2 §1).
- **Kind:** HARD MASK.
- **Data:** the two literal patterns above + the `C1 ≠ C2` predicate.

### K4. Attested vs. unactualized permutations (*Mustaʿmal* / *Muhmal*)
- **Rule (R5 §2, quoting Al-Khalīl):** "every triliteral root permutes into six forms [3! = 6]; among them are the **attested (Mustaʿmal)** and the **unactualized (Muhmal)**, and the orbit of meaning rotates with the permutations of that single root."
- **Kind:** HARD MASK (per-permutation attested/unactualized flag) + SCORING BIAS (orbit sharing: attested permutations of one radical set should share mass).
- **Data:** per-root, per-permutation `attested ∈ {0,1}` flags; `S3` permutation index `σ` (§6.1).
- **Note:** the documents do **not** supply the attestation table; it must come from the 9,114/9,856 root inventory.

### K5. $\mathbb{S}_3$ permutation orbit equivariance (*Al-Taqālīb* / *Al-Ishtiqāq al-Akbar*)
- **Rule (R2 §1 + Oversight 5):** "Every root belongs to an orbit of up to 6 permutations (3!). Roots within the same permutation orbit share acoustic articulation properties." Current code violates this (see §8.2). Fix:
  `E(Root) = E_orbit({c1,c2,c3}) + E_order(σ ∈ S3)`
- **Kind:** SCORING BIAS / weight-sharing prior (the report calls it "enforces weight sharing … enabling zero-shot generalization", i.e. an architectural prior, **not** a hard mask).
- **Data:** `S3` group table over 3 positions; orbit id per root; permutation index σ.
- **Orbit enumeration, verbatim (R4 §4.2):**
  `O(R) = { (c1,c2,c3), (c1,c3,c2), (c2,c1,c3), (c2,c3,c1), (c3,c1,c2), (c3,c2,c1) }`
- **Worked orbit example (R4 §4.2):** `عَقَدَ` (contracted/knotted) ↔ `قَعَدَ` (settled/founded) ↔ `قَدَعَ` (curbed/restrained).
- **Worked orbit example (R5 §3):** `{ع-ق-د}` knottedness ↔ `{ق-ع-د}` settled foundation.
- **Worked semantic-core example (R4 §2, Axiom 3):** `{ك-ل-م}`, `{ل-ك-م}`, `{م-ل-ك}` all share the invariant of "impact / strength / impression."

### K6. Makhraj as a continuous $\mathbb{R}^{17}$ acoustic space
- **Rule (R4 Axiom 1):** "letters are not discrete categorical IDs; they are continuous acoustic vectors in a **17-dimensional** articulatory manifold. **Two consonants cannot co-occur in the same root if their Euclidean distance in this manifold creates phonetic friction (*Tanafur*).**"
- **Kind:** HARD MASK **if** a distance threshold is fixed (the documents do **not** give the threshold → the numeric threshold is VAGUE); otherwise SCORING BIAS via a friction penalty.
- **Data needed:** 17-dim articulatory coordinates for every letter. The documents give only the *ordering principle*: ordered "by the physical depth of the vocal tract, starting at the deep larynx (`ع`, `ح`, `هـ`, `خ`, `غ`) up to the lips (`ف`, `ب`, `م`)" (R4 Axiom 1). **The full 17-dim coordinate table is not given in any of the five reports.**

### K7. Prosodic rhythmic cadence (*ʿArūḍ*)
- **Rule (R2 §1):** listed as authentic Al-Khalīlian theory ("4. Prosodic Rhythmic Cadence (*ʿArūḍ*)").
- **Kind:** VAGUE — the report lists it in the theory column but states **no operational rule**, and it appears in neither the "Currently Implements" nor the "Missed" column with any detail. **No implementable content in these five documents.**

### K8. Bilateral / Triliteral / Quadriliteral / Quintiliteral hierarchy
- **Rule (R5 §2, verbatim):** "The entire orbit of the speech of the Arabs revolves around the Radical Roots (*Al-Uṣūl*): **Bilateral, Triliteral, Quadriliteral, and Quintiliteral**… and the overwhelming majority of their speech is built upon the Triliteral, which is the most balanced and potent of architectures."
- **Kind:** SCORING BIAS (root-length prior favouring triliteral). The documents do **not** give an exclusion for non-triliteral roots.
- **Data:** root-length field ∈ {2,3,4,5}.

### K9. Strip affixes to recover the root
- **Rule (R5 §2, verbatim):** "if you wish to comprehend a word, strip it of its accidental affixes and return it to its Root; for **affixes are mere transient accidents**, whereas the roots are the immutable bearers of meaning."
- **Kind:** HARD MASK on the root stream given the surface form (affix letters may not be counted as radicals); equivalently a deterministic affix-stripping operator.
- **Data:** prefix list + suffix list (§7.3) to be removed before root matching.

### K10. Bilinear Wazn coupling (root ⊗ wazn)
- **Rule (R4 §4.1):** `m_i = r_i W_Farahidi w_i + GeLU(W_r r_i + W_w w_i)`; ensures e.g. "knowledge" in `اسْتِعْلام` is mathematically identical to `مَعْلُوم`, differing only by the wazn operator.
- **Kind:** SCORING BIAS / architectural prior (a coupling layer, not a filter).

---

## 2. Sībawayh — *Al-Kitāb*

R1 §2 gives the governing formalism directly usable as the mask equation:

```
P(w_t | w_<t, h_Arabic) = Softmax(z_t + M_t)
M_t,i =  0      if candidate i is syntactically admissible by the governing operator
M_t,i = -inf    if candidate i violates Sībawayh's syntactic laws
∀ i ∈ Allowed(w_<t):  z_t,i + M_t,i = z_t,i      # network stays sovereign on admissible choices
```

### S1. Hard Grammatical Exclusion Mask (global)
- **Rule:** Implement `M ∈ {0, −∞}` in **both** the NRMP Decoder and the Dual-Layer Transmutation Decoder. "All heuristic additive logit boosting (+7.0 / +8.0 static keyword hacks) has been permanently eradicated" (R1 §1, §2).
- **Kind:** HARD MASK (this is the framework rule).

### S2. Operator governance on morphological awzān (the three operator masks)
Verbatim from R1 §3.A.2:

| Operator | Arabic list given | Masked to −∞ | Allowed |
| :--- | :--- | :--- | :--- |
| Preposition (*Ḥarf Jarr*) | — (type named only) | **All 44 finite verbal awzān (past and imperfect)** | only nominal awzān, **masādir**, and **participles** |
| Jussive (*Ḥarf Jazm*) | `لم`, `لما`, `لا الناهية` | **All 81 nominal awzān and all 28 past-tense awzān** | (imperfect verbs only, by implication) |
| Subjunctive (*Ḥarf Naṣb*) | `لن`, `كي`, `أن` | **Nominal awzān and past verbs** | imperfect verbs exclusively |
| Future aspect marker | `سـ` prefix or `سوف` | — | **imperfect verbs exclusively** |

- **Kind:** HARD MASK (all four).
- **Data needed:** the explicit awzān partitions — **44 finite verbal awzān**, **28 past-tense awzān**, **81 nominal awzān** (plus masādir and participles as separate nominal-compatible classes). These counts are given; **the enumerated awzān lists themselves are NOT in any of the five documents.**

### S3. Prefix masks
Verbatim R1 §3.A.3:
- "Following a preposition, preposition prefixes (`بـ`, `لـ`, `كـ`, `في`) are **masked to −∞** (no double prepositions)."
- "Following jussive/subjunctive operators, definite article prefixes (`الـ`, `والـ`) are **masked to −∞**."
- **Kind:** HARD MASK.
- **Data:** `P_PREP = {بـ, لـ, كـ, في}`; `P_DEF = {الـ, والـ}`.

### S4. Suffix mask
Verbatim R1 §3.A.3: "For verbal awzān, feminine noun suffixes (`ة`, `ية`) are **masked to −∞**."
- **Kind:** HARD MASK.
- **Data:** `S_FEM = {ة, ية}`.

### S5. Unconditional purge of the catch-all meta-token `<PARTICLE>` (index 4)
- **Rule (R1 §3.A.1):** Un-categorized particles were mapped to index 4 (`<PARTICLE>`), producing non-words. "**Index 4 is classified as *Muhmal* (non-word) and masked to −∞ unconditionally.**"
- **Al-Khalīl's Principle (R1 §3.A.1):** "Every particle in Arabic is an actual lexical entity (*Harf Maʿnā*)", represented by dedicated particle roots, with the literal examples `<P:في>`, `<P:من>`, `<P:على>` (and `<P:أن>` listed as an emitted root in the benchmark).
- **Kind:** HARD MASK (index 4 always −∞).
- **Data:** particle-root vocabulary with members `<P:في>`, `<P:من>`, `<P:على>`, `<P:أن>`.

### S6. Inqiṭāʿ al-ʿAmal (scope/constituent termination)
- **Rule (R1 §3.B.3):** "Once all governed arguments of the sentence are realized (`... into the heart of whomever he wills`), the syntactic nexus is complete. **All tokens except `<EOS>` are masked to −∞.**"
- **Rule (R2 §2 Oversight 1, *Al-Kitāb* Vol. 1):** "An operator has a strict saturation valency. A preposition (*Ḥarf Jarr*) governs a Noun Phrase (*Jārr wa-Majrūr*). Once the noun phrase (and its attached genitive/adjective) concludes, the operator's governing force **terminates** (*Inqiṭāʿ al-ʿAmal*)."
- **Kind:** HARD MASK (terminal `{EOS}`-only mask) + POST-HOC/HARD filter per operator.
- **Data:** per-operator **valency** integer; a stack of `{type, remaining}` (R2 §2 gives the literal `SibawayhConstituentStack` with `push_operator(op_type, valency=1)`, `consume_argument()` decrementing and popping when `remaining <= 0`).

### S7. Structural valency & saturation (*Al-Iktifāʾ*) with cliticized object pronouns
- **Rule (R2 §1, "Structural Valency & Saturation (*Al-Iktifā'*)"; R1 §5.2 "Al-Rutbah wa al-Aṣālah"):** "In transitive verbal constructions with cliticized object pronouns (`يقذفه`), the direct object is already saturated inside the verb. **A bare accusative noun following it is impossible unless it is a circumstantial specification (*Tamyīz*) or hal (*Ḥāl*).**"
- **Kind:** HARD MASK (accusative-noun candidate blocked after a verb whose object slot is saturated) — with an exception set `{Tamyīz, Ḥāl}`.
- **Data:** per-verb transitivity; clitic object-pronoun suffix list; a saturation state counter updated per argument consumed (R1 §5.2 "Action: Implement structural valency tracking in the NRMP generator to dynamically update argument saturation states").

### S8. Structural rank (*Al-Rutbah wa al-Aṣālah*) — Noun is *Aṣl*, Verb is *Farʿ*
- **Rule (R1 §5.2, *Al-Kitāb*):** "The Noun is the Origin (*Al-Aṣl*), and the Verb is the Derivative Branch (*Al-Farʿ*)."
- **Kind:** SCORING BIAS (derivational priority) — used in R1 as the justification for S7, not as a standalone mask.

### S9. POS transition automaton / "Theory of the Operator" + Syntactic Followers (Tawābiʿ)
- **Rule (R2 §1):** "Syntactic Followers (*Al-Tawābiʿ*: **Naʿt, ʿAṭf, Badal**)" and "Five Validity States (*Awjuh al-Kalām*)". The audit says operator state must not reset after one word but must track "multi-word prepositional phrases, genitive chains (**Iḍāfah**), or case inheritance across *Al-Tawābiʿ* (adjectives inheriting noun case)."
- **Kind:** HARD MASK for case inheritance; the "Five Validity States (*Awjuh al-Kalām*)" are named but **never enumerated in the five documents → VAGUE**.
- **Data:** follower-type enum `{Naʿt, ʿAṭf, Badal}`; case values `{Nominative, Accusative, Genitive, Jussive}` (R5 §1 lists exactly these four as `V_Iʿrāb`).

### S10. Naʿt al-Nakirah («الجمل بعد النكرات صفات») — adjectival relative clauses
- **Rule (R1 §3.B.2; the report calls it "Sībawayh's Canon"):** "When an indefinite noun is followed by a verbal sentence (`نور يقذفه الله`), the English decoder is constrained:
  - Directly after `light`, **all non-relative tokens are masked to −∞** → enforces relative pronoun `that`.
  - Directly after `that`, **passive participle forms and prepositions are masked to −∞** → enforces active agent `god`.
  - Directly after `god`, **non-verbs are masked to −∞** → enforces transitive active verb `casts`."
- **Kind:** HARD MASK (stepwise, state-conditioned on the previous emitted token).

### S11. Al-Ibtidāʾ / equational nominal sentence
- **Rule (R1 §3.B.1):** "When translating an equational sentence starting with a Mubtada' (`العلم`, `الجوهر`, `الوجود`), the English subject must be predicated by the copula (`is` / `is a`). **Prepositions (`of`, `in`, `to`) directly following the subject are masked to −∞**, preventing corruptions like `"knowledge of the light"`."
- **Kind:** HARD MASK.
- **Data:** Mubtadaʾ trigger list (examples given: `العلم`, `الجوهر`, `الوجود`); English preposition blocklist `{of, in, to}`.

### S12. Operator→Patient potential field with latent/elided operators
- **Rule (R4 §4.3):** `h_patient^(l+1) = h_patient^(l) + Σ_{k∈A(patient)} Φ_amal(h_operator_k, h_patient)`, where:
  - "A Transitive Verb projects an **Accusative Gradient (∇_naṣb)** directly to its Object."
  - "A Subject Operator (*Ibtidāʾ*) projects a **Nominative Gradient (∇_rafʿ)** to both Subject and Predicate (*Mubtadaʾ wa Khabar*)."
  - "An Ellipsis Detector (*Taqdīr al-Maḥdhūf*) injects the latent operator into zero-copula sentences (e.g. `زيدٌ قائمٌ`), **without generating explicit phantom words**."
  - Also (R4 Axiom 4): operators radiate potential to patients "even across long distances or when the operator is deleted but latent (*Maḥdhūf Muqaddar*)."
- **Kind:** SCORING BIAS (vector field) + HARD MASK via the latent operator's case assignment (∇_naṣb → accusative on object; ∇_rafʿ → nominative on both terms).
- **Data:** operator→governed-argument adjacency `A(patient)`; gradient/case projection per operator class.

### S13. Operator state names that exist in code (R2 §1, "Currently Implements")
`HARF_JARR`, `HARF_JAZM`, `HARF_NASB`, `INNA`, `KANA` — a **bigram** operator state tracker.
**Current defect (R2 §1 + Oversight 1):** "`op_state` is a flat 1-step memory. It resets immediately after token t+1, failing when a noun phrase has an adjective (`في الأجسام الشفافة`)."

---

## 3. Ibn Jinnī — *Al-Khaṣāʾiṣ* (and *Sirr Ṣināʿat al-Iʿrāb*)

### J1. Root consonants = *Jawhar*; vowels/affixes = *Aʿrāḍ*
- **Rule (R5 §2, verbatim):** "The root consonants are the substance (Al-Jawhar), and the vowels and affixes are the accidents (Al-Aʿrāḍ) subsisting within it. Whoever seeks meaning from the external surface word without reducing it to its root is like one who pursues a shadow while remaining blind to the physical body that cast it."
- **Kind:** HARD MASK (root orthogonality: root stream must be invariant across `مُسْتَحْدَث` / `حَادِث` / `حُدُوث` / `أَحْدَثَ`, which all share `{ح-د-ث}` — R5 §4.1) + SCORING BIAS (dual-stream separation, R4 Axiom 2).

### J2. Greater Derivation (*Al-Ishtiqāq al-Akbar*)
- **Rule (R5 §2, verbatim):** "it is to take a triliteral root and bind it and all its six permutations to a single unifying semantic core testified to by all derived branches of speech."
- **Kind:** SCORING BIAS (see K5; R4 §2 Axiom 3 attributes the shared-core result to Ibn Jinnī).

### J3. Vocalic physics: *Ḥarakāt*, *Sukūn*, *Iʿlāl*, *Iltiqāʾ al-Sākinayn*
- **Rule (R1 §5.3, *Sirr Ṣināʿat al-Iʿrāb* & *Al-Khaṣāʾiṣ*):** "Short vowels are acoustic motions (*Ḥarakāt*); sukūn is stillness. **Weak letters (`و`, `ي`, `ا`) are elongated vowels susceptible to elision (*Iʿlāl*) when meeting another quiescent letter (*Iltiqāʾ al-Sākinayn*).**"
- **Required action (R1 §5.3):** "Expand `farahidian_syntactic_realizer.py` into a full dynamic vocalic physics simulation that **automatically executes weak-letter apocope in jussive moods (`لَمْ يَقُلْ` instead of `*لم يقول`)**."
- **Kind:** POST-HOC FILTER (surface realization rewrite) + HARD MASK on the illegal `*لم يقول` surface.
- **Data:** `WEAK = {و, ي, ا}`; quiescence/sukūn flag per letter position; jussive-mood marker.

### J4. Morphosemantic intensity scaling (Form II / Form V)
- **Rule (R2 §1, "Phonetic Mimetic Scaling (Form II/V semantic intensification)").** Current defect: "Form II (`فَعَّلَ` — extensive) and Form V (`تَفَعَّلَ` — reflexive-gradual) are translated identically to Form I (`فَعَلَ`), losing morphological intensity."
- **Kind:** SCORING BIAS / generation-side constraint (needs a per-form intensity feature; the documents do not give a numeric intensity table → magnitude VAGUE, the form labels are exact).

---

## 4. Ibn Mālik — *Al-Alfiyyah* / *Al-Khulāṣah* and *Lāmiyyat al-Afʿāl*

### M1. Guttural imperfect-verb decision tree (*Lāmiyyah* v. 49 / verses 17–55)
- **Verse quoted (R3 §2, Alg. 2):** «وفتح ما حرف حلق غير أوله ... اشع بالاتفاق كآت صيغ من سألا» — "Fatḥah is mandatory whenever the non-initial radical is a guttural letter, unanimously agreed upon, like the imperfect of 'sa'ala'."
- **Kind:** HARD MASK on `W_{t+1}`.
- **Data:** `THROAT_LETTERS` (the report uses this constant name; **the enumeration of the set is not given in R3**; R1 §5.1 supplies a guttural set `{ء, هـ, ع, ح, غ, خ}` which is the closest literal list in the corpus).
- **Exact algorithm, verbatim (R3 §2):**

```python
def predict_imperfect_wazn(r1, r2, r3, past_vowel):
    if past_vowel == 'u': return "يَفْعُلُ"  # fa'ula -> yaf'ulu (100%)
    if past_vowel == 'i': return "يَفْعَلُ"  # fa'ila -> yaf'alu (100%)
    # past_vowel == 'a' (fa'ala)
    if r2 in THROAT_LETTERS or r3 in THROAT_LETTERS:
        return "يَفْعَلُ"  # Guttural Fatḥah Rule (100% deterministic)
    if r1 == 'و' or r2 == 'ي':
        return "يَفْعِلُ"  # Assimilated / Hollow Kasrah Rule
    if r2 == 'و':
        return "يَفْعُلُ"  # Hollow Waw Dammah Rule
    return "يَفْعُلُ"      # General transitive default
```

- **Verified examples (R3 §3.1, verbatim with reasons):**
  `فتح → يَفْعَلُ` (R3 = ح), `سأل → يَفْعَلُ` (R2 = ء), `ذهب → يَفْعَلُ` (R2 = ه), `شرح → يَفْعَلُ` (R3 = ح), `كرم → يَفْعُلُ` (Fa'ula Dammah), `فرح → يَفْعَلُ` (Fa'ila Fathah), `ورث → يَفْعِلُ` (Assimilated Kasrah). All marked **[100% Match]**.
- **Stated impact:** "Masks impossible `W_{t+1}` candidates, reducing wazn search entropy by **>60%**."

### M2. Part-of-speech transition automaton (*Alfiyyah* v. 8–15)
- **Verse quoted (R3 §2, Alg. 3):** «كلامنا لفظ مفيد كاستقم ... واسم وفعل ثم حرف الكلم» — "Our speech is a beneficial utterance: Noun (*Ism*), Verb (*Fiʿl*), and Particle (*Ḥarf*)."
- **Rules (R3 §2, Alg. 3), literal:**
  1. `S3(Ḥarf) → S3(Ḥarf)` is **FORBIDDEN** (no consecutive unattached particles).
  2. `S2(Fiʿl) → S2(Fiʿl)` is **FORBIDDEN** without coordinating particles (`و`, `فـ`).
  3. Every proposition must satisfy binary predication closure: `Proposition ⟹ {Musnad, Musnad Ilayh}`.
- **Kind:** HARD MASK.
- **Data:** 3-state enum {Ism, Fiʿl, Ḥarf}; coordinating-particle set `{و, فـ}`; Musnad/Musnad-Ilayh slot pair.
- **Verified outputs (R3 §3.2):** `Harf→Harf` **STRICTLY BLOCKED**; `Fi'l→Fi'l` **STRICTLY BLOCKED**; `Harf→Ism` **PERMISSIBLE**; `Fi'l→Ism` **PERMISSIBLE**. "Stutter rate: **0.0%**".
- **Reported in R2 §1 as currently existing:** "POS constraint blocking consecutive particles."

### M3. Definiteness hierarchy (*Marātib al-Maʿārif*)
- **Rule (R2 §1):** "subject (*Mubtada'*) must be **≥** predicate (*Khabar*) in definiteness rank" with the exact rank order given:
  **Pronoun > Proper > Demonstrative > Definite > Indefinite.**
- **Kind:** HARD MASK (mask all wazn/prefix/suffix candidates that would produce a khabar of strictly higher definiteness rank than the mubtadaʾ) — or SCORING BIAS if violated candidates are merely penalized. R2 lists it under "**What We Oversaw / Missed**", i.e. it is **not implemented**.
- **Data:** definiteness-rank table for the five classes above; per-word class label.

---

## 5. Al-Suhaylī — *Natāʾij al-Fikr*

### Su1. Latent subject pronoun (*Al-Ḍamīr al-Mustatir*) unpacking
- **Quotation (R2 §2 Oversight 2, *Natāʾij al-Fikr* lines 2031–2038, verbatim):** «وتحقيق القول أن الفاعل مضمر في نفس المتكلم، ولفظ الفعل متضمن له دال عليه، واستغني عن إظهاره لتقدم ذكره...» ("The subject is latent in the speaker's mind, and the verb's verbal form contains and indicates it…").
- **Rule:** "If verb `V_t` has no overt nominal argument in its local constituent bracket: unpack the verbal conjugation prefix/suffix into its English pronominal equivalent". **Exact mapping table given:**
  - `يَفْعَلُ` → `it / he` + Verb
  - `تَفْعَلُ` → `it / she` + Verb
  - `أَفْعَلُ` → `I` + Verb
  - `نَفْعَلُ` → `we` + Verb
- **Kind:** HARD MASK (force insertion of a subject pronoun; block verb-only emission) — the report calls it "the Algorithmic Fix"; current defect is telegraphic English (`obtained by practice` instead of e.g. `it is attained`).
- **Data:** the 4-row conjugation-prefix→pronoun map above.

### Su2. Tanwīn as syntactic disconnection (*Al-Infisāl*) and mutual exclusion of `الـ` and tanwīn
- **Rule (R2 §1):** "1. Tanwīn as Syntactic Disconnection (*Al-Infisāl*) 2. **Mutual Exclusion of `الـ` and Tanwīn**".
- **Kind:** HARD MASK (cannot co-occur: definite article `الـ` with tanwīn suffix on the same word).
- **Data:** `P_DEF` (see S3); tanwīn suffix set (the documents do not enumerate the tanwīn glyphs — implementable as the standard `ً ٌ ٍ` set, but the exact list is **not given** in the five reports → the *pair-exclusion predicate* is concrete, the *suffix inventory* is VAGUE).
- **Reported as currently existing (R2 §1):** only "Subword-free prefix/suffix tokenization".

---

## 6. ʿAbd al-Qāhir al-Jurjānī — *Dalāʾil al-Iʿjāz*

### Ju1. Indivisible bipartite exclusivity frames (*Al-Ḥaṣr wa-l-Qaṣr*)
- **Rule (R2 §1 + Oversight 3):** "Particles of restriction (`إنما`, `ما ... إلا`, `ليس ... إلا`) do not operate as isolated words; they form a **discontinuous bipartite frame** enclosing the predicate." Exact transforms:
  - `Negation + X + إلا + Y` ⟹ `X is nothing other than Y`
  - `إنما + X + Y` ⟹ `X is exclusively / none other than Y`
- **Kind:** POST-HOC FILTER over the generated token sequence (pattern match + rewrite) and HARD MASK (block token-by-token literal rendering inside the frame).
- **Data:** frame patterns `إنما`; `ما … إلا`; `ليس … إلا`. Current defect: `ليس النظم إلا توخي معاني النحو` → `is the نظم except...`.

### Ju2. Naẓm — dependency isomorphism
- **Rule (R1 §5.4, *Dalāʾil al-Iʿjāz*):** "Eloquence is neither in the word nor in the raw thought; it is in the **exact correspondence of syntactic dependencies**."
- **Required action:** "Generalize the Jurjānī dependency planner from clause-level patterns to arbitrary tree-structured dependency graphs across multi-sentence paragraphs."
- **Kind:** SCORING BIAS / architectural objective. **VAGUE:** no concrete constraint set is given.

### Ju3. Information structure & focus (*Al-Taqdīm wa-l-Taʾkhīr*)
- **Rule (R2 §1):** listed as authentic Jurjānī theory ("2. Information Structure & Focus (*Al-Taqdīm wa-l-Taʾkhīr*)").
- **Kind:** SCORING BIAS (fronting/inversion preference). No operational rule stated in any of the five documents.

### Ju4. 8-Layer Jurjānī Transmuter
- **Reported as currently existing (R2 §1):** "8-Layer Jurjānī Transmuter with multi-stage cross-attention from Backbone Layers **8, 14, 24**."

---

## 7. Ibn Maḍāʾ al-Qurṭubī — *Kitāb al-Radd ʿalā al-Nuḥāt*

### Md1. Direct Semantic Realism (anti-phantom-operator)
- **Quotation (R3 §2, Alg. 1, lines 46–120, verbatim):** «فالعمل من النصب والرفع والجر والجزم، إنما هو للمتكلم نفسه لا لشيء غيره... وأما القول بأن الألفاظ يحدث بعضها بعضا فباطل عقلا وشرعا.» ("Inflection is produced by the communicative intention of the speaker himself, not by adjacent words… The claim that words generate one another like causal dominoes is rationally void.")
- **Rules:**
  1. "Reject token-to-token autoregressive domino drift."
  2. "Project directly from the global sentence representation to the target semantic concept: `L_Ibn Maḍāʾ = 1 − cos(h_Backbone^(24), e_concept)`."
  3. "Drop spurious phantom insertions (reduces semantic divergence by **80.4%**)."
- **Kind:** SCORING BIAS (cosine alignment loss) + POST-HOC FILTER (duplicate/phantom token removal).
- **Reported as currently existing (R2 §1):** "Direct Cosine Realism Loss between Layer 24 and Concept Space (`L_Maḍāʾ = 1 − cos`). 2. **Duplicate token filter**."

### Md2. Elimination of virtual operators (*Al-ʿAwāmil al-Muqaddarah*)
- **Rule (R2 §1 / R3 Alg. 1):** reject invented hidden governors and hypothetical deleted verbs.
- **Kind:** HARD MASK (forbid insertion of phantom operators/tokens not licensed by surface material) — the concrete negative list is **VAGUE**; only the general prohibition is stated.
- **Verified (R3 §3.4):** "Spurious phantom elimination: `'the the'` → `'the'`, `'is is'` → `'is'`, phantom copula collapse → clean natural output."

### Md3. Rejection of artificial latent restorations in idioms
- **Rule (R2 §1):** "Rejection of Artificial Latent Restorations in Idioms."
- **Kind:** HARD MASK / POST-HOC FILTER. **VAGUE:** no idiom list or operational predicate is supplied. **Tension with S12/Ju-b:** R4's "Ellipsis Detector (*Taqdīr al-Maḥdhūf*) injects the latent operator … without generating explicit phantom words" is the compatible reading; the five documents do not resolve the boundary beyond that.

---

## 8. Abū Ḥayyān al-Gharnāṭī — *Kitāb al-Idrāk li-Lisān al-Atrāk* & *Irtishāf al-Ḍarab*

### A1. Tripartite polyglot syntactic bridge (Arabic operator → Turkish case → English realization)
- **Exact table (R3 §2, Alg. 4), transcribed verbatim:**

| Arabic Syntactic State | Ottoman Turkish Pivot | English Analytical Realization | Example Arabic | Example Turkish | Example English |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Mubtada' (*Rafʿ*)** | Yalın / `-dir` (Copula) | `Subject + is/are` | العلمُ نورٌ | ilim nurdur | **knowledge is light** |
| **Mafʿūl Bih (*Naṣb*)** | Belirtme (`-i / -ı`) | `Verb + Direct Object` | عرفتُ الحقَّ | hakkı bildim | **I knew the truth** |
| **Iḍāfah (*Jarr*)** | İlgi hâli (`-in / -ın`) | `Noun + of + Complement` | رأسُ الحكمةِ | hikmetin başı | **the head of wisdom** |
| **Preposition (*Bi-*)** | Vasıta (`ile`) | `through / by means of` | بالممارسة | idman ile | **through practice** |
| **Preposition (*Min*)** | Ayrılma (`-den / -dan`) | `from / out of / by reason of` | من الإيمان | imandandır | **is from faith** |
| **Preposition (*Ilā*)** | Yönelme (`-e / -a`) | `to / toward` | إلى الحق | hakka | **to the truth** |
| **Preposition (*Fī*)** | Bulunma (`-de / -da`) | `in / within` | في النفس | nefiste | **in the soul** |

- **Kind:** HARD MASK on the prefix/operator stream (Arabic case state → required English realization) + deterministic surface templates.
- **Data:** the 7-row pivot table above; the four-valued case enum {Rafʿ, Naṣb, Jarr, Jazm} (R5 §1).
- **Reported as currently existing (R2 §1):** "Tripartite Turkish case bridge for prepositions and copula."

### A2. Typological relational word-order inversion (SOV ↔ SVO) / prepositional adjunct inversion
- **Rule (R2 §1 + §1 row for Abū Ḥayyān):** "**Prepositional Adjunct Inversion**: Arabic fronted adjuncts (`بالشك لا يزول`) are emitted fronted rather than inverted to English post-verbal order (`is not dispelled by doubt`)." Classical theory entry: "Typological Relational Word Order Inversion (SOV ↔ SVO)".
- **Kind:** POST-HOC FILTER (reorder) and/or HARD MASK preventing fronted-adjunct surface order in the English decoder.
- **Data:** adjunct-type tag (prepositional phrase before verb); target verb position.

---

## 9. Al-Shāṭibī — *Al-Maqāṣid al-Shāfiyah* & *Al-Muwāfaqāt*

### Sh1. Discourse functions of the coordinator `و`
- **Rule (R2 §1 + Oversight 4):** "The letter `و` (*waw*) is not a uniform connective; it has **8 distinct syntactic functions**." *Al-Maqāṣid al-Shāfiyah*, Vol. 1.
- **Explicitly enumerated in the document — only 4 of the 8 (verbatim):**
  1. *Waw al-ʿAṭf* (Conjunction): connects words in a list → English `and`.
  2. *Waw al-Ḥāl* (Circumstantial): introduces a subordinate clause → English `while / as`.
  3. *Waw al-Isti'nāf* (Discourse Resumption): begins a new proposition → English `.` (Period / Sentence Break).
  4. *Waw al-Qasam* (Oath): `والله` → English `By [God]`.
- **⚠️ The report asserts 8 types but names only these 4. The remaining 4 are NOT in any of the five documents.**
- **Operational rule actually given (R2 §2 Oversight 4):** "If `و` occurs at the start of a new independent proposition (followed by *Ism* or *Fiʿl* with completed previous valency), treat as `Waw al-Isti'nāf` → capitalize next word and emit a period `.` instead of `and`."
- **Kind:** HARD MASK / POST-HOC FILTER (per-`و` function classifier + rewrite). Current defect: every `و` → `and` ("`and the knowledge and the intellect and it guides`").

### Sh2. Information-theoretic economy: ellipsis (*Ḥadhf*) vs. explication (*Iẓhār*)
- **Quotation (R3 §2, Alg. 5, *Al-Maqāṣid al-Shāfiyah* vol. 1, p. 424 & *Al-Muwāfaqāt*):** «فإن المفرد لا إفادة له من حيث هو مفرد، وإنما تحصل الفائدة بالإسناد.» ("An isolated word has no communicative utility of itself; utility exists strictly in the predication nexus.")
- **Rule (formal):**
  `Realization State = Ellipsis (Ḥadhf) if I(Token | Context) < τ ; Explication (Iẓhār) if I(Token | Context) ≥ τ`
  "Governs when to emit English explicit relative pronouns (*'that which'*) vs. concise scholastic terms."
- **Kind:** HARD MASK driven by a thresholded information score (τ is a tunable; **the value of τ is not given**).
- **Data:** per-candidate conditional information `I(Token | Context)`; threshold τ; the explicit-relative-pronoun realization set (`that which`, etc.).
- **Reported as currently existing (R2 §1):** the project has a "**75:25 Balanced Basran-Andalusian Synthesis dataset**" — the only Al-Shāṭibī-related item in the "currently implements" column.

---

## 10. Consolidated tables and lists appearing in the five documents

### 10.1 Consonant / letter sets (only these are literal in the corpus)
| Set | Members | Source |
| :--- | :--- | :--- |
| Gutturals (*Ḥurūf al-Ḥalq*) | `ء`, `هـ`, `ع`, `ح`, `غ`, `خ` | R1 §5.1 |
| Labials (*Ḥurūf al-Shafah*) | `ف`, `ب`, `م` | R1 §5.1 |
| Weak letters (elongated vowels) | `و`, `ي`, `ا` | R1 §5.3 |
| `THROAT_LETTERS` | **not enumerated** (constant named only) | R3 §2 Alg. 2 |
| Imperfect-rule special radicals | `r1 == 'و'`, `r2 == 'ي'`, `r2 == 'و'` | R3 §2 Alg. 2 |
| Impossible-root examples | `*عحـ`, `*حخـ` | R1 §5.1 |

### 10.2 Operator lists (literal)
| Operator class | Literal members given | Source |
| :--- | :--- | :--- |
| Jussive (*Ḥarf Jazm*) | `لم`, `لما`, `لا الناهية` | R1 §3.A.2 |
| Subjunctive (*Ḥarf Naṣb*) | `لن`, `كي`, `أن` | R1 §3.A.2 |
| Future aspect markers | `سـ` (prefix), `سوف` | R1 §3.A.2 |
| Preposition prefixes | `بـ`, `لـ`, `كـ`, `في` | R1 §3.A.3 |
| Definite article prefixes | `الـ`, `والـ` | R1 §3.A.3 |
| Dedicated particle roots | `<P:في>`, `<P:من>`, `<P:على>`; plus `<P:أن>` emitted in benchmark | R1 §3.A.1, §3.A.1 result bullet |
| Operator state names in code | `HARF_JARR`, `HARF_JAZM`, `HARF_NASB`, `INNA`, `KANA` | R2 §1 |
| Coordinating particles | `و`, `فـ` | R3 §2 Alg. 3 |
| Restriction particles / frames | `إنما`, `ما ... إلا`, `ليس ... إلا` | R2 Oversight 3 |
| Mubtadaʾ trigger examples | `العلم`, `الجوهر`, `الوجود` | R1 §3.B.1 |
| English prepositions blocked after subject | `of`, `in`, `to` | R1 §3.B.1 |
| Clitic object-pronoun example | `يقذفه` | R1 §5.2 |

### 10.3 Pattern (wazn) inventory — every literal pattern string in the five documents
| Pattern | Meaning/role | Source |
| :--- | :--- | :--- |
| `يَفْعُلُ`, `يَفْعِلُ`, `يَفْعَلُ` | the three imperfect stems of `فَعَلَ` | R3 §2 Alg. 2 |
| `فَعَلَ` | Form I past | R2 §1; R3 Alg. 2 |
| `فَعَّلَ` | Form II, "extensive" | R2 §1 |
| `تَفَعَّلَ` | Form V, "reflexive-gradual" | R2 §1 |
| `استفعل` / `اسْتِفْلَحَ` / `اسْتِعْلام` | Form X, "seeking" | R4 §7; R5 §3 |
| `تَفَعْلَلَ` / `المُتَجَوْهِر` | quadriliteral template example | R4 §7 |
| `مُسْتَحْدَث`, `حَادِث`, `حُدُوث`, `أَحْدَثَ` | inflectional family of `{ح-د-ث}` | R5 §4.1 |
| `مَعْلُوم`, `مُحْدَث`, `وُجُود`, `مُسْتَحْدَث` | participles / maṣdar | R4 §4.1; R5 §4 |

**Counts given (no enumerated list):** `~120` canonical templates (R5 §1), `128` in the PyTorch head (R5 §5), split into **44 finite verbal awzān**, **28 past-tense awzān**, **81 nominal awzān** (R1 §3.A.2), `~15`/`16` iʿrāb classes (R5 §1, §5).

### 10.4 The four-value iʿrāb enum (R5 §1)
`Marfūʿ` (Nominative), `Manṣūb` (Accusative), `Majrūr` (Genitive), `Majzūm` (Jussive).

### 10.5 The 4-tuple factorization and vocabularies
- `(P_t, R_t, W_t, S_t)` = (prefix, root, wazn, suffix) — R1 §3.A.
- `R ∈ V_Roots` ≈ **10,000** valid Arabic roots (R5 §1); `V_Roots` deployed = **9,856** (9,114 classical) (R2 §1).
- `W ∈ V_Awzān` ≈ **120** canonical templates (R5 §1).
- `I ∈ V_Iʿrāb` ≈ **15** syntactic markers (R5 §1).
- Factorized objective (R5 §3): `P(Word) = P(R)·P(W|R,Context)·P(I|R,W,Context)`; `L_Total = λ_R L_Root + λ_W L_Wazn + λ_S L_Syntax`.

### 10.6 Sībawayh's structural valencies and the stack (R2 §2)
```python
class SibawayhConstituentStack:
    def __init__(self):
        self.stack = []  # Stack of active operators and their valencies
    def push_operator(self, op_type, valency=1):
        self.stack.append({'type': op_type, 'remaining': valency})
    def consume_argument(self):
        if self.stack:
            self.stack[-1]['remaining'] -= 1
            if self.stack[-1]['remaining'] <= 0:
                self.stack.pop() # Inqiṭāʿ al-ʿAmal (Bracket Closure)
```

### 10.7 The 4 Axioms / 4 Functional Hegemonies (R4 §2–3)
1. **Acoustic Physics** (*Al-Makhārij wa-l-Ṣifāt*) — articulatory vocal-tract coordinates in ℝ¹⁷.
2. **Invariant Radical** (*Al-Jawhar wa-l-ʿAraḍ*) — separation of root matrix from morphological wazn.
3. **Group Orbits** (*Al-Taqālīb al-Sittah*) — permutation symmetry S₃ across all root radicals.
4. **Sībawayhian Dynamics** (*Naẓariyyat al-ʿĀmil wa-l-ʿAmal*) — directed operator-to-patient gradient potential field.

Layer assignment (R4 §3): Layer 0 = phonetic acoustic manifold (17D) + consonant-vowel decomposer; **Layers 1–4** = Farāhīdian articulatory resonance & root extraction ("Consonant Friction Loss penalizes un-Farāhīdian phonotactics"); **Layers 5–8** = bilinear morphological tensor lattice (*Al-Mīzān al-Ṣarfī*), `h_l = (W_r r) ⊗ (W_w w)`; **Layers 9–12** = S₃ Taqālīb orbit attention; **Layers 13–16** = Sībawayhian syntactic flow, latent zero-copula reconstruction (*Al-Muqaddar fī al-Niyyah*), scholastic metaphysical projection.

---

## 11. Implementation-status audit (what the documents claim exists / is missing / is wrong)

### 11.1 Named as **already implemented** (with the name and file literally given)
| Named algorithm / class / file | Claim | Source |
| :--- | :--- | :--- |
| **Sībawayh's Hard Grammatical Exclusion Mask** (`M ∈ {0,−∞}`) | "implemented … across both the NRMP Decoder … and the Dual-Layer Transmutation Decoder"; status line says "100% Formalized, Integrated & Empirically Verified"; zero additive bias | R1 header, §1, §2 |
| **NRMP Decoder** (`rootformer_v18_nrmp_model.py`) | `<PARTICLE>` index-4 purge; operator governance on awzān; prefix/suffix masks | R1 §3.A |
| **Transmuter Decoder** (`basran_guided_transmuter.py`) | Ibtidāʾ/copula nexus; Naʿt al-Nakirah relative-clause masks; Inqiṭāʿ al-ʿAmal EOS mask | R1 §3.B |
| **`farahidian_syntactic_realizer.py`** | **Exists but incomplete** — R1 §5.3 says to "**expand**" it into a full dynamic vocalic physics simulation | R1 §5.3 |
| **`andalusian_grammatical_algorithms.py`** | "All five algorithms were implemented in `andalusian_grammatical_algorithms.py` and executed on the NVIDIA RTX PRO 4500 Blackwell GPU" | R3 §3 |
| **Ibn Mālik Verb Rule** (guttural imperfect tree) | implemented + 7/7 verified examples [100% Match] | R3 §2–3 |
| **Ibn Mālik POS Automaton** | implemented; Harf→Harf and Fi'l→Fi'l STRICTLY BLOCKED | R3 §2–3 |
| **Abū Ḥayyān Syntactic Bridge** | implemented; 3 verified row outputs | R3 §2–3 |
| **Ibn Maḍāʾ Realism** (`L = 1 − cos`) | implemented; duplicate/phantom elimination verified | R3 §2–3; R2 §1 |
| **Ibn Maḍāʾ duplicate token filter** | implemented | R2 §1 |
| **Abū Ḥayyān tripartite Turkish case bridge** (prepositions + copula) | implemented | R2 §1 |
| **8-Layer Jurjānī Transmuter** | implemented, "multi-stage cross-attention from Backbone Layers 8, 14, 24" | R2 §1 |
| **`SibawayhConstituentStack`** (bracket-closure stack) | given as the *fix to write* (code block) — presented as the algorithm, not as verified-deployed; the "Current Defect" (flat 1-step `op_state`) is the deployed state | R2 §2 Oversight 1 |
| **Bigram operator state tracking** (`HARF_JARR`, `HARF_JAZM`, `HARF_NASB`, `INNA`, `KANA`) | implemented but **defective** | R2 §1–2 |
| **Single-step affix and wazn exclusion masks** | implemented | R2 §1 |
| **Pharyngeal Fatḥah rule for throat letters** (`R2, R3 ∈ Throat ⟹ يَفْعَلُ`) | implemented | R2 §1 |
| **POS constraint blocking consecutive particles** | implemented | R2 §1 |
| **Hard exclusion mask for `C1 == C2` and throat clashes** | implemented | R2 §1 |
| **9,856 Root Vocab (9,114 classical)** | implemented | R2 §1 |
| **Subword-free prefix/suffix tokenization** | implemented | R2 §1 |
| **Morphemic 4-tuple factorization `(P, R, W, S)` embedding** | implemented | R2 §1 |
| **75:25 Balanced Basran-Andalusian Synthesis dataset** | implemented | R2 §1 |
| **FarahidianDualStreamBlock / FarahidianGenerativeHead / CrossLingualRootBridge** (PyTorch blueprints) | **blueprints only** ("Implementation Blueprint in PyTorch"); not claimed as deployed | R4 §6; R5 §5 |
| **Backbone:** 24-Layer DeepSeek-V4.1-Flash, layers 0–12 & 15–23 frozen, LoRA on layers 13 & 14 (`r=16, α=32`) at Rootformer v18.3 | stated as the live configuration | R1 header |

### 11.2 Named as **missing, stubbed, or wrong**
| Missing / wrong item | Exact complaint | Source |
| :--- | :--- | :--- |
| **Radical Permutation Group Symmetry** (*Al-Taqālīb*) | "Embeddings for `ع-ل-م` and `ع-م-ل` are treated as independent orthogonal IDs with **0 parameter sharing**, ignoring Al-Khalīl's shared acoustic sub-manifold"; "Roots are embedded as independent one-hot lookups into a **9,856-row** matrix" | R2 §1, Oversight 5 |
| **Constituent Scope & Valency Closure** | "`op_state` is a **flat 1-step memory**. It resets immediately after token t+1, failing when a noun phrase has an adjective (`في الأجسام الشفافة`)"; fails "multi-word prepositional phrases, genitive chains (*Iḍāfah*), or case inheritance across *Al-Tawābiʿ*" | R2 §1, Oversight 1 |
| **Semantic Intensity Scaling in Transmutation** | "Form II (`فَعَّلَ` — extensive) and Form V (`تَفَعَّلَ` — reflexive-gradual) are translated **identically** to Form I (`فَعَلَ`), losing morphological intensity" | R2 §1 |
| **Exclusivity Frame Transmutation** (*Al-Qaṣr wa-l-Ḥaṣr*) | "Translates `ليس النظم إلا...` as literal word-by-word copula rather than recognizing `ليس ... إلا` as an indivisible restrictive frame"; concrete failure: `is the نظم except...` | R2 §1, Oversight 3 |
| **Ellipsis Non-Restoration in Transmutation** | "Model still attempts to invent phantom subjects where classical Arabic uses concise pragmatic ellipsis" | R2 §1 (Ibn Maḍāʾ row) |
| **Definiteness Hierarchy** (*Marātib al-Maʿārif*) | "Did not enforce that subject (*Mubtada'*) must be ≥ predicate (*Khabar*) in definiteness rank" | R2 §1 (Ibn Mālik row) |
| **Latent Subject Pronoun Unpacking** (Al-Suhaylī) | "the transmuter **drops** the English subject pronoun instead of unpacking the latent pronoun (`"it is attained"`)" — failure mode `obtained by practice` | R2 §1, Oversight 2 |
| **Prepositional Adjunct Inversion** (Abū Ḥayyān) | "Arabic fronted adjuncts (`بالشك لا يزول`) are emitted fronted rather than inverted to English post-verbal order" | R2 §1 |
| **`Waw` Disambiguation** (Al-Shāṭibī) | "Translating **every** Arabic `و` as English 'and', causing repetitive sentence clutter"; failure string `and the knowledge and the intellect and it guides` | R2 §1, Oversight 4 |
| **Al-Khalīl's Phonotactic Compatibility Matrix** (*I'tilāf wa Tanāfur al-Ḥurūf*) | listed under "What Else Al-Khalīl & His Students Tell Us (**Remaining Agenda**)" → "**Action: Compile** a binary 9,114 × 9,114 root compatibility adjacency tensor" ⇒ **not yet compiled** | R1 §5.1 |
| **Sībawayh's Theory of Structural Rank** (*Al-Rutbah wa al-Aṣālah*) / valency tracking with clitic objects | remaining agenda → "**Action: Implement** structural valency tracking in the NRMP generator" | R1 §5.2 |
| **Ibn Jinnī's Vocalic Physics** (*Iʿlāl* / *Iltiqāʾ al-Sākinayn*) | remaining agenda → "**Action: Expand** `farahidian_syntactic_realizer.py`"; the `*لم يقول` → `لَمْ يَقُلْ` apocope is **not** in place | R1 §5.3 |
| **Al-Jurjānī's Dependency Isomorphism** (*Naẓm*) beyond clause level | remaining agenda → "**Action: Generalize** the Jurjānī dependency planner from clause-level patterns to arbitrary tree-structured dependency graphs across multi-sentence paragraphs" | R1 §5.4 |
| **Prosodic Rhythmic Cadence** (*ʿArūḍ*) | listed as authentic Al-Khalīlian theory but appears in **neither** the implemented nor the missed column with content | R2 §1 |
| **Five Validity States** (*Awjuh al-Kalām*) | named as authentic Sībawayhian theory; never defined anywhere in the five documents | R2 §1 |
| **Farāhīdian 17D acoustic manifold / Consonant Friction Loss / S₃ orbit attention layers** | presented as the target architecture in "How Al-Khalīl **Would** Architect" (counterfactual framing) ⇒ not claimed as deployed; blueprint code only | R4 throughout |
| **Factorized Next-Root/Morph heads & CrossLingualRootBridge** | presented as "Al-Khalīl **would** build" and as an "Implementation Blueprint" ⇒ blueprint, not a deployment claim | R5 §3, §5 |
| **Formal contradiction:** R1 §5.3 agenda vs R2 §1 "implements" column | R1 says to *expand* `farahidian_syntactic_realizer.py`; R2 does not list the realizer at all. Neither report claims full vocalic physics is live. | R1 §5.3; R2 §1 |

### 11.3 Discrepancies between the documents (flagged, not resolved)
1. **Root inventory:** 9,114 (R1 §5.1, R2 §1) vs. 9,856 deployed (R2 §1) vs. "~10,000 valid Arabic roots" (R5 §1) vs. `num_roots=10240` (R5 §5).
2. **Awnzān count:** ~120 (R5 §1) vs. `num_awzan=128` (R5 §5) vs. the 44 + 28 + 81 partitions (R1 §3.A.2). The relation between the 128/120 set and the 44/28/81 partitions is **never stated**.
3. **Iʿrāb count:** ~15 (R5 §1) vs. `num_irab=16` (R5 §5).
4. **Guttural set:** R1 §5.1 gives `{ء, هـ, ع, ح, غ, خ}`; R3 §2 Alg. 2 uses an unenumerated `THROAT_LETTERS` whose verified members include `ح`, `ء`, `ه` (all in the R1 set) — consistent but never explicitly identified.
5. **Phantom operators:** R4 §4.3 wants a latent-operator (*Taqdīr al-Maḥdhūf*) injector; R2 §1 wants "Rejection of Artificial Latent Restorations in Idioms" (Ibn Maḍāʾ). R4 reconciles them ("without generating explicit phantom words"); R2 does not address the reconciliation.
6. **Sībawayh's mask state:** R1 declares the Hard Exclusion Mask "100% Formalized, Integrated & Empirically Verified"; R2 (dated one day later) reports the same subsystem's operator state as a defective flat 1-step memory. The documents do not reconcile this.

### 11.4 Explicitly VAGUE / under-specified (do not invent)
- The **9,114 × 9,114 compatibility tensor** is requested, never provided.
- The **17-dim articulatory coordinates** and the **Tanafur distance threshold** are described, never tabulated.
- The **complete awzān lists** (the 44 / 28 / 81 partitions) are counted, never enumerated.
- `THROAT_LETTERS` is used as a constant, never enumerated (R3).
- The **labial** co-occurrence regulation (R1 §5.1) is "strictly regulated" with no rule given.
- The **8 waw functions** — only 4 named (R2 §1, Oversight 4).
- The **Five Validity States** (*Awjuh al-Kalām*) — named only.
- The **tanwīn glyph inventory** — the *mutual-exclusion predicate* with `الـ` is concrete; the suffix set is not given.
- The **τ threshold** in Al-Shāṭibī's ellipsis rule and the **λ_R / λ_W / λ_S weights** in the factorized loss — no values.
- *ʿArūḍ* (prosody) — no operational content.
- Jurjānī's *Naẓm* beyond clause level, and *Al-Taqdīm wa-l-Taʾkhīr* — narrative only.

---

## 12. Minimal implementable core (a programmer can build this today from the documents alone)

**Hard masks, in dependency order:**
1. `R ∈ V_Roots` whitelist (9,856/9,114) — K1.
2. `R` phonotactics: `c1 ≠ c2`; no homorganic guttural pair within `{ء, هـ, ع, ح, غ, خ}`; forbid `*عحـ`, `*حخـ` — K2/K3.
3. `W` masks by governing operator: after *Ḥarf Jarr* → block the 44 finite verbal awzān, allow nominal/maṣdar/participle; after *Ḥarf Jazm* → block the 81 nominal + 28 past awzān; after *Ḥarf Naṣb* or `سـ`/`سوف` → allow imperfect only — S2.
4. `P` masks: after a preposition block `{بـ, لـ, كـ, في}`; after jussive/subjunctive block `{الـ, والـ}` — S3.
5. `S` masks: with verbal awzān block `{ة, ية}`; never allow `الـ` + tanwīn on one word — S4/Su2.
6. Vocab index 4 (`<PARTICLE>`) always −∞ — S5.
7. Ibn Mālik imperfect stem selection (deterministic function, fully specified) — M1.
8. POS automaton: no `Ḥarf→Ḥarf`; no `Fiʿl→Fiʿl` unless previous is `{و, فـ}`; require `{Musnad, Musnad Ilayh}` closure — M2.
9. Definiteness: rank(mubtadaʾ) ≥ rank(khabar) with `Pronoun > Proper > Demonstrative > Definite > Indefinite` — M3.
10. Operator bracket stack with `{type, remaining}`; pop at saturation; after closure mask everything but `<EOS>` — S6; object-slot saturation blocks a bare accusative after a clitic object pronoun unless `{Tamyīz, Ḥāl}` — S7.
11. Latent-pronoun forcing map `يَفْعَلُ→it/he`, `تَفْعَلُ→it/she`, `أَفْعَلُ→I`, `نَفْعَلُ→we` when no overt subject — Su1.
12. Abū Ḥayyān 7-row case→realization table — A1.
13. Transmuter token-level masks: no `{of, in, to}` right after an equational subject; relative-clause chain `light → that → agent → verb` — S10/S11.

**Post-hoc filters:**
14. Bipartite frames `إنما X Y → X is exclusively Y`; `Neg + X + إلا + Y → X is nothing other than Y` — Ju1.
15. `و` classifier: new independent proposition → period, capital next word — Sh1.
16. Jussive apocope `*لم يقول → لَمْ يَقُلْ` — J3.
17. Duplicate/phantom token collapse (`the the → the`, `is is → is`) — Md1/Md2.
18. Fronted-prepositional-adjunct → post-verbal inversion — A2.

**Scoring biases (cannot be hard masks):**
19. S₃ orbit weight sharing `E(Root) = E_orbit({c1,c2,c3}) + E_order(σ∈S₃)` — K5/J2.
20. Ibn Maḍāʾ direct realism cosine loss `1 − cos(h^(24), e_concept)` — Md1.
21. Root-length prior favouring triliteral — K8.
22. Form II/V intensity scaling — J4.
23. Al-Shāṭibī ellipsis/explication threshold on `I(Token|Context)` — Sh2.
