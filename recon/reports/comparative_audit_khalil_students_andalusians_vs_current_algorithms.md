# Comprehensive Comparative Audit: Current Algorithms vs. Al-Khalīl, His Students & The Andalusian Grammarians

**Date:** September 30, 2026  
**Hardware:** NVIDIA RTX PRO 4500 (Blackwell 32GB VRAM)  
**Host:** RunPod `8r6b3vpms9ccmy` (IP: `213.173.109.111`)  
**Objective:** A rigorous, manuscript-level comparison between what Rootformer v19.2 currently implements versus what the Basran founders and Andalusian polishers actually formulated—identifying exact historical oversights and computational upgrades.

---

## 1. High-Level Comparison Matrix

| Classical Master | Authentic Classical Theory | What Rootformer Currently Implements | What We Oversaw / Missed |
| :--- | :--- | :--- | :--- |
| **Al-Khalīl ibn Aḥmad** (*Kitāb al-ʿAyn*) | 1. 9,114 Radical Root Inventory<br>2. Phonotactic Incompatibility Matrix ($C_1 \neq C_2$, homorganic gutturals)<br>3. Permutational Group Orbit (*Al-Taqālīb* $3!=6$)<br>4. Prosodic Rhythmic Cadence (*ʿArūḍ*) | 1. 9,856 Root Vocab (9,114 classical)<br>2. Hard exclusion mask for $C_1 == C_2$ and throat clashes. | **Radical Permutation Group Symmetry**: Embeddings for `ع-ل-م` and `ع-م-ل` are treated as independent orthogonal IDs with 0 parameter sharing, ignoring Al-Khalīl’s shared acoustic sub-manifold. |
| **Sībawayh** (*Al-Kitāb*) | 1. Theory of the Operator (*Al-ʿĀmil*)<br>2. Structural Valency & Saturation (*Al-Iktifā'*)<br>3. Scope Boundary Termination (*Inqiṭāʿ al-ʿAmal*)<br>4. Syntactic Followers (*Al-Tawābiʿ*: Naʿt, ʿAṭf, Badal)<br>5. Five Validity States (*Awjuh al-Kalām*) | 1. Bigram operator state tracking (`HARF_JARR`, `HARF_JAZM`, `HARF_NASB`, `INNA`, `KANA`).<br>2. Single-step affix and wazn exclusion masks. | **Constituent Scope & Valency Closure**: Operator state resets after a single word, failing to track multi-word prepositional phrases, genitive chains (*Iḍāfah*), or case inheritance across *Al-Tawābiʿ* (adjectives inheriting noun case). |
| **Ibn Jinnī** (*Al-Khaṣāʾiṣ*) | 1. Greater Derivation (*Al-Ishtiqāq al-Akbar*)<br>2. Morphosemantic Transparency of Derived Forms<br>3. Phonetic Mimetic Scaling (Form II/V semantic intensification) | 1. Morphemic 4-tuple factorization: $(P, R, W, S)$ embedding. | **Semantic Intensity Scaling in Transmutation**: Form II (`فَعَّلَ` - extensive) and Form V (`تَفَعَّلَ` - reflexive-gradual) are translated identically to Form I (`فَعَلَ`), losing morphological intensity. |
| **ʿAbd al-Qāhir al-Jurjānī** (*Dalāʾil al-Iʿjāz*) | 1. Theory of Construction (*Naẓm*)<br>2. Information Structure & Focus (*Al-Taqdīm wa-l-Taʾkhīr*)<br>3. Syntactic Exclusivity Frames (*Al-Qaṣr wa-l-Ḥaṣr*: `إنما`, `ما... إلا`, `ليس... إلا`) | 1. 8-Layer Jurjānī Transmuter with multi-stage cross-attention from Backbone Layers 8, 14, 24. | **Exclusivity Frame Transmutation**: Translates `ليس النظم إلا...` as literal word-by-word copula rather than recognizing `ليس ... إلا` as an indivisible restrictive frame (*"is nothing other than..."*). |
| **Ibn Maḍā' al-Qurṭubī** (*Kitāb al-Radd*) | 1. Elimination of Virtual Operators (*Al-ʿAwāmil al-Muqaddarah*)<br>2. Direct Communicative Realism<br>3. Rejection of Artificial Latent Restorations in Idioms | 1. Direct Cosine Realism Loss between Layer 24 and Concept Space ($\mathcal{L}_{\text{Maḍā'}} = 1 - \cos$).<br>2. Duplicate token filter. | **Ellipsis Non-Restoration in Transmutation**: Model still attempts to invent phantom subjects where classical Arabic uses concise pragmatic ellipsis. |
| **Ibn Mālik** (*Al-Alfiyyah* & *Lāmiyyah*) | 1. Pharyngeal Imperfect Verb Decision Tree (*Lāmiyyah* v.49)<br>2. POS State Transition Automaton (*Alfiyyah* v.8–15)<br>3. Definiteness Hierarchy (*Marātib al-Maʿārif*) | 1. Pharyngeal Fatḥah rule for throat letters ($R_2, R_3 \in \text{Throat} \implies \text{يَفْعَلُ}$).<br>2. POS constraint blocking consecutive particles. | **Definiteness Hierarchy (*Marātib al-Maʿārif*)**: Did not enforce that subject (*Mubtada'*) must be $\ge$ predicate (*Khabar*) in definiteness rank (Pronoun > Proper > Demonstrative > Definite > Indefinite). |
| **Al-Suhaylī** (*Natā'ij al-Fikr*) | 1. Tanwīn as Syntactic Disconnection (*Al-Infisāl*)<br>2. Mutual Exclusion of `الـ` and Tanwīn<br>3. Latent Pronoun Resolution (*Al-Ḍamīr al-Mustatir*) in verbs | 1. Subword-free prefix/suffix tokenization. | **Latent Subject Pronoun Unpacking**: When an Arabic verb has no overt noun subject (e.g. `يحصل بالممارسة`), the transmuter drops the English subject pronoun instead of unpacking the latent pronoun (*"it is attained"*). |
| **Abū Ḥayyān** (*Kitāb al-Idrāk*) | 1. Arabic-Turkish Agglutinative Case Mapping<br>2. Typological Relational Word Order Inversion (SOV $\leftrightarrow$ SVO) | 1. Tripartite Turkish case bridge for prepositions and copula. | **Prepositional Adjunct Inversion**: Arabic fronted adjuncts (`بالشك لا يزول`) are emitted fronted rather than inverted to English post-verbal order (`is not dispelled by doubt`). |
| **Al-Shāṭibī** (*Al-Maqāṣid* & *Muwāfaqāt*) | 1. Discourse Functions of the Coordinator `و` (8 distinct types: ʿAṭf, Ḥāl, Qasam, Isti'nāf)<br>2. Communicative Intention over Surface Tags | 1. 75:25 Balanced Basran-Andalusian Synthesis dataset. | **`Waw` Disambiguation**: Translating every Arabic `و` as English "and", causing repetitive sentence clutter instead of treating `Waw Isti'nāfiyyah` as a sentence/paragraph break. |

---

## 2. The 5 Major Oversights & Their Algorithmic Solutions

### Oversight 1: Sībawayh's Constituent Bracket Closure & Valency
* **The Classical Rule (*Al-Kitāb*, Vol. 1)**: An operator has a strict saturation valency. A preposition (*Ḥarf Jarr*) governs a Noun Phrase (*Jārr wa-Majrūr*). Once the noun phrase (and its attached genitive/adjective) concludes, the operator's governing force **terminates** (*Inqiṭāʿ al-ʿAmal*).
* **Current Defect**: Our `op_state` is a flat 1-step memory. It resets immediately after token $t+1$, failing when a noun phrase has an adjective (`في الأجسام الشفافة`).
* **The Algorithmic Fix**: Implement a **Syntactic Pushdown Bracket Stack**:
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

---

### Oversight 2: Al-Suhaylī's Latent Pronoun (*Ḍamīr Mustatir*) Unpacking
* **The Classical Rule (*Natā'ij al-Fikr*, lines 2031–2038)**:
  > *"وتحقيق القول أن الفاعل مضمر في نفس المتكلم، ولفظ الفعل متضمن له دال عليه، واستغني عن إظهاره لتقدم ذكره..."*  
  > (The subject is latent in the speaker's mind, and the verb's verbal form contains and indicates it...)
* **Current Defect**: When an Arabic sentence begins with a verb or has no overt noun after the verb (e.g. `يحصل بالممارسة`), the model produces telegraphic English: `obtained by practice`, omitting the subject pronoun.
* **The Algorithmic Fix**:
  If verb $V_t$ has no overt nominal argument in its local constituent bracket:
  Unpack the verbal conjugation prefix/suffix into its English pronominal equivalent:
  - `يَفْعَلُ` $\to$ `it / he` + Verb
  - `تَفْعَلُ` $\to$ `it / she` + Verb
  - `أَفْعَلُ` $\to$ `I` + Verb
  - `نَفْعَلُ` $\to$ `we` + Verb

---

### Oversight 3: Al-Jurjānī's Indivisible Exclusivity Frames (*Al-Ḥaṣr wa-l-Qaṣr*)
* **The Classical Rule (*Dalāʾil al-Iʿjāz*)**: Particles of restriction (`إنما`, `ما ... إلا`, `ليس ... إلا`) do not operate as isolated words; they form a **discontinuous bipartite frame** enclosing the predicate.
* **Current Defect**: `ليس النظم إلا توخي معاني النحو` was translated token-by-token: `is the نظم except...`.
* **The Algorithmic Fix**:
  Recognize the bipartite pattern:
  $$\text{Negation} + X + \text{إلا} + Y \implies X \text{ is nothing other than } Y$$
  $$\text{إنما} + X + Y \implies X \text{ is exclusively / none other than } Y$$

---

### Oversight 4: Al-Shāṭibī's Discourse `Waw` Disambiguation
* **The Classical Rule (*Al-Maqāṣid al-Shāfiyah*, Vol. 1)**: The letter `و` (*waw*) is not a uniform connective; it has 8 distinct syntactic functions:
  1. *Waw al-ʿAṭf* (Conjunction): connects words in a list $\to$ English `and`.
  2. *Waw al-Ḥāl* (Circumstantial): introduces a subordinate clause $\to$ English `while / as`.
  3. *Waw al-Isti'nāf* (Discourse Resumption): begins a new proposition $\to$ English `.` (Period / Sentence Break).
  4. *Waw al-Qasam* (Oath): `والله` $\to$ English `By [God]`.
* **Current Defect**: Every `و` was converted to `and`, resulting in repetitive strings: `and the knowledge and the intellect and it guides`.
* **The Algorithmic Fix**:
  If `و` occurs at the start of a new independent proposition (followed by *Ism* or *Fiʿl* with completed previous valency), treat as `Waw al-Isti'nāf` $\to$ capitalize next word and emit a period `.` instead of `and`.

---

### Oversight 5: Al-Khalīl's Radical Permutation Group Equivariance
* **The Classical Rule (*Kitāb al-ʿAyn*, Introduction)**: Every root belongs to an orbit of up to 6 permutations ($3!$). Roots within the same permutation orbit share acoustic articulation properties.
* **Current Defect**: Roots are embedded as independent one-hot lookups into a 9,856-row matrix.
* **The Algorithmic Fix**:
  Factorize root embeddings as:
  $$E(\text{Root}) = E_{\text{orbit}}(\text{Radical Set } \{c_1, c_2, c_3\}) + E_{\text{order}}(\text{Permutation Index } \sigma \in S_3)$$
  This enforces weight sharing across Al-Khalīl's permutation families, enabling zero-shot generalization across rare roots!

---

## 3. Summary of Upgrades to Apply to Rootformer

1. **Constituent Pushdown Stack** (Sībawayh Valency Saturation).
2. **Latent Pronoun Resolution** (Al-Suhaylī Subject Unpacking).
3. **Bipartite Restrictive Frames** (Al-Jurjānī *Naẓm*).
4. **Discourse `Waw` Classifier** (Al-Shāṭibī Contextual Economy).
5. **Permutation Orbit Weight Sharing** (Al-Khalīl *Taqālīb*).
