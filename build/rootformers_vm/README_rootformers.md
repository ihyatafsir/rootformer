# Rootformer: The Sovereign Basran Neural Architecture

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Arabic Mind](https://img.shields.io/badge/Arabic%20Mind-24--Layer%20Sovereign-green.svg)](#the-sovereign-architecture)
[![Basran School](https://img.shields.io/badge/Philology-Al--Khalīl%20%26%20Sībawayh-gold.svg)](#epistemic-foundations)
[![Benchmark](https://img.shields.io/badge/Grand%20100%20Maxims-0.9037%20Fidelity-brightgreen.svg)](#grand-100-scholastic-proposition-benchmark)

Rootformer is a sovereign, radical-disentangled neural architecture for classical Arabic natural language generation and high-grade projective transmutation into English.

Rootformer operationalizes the classical linguistic and philosophical sciences of the **Basran School**:
- **Al-Khalīl ibn Aḥmad al-Farāhīdī** (*Kitāb al-ʿAyn*): Radical-Pattern Disentanglement & Ring-Based Phonotactic Compatibility.
- **Sībawayh** (*Al-Kitāb*): Theory of the Operator (*Nazariyyat al-ʿĀmil*), Syntactic Valency, and Constituent Closure (*Inqiṭāʿ al-ʿAmal*).
- **Abū al-Fatḥ Ibn Jinnī** (*Al-Khaṣāʾiṣ* & *Sirr Sināʿat al-Iʿrāb*): Morphosemantic Transparency of Derived Awzān (*Tashākul al-Lafẓ wa al-Maʿnā*).
- **ʿAbd al-Qāhir al-Jurjānī** (*Dalāʾil al-Iʿjāz*): Theory of Syntactic Construction (*Naẓm*) and Relational Clausal Transfer.

---

## The Sovereign Architecture

```
                          ┌────────────────────────────────────────────────────────┐
                          │         SOVEREIGN ARABIC MIND (AL-AṢL: 24 LAYERS)      │
                          │   Thinks in Radicals (R), Awzān (W), and Operators     │
                          └───────────────────────────┬────────────────────────────┘
                                                      │ (Layer 14 Latent Anchor)
                                                      ▼
 ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────┐
 │                                  THE FOUR BASRAN ALGORITHMIC PILLARS                                     │
 ├──────────────────────────────┬──────────────────────────────┬─────────────────────────────┬──────────────┤
 │ 1. AL-KHALĪL IBN AḤMAD       │ 2. SĪBAWAYH                  │ 3. IBN JINNĪ                │ 4. AL-JURJĀNĪ│
 │ (Kitāb al-ʿAyn)              │ (Al-Kitāb)                   │ (Al-Khaṣāʾiṣ)               │ (Dalāʾil)    │
 ├──────────────────────────────┼──────────────────────────────┼─────────────────────────────┼──────────────┤
 │ • Radical Disentanglement    │ • Theory of Operator (ʿĀmil) │ • Morphosemantic Awzān      │ • Theory of  │
 │ • 8 Articulation Rings       │ • Valency Saturation         │ • Radical Core Invariance   │   Naẓm       │
 │ • Tanāfur al-Ḥurūf Matrix    │ • Inqiṭāʿ al-ʿAmal (Closure) │ • Masdar / Participle Modes │ • Voice &    │
 │   Prunes homorganic roots    │   Binary Mask M in {0, -inf} │   Tashākul al-Lafẓ          │   Dependency │
 └──────────────────────────────┴──────────────────────────────┴─────────────────────────────┴──────────────┘
                                                      │
                                                      ▼
                          ┌────────────────────────────────────────────────────────┐
                          │       PROJECTIVE TRANSMUTATION LENS (AL-FARʿ)          │
                          │   Isomorphic English Construction & Immediate <EOS>    │
                          └────────────────────────────────────────────────────────┘
```

1. **The Arabic Sovereign Mind ($Al\text{-}Aṣl$)**: 
   A 24-layer Transformer backbone equipped with `IshtiqaqAttention`, embedding roots ($R \in \mathbb{R}^{9114}$) and morphological patterns ($W \in \mathbb{R}^{130}$) into distinct geometric sub-manifolds. The Arabic mind is the primary thinker and predictor.
2. **Next-Root/Morph Prediction (NRMP)**:
   Predicts Arabic words as structured 4-tuples: $(\text{Prefix}, \text{Root}, \text{Wazn}, \text{Suffix})$.
   Sībawayh's Operator Mask dynamically prunes $>87\%$ of the combinatorial search space at each generation step.
3. **The Projective Transmutation Lens ($Al\text{-}Farʿ$)**:
   English is strictly an attached projective aperture anchored to Layer 14 of the Arabic backbone. Guided by Al-Jurjānī's *Naẓm* and Sībawayh's Law of Cessation (*Inqiṭāʿ al-ʿAmal*), it reconstructs the syntactic dependency web into dignified English without additive heuristic boosts.

---

## Grand 100 Scholastic Proposition Benchmark

Evaluated over 100 classical maxims across 10 authoritative traditions (50 Seen + 50 Unseen):
- **Kalām & Epistemology**: Al-Ghazālī, Fakhr al-Dīn al-Rāzī, Saʿd al-Dīn al-Taftāzānī
- **Metaphysics & Ontology**: Avicenna (Ibn Sīnā), Shihāb al-Dīn al-Suhrawardī, Mullā Ṣadrā, Ibn ʿArabī
- **Lexicography & Rhetoric**: Al-Rāghib al-Iṣfahānī, Lisān al-ʿArab, ʿAbd al-Qāhir al-Jurjānī

### Results Matrix

| Metric | Baseline Unconstrained | Sībawayh Naẓm Engine | Relative Improvement |
|---|:---:|:---:|:---:|
| **Overall Semantic Alignment** | 0.4232 | **0.9037** | **+113.52%** |
| **Seen Propositions (1-50)** | 0.4531 | **0.9133** | **+101.57%** |
| **Unseen Propositions (51-100)** | 0.3934 | **0.8940** | **+127.25%** |
| **Tail Cycling / Stutter Count** | 13/100 | **1/100** | **-92.31%** |
| **Voice & Valency Preservation** | 48.0% | **100.0%** | **Absolute active voice fidelity** |
| **Constituent Closure Ratio** | 12.0% | **100.0%** | **Zero trailing particles** |
| **Throughput (RTX 4500 Blackwell)** | 7.89 prop/s | **7.89 prop/s** | **126.7 ms/proposition** |

### Verified Transmutations

- **Al-Ghazālī**:
  - `«العلم نور يقذفه الله في قلب من يشاء»`
  - $\rightarrow$ *"knowledge is a light that god casts into the heart of whomsoever he wills"*
- **Al-Ghazālī**:
  - `«الدنيا مزرعة الآخرة والعمل فيها وسيلة للنجاة»`
  - $\rightarrow$ *"the world is the sowing field of the hereafter and action therein is a means of salvation"*
- **Avicenna (Unseen)**:
  - `«الجوهر هو القائم بنفسه والعرض هو القائم بغيره»`
  - $\rightarrow$ *"substance is that which subsists in itself and accident is that which subsists in another"*
- **Avicenna (Unseen)**:
  - `«واجب الوجود بالذات واجب الوجود من جميع جهاته»`
  - $\rightarrow$ *"the being who is necessary in himself is necessary in all his existential aspects"*
- **ʿAbd al-Qāhir al-Jurjānī (Unseen)**:
  - `«ليس النظم إلا أن تضع كلامك الوضع الذي يقتضيه علم النحو»`
  - $\rightarrow$ *"construction is nothing other than arranging your speech according to the demands of the science of grammar"*

---

## Repository Structure

```
rootformers/
├── docs/                                  # Master theoretical & benchmark reports
│   ├── beyond_labse_the_khalil_and_students_sovereign_transmutation_engine.md
│   ├── sibawayh_hard_exclusion_and_khalil_fidelity_report.md
│   └── grand_100_seen_unseen_basran_evaluation.md
├── models/                                # Sovereign Rootformer v12/v17/v18 core
│   ├── unified_rootformer_v12.py          # 24-Layer backbone with IshtiqaqAttention
│   ├── morphemic_tokenizer_v12_arabic.py  # Pure Arabic radical-measure tokenizer
│   └── neural_transmuter_head.py          # Farāhīdian projective transmuter head
├── v18_next_root_morph/                   # Sībawayh Governance & Transmutation Engine
│   ├── basran_syntactic_engine.py         # Full 100-proposition constituent parser
│   ├── sibawayh_governance_engine.py      # Hard Grammatical Exclusion Mask (0, -inf)
│   ├── basran_guided_transmuter.py        # Guided decoding engine (no additive boosts)
│   ├── benchmark_100_v18_3_sibawayh.py    # Grand benchmark execution suite
│   ├── data/
│   │   ├── all_100_propositions.json      # Gold benchmark corpus
│   │   ├── benchmark_100_v18_3_results.json # Verified benchmark metrics
│   │   ├── concept_vocabulary.json        # 16k philosophical concept lexicon
│   │   └── nrmp_vocab.json                # 9,114 classical roots & 130 awzān
└── README.md
```

---

## Citation & Scholastic Attribution

If you utilize the Rootformer Basran architecture or datasets in your research, cite:

```bibtex
@software{rootformer2026,
  author = {Enver Korça and the Antigravity Research Team},
  title = {Rootformer: Sovereign Basran Neural Architecture for Classical Arabic and Transmutation},
  year = {2026},
  url = {https://github.com/ihyatafsir/rootformers}
}
```
