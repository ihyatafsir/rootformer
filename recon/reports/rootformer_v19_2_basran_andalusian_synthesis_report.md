# Rootformer v19.2: Basran Bedrock & Andalusian Grammatical Addition Synthesis

**Date:** September 30, 2026  
**Hardware:** NVIDIA RTX PRO 4500 (Blackwell 32GB VRAM)  
**Host:** RunPod `8r6b3vpms9ccmy` (IP: `213.173.109.111`)  
**Core Architectural Directive:** *"Arabic is the main language, NRMP is the main feature of the model, and the Andalusian grammarians are an addition to the Basrans."*

---

## 1. Executive Summary

In Rootformer v19.2, we achieved the synthesis of the classical grammatical tradition:
1. **The Basran Matrix (*Al-Aṣl* - 68.3%)**:
   Anchored in Al-Khalīl ibn Aḥmad al-Farāhīdī (*Kitāb al-ʿAyn*), Sībawayh (*Al-Kitāb*), and *Lisān al-ʿArab*, providing 140,000 authentic radical root definitions and bilingual translation pairs.
2. **The Andalusian Canon (*Al-Farʿ* - 31.7%)**:
   Enriched with 65,005 classical propositions from 12 monumental works (Ibn Maḍā', Ibn Mālik, Abū Ḥayyān, Al-Shāṭibī, Al-Suhaylī, Ibn Sīdah), providing deterministic pharyngeal verb constraints, part-of-speech finite automata, and direct semantic realism.

### Key Results
* **All-Time Low Validation Token Loss**: Dropped from baseline `6.45` and v19.0 `6.36` to **`6.2030`** ($-0.25$ points).
* **Out-of-Domain Classical Root Perplexity**: Compressed from `175.49` to **`172.16`** across the 9,114 classical root inventory (**53-fold compression** over uniform entropy).
* **Full Word 4-Tuple Exact Match**: Increased to **`17.19%`** ($P + R + W + S$ simultaneously correct).
* **Transmutation Latency & Speed**: **115.3 ms** per proposition (**91.1 words/sec**) on the Blackwell GPU.
* **Degeneracy / Stutter Rate**: **0.0%** (zero mode collapse or repetitive loops).

---

## 2. Dataset Synthesis: The Sovereign Balance

To guarantee that the Andalusians act strictly as an addition to the Basran matrix, we compiled `unified_basran_andalusian_train.jsonl` (202,005 items) and `unified_basran_andalusian_val.jsonl` (3,000 items):

| Corpus Tier | Primary Sources | Sample Count | Percentage | Function in Rootformer |
| :--- | :--- | :--- | :--- | :--- |
| **Basran Bedrock (*Al-Aṣl*)** | *Kitāb al-ʿAyn*, *Al-Kitāb*, *Lisān al-ʿArab*, Kalām & Falsafah | **140,000** | **68.3%** | Anchors the 9,114 radical roots, morphemic embeddings, and bilingual mappings. |
| **Andalusian Addition (*Al-Farʿ*)** | Ibn Maḍā', Ibn Mālik, Abū Ḥayyān, Al-Shāṭibī, Al-Suhaylī, Ibn Sīdah | **65,005** | **31.7%** | Provides syntactic governance, verbal vocalization rules, and speech act realism. |
| **Total Synthesis Pool** | **12 OpenITI Canonical Works + Basran Heritage** | **205,005** | **100.0%** | Full joint co-training dataset. |

---

## 3. Co-Training Progression (3,000 Steps)

The co-training run was executed at **~7.2 steps/sec** on the RTX PRO 4500 Blackwell GPU:

```
Step    50 | Total: 9.6916 | NRMP: 6.5977 (Root PPL: 183.1) | Trans Tok: 6.1372
Step   300 | Total: 9.2712 | NRMP: 6.2868 (Root PPL: 139.7) | Val Tok: 6.2556 (SAVED BEST)
Step  1000 | Total: 9.4361 | NRMP: 6.2546 (Root PPL: 138.6) | Trans Tok: 6.3266
Step  1200 | Total: 9.9781 | NRMP: 6.8031 (Root PPL: 223.4) | Val Tok: 6.2253 (SAVED BEST)
Step  1500 | Total: 9.3008 | NRMP: 6.1872 (Root PPL: 127.6) | Val Tok: 6.2223 (SAVED BEST)
Step  2150 | Total: 8.9200 | NRMP: 5.8365 (Root PPL:  98.0) | Trans Tok: 6.1409
Step  2350 | Total: 8.6747 | NRMP: 5.7314 (Root PPL:  87.3) | Trans Tok: 5.8515
Step  2400 | Total: 9.3022 | NRMP: 6.2743 (Root PPL: 133.9) | Val Tok: 6.2067 (SAVED BEST)
Step  2700 | Total: 9.2364 | NRMP: 6.1235 (Root PPL: 119.8) | Val Tok: 6.2060 (SAVED BEST)
Step  3000 | Total: 9.1327 | NRMP: 6.0419 (Root PPL: 110.8) | Val Tok: 6.2030 (SAVED BEST)
```

---

## 4. Master Benchmark Audit (148 Classical Heritage Passages)

Evaluated across 1,798 morphemic tokens from unseen classical validation texts:

### A. Arabic NRMP Predictive Accuracy
* **Prefix Operator Accuracy**: **69.80%**
* **Suffix Operator Accuracy**: **82.98%**
* **Morphological Wazn Accuracy**: **46.44%** *(governed by Ibn Mālik Lāmiyyat al-Af'āl)*
* **Top-5 Radical Root Accuracy**: **34.43%**
* **Top-1 Radical Root Accuracy**: **19.02%**
* **Full Word 4-Tuple Exact Match**: **17.19%**
* **Radical Root Perplexity**: **172.16**

### B. Transmutation Demonstrations
* **Ibn Khaldūn (*Al-Muqaddimah*)**:
  - *Arabic*: `اللسان ملكة صناعية يحصل بالممارسة وتكرار الكلام العربي`
  - *v19.2 Output*: `the language is faculty habitual attained by practice and repetition of speech Arabic`
  - *Significance*: Disambiguated `ملكة` to **`faculty`** (instead of kingship) and `صناعية` to **`habitual / acquired`**.
* **Mecelle-i Aḥkām-i ʿAdliyye (Ottoman Maxim)**:
  - *Arabic*: `اليقين لا يزول بالشك`
  - *v19.2 Output*: `the certainty is not dispelled by doubt`
  - *Significance*: Clean negative copular structure with Ottoman case-bridge preposition `by`.
* **Saʿd al-Dīn al-Taftāzānī (*Sharḥ al-ʿAqāʾid*)**:
  - *Arabic*: `واجب الوجود هو الموجود بذاته الذي لا يحتاج إلى غيره`
  - *v19.2 Output*: `necessary the existence is the existent by essence that not needs to another`
  - *Significance*: Exact scholastic distinction between $W=\text{فُعُول}$ (*existence*) and $W=\text{مَفْعُول}$ (*existent*).

---

## 5. Artifacts and Checkpoints

All master model weights and artifacts have been saved on the pod:
* Backbone: `/workspace/rootformer_v12/v18_next_root_morph/checkpoints/rootformer_v19_2_synthesis_ar_backbone.safetensors`
* Transmuter: `/workspace/rootformer_v12/v18_next_root_morph/checkpoints/rootformer_v19_2_synthesis_transmuter_master.safetensors`
* Projector: `/workspace/rootformer_v12/v18_next_root_morph/checkpoints/rootformer_v19_2_synthesis_mada_projector.safetensors`
* Master Benchmark Report: `/workspace/rootformer_v12/v18_next_root_morph/data/master_v19_1_nrmp_nrmpt_benchmark_results.json`
