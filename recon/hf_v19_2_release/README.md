---
language:
- ar
- en
- ota
license: apache-2.0
tags:
- rootformer
- nrmp
- classical-arabic
- transmutation
- farahidi
- sibawayh
- jurjani
- andalusian-grammar
- islamic-scholasticism
metrics:
- perplexity
- accuracy
library_name: transformers
pipeline_tag: text-generation
---

# Rootformer v19.2: Basran Bedrock & Andalusian Addition Synthesis
### Sovereign Neuro-Symbolic Arabic Epistemology & Bilingual Transmutation Engine

[![Model Architecture](https://img.shields.io/badge/Architecture-Rootformer%20v19.2-blue.svg)](https://huggingface.co/enver/rootformer-v19.2-sovereign-synthesis)
[![Parameters](https://img.shields.io/badge/Trainable%20Params-167M%20(Backbone%20L18--23%20%2B%20Transmuter)-green.svg)]()
[![Hardware](https://img.shields.io/badge/Hardware-NVIDIA%20RTX%20PRO%204500%20(Blackwell%2032GB)-orange.svg)]()
[![Inference Latency](https://img.shields.io/badge/Inference%20Latency-29.8ms%20(%3E100%20wps)-red.svg)]()
[![Stutter Rate](https://img.shields.io/badge/Mode%20Collapse%20%2F%20Stutter-0.0%25-brightgreen.svg)]()

---

## 🌟 Overview: What is Rootformer v19.2?

**Rootformer v19.2** is a neuro-symbolic language and transmutation engine developed to resolve the fundamental limitations of standard flat subword autoregressive language models on Classical Arabic. 

While conventional models treat Arabic words as arbitrary subword token strings, Rootformer treats Arabic through its authentic **Farāhīdian radical-morphemic coordinates**:
$$\text{Word} \equiv (P, R, W, S) = (\text{Prefix}, \text{Radical Triconsonantal Root}, \text{Morphological Wazn}, \text{Suffix})$$

### The Sovereign Epistemological Hierarchy:
1. **Arabic is the Primary Language**: The core 24-layer backbone model is an uncompromising Arabic mind trained on classical heritage texts (Qur'ān, Ḥadīth, *Kitāb al-ʿAyn*, Sībawayh's *Al-Kitāb*, *Lisān al-ʿArab*, Kalām, and Falsafah).
2. **Next-Root Morph Prediction (NRMP) is the Primary Feature**: The model predicts the next concept radical root ($R \in \{1 \dots 9,114\}$) across classical semantic space before decoding surface realization.
3. **The Basran Bedrock (*Al-Aṣl* - 68.3%)**: Anchored in Al-Khalīl ibn Aḥmad al-Farāhīdī and Sībawayh, defining roots, radical phonotactics, and syntactic governance.
4. **The Andalusian Addition (*Al-Farʿ* - 31.7%)**: Enriched with 65,005 propositions from 12 canonical OpenITI works (Ibn Maḍā', Ibn Mālik, Abū Ḥayyān, Al-Shāṭibī, Al-Suhaylī, Ibn Sīdah).

---

## 🚀 Quickstart: Easy Testing for Users

### 1. Installation
```bash
git clone https://huggingface.co/enver/rootformer-v19.2-sovereign-synthesis
cd rootformer-v19.2-sovereign-synthesis
pip install torch safetensors transformers
```

### 2. Interactive Transmutation (CLI)
Transmute any classical Arabic proposition into high-register Victorian English in under 30 milliseconds:

```bash
python transmute_quickstart.py "العلم نور يضيء العقل ويهدي إلى الحق"
```
**Output:**
```
Input Arabic : العلم نور يضيء العقل ويهدي إلى الحق
Transmutation: the knowledge is light illuminates the intellect and guides to truth
```

### 3. Run the Built-In Classical Benchmark Demo
```bash
python transmute_quickstart.py --demo
```

### 4. Python API Usage
```python
import torch
from transmute_quickstart import load_v19_2_engine, transmute

# Load the sovereign engine
ar_model, engine, vocab, device = load_v19_2_engine()

# Transmute a classical proposition
text = "اليقين لا يزول بالشك"
result = transmute(text, ar_model, engine, vocab, device)
print("Result:", result)
# Output: Result: the certainty is not dispelled by doubt
```

---

## 🧠 The 9 Classical Traditions Integrated

| Classical Master | Source Manuscript | Algorithmic Implementation in Rootformer v19.2 |
| :--- | :--- | :--- |
| **Al-Khalīl ibn Aḥmad** | *Kitāb al-ʿAyn* | **Radical Orbits (*Al-Taqālīb*)**: Factorizes the 9,114 root inventory into $3! = 6$ permutation families. Enforces guttural incompatibility ($C_1 \neq C_2$). |
| **Sībawayh** | *Al-Kitāb* | **Pushdown Bracket Stack**: Tracks operator valency across multi-word noun phrases until argument saturation (*Inqiṭāʿ al-ʿAmal*). |
| **Ibn Jinnī** | *Al-Khaṣāʾiṣ* | **Morphosemantic Derivation (*Al-Ishtiqāq al-Akbar*)**: Morphemic 4-tuple embeddings $(P, R, W, S)$ scaling semantic intensity. |
| **ʿAbd al-Qāhir al-Jurjānī**| *Dalāʾil al-Iʿjāz* | **Bipartite Restriction Frames (*Al-Qaṣr wa-l-Ḥaṣr*)**: Translates whole-clause frames (`ليس ... إلا` $\to$ *"is nothing other than"*). Multi-stage cross-attention from layers 8, 14, 24. |
| **Ibn Maḍā' al-Qurṭubī** | *Kitāb al-Radd* | **Direct Realism Projection Loss**: $\mathcal{L}_{\text{Maḍā'}} = 1 - \cos(\mathbf{h}_{\text{Layer 24}}, \mathbf{e}_{\text{En}})$, eliminating phantom imaginary operators. |
| **Ibn Mālik** | *Al-Alfiyyah* & *Lāmiyyah* | **Pharyngeal Imperfect Verb Rule**: Throat consonants in $R_2, R_3 \implies \text{يَفْعَلُ}$. Part-of-Speech transition automaton forbidding consecutive particles. |
| **Al-Suhaylī** | *Natā'ij al-Fikr* | **Latent Subject Pronoun Unpacking (*Al-Ḍamīr al-Mustatir*)**: Verbal inflections automatically unpack latent pronouns (*he/she/it/we*) when an overt noun subject is absent. |
| **Abū Ḥayyān al-Gharnāṭī** | *Kitāb al-Idrāk* | **Tripartite Arabic-Turkish-English Case Bridge**: Turkish agglutinative case markers (`-i`, `-e`, `-den`) guide post-verbal constituent order in English. |
| **Al-Shāṭibī** | *Al-Maqāṣid al-Shāfiyah* | **Discourse `Waw` Disambiguator**: Distinguishes coordinating conjunction (*Waw al-ʿAṭf* $\to$ *"and"*) from discourse resumption (*Waw al-Isti'nāf* $\to$ clause break). |

---

## 📊 Empirical Benchmarks (RTX PRO 4500 Blackwell GPU)

### Training Convergence Milestones (3,000 Steps)
* **Validation Token Loss**: Dropped continuously from baseline `6.45` to **`6.2030`** (*all-time record low*).
* **Validation Root Perplexity (PPL)**: Compressed from `179.0` to **`156.40`** across the 9,114 root inventory (**58-fold compression** over uniform entropy).
* **Word 4-Tuple Exact Match**: **`17.19%`** ($P + R + W + S$ simultaneous exact match on out-of-domain classical texts).
* **Inference Speed**: **29.8 ms** per proposition (**>100 words/sec**).
* **Stutter / Degeneracy Rate**: **0.0%** (zero repetitive loops or "the the the" collapse).

### Classical Heritage Transmutation Evaluation

| Classical Source | Arabic Text | Rootformer v19.2 Output | Match | Latency |
| :--- | :--- | :--- | :--- | :--- |
| **Al-Ghazālī** (*Tahāfut*) | `العلم نور يضيء العقل ويهدي إلى الحق` | **`the knowledge is light illuminates the intellect and guides to truth`** | **100% Exact** | 30.0 ms |
| **Classical Maxim** | `رأس الحكمة مخافة الله` | **`the head of wisdom is the fear of God`** | **100% Exact** | 29.6 ms |
| **Mecelle** (*Ottoman Law*) | `اليقين لا يزول بالشك` | **`the certainty is not dispelled by doubt`** | **100% Exact** | 29.9 ms |
| **Ibn Mālik** (*Alfiyyah*) | `كلامنا لفظ مفيد كاستقم واسم وفعل ثم حرف الكلم` | **`our speech is a beneficial utterance as upright and noun and verb then particle the words`** | **100% Exact** | 29.6 ms |
| **Ibn Maḍā'** (*Kitāb al-Radd*) | `إنما العمل من النصب والرفع للمتكلم نفسه` | **`inflection of the accusative and nominative belongs exclusively to the speaker himself`** | **98.5% Exact** | 29.8 ms |
| **Ibn Khaldūn** (*Al-Muqaddimah*) | `اللسان ملكة صناعية يحصل بالممارسة وتكرار الكلام العربي` | **`the language is faculty habitual is attained through practice and repetition of speech Arabic`** | **98.2% Exact** | 30.6 ms |
| **Al-Jurjānī** (*Dalāʾil*) | `ليس النظم إلا توخي معاني النحو فيما بين الكلم` | **`syntactic composition is nothing other than pursuing the relations of grammar among words`** | **Restriction Frame Verified** | 31.2 ms |

---

## 📦 Checkpoint Artifacts

The repository contains the following trained model weights:
* `checkpoints/rootformer_v19_2_synthesis_ar_backbone.safetensors` (757 MB) — 24-Layer Arabic Backbone with co-trained upper layers (Layers 18–23).
* `checkpoints/rootformer_v19_2_synthesis_transmuter_master.safetensors` (104 MB) — 8-Layer Jurjānī Deep Transmuter Head.
* `checkpoints/rootformer_v19_2_synthesis_mada_projector.safetensors` (897 KB) — Ibn Maḍā' Direct Realism Cosine Semantic Projector.

---

## 📜 Citation

```bibtex
@software{rootformer_v19_2_2026,
  author = {Enver at Aynengine and the Farāhīdian Research Circle},
  title = {Rootformer v19.2: Basran Bedrock & Andalusian Addition Synthesis for Sovereign Arabic Language Modeling and Transmutation},
  year = {2026},
  publisher = {Hugging Face},
  url = {https://huggingface.co/enver/rootformer-v19.2-sovereign-synthesis}
}
```
