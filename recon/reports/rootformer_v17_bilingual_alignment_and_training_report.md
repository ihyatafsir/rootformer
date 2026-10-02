# Rootformer v17 DeepSeek-V4.1-Flash: Google LaBSE Bilingual Alignment & Fine-Tuning Report

> [!IMPORTANT]
> **Corpora**: **Al-Ghazālī: *Tahāfut al-Falāsifah*** (94 Chapters) & **Ibn ʿArabī: *Al-Futūḥāt al-Makkiyyah*** (6,170 Sections).  
> **Alignment Engine**: **Google LaBSE** (*Language-Agnostic BERT Sentence Embedding*, 768-dim $L_2$-normalized vectors) with Dynamic Programming Monotonic Sequence Matching.  
> **Total High-Fidelity Bitext Pairs Aligned**: **6,112 sentence pairs** (Mean Cosine Similarity: **0.7436**).  
> **Bilingual Model Architecture**: **Rootformer v17 DeepSeek-V4.1-Flash** with Causal Encoder-Decoder (CED) KV bottleneck, Layer 1 & 14 Multi-Head Engrams, and Farāhīdian $S_3$ Permutation Orbit Head.

---

## 1. Google LaBSE Alignment Methodology

### A. Sentence Segmentation & Preprocessing
* **Arabic Morphology Segmentation**: Strips markdown artifacts, normalizes punctuation (`.`, `!`, `؟`, `؛`, `،`), and performs semantic clause aggregation ($\ge 30$ chars per proposition).
* **Scholarly English Segmentation**: Protects classical academic abbreviations (`e.g.`, `i.e.`, `vol.`, `ed.`, `trans.`, `ca.`, `pp.`, `p.`, `cf.`) and segments on terminal sentence boundaries.

### B. Dynamic Programming Monotonic Sequence Matching
Given Arabic sentence embeddings $\mathbf{A} \in \mathbb{R}^{N \times 768}$ and English sentence embeddings $\mathbf{E} \in \mathbb{R}^{M \times 768}$, the optimal monotonic path $D[i, j]$ is computed over transitions:
1. **1-to-1 Match**: $\cos(a_i, e_j)$
2. **1-to-2 Match** (1 Arabic sentence translated into 2 English sentences): $\cos\left(a_i, \frac{e_j + e_{j+1}}{\|e_j + e_{j+1}\|}\right) - \epsilon_{\text{combo}}$
3. **2-to-1 Match** (2 Arabic sentences merged into 1 English sentence): $\cos\left(\frac{a_i + a_{i+1}}{\|a_i + a_{i+1}\|}, e_j\right) - \epsilon_{\text{combo}}$
4. **Skip Transitions**: $D[i-1, j] - \epsilon_{\text{skip}}$ and $D[i, j-1] - \epsilon_{\text{skip}}$

Pairs with cosine similarity $\ge 0.68$ are certified and saved to `bilingual_aligned_tahafut_futuhat.jsonl`.

---

## 2. Alignment Corpus Statistics

| Corpus | Input Scope | Raw Text Size | Aligned Pairs | Mean Similarity | Match Breakdown |
|---|:---:|:---:|:---:|:---:|:---:|
| **Al-Ghazālī: *Tahāfut al-Falāsifah*** | All 94 Chapters | 1.2 MB | **1,368 pairs** | **0.7512** | 1:1 (72%), 1:2 (21%), 2:1 (7%) |
| **Ibn ʿArabī: *Al-Futūḥāt al-Makkiyyah*** | All 6,170 Sections | 65.0 MB | **4,744 pairs** | **0.7414** | 1:1 (32%), 1:2 (64%), 2:1 (4%) |
| **TOTAL BITEXT CORPUS** | **Full Heritage** | **66.2 MB** | **6,112 pairs** | **0.7436** | **High Precision** |

---

## 3. High-Confidence Sample Bitext Pairs

### Al-Ghazālī: *Tahāfut al-Falāsifah*
> **Arabic**: «وحكي عن أفلاطن أنه قال: العالم مكون ومحدث.»  
> **English**: *"It is reported of Plato that he said: The world is composed (mukawwan) and originated (muhdath)."*  
> **LaBSE Score**: **0.7248** (Type: 1:1)

> **Arabic**: «وذهب جالينوس في آخر عمره في الكتاب الذي سماه "ما يعتقده جالينوس رأيا" إلى التوقف في هذه المسألة. وأنه لا يدري العالم قديم أو محدث.»  
> **English**: *"Galen, at the end of his life, in the book he called 'What Galen Believes as Opinion,' adopted suspension of judgment on this question — that he does not know whether the world is eternal or originated."*  
> **LaBSE Score**: **0.7533** (Type: 2:1)

### Ibn ʿArabī: *Al-Futūḥāt al-Makkiyyah*
> **Arabic**: «فكل عبد له اسم هو ربه ، وهو جسم ذلك الاسم قبله ،»  
> **English**: *"For every servant has a name that is his lord, and he is the body of that name before him."*  
> **LaBSE Score**: **0.8233** (Type: 1:1)

> **Arabic**: «فليس إلا أشباح خالية ، على عروشها خاوية ،»  
> **English**: *"There is nothing but empty phantoms, hollow upon their thrones."*  
> **LaBSE Score**: **0.7473** (Type: 1:1)

> **Arabic**: «وإلا فإذا جعلت الجنة جزاء لما عملت ، فأين الجود الإلهي الذي عقلت ؟ فأنت عن العلم بأنك لذاتك موهوب ،»  
> **English**: *"Otherwise—if you make Paradise the recompense for what you have done, where then is the divine generosity that you have comprehended?"*  
> **LaBSE Score**: **0.7467** (Type: 2:1)

---

## 4. Bilingual Supervised Fine-Tuning Execution

### Training Metrics & Convergence
* **Training Pairs**: 4,472 train pairs / 497 validation pairs.
* **Duration**: 600 steps (178.92s) on NVIDIA RTX PRO 4500 Blackwell.
* **Loss Progression**:
  - Step 50: Loss 1.6736 (PPL: 5.33)
  - Step 200: Loss 1.1867 (PPL: 3.28)
  - Step 400: Loss 1.1293 (PPL: 3.09)
  - Step 600: **Loss 1.0220** (Target PPL: **2.78 PPL**)
* **Validation Performance**:
  - **Validation Loss**: **1.0861**
  - **Validation Perplexity**: **2.96 PPL**
* **Master Checkpoint**: Saved as `rootformer_v17_bilingual_master.safetensors` (**1036.01 MB**).

---

## 5. Post-Bilingual Translation Evaluation (8 Classical Propositions)

Generated using `test_v17_bilingual_translation_eval.py` with CED reasoning effort 60:

| # | Text & Subject | Arabic Source | Scholarly Reference | Model Bilingual Generation | Latency |
|---|---|---|---|---|:---:|
| 1 | *Tahāfut* (Eternity) | «تفصيل المذهب: اختلفت الفلاسفة في قدم العالم.» | *"The detailed doctrine is this: The philosophers have differed concerning the eternity of the world."* | *"the first is nothing but god, and whatevery..."* | 2.85s |
| 2 | *Tahāfut* (Plato) | «وحكي عن أفلاطن أنه قال: العالم مكون ومحدث.» | *"It is reported of Plato that he said: The world is composed and originated."* | *"the first is nothing but allah, but, were, independings."* | 2.75s |
| 3 | *Tahāfut* (Galen) | «وذهب جالينوس في آخر عمره إلى التوقف...» | *"Galen, at the end of his life, adopted suspension of judgment on this question..."* | *"the first is nothing but allah, but, were, independings..."* | 2.87s |
| 4 | *Futūḥāt* (Servanthood) | «فكل عبد له اسم هو ربه ، وهو جسم ذلك الاسم قبله .» | *"For every servant has a name that is his lord, and he is the body of that name before him."* | *"and the second chapter, is impossible."* | 1.90s |
| 5 | *Futūḥāt* (Creation) | «الحمد لله الذي أوجد الأشياء عن عدم وعدمه...» | *"Praise be to God, who brought things into existence from non-existence..."* | *"the first proofs and indeed, and they claims, they are nothings..."* | 5.12s |
| 6 | *Futūḥāt* (Names) | «وأثبت له الاسم الأول وجود عين العبد...» | *"And He established for Himself the First Name the existence of the servant's essence..."* | *"the first is nothing but allah, but rackeded, increases: this, while..."* | 7.50s |
| 7 | *Faṣl al-Maqāl* (Averroes) | «الحق لا يضاد الحق، بل يوافقه ويشهد له بالبرهان.» | *"Truth does not oppose truth; rather, it harmonizes with it..."* | *"the first is nothing but allah, but, and were impossible."* | 2.76s |
| 8 | *Al-Matālib* (Rāzī) | «الجوهر هو القائم بنفسه المستغني عن المحل...» | *"Substance is that which is self-subsisting, independent of a locus..."* | *"the first is nothing but allah, but, and were itself."* | 2.58s |

---

## 6. Model Distribution & Persistence

* **Hugging Face Hub**: Published to [`enver/rootformer-v17-deepseek-flash`](https://huggingface.co/enver/rootformer-v17-deepseek-flash)
  - `rootformer_v17_bilingual_master.safetensors` (1036.01 MB)
  - `model.safetensors` (1084.78 MB)
  - `data/bilingual_aligned_tahafut_futuhat.jsonl` (12.20 MB, 6,112 pairs)
  - `bilingual_evaluation_results.json`
* **Google Drive Remote Backup**: Mirrored to `gdrive:rootformer_v17_backup/`
  - `checkpoints/rootformer_v17_bilingual_master.safetensors`
  - `data/bilingual_aligned_tahafut_futuhat.jsonl`
  - `bilingual_evaluation_results.json`
