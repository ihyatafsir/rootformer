---
language:
- ar
- en
license: apache-2.0
tags:
- rootformer
- nrmp
- next-root-morph-prediction
- classical-arabic
- morphology
- farahidi
- sibawayh
- ibn-malik
- basran
- andalusian-grammar
- grammar-constrained-decoding
library_name: transformers
pipeline_tag: text-generation
metrics:
- accuracy
- perplexity
---

# Rootformer v20 — Grammar-Governed Next-Root/Morph Prediction (NRMP)

Rootformer v20 makes **Next-Root/Morph Prediction** actually work end to end: the NRMP heads are
retrained and evaluated on a leak-free split, and decoding is governed by the project's own
implementations of the classical Arabic grammatical algorithms.

**This release is a correction as much as an upgrade.** In v19.2 the NRMP heads were shipped but
never invoked, and the advertised Arabic→English "transmutation" was a hardcoded dictionary plus
sentence-specific regexes, not model output — see *Correction notice* below for the falsification test.

---

## What NRMP is

Instead of predicting the next subword token, the model predicts the next word as a structured
morphological event:

```
P(Word_{t+1}) = P(Root_{t+1} | context) · P(Wazn_{t+1} | Root, context) · P(Affix_{t+1} | Root, Wazn, context)
Word ≡ (P, R, W, S) = (Prefix, triconsonantal Root, Wazn/pattern, Suffix)
```

Four factorized heads read a 896-d hidden state from a 24-layer Qwen2.5-0.5B-derived backbone:

| head | shape |
|---|---|
| `nrmp_head.root_head` | 896 → **9114** roots |
| `nrmp_head.wazn_head` | 896 → 130 awzān |
| `nrmp_head.prefix_head` | 896 → 26 prefixes |
| `nrmp_head.suffix_head` | 896 → 22 suffixes |
| `morphemic_embed.{root,wazn,prefix,suffix}_embed` | 9114×448 · 130×224 · 26×112 · 22×112 |

Inference is one forward pass per word plus a deterministic morphological realizer.

---

## Measured results

Task: predict the next word's **triconsonantal radical root** over 9114 classes, on held-out text
(150 sentences per corpus, split by source file). Particles (`<P:…>`) and the unanalysed catch-alls
(`<PARTICLE>`, `<UNK>`, which are ~21–30 % of tokens) are **excluded** from the metric — otherwise a
constant predictor scores well.

| metric | v19.2 shipped | **v20 (this release)** |
|---|---|---|
| radical-root acc@1 — Andalusian canon | 1.86 % | **3.85 %** |
| radical-root acc@1 — scholastic corpus | 1.15 % | **6.11 %** |
| radical-root acc@5 — Andalusian / scholastic | 7.70 / 6.20 % | **9.81 / 10.21 %** |
| root perplexity — Andalusian / scholastic | 1073 / 1156 | **756 / 653** |
| context gain (real vs shuffled context) | +1.86 / +0.77 pp | **+3.61 / +4.15 pp** |
| *reference:* unigram baseline | 3.35 / 3.05 % | 3.35 / 3.05 % |
| *reference:* oracle bigram over previous root | 38.63 / 39.50 % | 38.63 / 39.50 % |
| uniform over 9114 roots | 0.011 % | 0.011 % |

**Interpretation.** The v19.2 heads scored *below a context-free unigram root-frequency predictor*.
The v20 heads beat it and roughly doubled the measurable contribution of context. **Large headroom
remains:** an oracle bigram over the previous root reaches ~39 %, so the root head still badly
underuses even first-order root statistics.

### Grammar-engine ablation

Same weights and prompts, engines on vs off, 42 generated words each:

| violation of the classical rules | **governed** | engines off |
|---|---|---|
| bare `<NONE>` wazn on a radical root | **0** | 8 |
| definite article on a verb (`اليعلم`) | **0** | 3 |
| Harf → Harf (Alfiyyah-illegal particle chain) | **0** | 7 |
| stutter / degenerate loops | 0.024 | 0.000–0.070 |

Grammar firings in a governed run: `HARF_JARR`×5, `HARF_NASB`×3, `HARF_JAZM`×3, Ibn Mālik imperfect
wazn applied ×6, Sībawayh suffix masks ×15, Alfiyyah blocks Harf→Ṣifah ×32 and Fiʿl→Fiʿl ×2.

---

## The classical grammar engines

Decoding is delegated to implementations of the classical algorithms rather than ad-hoc rules:

| Engine | Source | Role |
|---|---|---|
| `SibawayhNRMPGovernance` | Al-Khalīl ibn Aḥmad, *Kitāb al-ʿAyn* | 311 phonotactically impossible roots excluded (C₁=C₂, deep-guttural incompatibility, bare alif) |
| `SibawayhNRMPGovernance` | Sībawayh, *Al-Kitāb* — *Naẓariyyat al-ʿĀmil* | operator states `HARF_JARR / HARF_JAZM / HARF_NASB / INNA / KANA / FUTURE` drive hard exclusion masks on the root, wazn, prefix and suffix heads |
| `IbnMalikVerbTransmuter` | Ibn Mālik, *Lāmiyyat al-Afʿāl* | pharyngeal / assimilated / hollow decision tree selects the imperfect wazn |
| `IbnMalikPOSAutomaton` | Ibn Mālik, *Al-Alfiyyah* | `كلامنا لفظ مفيد كاستقم واسم وفعل ثم حرف الكلم` — forbids Harf→Harf and Fiʿl→Fiʿl |
| `IbnMadaRealismFilter` | Ibn Maḍāʾ, *Kitāb al-Radd* | phantom-token elimination |
| `BasranSyntacticRealizer` | Baṣran school | *al-Ṣarf* + *al-Naḥw* surface realization |

---

## Quickstart

```bash
pip install torch transformers safetensors
# (this release bundles data/ and models/; Qwen2.5-0.5B config is fetched once)

# 1. load fidelity + held-out next-root evaluation
python nrmp_run.py --checkpoint checkpoints/rootformer_v20_nrmp_master.safetensors --eval

# 2. grammar-governed NRMP generation
python nrmp_generate.py --checkpoint checkpoints/rootformer_v20_nrmp_master.safetensors \
    "العلم نور يضيء العقل ويهدي إلى الحق"

# 3. the same decoder with the classical engines disabled (ablation)
python nrmp_generate.py --no-grammar \
    --checkpoint checkpoints/rootformer_v20_nrmp_master.safetensors \
    "العلم نور يضيء العقل ويهدي إلى الحق"

# 4. retrain / fine-tune the NRMP heads
python nrmp_train.py --prepare
python nrmp_train.py --train --steps 6000 --batch-size 32
```

`nrmp_run.py` prints an explicit load-fidelity report (`missing` / `unexpected` / shape mismatches)
rather than quietly accepting a partial state dict.

---

## Correction notice: the v19.2 "transmutation"

`transmute_quickstart.py` in v19.2 ran one encoder pass, **discarded the hidden states**, and returned
a hardcoded Arabic→English dictionary lookup. Verified by `transmute_control.py`, which calls the
shipped engine with (i) real hidden states, (ii) zero hidden states, (iii) `None`, and (iv) hidden
states from a *different* sentence — the output is **byte-identical** in all four cases:

```
العلم نور يضيء العقل ويهدي إلى الحق
  all four -> "the knowledge is light illuminates the intellect and guides to truth"
القط يشرب الحليب في الصباح            (out-of-distribution)
  all four -> "the قطط is شرب حلب in صبح"
تويتر منصة اجتماعية حديثة              (out-of-distribution)
  all four -> "<PARTICLE is <PARTICLE جمع حدث"
```

The fluent rows correspond one-to-one with entries in `SURFACE_SCHOLASTIC_LEXICON` and
sentence-specific regexes, so the v19.2 benchmark table is memorised text. English transmutation is
**not** repaired in v20 — the neural English decoder remains unexercised. v20 fixes and governs the
NRMP path only.

---

## Training

- **Data:** 691,127 unique sentences from 6 sanitized corpora → 7.06 M train / 838 k val word events,
  split **by source file** (no leakage).
- **Objective:** `L_root + 0.5·L_wazn + 0.25·L_prefix + 0.25·L_suffix`, with `<PARTICLE>`/`<UNK>`
  masked out of the root loss.
- **Phase 1** heads only (14.45 M params, 6000 steps, batch 32, lr 3e-4) → val acc@1 9.6 %.
- **Phase 2** upper 8 backbone layers unfrozen (138.96 M params, lr 1e-4 / 5e-6) → plateau ≈ 9.5 %,
  i.e. the ceiling is the data/task, not capacity.

---

## Limitations

1. **Root prediction is weak in absolute terms** (3.85 / 6.11 % acc@1 vs a ~39 % oracle bigram).
2. **~21–30 % of tokens are unanalysed catch-alls.** `rootformer_analyzer_v2.py` shows this is
   *structural*, not a missing affix rule: a strict reconstructing fallback repairs only **2.9 %** of
   failures, because the gap is dominated by *muʿtall* (weak-radical) morphology and by roots absent
   from the blueprint. Fixing it properly means extending the root inventory and retraining
   `morphemic_embed`/`nrmp_head` from scratch.
3. **Morphophonology is incomplete.** The realizer is largely an `if/elif` chain; hollow, defective and
   hamzated roots are partly wrong (e.g. سبب + فَاعِل + ة → `سابة`). The decoder repairs hamza+alif
   (`أا`→`آ`) and enforces wazn/affix agreement, but full *al-Ṣarf* is not implemented.
4. **Evaluation is in-domain** (the corpora overlap v19.x training material) — treat the numbers as
   upper bounds. The bigram reference is an oracle fitted on the evaluation set.
5. **English transmutation remains the v19.2 hardcoded lookup.** Unchanged, and labelled as such above.
6. Two root spaces coexist: 9114 (NRMP heads) and 9015 (per-attention `IshtiqaqAttention` +
   layer-14 `morphological_heads`). Do not conflate them.

---

## Citation

```bibtex
@software{rootformer_v20_2026,
  author = {Enver at Aynengine and the Farāhīdian Research Circle},
  title  = {Rootformer v20: Grammar-Governed Next-Root/Morph Prediction},
  year   = {2026},
  url    = {https://huggingface.co/enver/rootformer-v20}
}
```
