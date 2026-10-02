---
language:
- ar
- en
license: apache-2.0
tags:
- rootformer
- arabic
- classical-arabic
- translation
- root-aware
- nrmt
- leak-free-evaluation
- negative-result
library_name: transformers
pipeline_tag: translation
metrics:
- chrf
- bleu
---

# Rootformer v20.2 — Arabic→English: leak-free evaluation and the root-awareness verdict

v20.2 is the **evaluation** release. It contains the honest Arabic→English test harness, the
leak-free split, the baselines, and a controlled test of whether the Farāhīdian root representation
helps translation. **It does not** — in either form tested — and this release documents that with the
evidence.

Companion: the work-level leak-free split is published as a dataset,
[`enver/ar-en-rootformer-leakfree-split`](https://huggingface.co/datasets/enver/ar-en-rootformer-leakfree-split).

---

## The result

Evaluated on held-out **works** (31 whole works absent from training; 17,916 test sentences whose
13-grams never occur in train). Metrics: chrF++ / BLEU (sacrebleu).

| system | chrF++ | BLEU |
|---|---|---|
| identity (floor) | 1.98 | 1.30 |
| the project's shipped v19.2 lexicon/rule "transmutation" path | **1.53** | **0.02** |
| `Helsinki-NLP/opus-mt-ar-en`, zero-shot (subword NMT) | 28.14 | 9.43 |
| opus-mt fine-tuned in-domain — **subword** | **44.87** | **24.43** |
| …+ **textual** Farāhīdian root-concept string in the prompt | 44.59 | 23.91 |
| …+ **architectural** root vectors (1,500 steps) | 45.14 | 24.68 |
| …+ architectural root vectors (6,000 steps) | 45.91 | 25.28 |
| …+ architectural root vectors, **taken from a different sentence** (6,000 steps) | **45.91** | **25.45** |

**Reading it.**

1. **The shipped rule engine scores below the identity floor.** It generalises not at all; its
   published benchmark was memorisation.
2. **In-domain data is the real driver: +16.7 chrF++ over zero-shot.**
3. **Textual root conditioning: −0.28 chrF++** — no gain.
4. **Architectural root conditioning: no gain, and provably inert.** At 6,000 steps the model scores
   **identically (45.91 chrF++) whether it receives the sentence's own root vectors or another
   sentence's roots**, with the shuffled arm even edging BLEU. It cannot distinguish correct roots
   from mismatched ones, so the conditioning contributes nothing. The +0.12 over the matched
   no-conditioning control comes from the extra parameters and steps.

Two independent mechanisms, both negative, on a split built so that memorisation cannot help.

---

## What is included

| file | contents |
|---|---|
| `build_parallel_split.py` | assembles the work-level leak-free split and **verifies** it by counting 13-gram overlap in both directions |
| `mt_baselines.py` | identity / shipped-lookup / opus-mt / LLM(±concepts) baselines on the clean test set |
| `train_mt_inomain.py` | the in-domain fine-tune arms (subword vs textual root concepts) |
| `train_mt_root_arch.py` | **architecture**: a `RootEncoder` turning the morphemic `(prefix, root, wazn, suffix)` stream into K prepended encoder vectors, with a shuffled-roots null control |
| `MT_RESULTS.md` | the full write-up, including the two earlier correction notices |
| `models/mt_subword_model/` | the in-domain subword model (44.87 chrF++) |
| `models/mt_root_model/` | the textual root-concept model (44.59 chrF++) |
| `results/*.json` | every score, with per-system samples |

### Method notes

* The split holds out **whole works**, never sentences from a work that also contributes training
  pairs. Raw overlap before filtering was 0.98 % (Arabic) / 1.33 % (English); the published
  `test_clean` removes every offending row, leaving **0.000 %**.
* Both conditioning forms are compared against a **matched no-conditioning control** *and* a
  **shuffled/null control**, from the same checkpoint, on the same data slice, with the same schedule.
* **COMET is not reported.** Installing `unbabel-comet` pulled numpy 1.26.4 over the working
  numpy 2.1.2 and broke the environment; it was reverted and repaired rather than risk the host. The
  metrics are chrF++ and BLEU only.
* The English references are machine translations (`*_v4_translated`), which caps absolute scores.
  Comparisons between systems on identical references remain valid; a human-translated test set is
  needed for a publishable "high-grade" claim.

---

## Correction notices carried forward

* **The v19.2 "transmutation" was a hardcoded dictionary plus sentence regexes, not model output** —
  it returns byte-identical text under real hidden states, zero hidden states, `None`, and hidden
  states from a *different* sentence.
* **The Grand-100 LaBSE score of 0.9037 is memorisation** — all 100/100 Arabic sentences and 100/100
  English references occur verbatim in the training corpora.
* **One bilingual corpus is randomly paired** — an independent-MT agreement audit with a shuffled
  null gives `pure_gold` +17, `sovereign_classical_transmute` +21, `unified_basran_andalusian` +20,
  but `grand_scholastic_bilingual` **+4.7** (effectively random).

## Citation

```bibtex
@software{rootformer_v20_2_2026,
  author = {Enver at Aynengine and the Farāhīdian Research Circle},
  title  = {Rootformer v20.2: Arabic-to-English Leak-Free Evaluation and the Root-Awareness Verdict},
  year   = {2026},
  url    = {https://huggingface.co/enver/rootformer-v20.2-mt}
}
```
