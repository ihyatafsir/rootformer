---
language:
- ar
- en
license: apache-2.0
task_categories:
- translation
tags:
- arabic
- classical-arabic
- translation
- leak-free
- work-level-split
- rootformer
pretty_name: Classical Arabic → English (work-level leak-free split)
size_categories:
- 100K<n<1M
---

# Classical Arabic → English — work-level leak-free split

An Arabic→English parallel set assembled so that **evaluation cannot be won by memorisation**.

## Why it exists

The bilingual material this was drawn from could not support a benchmark:

* its largest "parallel" sources are **lexicons, not sentences** (`lisan_farahidian` 57,886,
  `semantic_anchor` 7,583, `Basran_Heritage_Matrix` 137,961 held lexeme rows mixed with sentences);
* the same work appears in several source files, so pairs had to be de-duplicated across files;
* an earlier in-project evaluation set (Grand-100) turned out to be **100 % contained in the training
  data** — all 100 Arabic sentences and all 100 English references appear verbatim in the corpora.

## How the split is made

1. keep sentence-level pairs only — the form heuristic rejects a lexeme (single-token Arabic side
   with a short English gloss);
2. de-duplicate on `(arabic, english)` **across all source files**;
3. hold out **whole works** (31 of them) so no sentence from a test work can contribute training
   pairs;
4. **verify** by counting 13-gram overlap between test and train in both directions, then drop every
   test row that still shares one.

Held-out works include Ġazālī (*al-Mustasfā*, *Mishkāt al-Anwār*, *al-Munqidh min al-Ḍalāl*,
*Maqāṣid al-Falāsifa* and others), several Nawawī works, and *Rāzī, Asrār al-Tanzīl*.

## Files

| file | rows | note |
|---|---|---|
| `train.jsonl` | 117,264 | |
| `dev.jsonl` | 6,171 | |
| `test.jsonl` | 19,069 | held-out works |
| `test_clean.jsonl` | **17,916** | **zero 13-gram overlap with train — use this one** |
| `test_clean.ar`, `test_clean.en` | 17,916 | plain text for sacrebleu |
| `split_report.json` | — | the overlap verification numbers |

Fields: `arabic`, `english`, `work`.

## Verified overlap

Before filtering: Arabic 0.98 %, English 1.33 % of test 13-grams also occur in train.
`test_clean.jsonl` removes every offending row (1,153), leaving **0.000 %**.

## Baselines on `test_clean` (chrF++ / BLEU, sacrebleu)

| system | chrF++ | BLEU |
|---|---|---|
| identity (floor) | 1.98 | 1.30 |
| a shipped lexicon/rule “transmutation” engine | 1.53 | 0.02 |
| `Helsinki-NLP/opus-mt-ar-en`, zero-shot | 28.14 | 9.43 |
| the same model fine-tuned on `train.jsonl` (subword) | **44.87** | **24.43** |
| the same, with a Farāhīdian root-concept string prefixed to the source | 44.59 | 23.91 |

So: in-domain data is worth **+16.7 chrF++** over zero-shot, while textual root-concept conditioning
is worth **−0.28** — no gain. Reference quality is machine translation, which caps the absolute
scores; comparisons between systems on identical references remain valid.

## Caveats

* The English side is machine-translated (`*_v4_translated`) — usable for *comparative* evaluation,
  not as a gold standard of human translation.
* Domain is classical/scholastic Arabic (Ġazālī, Rāzī, Nawawī, the Andalusian grammarians), so
  absolute numbers are not comparable to modern-MSA benchmarks such as FLORES.
* The split is by work, not by genre or century.
