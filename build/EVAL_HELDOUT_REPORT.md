# Held-out evaluation of the NRMP morphemic tokenizer

Harness: `/workspace/eval_heldout.py`  |  results: `/workspace/eval_heldout_results.json`  |  built 2026-10-01T16:46:49

## Sample

- 92 held-out files, cap 50000 word forms/file by uniform stride over each file
- **3298362 word forms sampled**, 188722 distinct forms scored in every mode
- corpora: scholastic_sunni_sanitized, scholastic_falsafa_sanitized, scholastic_masters, chronological_mujtahid_corpus
- NOT evaluated (kathra sources): /workspace/scholastic_sanitized, /workspace/andalusian_canon_sanitized, /workspace/heritage_foundations

## Metrics by mode (occurrence-weighted over the whole held-out sample)

| metric | mode 0 | mode 1 | mode 2 | mode1 - mode0 |
|---|---|---|---|---|
| round-trip rate (exact) | 73.24% | 73.86% | 71.19% | +0.62 pp |
| round-trip rate (skeleton) | 73.24% | 73.86% | 71.19% | +0.62 pp |
| round-trip rate (types, exact) | 56.09% | 55.96% | 54.32% | -0.13 pp |
| UNK rate (any slot) | 5.41% | 4.19% | 5.02% | -1.22 pp |
| UNK rate (root slot) | 4.80% | 3.58% | 4.28% | -1.22 pp |
| morphemic coverage | 51.92% | 48.49% | 47.80% | -3.43 pp |
| collapse to <PARTICLE> | 41.38% | 46.06% | 46.07% | +4.67 pp |
| collapse to <NONE> wazn | 45.57% | 49.98% | 50.17% | +4.41 pp |
| root attestation (al-'Ayn) | 66.46% | 69.05% | 75.54% | +2.58 pp |
| root muhmal (condemned) | 0.74% | 0.76% | 0.48% | +0.02 pp |
| root unrecorded | 23.99% | 21.24% | 18.30% | -2.75 pp |

| denominator | mode 0 | mode 1 | mode 2 | |
|---|---|---|---|---|
| word forms | 3298362 | 3298362 | 3298362 | |
| distinct forms | 188722 | 188722 | 188722 | |
| forms given a real root | 1774906 | 1661045 | 1637784 | |
| diacritic-bearing forms | 1 | 1 | 1 | |
| distinct forms failing the round trip | 82860 | 83111 | 86209 | |

## Per corpus

### scholastic_sunni_sanitized

| metric | mode 0 | mode 1 | mode 2 |
|---|---|---|---|
| round-trip rate (exact) | 74.04% | 75.42% | 72.71% |
| round-trip rate (skeleton) | 74.04% | 75.42% | 72.71% |
| round-trip rate (types, exact) | 63.95% | 63.81% | 62.09% |
| UNK rate (any slot) | 5.52% | 3.31% | 4.21% |
| UNK rate (root slot) | 4.89% | 2.69% | 3.42% |
| morphemic coverage | 52.15% | 49.45% | 48.66% |
| collapse to <PARTICLE> | 41.10% | 46.08% | 46.16% |
| collapse to <NONE> wazn | 45.34% | 49.12% | 49.32% |
| root attestation (al-'Ayn) | 67.74% | 69.63% | 76.31% |
| root muhmal (condemned) | 0.74% | 0.75% | 0.35% |
| root unrecorded | 22.14% | 20.17% | 17.16% |
| word forms | 991302 | 991302 | 991302 |

### scholastic_falsafa_sanitized

| metric | mode 0 | mode 1 | mode 2 |
|---|---|---|---|
| round-trip rate (exact) | 71.93% | 71.53% | 69.65% |
| round-trip rate (skeleton) | 71.93% | 71.53% | 69.65% |
| round-trip rate (types, exact) | 54.18% | 54.00% | 52.28% |
| UNK rate (any slot) | 5.83% | 6.00% | 6.56% |
| UNK rate (root slot) | 5.16% | 5.33% | 5.75% |
| morphemic coverage | 50.00% | 45.30% | 44.97% |
| collapse to <PARTICLE> | 42.67% | 47.15% | 47.08% |
| collapse to <NONE> wazn | 47.25% | 52.89% | 53.10% |
| root attestation (al-'Ayn) | 66.50% | 70.55% | 75.47% |
| root muhmal (condemned) | 0.86% | 0.92% | 0.72% |
| root unrecorded | 26.59% | 22.55% | 19.83% |
| word forms | 1136880 | 1136880 | 1136880 |

### scholastic_masters

| metric | mode 0 | mode 1 | mode 2 |
|---|---|---|---|
| round-trip rate (exact) | 74.76% | 75.73% | 73.02% |
| round-trip rate (skeleton) | 74.76% | 75.73% | 73.02% |
| round-trip rate (types, exact) | 65.34% | 65.18% | 63.35% |
| UNK rate (any slot) | 4.65% | 3.04% | 3.84% |
| UNK rate (root slot) | 4.14% | 2.52% | 3.25% |
| morphemic coverage | 54.37% | 51.69% | 50.97% |
| collapse to <PARTICLE> | 39.86% | 44.20% | 44.22% |
| collapse to <NONE> wazn | 43.26% | 46.88% | 47.00% |
| root attestation (al-'Ayn) | 67.74% | 69.48% | 75.75% |
| root muhmal (condemned) | 0.73% | 0.73% | 0.44% |
| root unrecorded | 22.57% | 20.77% | 18.14% |
| word forms | 690977 | 690977 | 690977 |

### chronological_mujtahid_corpus

| metric | mode 0 | mode 1 | mode 2 |
|---|---|---|---|
| round-trip rate (exact) | 72.53% | 73.48% | 69.10% |
| round-trip rate (skeleton) | 72.53% | 73.48% | 69.10% |
| round-trip rate (types, exact) | 64.43% | 64.20% | 62.22% |
| UNK rate (any slot) | 5.31% | 3.40% | 4.75% |
| UNK rate (root slot) | 4.73% | 2.82% | 4.04% |
| morphemic coverage | 52.47% | 49.46% | 48.20% |
| collapse to <PARTICLE> | 41.11% | 46.09% | 46.15% |
| collapse to <NONE> wazn | 45.39% | 49.34% | 49.55% |
| root attestation (al-'Ayn) | 61.83% | 63.85% | 73.79% |
| root muhmal (condemned) | 0.48% | 0.48% | 0.27% |
| root unrecorded | 23.99% | 21.26% | 17.51% |
| word forms | 479203 | 479203 | 479203 |

## Per author (filename prefix; chronological corpus uses the file-stem prefix)

| author | mode | forms | distinct | round-trip | morphemic | unk | muhmal |
|---|---|---|---|---|---|---|---|
| Ghazali | 0 | 333819 | 45550 | 75.34% | 52.90% | 5.23% | 0.96% |
| Ghazali | 1 | 333819 | 45550 | 76.77% | 50.36% | 3.05% | 0.99% |
| Ghazali | 2 | 333819 | 45550 | 74.60% | 49.71% | 3.77% | 0.58% |
| Razi | 0 | 275838 | 32051 | 75.22% | 52.09% | 5.00% | 0.71% |
| Razi | 1 | 275838 | 32051 | 76.23% | 49.15% | 3.39% | 0.70% |
| Razi | 2 | 275838 | 32051 | 74.31% | 48.71% | 3.95% | 0.33% |
| Raghib | 0 | 87938 | 23302 | 72.85% | 55.46% | 4.22% | 0.55% |
| Raghib | 1 | 87938 | 23302 | 73.66% | 52.33% | 2.83% | 0.49% |
| Raghib | 2 | 87938 | 23302 | 70.31% | 51.27% | 3.94% | 0.30% |
| Al | 0 | 16196 | 6059 | 73.37% | 57.98% | 4.24% | 0.50% |
| Al | 1 | 16196 | 6059 | 73.68% | 54.93% | 3.24% | 0.51% |
| Al | 2 | 16196 | 6059 | 70.00% | 53.88% | 4.35% | 0.37% |
| Amidi | 0 | 96868 | 15823 | 74.08% | 49.69% | 5.75% | 0.54% |
| Amidi | 1 | 96868 | 15823 | 75.83% | 47.22% | 3.28% | 0.55% |
| Amidi | 2 | 96868 | 15823 | 74.29% | 46.81% | 3.83% | 0.27% |
| Baydawi | 0 | 46652 | 14960 | 70.70% | 50.72% | 4.75% | 0.71% |
| Baydawi | 1 | 46652 | 14960 | 71.60% | 47.52% | 3.06% | 0.53% |
| Baydawi | 2 | 46652 | 14960 | 69.29% | 46.75% | 3.95% | 0.32% |
| Biruni | 0 | 167317 | 41609 | 58.07% | 47.99% | 11.50% | 0.87% |
| Biruni | 1 | 167317 | 41609 | 56.62% | 43.52% | 12.94% | 0.94% |
| Biruni | 2 | 167317 | 41609 | 54.47% | 43.00% | 13.62% | 0.77% |
| Farabi | 0 | 92644 | 12965 | 75.82% | 47.57% | 5.66% | 0.97% |
| Farabi | 1 | 92644 | 12965 | 76.77% | 44.03% | 4.22% | 1.04% |
| Farabi | 2 | 92644 | 12965 | 75.28% | 43.71% | 4.81% | 0.84% |
| Hilyat | 0 | 48261 | 10212 | 69.24% | 60.75% | 3.73% | 0.37% |
| Hilyat | 1 | 48261 | 10212 | 69.64% | 59.00% | 2.39% | 0.36% |
| Hilyat | 2 | 48261 | 10212 | 63.73% | 57.66% | 3.74% | 0.24% |
| Ibn | 0 | 824770 | 60635 | 74.75% | 50.85% | 4.77% | 0.84% |
| Ibn | 1 | 824770 | 60635 | 74.37% | 45.86% | 4.90% | 0.90% |
| Ibn | 2 | 824770 | 60635 | 72.49% | 45.57% | 5.44% | 0.70% |
| Juwayni | 0 | 92350 | 17721 | 75.80% | 51.20% | 5.40% | 0.90% |
| Juwayni | 1 | 92350 | 17721 | 77.09% | 48.78% | 3.31% | 0.92% |
| Juwayni | 2 | 92350 | 17721 | 74.84% | 48.18% | 4.03% | 0.37% |
| Kasani | 0 | 48473 | 10368 | 76.20% | 53.07% | 5.42% | 0.68% |
| Kasani | 1 | 48473 | 10368 | 77.99% | 50.55% | 2.79% | 0.70% |
| Kasani | 2 | 48473 | 10368 | 75.18% | 49.79% | 3.66% | 0.35% |
| Kindi | 0 | 52149 | 15714 | 64.85% | 47.33% | 4.73% | 0.96% |
| Kindi | 1 | 52149 | 15714 | 65.15% | 44.44% | 4.29% | 1.00% |
| Kindi | 2 | 52149 | 15714 | 63.39% | 44.07% | 4.73% | 0.70% |
| Marghinani | 0 | 49435 | 11257 | 74.90% | 53.99% | 6.23% | 0.61% |
| Marghinani | 1 | 49435 | 11257 | 77.31% | 51.15% | 2.81% | 0.63% |
| Marghinani | 2 | 49435 | 11257 | 74.39% | 50.33% | 3.73% | 0.32% |
| Maturidi | 0 | 82330 | 16367 | 73.67% | 49.81% | 5.33% | 0.52% |
| Maturidi | 1 | 82330 | 16367 | 74.88% | 46.72% | 3.50% | 0.46% |
| Maturidi | 2 | 82330 | 16367 | 72.46% | 45.84% | 4.52% | 0.31% |
| Nasafi | 0 | 87118 | 15725 | 70.85% | 50.64% | 7.03% | 1.07% |
| Nasafi | 1 | 87118 | 15725 | 72.51% | 48.28% | 4.49% | 1.10% |
| Nasafi | 2 | 87118 | 15725 | 69.74% | 47.26% | 5.61% | 0.33% |
| Nawawi | 0 | 98293 | 19251 | 75.38% | 56.25% | 4.93% | 0.87% |
| Nawawi | 1 | 98293 | 19251 | 75.62% | 52.63% | 3.91% | 0.91% |
| Nawawi | 2 | 98293 | 19251 | 71.83% | 51.51% | 5.12% | 0.37% |
| Qushayri | 0 | 37970 | 10121 | 73.37% | 59.38% | 3.84% | 0.55% |
| Qushayri | 1 | 37970 | 10121 | 74.08% | 57.33% | 2.31% | 0.54% |
| Qushayri | 2 | 37970 | 10121 | 68.85% | 55.87% | 3.68% | 0.42% |
| Sarakhsi | 0 | 49585 | 10113 | 74.57% | 52.28% | 5.91% | 0.71% |
| Sarakhsi | 1 | 49585 | 10113 | 76.31% | 49.56% | 3.23% | 0.73% |
| Sarakhsi | 2 | 49585 | 10113 | 72.97% | 48.73% | 4.15% | 0.43% |
| Sari | 0 | 39636 | 10437 | 72.61% | 61.28% | 3.46% | 0.52% |
| Sari | 1 | 39636 | 10437 | 72.91% | 58.78% | 2.50% | 0.53% |
| Sari | 2 | 39636 | 10437 | 67.66% | 57.53% | 3.80% | 0.43% |
| Shafii | 0 | 48195 | 10460 | 71.92% | 50.15% | 6.10% | 0.92% |
| Shafii | 1 | 48195 | 10460 | 73.26% | 47.53% | 3.47% | 0.94% |
| Shafii | 2 | 48195 | 10460 | 68.64% | 46.75% | 4.51% | 0.47% |
| Taftazani | 0 | 96811 | 17924 | 77.89% | 53.50% | 4.60% | 0.65% |
| Taftazani | 1 | 96811 | 17924 | 79.31% | 51.45% | 2.65% | 0.67% |
| Taftazani | 2 | 96811 | 17924 | 77.51% | 50.96% | 3.25% | 0.34% |
| Tahawi | 0 | 46511 | 11773 | 71.66% | 52.23% | 5.15% | 0.45% |
| Tahawi | 1 | 46511 | 11773 | 72.76% | 49.18% | 3.04% | 0.45% |
| Tahawi | 2 | 46511 | 11773 | 69.88% | 48.04% | 4.21% | 0.24% |
| abu | 0 | 60892 | 8558 | 69.34% | 53.99% | 5.90% | 0.41% |
| abu | 1 | 60892 | 8558 | 72.08% | 52.15% | 2.12% | 0.43% |
| abu | 2 | 60892 | 8558 | 64.99% | 49.83% | 4.49% | 0.31% |
| ashcari | 0 | 70751 | 10524 | 73.25% | 52.17% | 4.90% | 0.36% |
| ashcari | 1 | 70751 | 10524 | 74.22% | 49.51% | 3.26% | 0.33% |
| ashcari | 2 | 70751 | 10524 | 71.42% | 48.39% | 4.43% | 0.21% |
| maturidi | 0 | 82959 | 16430 | 73.62% | 49.49% | 5.43% | 0.52% |
| maturidi | 1 | 82959 | 16430 | 74.82% | 46.33% | 3.61% | 0.45% |
| maturidi | 2 | 82959 | 16430 | 72.42% | 45.43% | 4.65% | 0.32% |
| shafici | 0 | 90856 | 15597 | 72.86% | 52.00% | 5.89% | 0.67% |
| shafici | 1 | 90856 | 15597 | 74.17% | 49.29% | 3.50% | 0.67% |
| shafici | 2 | 90856 | 15597 | 69.95% | 48.26% | 4.75% | 0.30% |
| shaybani | 0 | 124112 | 14852 | 74.51% | 53.42% | 4.83% | 0.48% |
| shaybani | 1 | 124112 | 14852 | 74.06% | 49.15% | 4.40% | 0.50% |
| shaybani | 2 | 124112 | 14852 | 69.53% | 48.13% | 5.44% | 0.23% |
| tahawi | 0 | 49633 | 7979 | 68.09% | 54.50% | 5.10% | 0.38% |
| tahawi | 1 | 49633 | 7979 | 69.23% | 52.40% | 2.08% | 0.39% |
| tahawi | 2 | 49633 | 7979 | 62.63% | 50.60% | 3.98% | 0.23% |

## Top-10 round-trip failures (occurrence-weighted)

### mode 0 -- 0 greedy (validator OFF)

| # | word | count | decoded | root | wazn | suffix |
|---|---|---|---|---|---|---|
| 1 | الله | 31938 | أله | اله | فَعَلَ | <NONE> |
| 2 | عليه | 18068 | على<UNK> | <UNK> | <NONE> | <NONE> |
| 3 | فى | 17526 | <PARTICLE> | <PARTICLE> | <NONE> | <NONE> |
| 4 | له | 14886 | ل<UNK> | <UNK> | <NONE> | <NONE> |
| 5 | أنه | 13307 | أن<UNK> | <UNK> | <NONE> | <NONE> |
| 6 | به | 12194 | ب<UNK> | <UNK> | <NONE> | <NONE> |
| 7 | فيه | 11535 | في<UNK> | <UNK> | <NONE> | <NONE> |
| 8 | إلا | 10098 | <UNK> | <UNK> | <NONE> | <NONE> |
| 9 | تعالى | 9942 | <PARTICLE> | <PARTICLE> | <NONE> | <NONE> |
| 10 | عنه | 6783 | عن<UNK> | <UNK> | <NONE> | <NONE> |

### mode 1 -- 1 shipped DEFAULT (closed-class only)

| # | word | count | decoded | root | wazn | suffix |
|---|---|---|---|---|---|---|
| 1 | الله | 31938 | أله | اله | فَعَلَ | <NONE> |
| 2 | عليه | 18068 | علىه | <P:على> | <NONE> | ه |
| 3 | فى | 17526 | <PARTICLE> | <PARTICLE> | <NONE> | <NONE> |
| 4 | له | 14886 | ل<UNK> | <UNK> | <NONE> | <NONE> |
| 5 | به | 12194 | <UNK> | <UNK> | <NONE> | <NONE> |
| 6 | ان | 10138 | <UNK> | <UNK> | <NONE> | <NONE> |
| 7 | إلا | 10098 | <UNK> | <UNK> | <NONE> | <NONE> |
| 8 | تعالى | 9942 | <PARTICLE> | <PARTICLE> | <NONE> | <NONE> |
| 9 | صلى | 6082 | <PARTICLE> | <PARTICLE> | <NONE> | <NONE> |
| 10 | و | 5740 | <UNK> | <UNK> | <NONE> | <NONE> |

### mode 2 -- 2 root re-segmentation (needs kathra)

| # | word | count | decoded | root | wazn | suffix |
|---|---|---|---|---|---|---|
| 1 | الله | 31938 | أله | اله | فَعَلَ | <NONE> |
| 2 | قال | 22432 | قول | قول | فَعَلَ | <NONE> |
| 3 | عليه | 18068 | علىه | <P:على> | <NONE> | ه |
| 4 | فى | 17526 | <PARTICLE> | <PARTICLE> | <NONE> | <NONE> |
| 5 | له | 14886 | ل<UNK> | <UNK> | <NONE> | <NONE> |
| 6 | به | 12194 | <UNK> | <UNK> | <NONE> | <NONE> |
| 7 | ان | 10138 | <UNK> | <UNK> | <NONE> | <NONE> |
| 8 | إلا | 10098 | <UNK> | <UNK> | <NONE> | <NONE> |
| 9 | تعالى | 9942 | <PARTICLE> | <PARTICLE> | <NONE> | <NONE> |
| 10 | صلى | 6082 | <PARTICLE> | <PARTICLE> | <NONE> | <NONE> |

## Paired comparison (same word sample, mode A -> mode B)

Marginal rates can hide offsetting changes. These are paired: each distinct form is
scored in both modes and its round-trip fate compared.

| pair | ok -> ok | ok -> FAIL (newly broken) | FAIL -> ok (newly fixed) | FAIL -> FAIL | NET occ | NET forms |
|---|---|---|---|---|---|---|
| mode 0 -> mode 1 | 2382742 | 33126 | 53462 | 829032 | +20336 | -251 |
| mode 1 -> mode 2 | 2346571 | 89633 | 1665 | 860493 | -87968 | -3098 |
| mode 0 -> mode 2 | 2293109 | 122759 | 55127 | 827367 | -67632 | -3349 |

### mode 0 -> mode 1 detail

Newly broken (top by occurrences):

| word | occurrences | mode 0 decode | mode 1 decode | mode 1 root / wazn / suffix |
|---|---|---|---|---|
| ان | 10138 | ان | <UNK> | <UNK> / <NONE> / <NONE> |
| لهم | 2648 | لهم | <UNK> | <UNK> / <NONE> / <NONE> |
| انه | 2392 | انه | <UNK>ه | <UNK> / <NONE> / ه |
| فان | 2079 | فان | ف<UNK> | <UNK> / <NONE> / <NONE> |
| وان | 1683 | وان | و<UNK> | <UNK> / <NONE> / <NONE> |
| لك | 1133 | لك | <UNK> | <UNK> / <NONE> / <NONE> |
| فانه | 833 | فانه | ف<UNK>ه | <UNK> / <NONE> / ه |
| انها | 569 | انها | <UNK>ها | <UNK> / <NONE> / ها |
| هناك | 564 | هناك | <UNK> | <UNK> / <NONE> / <NONE> |
| لهما | 514 | لهما | <UNK> | <UNK> / <NONE> / <NONE> |

Newly fixed (top by occurrences):

| word | occurrences | mode 0 decode | mode 1 decode | mode 1 root / wazn / suffix |
|---|---|---|---|---|
| أنه | 13307 | أن<UNK> | أنه | <P:أن> / <NONE> / ه |
| فيه | 11535 | في<UNK> | فيه | <P:في> / <NONE> / ه |
| عنه | 6783 | عن<UNK> | عنه | <P:عن> / <NONE> / ه |
| منه | 6589 | من<UNK> | منه | <P:من> / <NONE> / ه |
| لأنه | 4451 | ل<UNK> | لأنه | <P:أن> / <NONE> / ه |
| فإنه | 3900 | ف<UNK> | فإنه | <P:إن> / <NONE> / ه |
| إنه | 1980 | إن<UNK> | إنه | <P:إن> / <NONE> / ه |
| معه | 937 | مع<UNK> | معه | <P:مع> / <NONE> / ه |
| فإنها | 732 | ف<UNK> | فإنها | <P:إن> / <NONE> / ها |
| فيهما | 461 | فهما | فيهما | <P:في> / <NONE> / هما |

REAL-root -> PARTICLE reassignments: 82302 occurrences (FAIL->ok 1537, ok->FAIL 1120, still ok 76294, still FAIL 3351)

### mode 1 -> mode 2 detail

Newly broken (top by occurrences):

| word | occurrences | mode 1 decode | mode 2 decode | mode 2 root / wazn / suffix |
|---|---|---|---|---|
| قال | 22432 | قال | قول | قول / فَعَلَ / <NONE> |
| كانت | 5986 | كانت | كونت | كون / فَعَلَ / ت |
| فقال | 4499 | فقال | فقول | قول / فَعَلَ / <NONE> |
| رضي | 3512 | رضي | <UNK> | <UNK> / فَعِلَ / <NONE> |
| قلنا | 2512 | قلنا | وقلنا | وقل / فَعَلَ / نا |
| الناس | 2482 | الناس | النوس | نوس / فَعَلَ / <NONE> |
| علي | 1884 | علي | <UNK> | <UNK> / فَعِلَ / <NONE> |
| حال | 1816 | حال | حول | حول / فَعَلَ / <NONE> |
| وجل | 1557 | وجل | <UNK> | <UNK> / فَعَلَ / <NONE> |
| النار | 1341 | النار | النور | نور / فَعَلَ / <NONE> |

Newly fixed (top by occurrences):

| word | occurrences | mode 1 decode | mode 2 decode | mode 2 root / wazn / suffix |
|---|---|---|---|---|
| منا | 312 | من<UNK> | منا | <P:من> / <NONE> / ا |
| نظرا | 117 | ظرا | نظرا | نظر / فَعَلَ / ا |
| بصره | 81 | <UNK> | بصره | بصر / فَعَلَ / ه |
| نسخا | 65 | سخا | نسخا | نسخ / فَعَلَ / ا |
| نفعا | 60 | فعا | نفعا | نفع / فَعَلَ / ا |
| مستحقا | 55 | حقا | مستحقا | سحق / مُفْتَعِل / ا |
| بممكن | 49 | بمكن | بممكن | مكن / مَفْعَل / <NONE> |
| لأجلها | 41 | لجللها | لأجلها | جله / أَفْعَلَ / ا |
| نهيه | 27 | هيه | نهيه | نهي / فَعِلَ / ه |
| مستقلا | 25 | قلا | مستقلا | سقل / مُفْتَعِل / ا |

REAL-root -> PARTICLE reassignments: 7078 occurrences (FAIL->ok 41, ok->FAIL 2393, still ok 4498, still FAIL 146)

### mode 0 -> mode 2 detail

Newly broken (top by occurrences):

| word | occurrences | mode 0 decode | mode 2 decode | mode 2 root / wazn / suffix |
|---|---|---|---|---|
| قال | 22432 | قال | قول | قول / فَعَلَ / <NONE> |
| ان | 10138 | ان | <UNK> | <UNK> / <NONE> / <NONE> |
| كانت | 5986 | كانت | كونت | كون / فَعَلَ / ت |
| فقال | 4499 | فقال | فقول | قول / فَعَلَ / <NONE> |
| رضي | 3512 | رضي | <UNK> | <UNK> / فَعِلَ / <NONE> |
| لهم | 2648 | لهم | <UNK> | <UNK> / <NONE> / <NONE> |
| قلنا | 2512 | قلنا | وقلنا | وقل / فَعَلَ / نا |
| الناس | 2482 | الناس | النوس | نوس / فَعَلَ / <NONE> |
| انه | 2392 | انه | <UNK>ه | <UNK> / <NONE> / ه |
| فان | 2079 | فان | ف<UNK> | <UNK> / <NONE> / <NONE> |

Newly fixed (top by occurrences):

| word | occurrences | mode 0 decode | mode 2 decode | mode 2 root / wazn / suffix |
|---|---|---|---|---|
| أنه | 13307 | أن<UNK> | أنه | <P:أن> / <NONE> / ه |
| فيه | 11535 | في<UNK> | فيه | <P:في> / <NONE> / ه |
| عنه | 6783 | عن<UNK> | عنه | <P:عن> / <NONE> / ه |
| منه | 6589 | من<UNK> | منه | <P:من> / <NONE> / ه |
| لأنه | 4451 | ل<UNK> | لأنه | <P:أن> / <NONE> / ه |
| فإنه | 3900 | ف<UNK> | فإنه | <P:إن> / <NONE> / ه |
| إنه | 1980 | إن<UNK> | إنه | <P:إن> / <NONE> / ه |
| معه | 937 | مع<UNK> | معه | <P:مع> / <NONE> / ه |
| فإنها | 732 | ف<UNK> | فإنها | <P:إن> / <NONE> / ها |
| فيهما | 461 | فهما | فيهما | <P:في> / <NONE> / هما |

REAL-root -> PARTICLE reassignments: 89380 occurrences (FAIL->ok 1578, ok->FAIL 3513, still ok 80792, still FAIL 3497)

## Contamination check

The kathra frequency table was built from ['/workspace/heritage_foundations/*.txt', '/workspace/andalusian_canon_sanitized/*.txt', '/workspace/scholastic_sanitized/*.txt'] (min_count=2, 12403007 tokens).

| corpus | form overlap with kathra | type overlap with kathra |
|---|---|---|
| chronological_mujtahid_corpus | 97.71% | 90.03% |
| scholastic_falsafa_sanitized | 94.84% | 80.68% |
| scholastic_masters | 98.72% | 94.64% |
| scholastic_sunni_sanitized | 97.39% | 90.06% |

Caveat: hapax pruned at min_count, so absence from kathra is uninformative for rare forms; overlap is a LOWER bound on prior exposure

