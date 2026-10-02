# Grand-100 Benchmark: contamination audit

**Verdict: the headline LaBSE score of 0.9037 measures memorisation, not translation.**

Source artefact: `v18_next_root_morph/data/benchmark_100_v18_3_results.json` on the `absolut7` VM
(`10.20.102.177:/home/absolut7/rootformers/`).

## What the benchmark claims

```
model: Rootformer v18.3 (Sibawayh Hard Grammatical Exclusion Mask + LoRA Layers 13/14)
total_propositions: 100
overall_labse:  baseline_unconstrained 0.4232  ->  sibawayh_hard_exclusion 0.9037  (+113.52%)
seen_propositions_labse:    0.4531 -> 0.9133
unseen_propositions_labse:  0.3934 -> 0.8940
stutter_and_cycle_count:    baseline 13 -> sibawayh 1   (-92.31%)
```

## The audit

Scanned all 244 corpus files under `/home/absolut7/rootformers/**/*.json{,l}`
(excluding the benchmark file itself), plus the bilingual/translation training sets:

| check | result |
|---|---|
| benchmark **Arabic** sentences found verbatim in training data | **100 / 100** |
| benchmark **English** references found verbatim in training data | **100 / 100** |
| exact `output == reference` after case/punctuation normalisation | effectively all |

The `SEEN` (50) / `UNSEEN` (50) partition therefore does **not** mean train/test. Both halves are
present in the training corpora; "unseen" means only "not among the 50 memorised exemplars used as
the seen list".

## Evidence that the outputs are copies

```
AR : الجوهر هو القائم بنفسه والعرض هو القائم بغيره
REF: Substance is that which subsists in itself, and accident is that which subsists in another.
OUT: substance is that which subsists in itself and accident is that which subsists in another
LaBSE 0.8897     <- identical modulo capitalisation and the final period

AR : الوجود عارض للماهية في الذهن والخارج
REF: Existence is an accident supervening upon quiddity in both the mind and external reality.
OUT: existence is an accident supervening upon quiddity in both the mind and external reality
LaBSE 0.9347
```

LaBSE is not reaching ~1.0 purely because of casing and punctuation. A genuine translation system
would not reproduce the reference wording this exactly across 50 supposedly unseen items.

## Consequence

1. The 0.9037 figure cannot be cited as translation quality. Any claim built on it — including
   "high-grade Arabic→English transmutation" — currently rests on memorised pairs.
2. It also explains the shape of the earlier results: the "unconstrained baseline" (0.4232) and the
   "Sibawayh" arm (0.9037) are both reading the same memorised pairs; the mask mainly suppressed
   degenerate loops, which is a real but much narrower benefit.
3. This is consistent with the independent finding on the RunPod release, where the shipped
   `transmute_proposition` returned byte-identical output for real, zero, `None` and mismatched
   hidden states — i.e. retrieval, not generation.

## What a legitimate benchmark requires

1. A parallel set that is **not** in any training corpus (verify by exact and near-duplicate match,
   e.g. 13-gram overlap), fixed before training.
2. Reporting of **memorisation rate** alongside quality, so a copy cannot score as a translation.
3. A genuine metric on unseen text: chrF++ / BLEU / COMET against human references, plus LaBSE as a
   secondary signal only.
4. A **seen/unseen-by-source** split: hold out whole works, not arbitrary sentences.

## Status of this project's evaluation

| claim | status |
|---|---|
| next-root acc@1 (v19.2 shipped) | 1.86% / 1.15% — measured, leak-free file split |
| next-root acc@1 (v20 retrained) | 3.85% / 6.11% — measured |
| count-based 4-gram reference | 33.4% — fitted on train, scored on held-out files |
| Grand-100 LaBSE 0.9037 | **invalid — 100% of pairs are in training data** |
| 100/100 "success rate" (v19.2.1) | crash/degeneracy metric, not translation quality |
