# First held-out evaluation of the NRMP morphemic tokenizer — findings

Harness: `/workspace/eval_heldout.py` (mirrored at `rootformer/build/eval_heldout.py`)
Results: `/workspace/eval_heldout_results.json` (per-file + per-author + per-corpus, all 3 modes)
Report: `/workspace/EVAL_HELDOUT_REPORT.md`
Sample cache: `/workspace/eval_heldout_sample.json`
Run logs: `/workspace/eval_heldout_run.log`, `/workspace/eval_cmp.log`
Per-word dumps (paired evidence): `/workspace/du_m0.json`, `/workspace/du_m1.json`, `/workspace/du_m2.json`

## Exact command

```bash
SSH="ssh -i /home/grem3/Documents/deepseek-harness/default-workspace/rootformer/.secrets/id_ed25519_runpod \
     -p 22231 -o StrictHostKeyChecking=no -o BatchMode=yes root@213.173.107.14"
cd /home/grem3/Documents/deepseek-harness/default-workspace/rootformer/build
scp -i ../.secrets/id_ed25519_runpod -P 22231 -o StrictHostKeyChecking=no eval_heldout.py \
    root@213.173.107.14:/workspace/eval_heldout.py

# the three modes (each evaluates the IDENTICAL cached sample; first run builds the scan)
for m in 0 1 2; do
  $SSH "cd /workspace && ROOTFORMER_VALIDATED_SEG=$m \
        /workspace/venvs/rootformer/bin/python eval_heldout.py --per-file-cap 50000"
done

# paired transitions (optional but this is the load-bearing evidence)
for m in 0 1 2; do
  $SSH "cd /workspace && ROOTFORMER_VALIDATED_SEG=$m \
        /workspace/venvs/rootformer/bin/python eval_heldout.py --dump-words /workspace/du_m$m.json"
done
$SSH "cd /workspace && /workspace/venvs/rootformer/bin/python eval_heldout.py --compare /workspace/du_m0.json /workspace/du_m1.json"
$SSH "cd /workspace && /workspace/venvs/rootformer/bin/python eval_heldout.py --compare /workspace/du_m1.json /workspace/du_m2.json"
$SSH "cd /workspace && /workspace/venvs/rootformer/bin/python eval_heldout.py --report"
```

CPU-only, single process, `nice -n 10`; the concurrent GPU job was not touched.

## Sample

3,298,362 word forms / 188,722 distinct forms from 92 held-out files, uniform stride,
cap 50,000 forms/file.

| corpus | files | forms | distinct | mode-1 round trip |
|---|---|---|---|---|
| scholastic_sunni_sanitized | 21 | 991,302 | 84,967 | 75.42% |
| scholastic_falsafa_sanitized | 40 | 1,136,880 | 101,462 | 71.53% |
| scholastic_masters | 17 | 690,977 | 71,157 | 75.73% |
| chronological_mujtahid_corpus (sanitized/*.jsonl) | 14 | 479,203 | 44,145 | 73.48% |
| **total** | **92** | **3,298,362** | **188,722** | **73.86%** |

NOT evaluated (kathra frequency-table sources, asserted out of the scan):
`scholastic_sanitized/`, `andalusian_canon_sanitized/`, `heritage_foundations/`.

## Metrics, all three modes (occurrence-weighted over the whole sample)

| metric | mode 0 greedy | mode 1 default | mode 2 root-reseg | m1 − m0 |
|---|---|---|---|---|
| **round-trip exact** | 73.24% | **73.86%** | 71.19% | **+0.62 pp** |
| round-trip skeleton | 73.24% | 73.86% | 71.19% | +0.62 pp |
| round-trip, distinct forms | **56.09%** | 55.96% | 54.32% | −0.13 pp |
| UNK (any slot) | 5.41% | **4.19%** | 5.02% | −1.22 pp |
| UNK (root slot) | 4.80% | **3.58%** | 4.28% | −1.22 pp |
| morphemic coverage | **51.92%** | 48.49% | 47.80% | −3.43 pp |
| collapse to `<PARTICLE>` | **41.38%** | 46.06% | 46.07% | +4.67 pp |
| collapse to `<NONE>` wazn | **45.57%** | 49.98% | 50.17% | +4.41 pp |
| root attestation (al-ʿAyn) | 66.46% | 69.05% | **75.54%** | +2.58 pp |
| root muhmal (condemned) | **0.74%** | 0.76% | 0.48% | +0.02 pp |
| root unrecorded | 23.99% | 21.24% | **18.30%** | −2.75 pp |

## Per-author (mode 1; filename prefix)

| author | files | forms | round trip | morphemic | UNK |
|---|---|---|---|---|---|
| **Ghazālī** | 6 | 333,819 | 76.77% | 50.36% | 3.05% |
| **Rāzī** | 6 | 275,838 | 76.23% | 49.15% | 3.39% |
| **al-Rāghib al-Iṣfahānī** | 3 | 87,938 | 73.66% | 52.33% | 2.83% |
| Taftazānī | 2 | 96,811 | 79.31% | – | – |
| Juwaynī | 2 | 92,350 | 77.09% | – | – |
| Farābī | 7 | 92,644 | 76.77% | – | – |
| **Bīrūnī** | 4 | 167,317 | **56.62%** | 43.7% | 12.6% |
| Kindī | 3 | 52,149 | 65.15% | – | – |
| Ṭaḥāwī (chrono) | 2 | 49,633 | 69.23% | – | – |

Author keys are filename prefixes, so `Ibn_*` (Ibn Sīnā, Ibn Rushd, Ibn al-Haytham) collapses
into one `Ibn` bucket (26 files, 74.37%), and the chronological lowercase keys are separate from
the capitalised ones (`Maturidi` 74.88% vs `maturidi` 74.82%). Per-file rows are in the JSON.

**Bīrūnī is the real outlier**, driven by domain not author: the three scientific works collapse —
`Al-Athar_al-Baqiyah` 49.73% (UNK 12.2%), `Al-Qanun_al-Masudi` 50.93% (UNK **20.0%**),
`Kitab_al-Zilal` 57.46% — while `Tahqiq_ma_lil_Hind` reaches 70.46%. Worst files are all
astronomical/technical; the best (80–81%) are Ibn Sīnā *Ilāhiyyāt*, Rāzī *Matālib*, Ghazālī
*Tahāfut*.

## Top-10 round-trip failures (mode 1, occurrence-weighted)

| # | word | count | decoded | root | wazn | suffix |
|---|---|---|---|---|---|---|
| 1 | الله | 31,938 | أله | اله | فَعَلَ | `<NONE>` |
| 2 | عليه | 18,068 | علىه | `<P:على>` | `<NONE>` | ه |
| 3 | فى | 17,526 | `<PARTICLE>` | `<PARTICLE>` | `<NONE>` | `<NONE>` |
| 4 | له | 14,886 | ل`<UNK>` | `<UNK>` | `<NONE>` | `<NONE>` |
| 5 | به | 12,194 | `<UNK>` | `<UNK>` | `<NONE>` | `<NONE>` |
| 6 | ان | 10,138 | `<UNK>` | `<UNK>` | `<NONE>` | `<NONE>` |
| 7 | إلا | 10,098 | `<UNK>` | `<UNK>` | `<NONE>` | `<NONE>` |
| 8 | تعالى | 9,942 | `<PARTICLE>` | `<PARTICLE>` | `<NONE>` | `<NONE>` |
| 9 | صلى | 6,082 | `<PARTICLE>` | `<PARTICLE>` | `<NONE>` | `<NONE>` |
| 10 | و | 5,740 | `<UNK>` | `<UNK>` | `<NONE>` | `<NONE>` |

Mode 0's list differs only in rank; mode 2's adds **قال → قول (22,432)** at #2.

## Honest reading

**The shipped default IS better than greedy on the headline.** Paired over the identical
188,722 forms: mode 0 → mode 1 newly fixes 53,462 occurrences and newly breaks 33,126,
**net +20,336 occurrences (+0.62 pp)**, and better UNK (−1.22 pp) and attestation (+2.58 pp).

**But the default is worse than greedy on four metrics, and that must be said plainly:**
distinct-form round trip (−0.13 pp; 306 forms broken vs 55 fixed, net **−251 forms**),
morphemic coverage (−3.43 pp), `<PARTICLE>` collapse (+4.67 pp), muhmal (+0.02 pp).

These are two different stories and both are true:
* The coverage/particle loss is mostly the closed-class rule doing its job. Of 82,302
  occurrences reassigned REAL-root → `<PARTICLE>`, **76,294 already decoded correctly either
  way**, 1,537 newly decode, only 1,120 newly fail. Those were spurious "roots" invented out of a
  clitic + particle (أنه, فيه, عنه, منه, لأنه …) and are now honestly particles. Losing morphemic
  coverage there is *correct*.
* The distinct-form regression is a genuine cost. The rule over-fires on `إن/أن` written without
  hamza and on pronoun-suffix forms: **ان (10,138) → `<UNK>`** is the single largest regression,
  then لهم 2,648, انه 2,392, فان 2,079, وان 1,683, لك 1,133.

**Mode 2 must not ship.** Paired mode 1 → mode 2: newly broken 89,633 occurrences vs newly fixed
1,665, **net −87,968 (−2.67 pp)**, −3,098 distinct forms. It is a hollow-verb (ajwaf)
over-application: قال→قول (22,432), كانت→كونت (5,986), فقال→فقول, حال→حول, الناس→النوس,
المال→المول, قلنا→وقلنا, رضي→`<UNK>`. Its attestation gain (+6.50 pp) is real but not worth
destroying 88k words of surface. This reproduces, on held-out text, the module's own warning that
mode 2 is "not good enough to enable".

**What the absolute numbers mean.** 73.86% round trip ⇒ **26.1% of running word forms — about
1 in 4 — do not survive `decode(encode(w))`**. The al-ʿAyn 70.63% figure was pessimistic; the
real held-out number is better but the same order. The more damaging number for a tokenizer used
as an LM vocabulary is the **type-level 55.96%**: **44% of distinct word forms fail**, because
failures concentrate in the rarer, more inflected forms while the frequent ones survive.

The 26% is not diffuse morphological inadequacy; it is dominated by four fixable, systematic bugs:
1. **Control tokens are emitted as literal text.** `decode_word` returns the strings
   `<PARTICLE>` and `<UNK>`. Every word hitting the generic `<PARTICLE>` fallback (فى 17,526,
   تعالى 9,942, صلى 6,082, يا …) or `<UNK>` (و, إلا, ان, به …) fails by construction. This is
   probably the single largest bucket of the 26%.
2. **الله** — 31,938 occurrences, the commonest word in the corpus — analyzes as root `اله` +
   `فَعَلَ` and realizes `أله`. Definite-article / hamzat al-waṣl handling in realization.
3. **Particle + pronoun orthographic contraction**: `<P:على>` + ه → `علىه`, not `عليه`.
4. **Mode 2's ajwaf rewriting** (above).

**Bug signal requested: muhmal is non-zero in every mode** — 13,179 occurrences (0.74%) in mode 0,
**12,656 (0.76%) in the shipped default**, 7,833 (0.48%) in mode 2. al-ʿAyn condemns those roots,
so the analyzer is still assigning roots the record marks مهمل. A further 21.24% of assigned roots
(mode 1) are not in al-ʿAyn's record at all; that is not necessarily wrong — the record holds
4,381 attested roots against a 9,114-root lexicon — but it bounds how much the attestation metric
can certify.

## Contamination — measured, not assumed

The kathra table (`data/kathra_counts.json`) was built from exactly
`heritage_foundations/*.txt`, `andalusian_canon_sanitized/*.txt`, `scholastic_sanitized/*.txt`
(its own meta records this; 51 files, 12,403,007 tokens, 294,926 kept forms, min_count 2).
Those three directories are asserted out of the scan.

Measured overlap of the held-out sample with the kathra table:

| corpus | form overlap | type overlap |
|---|---|---|
| scholastic_sunni_sanitized | 97.39% | 90.07% |
| scholastic_falsafa_sanitized | 94.84% | 80.68% |
| **scholastic_masters** | **98.72%** | **94.64%** |
| chronological_mujtahid_corpus | 97.71% | 90.03% |

So "held out" is true in the strict sense — the table never read these files — but the
*vocabulary* is not disjoint: 81–95% of distinct forms are also kathra keys, because
high-frequency classical Arabic is shared across the genre. Hapax pruning (min_count 2) makes
this a lower bound.

**Recommendation:** treat `scholastic_masters/` as contaminated for any future kathra-related
claim. Its 94.64% type / 98.72% form overlap is the highest of all four corpora, and 16 of its 17
filenames are the same works as the kathra-source `scholastic_sanitized/` (bytes differ only by
sanitisation). The user's table marked it "yes"; the measurement says otherwise.

## Limitations

* Morphological correctness is not hand-annotated anywhere here — these are objective round-trip /
  coverage / attestation metrics only, by design. They bound surface fidelity, not linguistic
  correctness.
* Author keys come from filename prefixes, so `Ibn_*` merges several authors and chronological
  lowercase keys are separate from the capitalised ones.
* The sample is stride-sampled at ≤50k forms/file, not a full census; the four largest
  chronological works are truncated most.
* `scholastic_masters` files are byte-different but work-identical to kathra sources (above).
