# Rootformer backup — 2026-10-01

Google Drive target: `gdrive:rootformer_backup_2026-10-01/`

Created from the RunPod host `213.173.107.14:22231` and the local workstation tree at
`~/Documents/deepseek-harness/default-workspace/rootformer`.

## What is in here

| path | contents | files |
|---|---|---|
| `local_analysis/` | the local workstation's analysis tree, as of this backup | 235 |
| `pod_workspace/` | top-level `/workspace` scripts, results and notes | 72 |
| `pod_release/` | `hf_v19_2_release` **code only** (checkpoints excluded) | 90 |

`local_analysis/` holds `build/`, `release/`, `results/`, `recon/`, `corpus/`, `local_corpus/`
and the top-level reports (`NRMP_REPORT.md`, `IMPLEMENTATION_FIDELITY.md`,
`CLASSICAL_SOURCE_VERIFICATION.md`, `BENCHMARK_CONTAMINATION.md`, `ARCHITECTURE_FIX.md`,
`MT_RESULTS.md`).

## What this backup adds over `rootformer_v20_backup` (2026-09-30)

The v20 backup already holds `analysis/` (older revisions), `hf_v20_release`, `nrmp_cache`,
`corpus_basran`, `corpus_andalusian`, `corpus_v18_data`, `v18_masters`, `texts`,
`v19_2_baseline_checkpoints`. This one adds the work done on 2026-10-01:

**The complete al-ʿAyn attestation record.**

| | v3 (in the v20 backup) | v4 (here) |
|---|---|---|
| attested permutations | 2,483 | **4,381** |
| unused (مهمل) permutations | 106 | **2,750** |
| 2-letter pairs marked مهمل | 0 | **41** |
| chapters with a usable enumeration | 1,183 | **1,428** |

Files: `khalil_attest_v4.py` (extractor, reproducible), `khalil_attest_v4.json` (the record),
`KHALIL_ATTEST_V4_FINDINGS.md` (the five defects, the guards, what was rejected, and the results).

**The analyzer chain**, in `analyzer_segmentation_v4/v5/v9.py`:

- v4 — the record wired in, **no rule change: null result** (the analyzer trusted the lexicon
  before al-ʿAyn, so a 26× larger negative record was invisible). This negative result is the
  most useful finding of the session.
- v5 — the **mudaʿʿaf pair gate**: `C1 C2 C2` is the doubling of the pair `(C1,C2)`, so `نتت` is
  condemned by the pair `نت` being مهمل.
- v9 — **ranked evidence** (al-ʿAyn `+8` > lexicon `+3`), root candidates from al-Khalīl's
  *calculation* rather than the lookup, and elision constrained by the ajwaf/nāqiṣ division.

Result: `كنت → كون`, `قال → قول`, `دعا → دعو`, with `فكم → ف+كم` and `عنهم → عن+هم` retained.

## Deliberately excluded, and where it already lives

| excluded | size | already at |
|---|---|---|
| `local_prep/data/` | 1.9 GB | `gdrive:rootformer_v20_backup/corpus_v18_data/` (same filenames) |
| `hf_v19_2_release/checkpoints/` | 863 MB | `gdrive:rootformer_v20_backup/v19_2_baseline_checkpoints/` |
| `/workspace/rootformer_v12` | 19 GB | `gdrive:rootformer_v20_backup/` |
| local `.secrets/` | — | never backed up; SSH keys are not uploaded |

## Known open items at the time of this backup

1. **The segmentation fix is not wired in.** `analyzer_segmentation_v9.py` has **0 importers** and
   `morphemic_tokenizer_v12_arabic_PATCHED.py` contains no reference to it. `كنت → كون` holds only
   when the wrapper is called directly.
2. **The grammar audit reports 9/20 on the shipped tree.** `verify_v2.py` passes, but it tests the
   corrected helper modules, not the path the model runs. The audit's own failures say
   `SibawayhConstituentStack ... is not called from the NRMP path`,
   `state is recomputed from t-1 only`, `roots are independent one-hot ids`.
3. **`irab_realizer`, `iktifa_apocope`, `constituent_stack`, `khalil_root_calculator`,
   `loanword_rule`, `ilal_conditioned` have 0 importers** — implemented and verified, currently
   dead code.
4. **No gold set, therefore no honest measurement on real text.** The 95.1% figure remains
   withdrawn as circular.

## Restore

```bash
# from Google Drive
rclone copy gdrive:rootformer_backup_2026-10-01/local_analysis/ ./rootformer/
rclone copy gdrive:rootformer_backup_2026-10-01/pod_workspace/ /workspace/
rclone copy gdrive:rootformer_backup_2026-10-01/pod_release/  /workspace/hf_v19_2_release/
```

The record is rebuilt from the primary text in about 30 s:

```bash
python khalil_attest_v4.py corpus/basran/Al_Khalil_Al_Ayn.txt results/khalil_attest_v4.json
```

Source text: `corpus/basran/Al_Khalil_Al_Ayn.txt` (5.2 MB, 36,381 lines) — included in
`local_analysis/corpus/`.
