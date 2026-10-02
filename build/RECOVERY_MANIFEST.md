# Recovery manifest — rootformer, end of session 2026-10-01

The pod (`root@213.173.107.14:22231`, `/workspace/hf_v19_2_release`) went down with
`Connection refused` while the model-scale fix, the lexicon rebuild and a Drive backup were
running. This file records what survived, what is reproducible, and how.

## Remote environment (to recreate)

```
SSH="ssh -i rootformer/.secrets/id_ed25519_runpod -p 22231 -o StrictHostKeyChecking=no -o BatchMode=yes root@213.173.107.14"
python: /workspace/venvs/rootformer/bin/python
GPU:    RTX PRO 4500 Blackwell, 32623 MiB
rclone: RCLONE_CONFIG=/workspace/.rclone/rclone.conf   remote: gdrive:
```

## Safe locally (verified present, do not lose these)

| path (under `rootformer/`) | size | what |
|---|---|---|
| `corpus/basran/IbnManzur_Lisan_al_Arab.txt` | 30.2 MB | Lisān al-ʿArab, canonical (Shamela0001687 recension), md5 `998201d7b4e1361e256b88979a9e72df` |
| `corpus/basran/Zamakhshari_Asas_al_Balaghah.txt` | 2.9 MB | Asās al-Balāghah, md5 `3a7363bab8ddcbc47903745c404de1be` |
| `build/lisan_fetch/` | 31 MB | the other three Lisān recensions + web scrape + parsers |
| `build/asas_fetch/` | 37 MB | Asās acquisition: recensions, verification, derived-corpus audit |
| `build/raghib_audit/` | 11 MB | al-Rāghib fidelity audit (the fabricated-bibliography finding) |
| `build/taxonomy_out/` | 25 MB | closed taxonomy of all 83,111 failing forms |
| `build/ppl_trace/` | 8.3 MB | per-step loss/PPL, 5,000 and 20,000 steps, + `nrmt_train.py.orig` |

## Reproducible on a fresh pod (rebuild in this order)

State at time of loss: `num_roots=9490`, `total_vocab_size=10052`, `num_awzan=142`.

1. **Restore the release tree** from `gdrive:rootformer_backup_2026-10-01_night/` (the earlier,
   verified backup: 434 + 38 objects, ~871 MiB).
2. **Original checkpoint** `checkpoints/rootformer_v19_2_synthesis_ar_backbone.awzan142.safetensors`
   md5 `055411f54e6f5ad7ef42855e11dd40eb` — 793,724,088 B. Must be present and unchanged.
3. **Extend the root tables** (append-only, 9114 → 9490):
   ```
   python extend_root_checkpoint.py --src <original> \
     --dst checkpoints/...awzan142.roots9490.safetensors \
     --target-roots 9490 --init mean+noise --noise-frac 1.0 \
     --against <original>
   ```
   then append the token table (9868 → 10052):
   ```
   ... --append backbone.embed_tokens.weight=10052
   ```
   Target md5: `3335a3d39091535d0fdd047dca842715`
   (`.roots9490.tok10052.safetensors`). **Use this one** — `.roots9490.safetensors`
   (`1005eb9a…`) has `embed_tokens` 9868 and will not load against the live 10052 blueprint.
   The script is parameterised and idempotent; intermediates are `1e832105…` (9313).
   **Safety property: rows 0..9113 must be bit-identical to the original.**
4. **Rebuild the token cache** — `/workspace/nrmp_cache_9490/{train,val}.pt`.
   The split is NOT a stored artifact: it is **derived by `nrmp_train.py --prepare`
   (`cmd_prepare`, lines 86-141)** — corpora → sentences 4-64 words, `limit=120000` → dedup →
   `sorted(files)` → `random.Random(1337).shuffle` → `n_val = int(len*0.08)`, file-level.
   Reproduced: 47 files, 3 val files, 691,127 unique sentences; train prefix stream 8,311,213;
   val 972,383 — byte-identical file sizes to the shipped cache.
   Target md5s: train `214959e528c8e5d8a33ac38a1c84afa2`, val `2f17167779f2e9465351f23309f40701`.
5. **Vocab/blueprint**: `nrmp_vocab.py` md5 `0ee0ab6301d7f26eddc7bd079f7fede7`,
   blueprint `data/rootformer_v12_arabic_blueprint.json` md5 `5bd3e828b41fb1a6a6039d47c2f359fc`
   (+177 Lisān roots, `classical_roots_lisan_extension [9868,10052)`).
6. **Trainer**: `nrmt_train.py` md5 `41341576c1c3f0a52c0398832cb6ddce`.
   Per-step trace needs **no edit** — `--probe-out` already exists (line 202).

## What was lost with the pod (needs redoing)

- `/workspace/lexicon_rebuild/out/` — 8 sources parsed and hand-validated (17 MB):
  headings/unique roots — lisan 9,151/8,973 · tahdhib 6,005/6,005 · ayn 5,592/5,592 ·
  maqayis 4,278/4,276 · sihah 4,158/4,074 · jamhara 3,970/3,818 · asas 3,842/3,724 ·
  mufradat 1,512/1,506. **Union coverage 98.2 %** (9,031 of 9,196 roots covered; 165 uncovered).
  Per-source: lisan 96.9 · tahdhib 54.2 · ayn 51.5 · sihah 43.0 · maqayis 40.8 · asas 37.1 ·
  jamhara 27.4 · mufradat 13.7. Source paths are in `out/inventory_coverage.json → cat`.
  **The parsers and these numbers are the thing to reproduce first** — quotes/records were never
  extracted, so the rebuild had produced no citable output yet.
- The logit-scale (`extra_norm`) fix — never completed; `nrmt_train.py` was unmodified at loss time.
- Drive backup `rootformer_backup_2026-10-01_late/` — **status unconfirmed**; the agent ended
  without reporting, and the host died. Check Drive for the folder before assuming either way.

## Results established this session (so they need not be re-derived)

- **Round-trip, held-out**: mode 1 token `73.8610 % → 80.7170 % → 82.1065 %`; distinct-form
  `55.9611 % → 56.4423 % → 57.5884 %`; failing types 83,111 → 82,203 → 80,040.
  Union-key standard: newly-passing vs newly-failing reported separately.
- **Mode 2 must not ship**: net −87,968 occurrences.
- **E1** (`<PARTICLE>` echo, the dominant bucket) 69,509 → 67,315 types after the 177 roots;
  100 % of the 2,193 newly-passing forms carry one of the recovered roots.
- **Training**: 20,000 steps, train loss 11.698 → 2.222 (PPL 120,335 → 9.2), **no NaN/Inf**,
  584 s, peak 2742 MiB. **Val root acc@1 FLAT 5.42–5.95 %** while val PPL inflates 55×
  (1197 → 65394). `corr(extra_norm, val root CE) = +0.9757` — the PPL blow-up is a
  **logit-scale calibration artifact**, not incapability. **Report acc@1, never raw PPL.**
- **Step 2–4 loss explosion** is deterministic under `--head-init remap` (loss 90.53, grad_norm
  12,410, from warm logits driving the quadratic impossibility hinge). Scratch peaks at 146.
- **Do not trust a 3-step smoke test** — `OneCycleLR` (`pct_start=0.3`) peaks the LR within one
  step of three and inverts the ranking. Isolate the head with `--lr 0 --steps 1` instead.
- **Six classes of defect found**, all one family (*a step that failed without saying so*):
  dead data · missing data · unreachable rules · invented citations · non-executing wiring ·
  checks that cannot fail. Five separate "checks that couldn't fail" were found this session.
