# VERSIONS — the correct version

All hashes verified 2026-10-02 against `/workspace/hf_v19_2_release/` on the pod.
**If a file's md5 does not match, it is not the version that produced the results.**

## Pinned source (`src/`)

| file | md5 | note |
|---|---|---|
| `nrmp_vocab.py` | `0ee0ab6301d7f26eddc7bd079f7fede7` | vocabulary, 10,052 entries |
| `nrmt_train.py` | `56ef3f2fd9ddc352dee596f32590e7b5` | **root-attn patched** (original was `ceacdb60…`, diff 406 lines) |
| `nrmt_arch.py` | `ac7f895123583fc47f04f74df7914535` | |
| `root_cross_attn.py` | `b53ae702bde947ac66f933191f904c36` | the RCA module |
| `data/rootformer_v12_arabic_blueprint.json` | `5bd3e828b41fb1a6a6039d47c2f359fc` | config |

`nrmp_vocab.py`, the blueprint and the checkpoints are **read-only** — do not edit. Any experiment
that needs to change behaviour must wrap, patch, or hook them, and must prove flag-off inertness.

## Checkpoint — use this one

```
checkpoints/rootformer_v19_2_synthesis_ar_backbone...roots9490.tok10052.safetensors
md5  3335a3d39091535d0fdd047dca842715
size 795,064,504 B
```

The chain is **append-only** — rows `0..9113` are bit-identical across all three root extensions:

```
…awzan142.safetensors     055411f5…                              (baseline)
…roots9313.safetensors    1e832105…   +199 roots
…roots9490.safetensors    1005eb9a…   +177 (Lisān, incl. all 69 ي-roots)
…roots9490.tok10052.safetensors  3335a3d3…   ← USE THIS
```

Aligned cache: `/workspace/head_fix/nrmp_cache_9490_aligned/` (train 113,022,473 B, md5 `051d8e54…`).

## Experiment variants (in `build/`, not shipped)

| file | md5 | what it adds |
|---|---|---|
| `build/root_attn_width/nrmt_train_width.py` | `0c58829b78263c40e75b7fb11120d391` | `--rca-dim W` (internal attention width) |
| `build/ghazali_forget/nrmt_train_ewc.py` | — | `--ewc-lambda`, `--ewc-fisher`, `--ewc-scope` (inert by default) |
| `build/ishtiqaq_check/ishtiqaq_root_bias.py` | `fbcbc033c546adb7b5d4ec16119dc80b` | gateable native score bias |

Every variant must satisfy the same two properties the originals were held to:
**flag-off is bit-identical to shipped**, and **the added pathway is provably live when on**
(ablation changes the output).

## Model config

```
backbone      Qwen2.5-0.5B config, 24 layers, d_model 896
architecture  UnifiedRootformerV12 + IshtiqaqAttentionV12 (v12 lineage)
num_roots     9,490         total_vocab_size  10,052        num_awzan  142
```

## Known-drift files (do NOT trust local copies)

Some mirrors in `build/` are stale relative to the pod canonical. Observed drift:

```
build/nrmp_vocab.py    7a3f3a57…   ← STALE, canonical is 0ee0ab63…
build/nrmt_arch.py     0558fc99…   ← STALE, canonical is ac7f8951…
```

`src/` holds the verified copies. Always re-verify by md5 before trusting a number produced from a
file, and re-pull from `/workspace/hf_v19_2_release/` rather than a mirror.

## Three distinct root-id spaces — a known trap

```
9,490   morphemic_embed / nrmp_vocab          ← the live one; RCA reads root_embed (9490, 448)
9,015   models/unified_rootformer_v12.py:80,91 hardcoded
9,015   IshtiqaqAttentionV12.root_embed (9015, 64)   ← dead code, 0/24 calls carry root_ids
```

`ishtiqaq_attention_v12.py:169,208` does `torch.clamp(root_ids, 0, 9014)`, so 475 ids collapse onto
row 9014 and 5.01 % of roots have no row at all.
