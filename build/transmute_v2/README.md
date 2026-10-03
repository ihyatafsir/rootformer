# transmute_v2 — corrected transmutation + early root structure

Pod `g6exduq0bd17z8` (`213.173.104.76:46758`), Python `/workspace/venvs/rootformer/bin/python`.
Written to `/workspace/transmute_v2/` and mirrored here. `models/unified_rootformer_v12.py`,
`nrmp_vocab.py`, the blueprint, the checkpoints and the caches are **read-only and unmodified** —
this directory only *adds* files.

---

## Read this first: what is measured, and what is not

| deliverable | status |
|---|---|
| liveness assertion (flagged defects) | **RUN, complete** |
| constant fixes + one-space assertion | **RUN, complete — all 5 verdicts OK** |
| checkpoint resize + 290 real Qwen tensors | **RUN, complete — 290/290** |
| init-equivalence proof (gate 0 bit-identical) | **RUN, complete — PASS** |
| isolation ablation (gate 0 vs 1, rolled roots) | **RUN, complete — PASS** |
| seven verification suites | **RUN, complete — 20/20, ALL PASS, 67/67, 15/15, 46/46, 30/30, 24/24** |
| arms A (floor) / C (early root) at 20k | **NOT RUN** — no GPU slot (2 residents = the measured ceiling); launcher ready |
| length-1 ridge probe on trained arms | **NOT RUN** — needs a trained arm |

The two claims the brief says make future runs trustworthy (init-equivalence, isolation) are
proved. The arms were not run: the pod never had a free slot, and a single ungrounded arm is
uninterpretable.

---

## 1. Liveness (`liveness_assert.py` → `liveness_stdout.txt`, `liveness_report.json`)

Method: three independent measurements, none of which is grep — forward **hooks**, per-parameter
**gradient reach** after a real forward+backward, and **causal perturbation** (does changing this
parameter change the output). The perturbation test needed two fixes before it told the truth:
`std()` of a scalar is NaN (so `nan or 1e-4` poisoned every scalar), and `p + eps - eps != p`
(so a naive restore drifts the model and every later "inert" parameter inherits the drift). After
the fixes: determinism bit-equal, restore integrity bit-equal after 230 perturb/restore cycles,
**201 parameters causally INERT**.

### Flags

1. **`self_attn.root_embed` is 9015 rows against a live 9490-id space** (all 24 layers).
   475 ids have no row; `clamp(root_ids, 0, 9014)` puts 475 ids (**5.005 %**) onto row 9014, so
   row 9014 represents **476 different roots**. Not injective.
2. **`self_attn.wazn_embed` is 128 rows against a live 142.**
3. **193 parameters have `.grad is None` — never read in the live forward.** 24×
   `root_embed`, `root_q_proj`, `root_k_proj`, `wazn_embed`, `ishtiqaq_gamma`, `coverage_weight`,
   `coverage_threshold`, `governance_strength` — plus **`backbone.embed_tokens.weight`**, dead
   because the trainer passes `inputs_embeds` (confirmed INERT under perturbation too).
4. **`stream_mix[1]` has `grad == 0.0` exactly in all 24 layers** — the native root path has never
   received a single gradient. Resolved element-wise
   (`liveness_streammix_elements.py`): **element 0 LIVE in 24/24, element 1 INERT in 24/24**.
5. **Every `self_attn.active_root_ids is None` after a full live forward** — 0/24 attention calls
   carry root ids. The native IshtiqaqAttentionV12 root path is dead.
6. **`base.extract_morphemic_ids()` returns all-zero `wazn_ids`** — the stub at
   `unified_rootformer_v12.py:121` is dead, **but the wazn is LIVE** through
   `morphemic_embed(p, r, w, s)`. This is the finding previously misread as the architecture; the
   perturbation test settles it (perturbing `wazn_embed` moves the output).
7. **`base.id_to_root_table`: 9199 `<root_*>` tokens → 9000 distinct ids, max 9014,
   non-injective.** The `token_id - 349` offset arithmetic is wrong.
8. **NEW, not in the brief.** `deepseek_v4_1_flash_model.py` registers **live forward hooks on
   trunk layers 1/11/14** (`engram_layer1`, `engram_layer14`, `farahidi_orbit`) holding
   **151.95 M parameters that are in no checkpoint and no `model.parameters()`** — never
   optimised, though they accumulate `.grad` forever. Zero-init INERT today (`mem_proj` /
   `out_proj` are exactly 0), but they would inject random noise the moment anyone initialised
   their output projections.
9. `unified_rootformer_v12.py:51` comment says vocab 9,856; the live value is **10,052**.

### Two corrections to the brief

* The brief attributes md5 `ac7f8951…` to `models/unified_rootformer_v12.py`. `ac7f8951…` is
  **`nrmt_arch.py`** (verified on the pod). The pod's `unified_rootformer_v12.py` is
  `68646a897e7037e5219dfd0463fdcc3d`, identical to the local copy.
* The brief quotes the clamp as `clamp(root_ids, 0, 9014)`. The shipped source writes
  `clamp(root_ids, 0, self.root_embed.num_embeddings - 1)` — 9014 is **derived**, not literal, so
  the collision is a consequence of the wrong table size, not of a hardcoded bound.
* `src/` locally was missing `validated_segmentation.py` (and other release modules). Without it
  `nrmp_vocab` silently reports **9445** roots instead of **9490** — exactly the silent-shrink
  failure `authored_closed_inventory` documents. The missing release modules were pulled in.

---

## 2. Constant fixes (`unified_rootformer_v13.py`, `root_space.py`)

`unified_rootformer_v12.py` is **not edited**. `unified_rootformer_v13.py` is a generated sibling;
`make_trainer_v13.py` generates the trainer and writes a unified diff.

| v12 | v13 |
|---|---|
| line 26 `num_roots=9015, num_awzan=128` | `LIVE_ROOTS` 9490, `LIVE_AWZAN` 142 |
| line 51 `vocab_size` comment 9,856 | asserted `== LIVE_VOCAB` 10052 |
| lines 80/91 `num_roots=9015` | 9490 |
| lines 81/92 `num_awzan=128` | 142 |
| line 109 `root_part = token_id - 349`, `clamp(1, 9014)` | `build_token_root_table(vocab)` — **by string** through the live `root2id`, injectivity asserted |
| line 121 `wazn_ids = torch.zeros_like(...)` stub | `build_token_wazn_table(vocab)` — the real wazn id |
| lines 172/180 `view(-1, 9015)` / `view(-1, 128)` | `view(-1, LIVE_ROOTS)` / `view(-1, LIVE_AWZAN)` |
| line 277 `from_pretrained` (needs the network) | sealed safetensors, resized, 290 tensors asserted |

`assert_one_root_space(model, vocab)` raises on **any** 9015/128 survivor.
`assert_no_aliasing()` raises on the 475-id collision.

### One-space assertion result (`transmute_v13_stdout.txt`, EXIT=0)

```
constants 9490 / 142 / 10052 in every table           [OK]
token->id tables injective, in range, no offset math  [OK]
290/290 real Qwen tensors consumed, table resized     [OK]
exactly 48 stale-SPACE tables dropped, nothing else   [OK]
one root-id space after loading                       [OK]
```

* Token→root: 9197 `<root_X>` tokens → **9197 distinct** live root ids, min 0 max **9489**,
  injective. Token→wazn: 137 → 137 distinct.
* Root tables after loading: `[9490]` across all 24 layers.
* **290/290 real Qwen tensors**: 288 layer tensors + `embed_tokens` (resized 151936 → 10052) +
  final norm. `lm_head` is absent from a released NRMT checkpoint (the NRMT head replaces it).
* **Exactly 48 stale tables dropped** (24× `root_embed` 9015, 24× `wazn_embed` 128); all 506
  non-stale trunk keys loaded; 0 unexpected; the 14 `morphemic_embed` / `final_norm` / `nrmp_head`
  keys belong to the `RootformerNRMT` wrapper and are overlaid by the trainer.

Two traps found while doing this and written into the code as comments:
* the HF cache on this pod holds only the **681-byte config blob**; `/workspace/.hf_home` holds a
  **dangling symlink** for the weights; the only real **988,097,824-byte** copy lives in another
  agent's read-only cache (`/workspace/alt_base/cache/hub`). Resolving through the hub silently
  picks the broken link. The weights are now addressed as a file, with size-based discovery.
* a released checkpoint is a `RootformerNRMT` state dict (`backbone.layers.*`) while V13 holds the
  `CausalLM` (`backbone.model.layers.*`). A naive `load_state_dict` reports 555 missing / 520
  unexpected and loads **nothing**; the prefix remap is explicit.
* Saving the 795 MB corrected checkpoint fails with **disk quota exceeded** (the `/workspace`
  mount is 80 % used, 466 T free, so it is a per-user quota). It is **not needed**: the trainer
  drops the 48 stale tables and loads in-process.

---

## 3. Init-equivalence + isolation (`proof_init_and_isolation.py` → `proof_pathway.json`)

Schedule **ALL 24 layers (0..23)** — root structure from the input stage upward. Network: residual
injector **57.86 M** new params, native score bias **11.01 M**.

### Part 1 — gate 0 is BIT-IDENTICAL to no pathway (`torch.equal`, max|d| = 0)

```
residual injector  gate 0 vs NO pathway : bit_equal=true   max|d|=0.0
native score bias  gate 0 vs NO pathway : bit_equal=true   max|d|=0.0
BOTH               gate 0 vs NO pathway : bit_equal=true   max|d|=0.0
forward determinism: two identical forwards bit-equal = True
```

### Part 2 — isolation: both mechanisms ACT and both READ THE ROOT SEQUENCE

```
residual    gate 1 vs none                        : differ  max|d|=0.712
residual    gate 1, ROOTS ROLLED, trunk input FIXED: differ  max|d|=1.002   <- reads roots
residual    gate 0, roots rolled                   : bit_equal=true max|d|=0.0 <- no leak
score bias  gate 1 vs none                        : differ  max|d|=0.159
score bias  gate 1, roots rolled, input fixed      : differ  max|d|=0.0218  <- reads roots
score bias  gate 0, roots rolled                   : bit_equal=true max|d|=0.0
complementary: residual-only vs bias-only differ (0.631); residual effect RMS 0.112 vs bias
              0.0128; both vs sum-of-parts differ (0.027) => they interact
```

The roll test is the isolating one: the trunk input is **held fixed** and only the root sequence
the pathway reads is rolled. A pathway whose output does not move when the roots move is not
reading roots.

---

## 4. The two mechanisms, as built

```
score bias   total_score += mix*cond*score_root + identical_root_bonus      -> REWEIGHTS
             `ishtiqaq_root_bias.IshtiqaqRootBiasAttention` (gated subclass; shipped file untouched)
residual     h <- h + gate*LayerNorm(o_proj(softmax(Wq·h · Wk·E(r)) @ Wv·E(r)))  -> INJECTS
             `root_cross_attn.RootHistoryCrossAttention` (out_norm=True)
```

Both read the **448-dim `morphemic_embed.root_embed` (9490×448)**. `early_root_path.py` adds the
`all` / `earlyN` schedules and a `_guard_root_source()` that **refuses to attach** a pathway whose
root source is not 9490 rows.

---

## 5. Files

| file | what |
|---|---|
| `liveness_assert.py` | the liveness assertion (hooks + grad + causal perturbation) |
| `liveness_streammix_elements.py` | element-wise proof that `stream_mix[1]` is inert, `[0]` live |
| `root_space.py` | the ONE root-id space + the assertions |
| `unified_rootformer_v13.py` | the corrected architecture + resize/290-tensor transmutation |
| `transmute_v13_check.py` | the transmutation + one-space proof (EXIT=0) |
| `early_root_path.py` | schedules + attachment API + root-source guard |
| `proof_init_and_isolation.py` | init-equivalence + isolation proofs |
| `make_trainer_v13.py` / `nrmt_train_v13.py` / `nrmt_train_v13.diff` | the trainer (generated, diff auditable) |
| `run_matched_arms.sh` | matched A/C launcher with a **broadened** occupancy gate + VRAM check |
| `_ref_*.py`, `_ref_*.json` | the reference protocol and the recorded comparison numbers |

### Comparison numbers (recorded, from the pod)

| number | source | n |
|---|---|---|
| **29.30 % / 23.34 %** raw base, length-1 ridge | `_ref_probe_orig.json` (marginal 0.4047 %) | 63,496 / 21,243 unseen |
| **92.34 % / 83.92 %** current trunk, same protocol | `_ref_echo_probe_len1.json` | 73,502 / 14,390 unseen |
| **19.53 %** `ALL_val.acc@1` | `results_RCA_UNFREEZE_A2.json` (RCA `top4`), `NOVEL_only` 19.90 % | 18,869 |

Reference implementation of the length-1 ridge protocol: `build/alt_base/echo_probe_len1.py`
(plus `echo_identity.py` for `fit_ridge` / `eval_ridge`). The trained arms must be probed with the
**pathway active and `set_root_ids` / `active_root_ids` supplied**, or the probe measures a
pathway-off trunk.

---

## 6. Running the arms

```bash
STEPS=20000 ./run_matched_arms.sh --status        # occupants + free VRAM
STEPS=20000 ./run_matched_arms.sh                  # waits for a slot, runs A
STEPS=20000 ./run_matched_arms.sh --with-bias      # A, then C, then D
STEPS=20000 ./run_matched_arms.sh --concurrent     # A and C together (2 occupants = the ceiling)
```

`run_matched_arms.sh` counts occupants with a **broadened `ps`** (the standard
`pgrep -f "nrmt_train[.]py --checkpoint"` misses `nrmt_train_width.py`, `nrmt_train_ewc.py` and
`nrmt_train_v13.py`) **and** checks VRAM, waits rather than killing, and refuses to become a third
occupant (the measured-safe ceiling is two 32-batch arms ≈ 21 GB of 32.6 GB).

A and C differ in exactly one flag (`--root-cross-attn none` vs `all`); everything else — LR,
steps, batch, head, feature branch, eval cadence — is identical. D adds the score bias.

**LR note.** "Normal learning rate" is taken as the trainer's `--lr 1e-3` with
`--trunk-lr-scale 1.0`, i.e. the trunk trains at the head's LR. The brief's evidence is that
0.1× LR (= 1e-4) destroys the structure, so 1e-3 is the intended "normal" rate; `--grad-clip 1.0`
and the trace probe are the safety net, and A and C share it so the comparison is matched.
