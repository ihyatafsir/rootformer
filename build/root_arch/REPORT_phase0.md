# Phase 0 — engineering report (`/workspace/root_arch/`, pod `g6exduq0bd17z8`)

Mirror: `rootformer/build/root_arch/`. All artefacts are additive; no shipped file was modified.

---

## 0a. Eager attention → `F.scaled_dot_product_attention`

### What was wrong

`IshtiqaqAttentionV12.forward` built the score matrix by hand and every intermediate is a full
`[B, 14, T, T]`:

```
score_surface = matmul(q_s, k_s_rep.transpose(-2,-1)) * self.scale   # [B,14,T,T]
total_score  += causal_mask (-inf, float32)                          # -> FP32 [B,14,T,T]
attn_weights  = softmax(total_score, dim=-1)                         # FP32 [B,14,T,T]
attn_probs    = dropout(attn_weights).to(bf16)                       # bf16 [B,14,T,T]
context       = matmul(attn_probs, v_rep)
```

Softmax's input **and** output are both retained for backward, so ~0.85 GB of activation per layer
at B=32/T=487, plus ~1 GB transient. That — not the weights — is what pinned a 0.5 B model at
~11 GB.

### The replacement

`sdpa_attention.py` defines `SdpaIshtiqaqAttention` / `SdpaIshtiqaqRootBiasAttention` (subclasses of
the shipped classes; the shipped module is untouched) and `install()`, which swaps them into the
import namespaces **before** the backbone is constructed.

Four things the naive rewrite gets wrong, each found by measurement:

1. **`is_causal` is IGNORED whenever `attn_mask` is supplied.** Verified on this pod: passing both
   is bit-identical to passing the mask alone (`max|d| = 0.0`). So the causal `-inf` band is folded
   into the additive mask explicitly.
2. **`stream_mix[0]` is a *trained* parameter and SDPA's `scale` is a Python float.** Passing
   `scale * stream_mix[0]` would cut the gradient. The scalar is folded into `q` instead
   (`(c·q)·kᵀ == c·(q·kᵀ)`), which is gradient-preserving.
3. **`enable_gqa=True` cannot carry an additive mask.** Measured, per named backend (`prove_sdpa_
   equivalence.py` PART 6):

   | call | FLASH | EFFICIENT | MATH |
   |---|---|---|---|
   | expanded k/v, `is_causal` (arm A) | **yes** | yes | yes |
   | expanded k/v, additive mask (arm C) | no (flash rejects masks) | **yes** | yes |
   | `enable_gqa=True`, `is_causal` | yes | no | yes |
   | `enable_gqa=True`, additive mask (what arm C needs) | no | **no** | yes only |

   So with `enable_gqa=True` arm C's attention is accepted **only by `MATH`**, which materialises
   `[B,14,T,T]` — the exact tensor the rewrite exists to remove. `verify_dispatch` confirms the
   consequence and the fix (48 MB arm A / 58 MB arm C, against the 224 MB matrix). **GQA is now
   expanded by hand with `repeat_interleave`**, bit-matching the shipped expansion
   (heads 0..6 → kv 0, 7..13 → kv 1) at a cost of 7.3 MB per layer.
4. **The shipped forward is MIXED dtype.** `q_proj` returns bf16 but
   `Qwen2RMSNorm.__init__` builds `nn.Parameter(torch.ones(dim))` **with no dtype argument and
   nothing ever casts it**, so `q_norm.weight`/`k_norm.weight` are **float32**; after the norms
   `q`,`k` are fp32 and `v` is bf16. `F.scaled_dot_product_attention` requires ONE dtype, so the
   swap *forces* a policy choice the eager code never had to make. Two are implemented and measured
   (`ROOT_ARCH_SDPA_DTYPE`): `bf16` (flash; the arms use this) and `fp32`.

### Equivalence — `prove_sdpa_equivalence.py`, `proof_sdpa_equivalence.json`

Same weights, same inputs, two forwards. **Tolerance stated up front: top-1 ≥ 99 % and
`max|Δlogits| ≤ 1.5 × the eager path's own fp64 error`.**

| measurement | result |
|---|---|
| fp64 truth — **eager (shipped fp32 QKᵀ)** | max\|Δ\| **2.2699e-01** |
| fp64 truth — **sdpa dtype=fp32** | max\|Δ\| **1.9073e-05** |
| fp64 truth — **sdpa dtype=bf16** | max\|Δ\| **2.0493e-01** |
| eager vs sdpa (bf16), full trunk | max\|d\| 2.0075e-01, top-1 **99.316 %** |
| eager vs sdpa (fp32), full trunk | max\|d\| 2.1313e-01, top-1 99.316 % |
| eager vs sdpa + root bias, all 24 layers | max\|d\| 1.8603e-01, top-1 **99.316 %** |
| gradients, 166 significant params | min cosine **0.999992** |
| **noise floor** | the deviation is **1.22×** a ONE-ULP change in `layers.12.o_proj.weight` |

**The decisive line: the eager code SDPA replaces is the *less* accurate of the two.** Its
QKᵀ is silently downgraded to bf16 by autocast, its softmax runs in fp32, and its probabilities are
cast back to bf16 for `@v`. `sdpa dtype=bf16` is closer to the fp64 truth than eager is
(2.05e-01 vs 2.27e-01), and `sdpa dtype=fp32` is exact to 1.9e-05. The 0.20 logit difference is
therefore **eager's own error**, bounded by the model's representation noise (1.22 × 1 ULP).

0d flash hooks, perturbation-proved (the gap a prior agent flagged):

```
(a) hooks ON vs OFF, same weights          torch.equal = True   max|d| = 0.0
(b) engram_layer1.mem_proj + orbit.out_proj un-zeroed, hooks ON   output moves 5.07   <- hooks LIVE
(c) same weights, hooks DETACHED                                 output moves 0.0    <- true removal
```

### Measured gain — `bench_sdpa.py`, `bench_sdpa.json`

Same v13 trunk, same weights, same loss; the only variable is the attention implementation.
`max_memory_allocated` over the whole fwd+bwd step, per process.

**T = 128 (the trainer's window), gradient checkpointing OFF**

| batch | eager | sdpa | memory saved |
|---|---|---|---|
| 8 | 8.11 it/s @ 3.06 GB | **9.57 it/s @ 2.81 GB** | 0.25 GB |
| 16 | 7.05 it/s @ 5.06 GB | **8.93 it/s @ 4.58 GB** | 0.48 GB |
| 32 | 4.38 it/s @ 9.08 GB | **5.34 it/s @ 8.10 GB** | **0.98 GB (−11 %), +22 % it/s** |

**T = 512 (the shape that OOM'd), gradient checkpointing OFF**

| batch | eager | sdpa |
|---|---|---|
| 2 | 7.67 it/s @ 4.10 GB | 6.16 it/s @ 2.80 GB |
| 4 | 5.58 it/s @ 7.04 GB | **6.89 it/s @ 4.61 GB (−35 %)** |
| 8 | guard-skipped (>15.5 GB needed) | **4.85 it/s @ 8.12 GB** |
| **max batch** | **4** | **8 — 2× the batch in 15 % more memory** |

**Gradient checkpointing ON** — where the eager path collapses, because it *recomputes* the
`[B,14,T,T]` softmax:

| T | batch | eager | sdpa |
|---|---|---|---|
| 128 | 8 | 1.32 it/s | **5.11 it/s (3.9×)** |
| 128 | 16 | 1.11 it/s | **4.70 it/s (4.2×)** |
| 512 | 16 | 1.30 it/s @ 2.91 GB | **1.92 it/s @ 2.45 GB (1.5×)** |

A dispatch verifier (`verify_dispatch`) runs at the start of **every** arm and logs the fact:
at B=8/T=1024 the real `_sdpa` peaks at **48 MB (arm A) / 58 MB (arm C)** against the **224 MB**
`[B,14,T,T]` matrix a materialising kernel would need — ratio 0.21 / 0.26.

Caveat stated: the bench uses 3 iterations and the first configuration of each `(mode,T)` includes
kernel warm-up, so the small-batch rows are noisy; the B=32/T=128 row is the one that matches the
live arm (bench 5.34 it/s; FLOOR_A logs ~4.6 it/s including the NRMT head and the eval passes).

---

## 0b. The v13 fixes — deployed additively

`unified_rootformer_v13.py` and `root_space.py` were copied **as new files** into
`/workspace/hf_v19_2_release/models/` (they were never deployed; the trainer's
`from models.unified_rootformer_v13 import ...` could not resolve). Shipped files keep their md5:

```
ishtiqaq_attention_v12.py   d192ac9f3514f2a7f831a5ff03fe93b8   (unchanged)
unified_rootformer_v12.py   68646a897e7037e5219dfd0463fdcc3d   (unchanged)
root_cross_attn.py          b53ae702bde947ac66f933191f904c36   (unchanged; see below)
```

Every script asserts the **root count is 9490 / 142** before doing anything. A missing release
module (`validated_segmentation.py`) makes `nrmp_vocab` report **9445** and nothing downstream
raises; the pod is correct today and the runner refuses to launch otherwise.

**The checkpoint trap is closed and asserted.** The released checkpoint is a `RootformerNRMT` state
dict (`backbone.layers.*`) while v13 holds the CausalLM (`backbone.model.*`). `model_build.build()`
raises on any non-head, non-deliberately-dropped missing tensor, so the "555 missing / 520
unexpected, silently random model" failure cannot recur:

```
checkpoint loaded: dropped 48 stale-SPACE self_attn tables; missing=67 (0 unexpected-missing) unexpected=8
```

---

## 0c. The native `IshtiqaqAttentionV12` root branch — confirmed, not re-enabled

Read from the checkpoint directly, all 24 layers identical:

```
stream_mix            [0.75, 0.25]        <- __init__ values, never moved
ishtiqaq_gamma        0.25                <- __init__ value
coverage_weight       2.0                 <- LIVE (the brief's correction is right)
governance_strength   1.5                 <- LIVE
self_attn.root_embed  (9015, 64)          <- the stale space; 475 ids have no row
morphemic_embed.root_embed (9490, 448)    <- the LIVE source, used exclusively
```

The branch is never re-enabled: `_root_table()` returns the released 448-dim table for every
attached layer (`root_source='shared'`), and `early_root_path._guard_root_source()` refuses to
attach a pathway whose source is not 9490 rows.

**Pillar-I/II is held at exactly 0.** The `coverage_weight`/`governance_strength` penalties read the
*post-softmax attention weights*, which a fused kernel does not return, and they are not one of the
two declared mechanisms — leaving them live would confound A-vs-C with an undeclared third term.
`--pillar-freeze` zeroes `pillar_gate`, sets `requires_grad_(False)` and removes it from every
optimizer group, so the skip is **structural** rather than a `0.0 × finite` accident that could
drift. This does not regress the proven init-equivalence: it makes gate-0 == no-pathway hold for
every step instead of only at step 0.

---

## 0d. The 151.95 M orphan parameters — DECISION: disable them

`deepseek_v4_1_flash_model.py` registers live forward hooks on trunk layers 1/11/14 whose
`engram_layer1`, `engram_layer14` and `farahidi_orbit` parameters are in **no checkpoint and no
`model.parameters()`** — `RootformerNRMT.backbone` is `...backbone.model`, so the flash wrapper's own
modules are outside the optimizer while still firing on every forward.

**They are disabled (`--flash-hooks off`) for every arm.** Reasons, in order:

1. **Provably inert today and provably live if touched.** `mem_proj`/`out_proj` are exactly zero, so
   the hooks are an exact identity — the perturbation proof above shows bit-equality at load
   (`torch.equal`, max|d| = 0.0) *and* shows the hooks inject 5.07 the moment the output projection
   is un-zeroed. They are a loaded gun, not dead code.
2. **They cannot ever leave zero** (not in `model.parameters()`, so never optimised) yet their
   gradients accumulate forever: ~152 M parameters of weights plus gradients and autograd work per
   step.
3. **They cost wall-clock on every step** for an exactly-zero output.
4. They are not part of the A-vs-C question.

`--flash-hooks off` also **detaches the `_layer11_bottleneck_hook`**, so `cached_h12` is no longer
populated. That is safe: nothing reads `cached_h12` anywhere in the release, and `ced_bridge` (the
CED cross-layer KV bridge the docstring advertises) is **never called from any forward path** — it
is an unused module, not a disabled feature.

---

## The queue — `runner.sh`

Runs **on the pod** under `setsid`, so an SSH drop cannot kill it and no human decision gates GPU
use. Requirements met:

* `setsid` + `nohup` + all three std streams redirected; the arm survives the runner exiting.
* **checkpoint every 1 000 steps** — `--eval-every 1000` writes `head_<TAG>.pt` and
  `head_<TAG>.pt.trunk.pt` at each eval, **overwriting in place** so disk stays bounded.
* **append-only logs** (`>>`), a runner PID file and a per-arm PID file.
* **`ps`-based liveness** on the unique `--tag` (no PID-reuse hazard), plus per-arm
  `ps` liveness before any stage is retired.
* **broadened `ps` counter AND an `nvidia-smi` VRAM check.** The counter anchors on the interpreter
  path at the *start* of `argv`, so the pattern text inside the script can never match itself — and
  `grep -c` is never followed by `|| echo 0` (that yields `"0\n0"` and breaks every comparison).
* **Never a third occupant; never kills anything.** `EWC_Q20K_LHEAD` and `RCA_NATIVE_A2` were other
  agents' work and were left alone.
* Per-poll `nvidia-smi --query-compute-apps` snapshot to `queue/vram.csv`, which is how
  **peak VRAM per arm** is obtained without instrumenting the trainer.

### The measured occupancy ceiling changed for these arms

The brief's "two 32-batch arms ≈ 21 GB" was measured on `--unfreeze-last 4` arms (10.6 GB each).
With `--unfreeze-trunk-all` **all 24 layers retain activations**, and the CUDA OOM message from the
concurrent attempt states the real number:

```
GPU 0 has a total capacity of 31.37 GiB of which 14.25 MiB is free.
Process 405014 has 13.65 GiB in use. Including non-PyTorch memory, this process
has 17.69 GiB memory in use.
```

**17.69 GiB per arm — two cannot coexist on a 31.37 GiB card at batch 32.** The gate is therefore
`free ≥ 20000 MiB`, which makes the queue **sequential**. `FLOOR_A` was not harmed (CUDA OOM is
process-local). This is the outcome the brief calls for over an OOM.

---

## Operational findings that cost runs

**1. `/workspace` has a per-user disk quota.** `df` reports 466 T free (it is a MooseFS cluster
volume) while `dd` of 700 MB fails with **"Disk quota exceeded"**. A 640 MiB trunk checkpoint killed
a smoke arm mid-run **with no traceback** — the log simply stops after the eval line and the trace
jsonl stays 0 bytes. **All large checkpoints now go to `/tmp/root_arch_arms`** (a 50 GB overlay with
no quota; a 700 MiB write takes 0.08 s). The runner's preflight `dd`s 750 MiB into the real
checkpoint directory rather than trusting `df`.

**2. `root_cross_attn.py:136` does `safe = root_ids.clamp_(...)` IN PLACE on the caller's tensor.**
Every attached layer then saves a VIEW of that tensor in `IndexSelectBackward0`, and the next
layer's `clamp_` bumps its version, so the first backward dies:

```
RuntimeError: one of the variables needed for gradient computation has been modified by an
inplace operation: [torch.cuda.LongTensor [512]] is at version 24; expected version 23
```

Found with `torch.autograd.set_detect_anomaly(True)`, which names `IndexSelectBackward0` and
`root_cross_attn.py line 137`. It has never bitten before because `index_select` only builds a grad
node when `root_weight.requires_grad` — every earlier arm froze `morphemic_embed`, so the index was
never saved. It fires **the moment the input stage is trained**, i.e. only for this experiment.

**The verified module was left BYTE-IDENTICAL** (`b53ae702…`); an earlier edit was reverted.
`rca_compat.install()` wraps `RootHistoryCrossAttention.forward` so each layer clamps its own clone
(numerically identical, 24 int64 clones per step). Verified with anomaly detection on: 3/3 steps,
zero errors. The runner preflight installs-and-verifies it, so a missing workaround blocks a launch
instead of burning 70 minutes.

**3. The seven suites after the module edit** (run from `/workspace/taxpin`):

```
test_grammar_impl.py        20/20 checks pass
verify_v2.py                ALL v2 CHECKS PASS
andalusian_realizer.py      67/67 checks pass
test_sibawayh_governor.py   15/15 checks pass
ibn_malik_automaton.py      46/46 checks pass
test_khalil_orbits.py       30/30 checks pass
test_awzan_order.py         20/22 — 2 FAIL, PRE-EXISTING and unrelated
```

`test_awzan_order.py` fails on `missing /workspace/rootformer_v12/v18_next_root_morph/data/
nrmp_vocab.json`. That path **does not exist anywhere on the pod** (`find /` returns nothing), the
two failing checks both require it, and the suite **does not import `root_cross_attn`** (grep count
0). It is an environment gap, not a consequence of the `clamp_` workaround — reported plainly rather
than papered over.

---

## The length-1 ridge probe, re-implemented and validated

The brief's 29.30 % / 92.34 % comparators come from a *different* question than `ALL_val.acc@1`:
a free closed-form **ridge readout** fitted on the length-1 hidden state and asked for the word's own
root.  `probe_ridge_arm.py` re-implements it against the v13 trunk and, crucially, **validates itself
against the published numbers first**:

```
probe_ridge_arm.py --tag RELEASED_BASELINE --released --pathway none
   val_all            acc@1  92.34%  (n=73502)      published 92.34%
   val_unseen_words   acc@1  83.92%  (n=14390)      published 83.92%
   val_seen_words     acc@1  94.39%
   train_all          acc@1  83.83%
   marginal (majority root) val_all 0.43%
```

Every count and every accuracy matches the recorded comparator **exactly**, so a difference measured
on a trained arm is real and not a protocol artefact.

Three things make the arm probe honest, and each is a way the measurement could lie:

1. **The pathway must be ACTIVE for arm C.** Its trunk was *trained* with the root pathway in the
   forward pass; probing it with the pathway off measures a different network. `--pathway` re-creates
   the same mechanism and supplies `active_root_ids` — exactly the wiring the trainer's
   `live_h_windows` uses.
2. **The weights must come from the arm**, from
   `/tmp/root_arch_arms/head_<TAG>.pt.trunk.pt`, which holds all 577 trained tensors (24 layers ×
   `backbone.layers.*`, `morphemic_embed.*`, `final_norm.*`, `root_cross.*`, and the score-bias
   `root_gate`/`root_q_proj`/`root_k_proj` that live under `backbone.layers.*.self_attn.`). The
   loader reports `applied=577 skipped=0` and refuses to probe if fewer than 100 tensors apply.
3. **The word lists must match the reference** (`--cache /workspace/nrmp_cache_9490`, the cache the
   published numbers used). The arms were *trained* on `nrmp_cache_9490_aligned`, which changes their
   weights but must not change the split they are judged on.

The probe needs ~3 GB and 80 s, so it runs **alongside** a training arm without ever idling the card.
