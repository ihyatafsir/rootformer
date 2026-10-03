# The native root path in `IshtiqaqAttentionV12`: what it computes, why it is inert, and whether
# it is complementary to the residual root cross-attention

Artifacts, all under `/workspace/ishtiqaq_check/` (mirrored to
`rootformer/build/ishtiqaq_check/`):

| file | what it is |
|---|---|
| `probe_native_root_path.py` | the diagnostic: wiring, id spaces, decomposed force-enable, nats, exact reconstruction, weights |
| `probe_native.json` | its output |
| `ishtiqaq_root_bias.py` | the fix: a gated subclass + swap helper. The shipped module is NOT edited |
| `verify_ishtiqaq_fixed.py`, `verify_fixed.json` | the equivalence/liveness proof (ALL CHECKS PASS) |
| `make_ishtiqaq_trainer.py`, `nrmt_train.py`, `nrmt_train_ishtiqaq.diff` | the trainer patch (flag-guarded, asserted anchors) |
| `run_ishtiqaq_arm.sh`, `run_probe.sh`, `run_verify.sh`, `run_seven_suites.sh` | durable launchers |
| `suites_after_ishtiqaq.log` | the seven suites, re-run |
| `train_RCA_NATIVE_A2.log`, `results_RCA_NATIVE_A2.json`, `trace_RCA_NATIVE_A2.jsonl`, `vram_RCA_NATIVE_A2.txt` | the combined arm |

The shipped module was never modified: `models/ishtiqaq_attention_v12.py` md5 stays
`d192ac9f3514f2a7f831a5ff03fe93b8`. The fix is a subclass in a new file, so with the new flag off
the trainer executes the original code object.

---

## 1. What the native path computes

`models/ishtiqaq_attention_v12.py` (md5 `d192ac9f…`; the file the live trunk loads —
`/workspace/hf_v19_2_release/models/ishtiqaq_attention_v12.py`).

**Two streams are projected, but only one supplies values.**

```
l.126  q_s  = q_proj(hidden_states)          # surface query,  from h
l.127  k_s  = k_proj(hidden_states)          # surface key,    from h
l.128  v    = v_proj(hidden_states)          # SURFACE VALUE,  from h
l.160  score_surface = q_s @ k_s.T * scale
l.166  total_score   = stream_mix[0] * score_surface          # stream_mix[0] = 0.75
```

**The root stream contributes only an additive term on the score matrix.**

```
l.171  r_emb  = root_embed(safe_root_ids)                     # (9015, 64) table
l.172  q_r    = root_q_proj(r_emb)                            # 64 -> 896,  from ROOT EMBED
l.173  k_r    = root_k_proj(r_emb)                            # 64 -> 128,  from ROOT EMBED
l.177  score_root       = q_r @ k_r.T * scale                 # a function of (r_i, r_j) ONLY
l.179  has_root         = (safe_root_ids != 0)
l.180  ishtiqaq_cond    = has_root_i * has_root_j
l.184  identical_root_bonus = [r_i == r_j != 0] * ishtiqaq_gamma   # gamma = 0.25, a CONSTANT
l.186  total_score += stream_mix[1] * ishtiqaq_cond * score_root  # stream_mix[1] = 0.25
                    + identical_root_bonus
l.194  total_score += causal_mask                             # -inf, applied AFTER
l.262  attn_weights = softmax(total_score)
l.270  context = attn_weights @ v_rep                         # values are STILL surface
l.272  output  = o_proj(context)
```

**Where the root term enters:** the *attention score*, before the softmax — i.e. it reweights
which surface positions are attended. **What it is added to:** the score matrix
`total_score`, at line 186. **At what point in the layer:** inside self-attention, before
softmax; it never touches `v`, `context`, `o_proj` or the residual stream. There is no
`h ← h + …` anywhere in the file.

**What it is a function of.** `q_r` and `k_r` are built from `root_embed` only — there is no `h`
operand in the root sub-graph. So `score_root[i,j]` and `identical_root_bonus[i,j]` depend only
on `(r_i, r_j)`. Consequently the *entire* effect of the native root path on the layer output is

```
Δoutput = o_proj( (softmax(S + B) − softmax(S)) @ v_proj(h) )
```

Measured exactly (float32 rebuild of layer 0, same weights, real batch):
`max|Δ − o_proj((attn_on − attn_off) @ v)| / max|Δ| = 7.804e-06` — i.e. the whole delta is that
reweighting, to numerical precision. (`probe_native.json:C_reconstruction` reports the bf16
figure, 1.25e-01, which is the difference-of-two-bf16-tensors cancellation floor, not a
structural residual: the raw bf16 delta is 12.0 while the bf16 ulp near those magnitudes is
~3e-3 per element, and `max|o(roots) − o(rolled roots)| = 12.0` at the same scale.)

**Author's characterisation: ACCURATE.** The RCA docstring's three structural reasons
(`root_cross_attn.py:50-58`) are all correct as statements about this file:
1. Q comes from `root_embed`, not the token stream → root-to-root, not root-history-from-h. ✓
2. There, V is `v_proj(hidden_states)` — the value path is surface. ✓ (l.128, l.270)
3. The root term is a fused additive bias on the surface attention scores. ✓ (l.186)

---

## 2. Why it is inert — three independent causes, with numbers

### (a) It is not wired. `root_ids` is always `None`

`IshtiqaqAttentionV12.forward` takes `root_ids=None` and falls back to `self.active_root_ids`
(l.163-164), which is `None` at l.102 and assigned **nowhere in the file**. The only writers in
the repository are `models/unified_rootformer_v12.py:135-137` (`forward`) and `:218-219`
(`generate`). The live trainer never calls those: it drives the HF backbone directly at
`nrmt_train.py:101` (`extract_backbone`) and `:603` (`live_h_windows`).

Measured on the real checkpoint and batch, exactly as the trainer drives it:

```
attention calls = 24, carrying root_ids = 0, active_root_ids set = 0
```

The whole block l.168-186 and the whole Pillar-I/II block l.207-259 are skipped. This is also
why `stream_mix` and `ishtiqaq_gamma` are at their `__init__` values (§2c).

Two *secondary* wiring facts matter if someone tries to switch it on through the model instead:
* `extract_morphemic_ids` (`unified_rootformer_v12.py:118-123`) maps token id → root via
  `token_id − 349` clamped to `[1, 9014]`, so it fires only on literal `<root_…>` specials.
  Measured on the real token stream: **4 of 256 positions (1.5625 %)** are non-zero. Since
  `ishtiqaq_cond` requires *both* i and j non-zero, it is non-zero for ~0.02 % of causal pairs ⇒
  through the model's own wiring the bias is still a no-op. What *does* fire with an almost-all-
  zero root stream is Pillar II (governance damping on `is_generated`), which is why that route
  perturbs `h` by 5.63 % — entirely through the penalty, not the root bias
  (`B_decomposed.token_table_root_terms_only` = 0.31 %).
* `_set_flash(r, w)` (`nrmt_train.py:100`) sets `flash.current_root_ids`, which feeds the
  *engram* at layer 1 (`deepseek_v4_1_flash_model.py:203-208`) — a different pathway, not this
  module's `active_root_ids`.

### (b) The embedding table is in a stale id space and has never been trained

* Shape `(9015, 64)` against a **9490**-id root stream (`nrmp_vocab.num_roots = 9490`). l.169
  clamps: ids ≥ 9015 alias onto row 9014. Measured on a 256-position real batch: **32 positions
  (12.50 %)** and **21 of 138 distinct root ids (15.2 %)** collapse onto one row. **475 root ids
  (5.01 % of the inventory) have no row at all.** `extend_root_checkpoint.py:76-85` documents
  this table as a deliberately un-extended third root-id space.
* All 24 per-layer tables are distinct (`n_identical_to_layer0 = 1`, i.e. only itself) and sit
  at the `nn.Embedding` default N(0,1) init: std 0.9931, absmax 3.3281, row-norm mean 7.9149
  against the χ₆₄ expectation 7.9688 with sd 0.68898 (SE 0.00726) — 0.7 %, about **7 SE below**
  N(0,1). A 7-SE deficit with *bit-exactly zero* movement in the coefficients that multiply the
  branch (below) is the signature of **weight decay with an exactly zero loss gradient**, not of
  learning. `root_q_proj` / `root_k_proj` are at their kaiming inits (RMS 0.0723 / 0.0728).
* For contrast, the table the RCA reads, `morphemic_embed.root_embed`, is `(9490, 448)` — the
  released, synthesis-trained table, and the one in the correct id space.

### (c) It is scaled to 4.5 % of the surface score — and the learned part never got a gradient

Layer-0 magnitudes in nats, on the real batch (`B2_nats`), causal region only:

| quantity | value |
|---|---|
| surface score RMS | **1.7837** (absmax 5.976) |
| root raw bilinear score RMS | 0.3212 |
| root bias RMS (`stream_mix[1]·cond·score_root`) | **0.0803** (absmax 0.396) |
| **root bias / surface score** | **0.0450** |
| `identical_root_bonus` | 0.25 flat |
| max\|Δ attention weight\| | 0.09999 |
| RMS Δ attention weight | 0.002168 |
| mean max attention weight, off → on | 0.12438 → 0.12630 (**+1.55 %**) |

Whole-model effect, decomposed (`B_decomposed`, RMS(Δh)/RMS(h)):

| configuration | RMS(Δh)/RMS(h) | max\|Δh\| |
|---|---|---|
| everything ON (bias + bonus + Pillars) | **1.3763 %** | 0.25 |
| root score terms only, Pillars off | 1.3368 % | 0.25 |
| `identical_root_bonus` only | **1.2868 %** | 0.25 |
| root score bias only | 0.7123 % | 0.188 |
| Pillar I/II only | 0.5557 % | 0.125 |
| token-table root terms only (98.4 % zeros) | 0.3133 % | 0.125 |

The 1.3763 % reproduces the historical "force-enable Δh = 1.33 %" — the setup is the same one.
The decomposition is the new information: **the effect that exists is carried by
`identical_root_bonus`, a lookup-free constant +0.25 on the 5.31 % of causal pairs with
`r_i == r_j ≠ 0`. It uses no embedding at all.** The learned part — the bilinear form over
`root_embed` that the projections exist to compute — contributes 0.7123 %, and the whole thing
saturates at 1.37 % because the terms overlap.

The decisive gradient evidence for "the learned part never ran":

```
stream_mix      distinct across 24 layers = [(0.75, 0.25)]   __init__ = [0.75, 0.25]
ishtiqaq_gamma  distinct across 24 layers = [0.25]           __init__ = 0.25
coverage_weight distinct = [2.0]                             __init__ = 0.0
governance_strength distinct = [1.5]                         __init__ = 0.0
```

`stream_mix[1]` and `ishtiqaq_gamma` are the two multiplicative coefficients of the root score
terms. Bit-exact equality with `__init__` in all 24 layers is only possible if those terms never
produced a gradient. `coverage_weight`/`governance_strength` moved off 0.0, but their gradients
are *also* zero whenever `is_root ≡ 0` (Pillar I is masked by `is_root`; Pillar II's
`damping = (1−decay)·is_generated` and is 1 everywhere only when every root id is 0) — so they
were set externally, not learned. The shipped configuration is therefore
`0.75·surface + 0.25·(never-trained root score) + a fixed same-root bonus`, with Pillar weights
that were only ever switched on by hand.

### Summary of the mechanism

1. **First order — dead wiring:** 0 of 24 attention calls carry root ids, so nothing executes.
2. **Second order — even when forced on:** 4.5 %-of-surface-score bias + a fixed 0.25-nat
   bonus, over a table that is 5 % missing ids / 12.5 % aliased and never trained;
   total perturbation of `h` is 1.37 % RMS, and it is carried by the constant bonus, not by the
   learned projections.
3. **Third order — structurally:** the term reads no `h` and writes only the attention score, so
   it cannot introduce information; it can only reweight the surface value stream.

---

## 3. Complementary or superseded?

**Different mechanism, therefore in principle complementary — but information-poorer, so the
combination is a real hypothesis with an honest prior against it.**

*Insertion point.* Native: additive bias on the softmax logits of the *self-attention over
surface positions*. RCA: `h ← h + gate·o_proj(softmax(Q(h)K(E_r)ᵀ)V(E_r))` post-layer.
Reweighting versus injecting — different function classes, not two spellings of one thing.

*Information source.* This is the sharper distinction. The native path's values are
`v_proj(h)` (l.128, l.270), so everything it can deliver is a position-dependent convex
combination of surface value vectors — a subspace of what `h` already contains. The RCA's values
are `W_v·E_root(r_j)`, a task-specific subspace read straight from the released root table: that
is genuinely new information, not a reweighting of the old. So the native path is *strictly* the
weaker channel for the root-history lookup the RCA exists to approximate (order-4 ceiling
53.13 % vs single-`h` linear 6.48 %), while remaining a *different* operation.

*What the native path uniquely offers.* One thing the RCA does not obviously express: a **sharp
discrete same-root retrieval prior** — `identical_root_bonus` makes "the same root occurred
earlier" worth a flat +0.25 nats, and `ishtiqaq_gamma` is trainable. The RCA's 8-head softmax
over LayerNorm'd 448-dim root keys is a smooth average; a crisp indicator on root identity is a
different inductive bias, and it is the term that carries 94 % of the native path's measured
effect (1.2868 % of 1.3763 %).

Verdict on §3: **not redundant, but subsumed in usefulness unless the sharp same-root prior
earns its keep.** That is an empirical question, so it was tested.

---

## 4. The fix, and its ablation evidence

`ishtiqaq_root_bias.py` defines `IshtiqaqRootBiasAttention(IshtiqaqAttentionV12)` — the shipped
forward, with exactly two structural changes:

```
total_score = total_score
              + root_gate * (stream_mix[1] * ishtiqaq_cond * score_root)
              + root_gate * identical_root_bonus
...
total_score = total_score - pillar_gate * (coverage_weight * …)     # Pillar I
total_score = total_score - pillar_gate * (governance_strength * …) # Pillar II
```

* `root_gate`, `pillar_gate` are zero-initialised, so the whole native path is a no-op at step 0.
  They are not a dead saddle: `stream_mix[1]` and `ishtiqaq_gamma` are already 0.25, so the
  branch output is non-zero and `d(loss)/d(root_gate) ≠ 0` at step 0, exactly as
  `root_cross_attn.py:78-83` reasons for the RCA.
* The Pillars get their **own** gate because the shipped `coverage_weight = 2.0` and
  `governance_strength = 1.5` fire the instant root ids are supplied — the historical
  "force-enable" measurement conflated all three, and any clean attribution has to separate them.
* `--ishtiqaq-root-source shared` binds the released `morphemic_embed.root_embed` `(9490, 448)`
  and widens `root_q_proj`/`root_k_proj` to 448 inputs. This fixes the stale id space **and** the
  untrained table, and it makes the *only* remaining difference from the RCA the insertion point.

`verify_ishtiqaq_fixed.py` (ALL CHECKS PASS, `verify_fixed.json`):

| check | result |
|---|---|
| shipped `ishtiqaq_attention_v12.py` md5 unchanged | PASS `d192ac9f3514f2a7f831a5ff03fe93b8` |
| `gate 0, root_ids=None` == shipped `root_ids=None`, all 24 layers | PASS **bitwise**, worst\|d\| = 0.000e+00 |
| `gate 1, native source, root_ids=R` == shipped `root_ids=R`, all 24 layers | PASS **bitwise**, worst\|d\| = 0.000e+00 |
| whole model, gates 0 + no root ids == un-swapped trunk | PASS **bitwise**, max\|d\| = 0.000e+00 |
| *gate 1 is LIVE* | PASS RMS(Δh)/RMS(h) = **1.3542 %** |
| *gate 1 output depends on the root sequence* (125 rolled positions) | PASS max\|Δh\| = 0.25 |
| *gate 0 output is invariant to the root sequence* | PASS max\|Δh\| = **0.000e+00** |
| shared 448-dim source: gate 0 == trunk bitwise; gate 1 LIVE | PASS 0.000e+00 / LIVE |
| still a pure score bias: Δ == `o_proj((attn_on − attn_off) @ v)` | PASS float32 rel residual **7.804e-06** |

That is the same standard the RCA was held to: `gate = 0` a bitwise no-op, `gate = 1` live, a
rolled root sequence moves the output, and the mechanism is proved (not assumed) to be a score
bias. The fixed path is trainable: **1.835 M** new parameters for 4 layers at the shared source
(15.42 M if the shipped `native` 9015×64 source is used at all 24 layers).

Seven suites, re-run after the change (`suites_after_ishtiqaq.log`): `test_grammar_impl.py`
**20/20**, `verify_v2.py` pass, `andalusian_realizer.py` **67/67**, `test_sibawayh_governor.py`
**15/15**, `ibn_malik_automaton.py` **46/46**, `test_khalil_orbits.py` **30/30**,
`test_awzan_order.py` **24/24** — all exit 0.

### Wiring the arm

`nrmt_train_ishtiqaq.diff` (93 added lines, 4 replaced, every anchor asserted to occur exactly
once against the live `nrmt_train.py` md5 `56ef3f2f…`). With `--ishtiqaq-root-bias none` (the
default) every inserted block is either not entered or a no-op: `ishtiqaq_layers = []` ⇒
`live` unchanged, `trunk_params` unchanged (empty exclusion set), the optimizer groups and the
LR schedule unchanged, and the saved `.trunk.pt` payload unchanged (the four metadata keys are
added only when `ishtiqaq_layers` is non-empty). The single runtime hook is
`live_h_windows`:

```python
for _i in ishtiqaq_layers:
    model.backbone.layers[_i].self_attn.active_root_ids = Tr_w   # <- the only wiring ever needed
```

**Arm `RCA_NATIVE_A2` = `RCA_UNFREEZE_A2` + the fixed native score bias, one variable:**

```
--root-cross-attn top4 --unfreeze-last 4 --trunk-lr-scale 0.01 --rca-lr-scale 0.3
--rca-out-norm --rca-dropout 0.1 --rca-ablate-eval --h-drift-probe 500
--feat-gate --margin-ramp 10 --grad-warmup 10 --logit-scale ln --head-init remap
--steps 20000 --batch-size 32 --eval-every 1000
+ --ishtiqaq-root-bias top4 --ishtiqaq-root-source shared --ishtiqaq-gate-init 0.0
+ --ishtiqaq-pillar-gate-init 0.0 --ishtiqaq-lr-scale 0.3 --ishtiqaq-ablate-eval
```

Layers 20–23 for both mechanisms (so the native set equals the unfrozen set), `root_gate`
zero-init and trained, Pillars held off by their own gate. The in-loop
`ALL_val_ISHTIQAQ_OFF` / `NOVEL_only_ISHTIQAQ_OFF` evaluations are the causal ablation of the
native path on the same trunk weights at every 1000 steps.


## 4b. The native path alone, measured in held-out acc@1 (CPU, converged model, frozen trunk)

`repro_native_eval.py` (`native_eval.json`) is a second implementation of the trainer's
evaluation — the trainer's val starts, novelty mask, Sibawayh ʿāmil stream and specials mask —
run over the **exact control head for A2**, `/workspace/head_fix/head_ALIGNED_FIX.pt`, with the
FIX/ALIGNED_FIX construction flags (`feat_gate=True, hist=3, dropout=0.1, use_features=True`).
Val positions: **18869 ALL / 15046 NOVEL**, i.e. exactly the published denominators.

Harness validation: the native path OFF reproduces the published ALIGNED_FIX number to within
0.016 pp (mine **6.603 %**, published **6.619 %**). `model._set_flash(r, w)` is called, because
that is what the frozen-cache arms did; this is the *same* condition under which 6.619 % was
produced, so the only thing that varies below is the native path.

| variant | ALL@1 | Δ vs off | NOVEL@1 | Δ | CE_z (ALL) |
|---|---|---|---|---|---|
| `off` — frozen baseline | **6.603 %** | — | 6.427 % | — | 8.0995 |
| `root_terms_shared` — the FIXED path, gate = 1, 448-dim released table, Pillars off | **6.598 %** | **−0.005 pp** | 6.427 % | **0.000 pp** | 8.0995 |
| `bonus_only_shared` — only `identical_root_bonus` | 6.619 % | +0.016 pp | 6.454 % | +0.027 pp | 8.0995 |
| `shipped_force_enable` — the SHIPPED module, `active_root_ids` set (bias + bonus + Pillar I 2.0 + Pillar II 1.5) | 6.619 % | +0.016 pp | 6.454 % | +0.027 pp | 8.0995 |

Read this against the run-to-run noise the ladder itself reports: within-run eval sd over the
last 10 evals is **0.026–0.113 pp** (0.093 pp for A2). Every delta above is **smaller than one
sd**, and 0.016 pp is **3 positions out of 18 869**. `CE_z` is 8.0995 in all four variants, to
four decimals.

Three things this settles, that Δh alone could not:

1. **The fixed path is numerically live and behaviourally inert.** It perturbs the trunk by
   1.35 % RMS (§4) and moves held-out acc@1 by −0.005 pp with *identical* CE_z. So the historical
   "−0.016 pp" was not an artifact of the broken table, the stale id space or the aliasing: with
   a correct 448-dim table in the correct id space, gated and ablated, the score bias is still
   metric-inert.
2. **The whole effect is the lookup-free constant prior.** `bonus_only_shared` (+0.016/+0.027 pp)
   reproduces `shipped_force_enable` **exactly**, while the learned bilinear bias *subtracts*
   0.021 pp (`root_terms_shared` −0.005 vs bonus-only +0.016). The bilinear form over the root
   table is the part the projections exist to compute, and it is the part that does nothing.
3. **The sharp same-root prior is worth 3 positions.** The best available reading of the native
   path is "attend to positions with the same root", and on a converged model that is worth
   3/18 869 — no better than noise.

Caveat kept explicit: these variants run the native path with *untrained* projections, so the
bilinear term is meaningless at init. The trained question is the GPU arm below.


---

## 5. Verdict


**What it computes.** A fused additive bias on the *surface self-attention scores*, applied
before the softmax (l.186); a function of `(r_i, r_j)` alone; values stay `v_proj(h)`
(l.128, l.270). The author's characterisation — "a fused score bias, not a residual stream" — is
**accurate**, and so are all three of the structural reasons in `root_cross_attn.py:50-58`.

**Why it is inert.** Three causes, in order of force:
1. **Not wired.** 0 of 24 attention calls carry root ids; `active_root_ids` is written only where
   the live trainer never goes.
2. **When forced on, ~nothing.** A bias measuring **4.50 %** of the surface score RMS, computed
   over a table with **475 unreachable** root ids and **12.50 %** of positions aliased onto one
   row, at N(0,1) default init — while `stream_mix[1]` and `ishtiqaq_gamma` sit bit-exactly at
   their `__init__` values in all 24 layers, which is only possible if the learned terms never
   received a gradient. Total: **1.3763 %** RMS Δh, reproducing the historical 1.33 %.
3. **Structurally information-limited.** No `h` operand anywhere in the root sub-graph, so the
   entire effect is `o_proj((softmax(S+B) − softmax(S)) @ v_proj(h))` — a reweighting of surface
   values, carrying nothing new. Proved to a relative residual of **7.804e-06** in float32.

**Complementary or superseded?** **Not redundant — but superseded in usefulness.** The two are
different operations on different information: the native term reweights the *surface* stream
from a pairwise root prior; the RCA injects a residual built from the *trained 448-dim root
table*. That is genuinely complementary in form. But the native channel can only reweight what
`h` already contains, and what this model lacks is not the current root (92.34 % linearly
decodable from a single `h_t`) but a *history-dependent* readout (single-`h` ceiling 6.48 % vs
order-4 lookup 53.13 %). Only the residual channel addresses that. Measured, with the
implementation corrected on every axis the RCA author objected to:

* **−0.005 pp** ALL / **0.000 pp** NOVEL acc@1 for the fixed score-bias path, with **identical
  CE_z** (8.0995) — even though it is provably live at 1.35 % RMS Δh;
* the only sub-term with any magnitude, the constant +0.25 same-root bonus, is **+0.016 pp /
  +0.027 pp** = **3 positions out of 18 869**, against a within-run sd of 0.026–0.113 pp;
* the learned bilinear form — the part the projections exist to compute — *subtracts* 0.021 pp.

**Verdict: this is a correct implementation of a different, weaker idea — not a defective
implementation of the RCA.** The score bias is a real mechanism, it is now provably live and
bit-exactly ablatable, and it still buys nothing here. Abandon the score-bias *form* as
superseded by the residual RCA; keep the fix only because it is what makes that statement
provable rather than assumed.


---

## 6. What I would change


Ordered by value. Note that **"abandon it" requires no behavioural code change at all** — with
`--ishtiqaq-root-bias none` (the default) the shipped state is bit-identical to today's, and the
module is already inert. The changes below are about not paying for it.

1. **Stop shipping the dead parameters.** Measured from the shipped checkpoint: **15.62 M**
   parameters in this module can never affect the forward pass while `root_ids is None` —
   24 × `root_embed` (9015, 64) = 13.85 M, 24 × `wazn_embed` (128, 64) = 0.20 M, and
   24 × `root_q_proj`/`root_k_proj` = 1.57 M. Worse, `wazn_embed` is not merely unwired: it is
   declared at l.74, has row 0 re-initialised at l.78, accepts `wazn_ids` at l.119 — and is
   **never read anywhere in the file**, while `active_wazn_ids` (l.103) is written by
   `unified_rootformer_v12.py:137` and never read either. The "dual stream" has a wazn half with
   no forward path at all. Delete the module's root/wazn machinery, or keep it explicitly gated
   and excluded from the checkpoint.
2. **If anything is kept, fix the id space.** The 24 tables live in a third, older 9015-id space
   (documented at `extend_root_checkpoint.py:76-85`). Bind the released, synthesis-trained
   `morphemic_embed.root_embed` `(9490, 448)` instead — the fix does this, and it removes the 475
   unreachable ids and the 12.50 % aliasing at once, and replaces an N(0,1) table with a trained
   one.
3. **Give the two mechanisms separate switches.** The Pillar-I/II penalties are welded to
   `if root_ids is not None` (l.207), so they cannot be enabled independently of the root bias,
   and they are the part that fires destructively when root ids come from the model's own
   token→root table (98.4 % zeros ⇒ `is_generated` is 1 almost everywhere ⇒ governance damping
   over the whole sequence; that is where the historical 5.63 % Δh figure comes from, not from the
   root bias). `pillar_gate` in the fix separates them; a real fix would make them a separate
   opt-in module.
4. **Gate anything that can be attached.** l.186 has no gate, so "attached but untrained" is not
   a no-op. `root_gate` makes gate 0 bitwise-identical to the baseline (proved), which is what
   makes the ablation above a causal measurement rather than a comparison of two different
   models.
5. **Adopt the fix's ablation contract as the module's interface**, whatever is decided about the
   mechanism: `set_root_ids()` / `active_root_ids` supplied by the caller, `gate = 0` a bitwise
   no-op, `gate = 1` live, and the root-sequence sensitivity test. Without it the module cannot
   be measured at all — which is the actual reason it went unnoticed for so long.

**Separate finding, unrelated to this module but relevant to the comparator ladder.**
`nrmt_train.py:100` calls `model._set_flash(r, w)` inside `extract_backbone`, but the live
replacement `live_h_windows` (`nrmt_train.py:593-604`) does **not**. So every frozen-cache arm
(FIX 6.837 %, ALIGNED_FIX 6.619 %, RCA_FROZEN_CTL …) ran the layer-1 and layer-14 engrams with
the root ids as hash keys, while every live arm (A2 19.53 %, the depth/width arms, and mine) runs
them with the `arange` fallback. `_set_flash` appears exactly once in the trainer. This does not
affect a single-variable comparison between two live arms — `RCA_NATIVE_A2` vs `RCA_UNFREEZE_A2`
is unaffected — but it does mean "A2 vs FIX" changes the engram input as well as the RCA, and any
ladder reading that crosses that boundary should be treated as confounded until it is fixed.

