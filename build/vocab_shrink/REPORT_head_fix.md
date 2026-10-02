# Broken prediction head — diagnosis, fix, and what the ceiling actually is

Date: 2026-10-02 · Pod `6rnaag3vvrfpr` (213.173.108.47:22515) · all work under `/workspace/head_fix/`
mirrored to `rootformer/build/head_fix/`.

## 0. Verdict in one paragraph

The "broken head" premise is **refuted**, and the real defect is elsewhere. The head is not
broken: it is one linear readout aimed at `t+1`, and it sits at **88 %** of what a free closed-form
readout of the same frozen hidden state achieves on that task. The genuine, verified, high-value
defect is a **data-construction bug**: the cache's prefix stream was index-misaligned with
root/wazn/suffix, so the model was fed *another word's prefix* — measured to carry **zero
information** about the current word's prefix. That is fixed, the cache is rebuilt, and the fix is
worth **+0.43 pp acc@1** (5.72 → 6.15) on the CONTROL configuration. But the dominant finding is
item 2: **a plain order-4 n-gram over roots scores 53.13 % acc@1 on the identical positions where
the head scores 5.72 %** — and a closed-form linear readout of *exactly the features the head's
conditioning branch receives* scores **1.7 %**. The 12.8 M-parameter conditioning branch is
measurably inert, so the head is capped by the *form* of its input pathway, not its size.

## 1. The prefix stream was index-misaligned — verified, fixed, retrained

### 1.1 Root cause

`/workspace/hf_v19_2_release/nrmp_train.py` `cmd_prepare`, lines 124 and 133:

```python
st[0].append(vocab.BOS_ROOT)          # stream 0 ONLY, once per sentence
for (p, r, w, sf) in enc:
    st[0].append(p)                   # prefix
    st[1].append(r); st[2].append(w); st[3].append(sf)
st[0].append(vocab.EOS_ROOT)          # stream 0 ONLY, once per sentence
```

Stream 0 receives `n_k + 2` entries for an `n_k`-word sentence; streams 1–3 receive `n_k`. Both
consumers pair the streams **by index** — `nrmt_train.py:375`
(`Tr, Wr, Pr, Sr = (win_tensor(tr4[i], tr_st) for i in (1, 2, 0, 3))`) and `nrmp_train.py:262`
`cmd_train` (`chunk(load_stream('train'), ...)` slices all four). So after the first sentence,
stream 0 runs `2 × sentence_index` positions ahead: the model is fed
`(prefix of a different word, root_t, wazn_t, suffix_t)`.
`/workspace/nrmp_rebuild_cache.py` (lines 181/190) reproduced the same bug verbatim, which is why
the shipped `/workspace/nrmp_cache_9490` carries it.

The writer's own manifest already contains the proof:
`val: 972383 = 837575 + 2×67404`, `train: 8311213 = 7063767 + 2×623723`.

### 1.2 Independent verification

A fresh process re-derived the split and the streams from the corpus using the original writer's
own `sentences_from` / `encode_sentence`:

| check | train | val |
|---|---|---|
| reproduced **legacy** cache md5 vs shipped | `214959e5…` **identical** | `2f171677…` **identical** |
| first 20 000 word events, stream 0 equal to reproduction | **100.0 %** | **100.0 %** |
| root / wazn / suffix equal to reproduction | **100.0 %** | **100.0 %** |
| `consumed_prefix == true_prefix` | **58.775 %** | **59.405 %** |
| **random shuffle** of the same stream agrees | **58.84 %** | **60.03 %** |
| majority-prefix baseline | 79.44 % | 78.72 % |
| match on positions whose prefix is informative | **4.47 %** | **5.47 %** |
| positions reading an injected boundary marker | 20.06 % | 14.17 % |
| mean source-word displacement | 1681 words | 1437 words |
| `decode_word(prefix,r,w,s) == surface word` — correct prefix | **70.1 %** | **64.9 %** |
| same, as actually fed | **44.8 %** | **40.2 %** |

The consumed prefix agrees with the truth at exactly the rate of a **random permutation**, i.e. it
carries **zero mutual information**, and it is *below* a constant predictor. The semantic decode
confirms the direction of the fix: feeding the correct prefix raises exact surface-word
reconstruction from 44.8 % to 70.1 % (the residual is encode→decode non-invertibility).

### 1.3 Fix

`--prefix-layout {aligned,legacy,all}`, **default `aligned`**: exactly one prefix id per word
event, no stream-0 markers. `legacy` reproduces the historical bytes exactly; `all` puts BOS/EOS
into all four streams (PAD in wazn/suffix). Both writers were patched:

* `/workspace/hf_v19_2_release/nrmp_train.py` (md5 `c24675c5…` → `cdb50c48…`), diff in
  `artifacts/nrmp_train_prefix_align.diff`
* `/workspace/nrmp_rebuild_cache.py` (md5 `904b8fee…`), diff in
  `artifacts/nrmp_rebuild_cache_prefix_align.diff`

`.bak`s kept on the pod (`*.bak.prefix-align-<UTC>`) and mirrored
(`artifacts/nrmp_train.py.ORIGINAL.bak`). Patch validated by a 4-separate-process test: patched
`legacy` is **tensor-identical to the original on all four streams**; `aligned`/`all` have four
equal-length streams. (Note: `nrmp_train.py` derives `BLUEPRINT` from `__file__`, so an
out-of-tree copy silently falls back to a blueprint-less vocab and shifts 73 wazn ids — the test
pins the globals.)

Rebuilt cache: `/workspace/head_fix/nrmp_cache_9490_aligned`
(`train.pt` 7063767 × 4, `val.pt` 837575 × 4). Fresh-process verification of the **written** file:

* all four streams equal length; `prefix_stream == word_events`
* root / wazn / suffix **100.0 % identical to the shipped cache over their FULL length**
* `aligned_stream0 == true_prefix` **100.0 %** over 20 000 positions
* marker-count deltas: BOS_ROOT and EOS_ROOT each `67404 == val sentences`, `623723 == train
  sentences`

### 1.4 What the misalignment corrupted (and what it did not)

Measured on the exact 300 val / 3000 train windows of every 20k run:

| quantity | val | train |
|---|---|---|
| the head's prefix **input** differs | 40.47 % | 40.34 % |
| the prefix head's **target** differs | 40.46 % | 40.34 % |
| Sibawayh ʿāmil state differs | **0.0 %** | **0.0 %** |

The ʿāmil stream is invariant to the prefix argument (`gov.op_ids_for_ids(r, w, p)` ignores `p`
on this data), so a third suspected corruption path is ruled out. The corruption is confined to
the head's prefix feature and the prefix head's label. The prefix head did **not** improve with
the fix (73.10 % legacy → 73.17 % aligned, both below the 79 % majority), so prefix-label noise
was not the binding constraint.

### 1.5 Retrain on the aligned cache

Exact CONTROL invocation, only `--cache` changed; 20 000 steps, batch 32, `--head-init remap`.
**wall 900 s, peak VRAM 8848 MiB**, no NaN/Inf.

| arm (all 20k, same checkpoint) | ALL@1 | ALL@5 | NOVEL@1 | NOVEL@5 | CE_z | feat_scale |
|---|---|---|---|---|---|---|
| CONTROL (baseline) | 5.72 | 11.41 | 5.47 | 11.13 | 7.1678 | 24.36 |
| **ALIGNED (this work)** | **6.15** | **11.93** | **5.87** | **11.51** | 7.1552 | 24.22 |
| NOFEAT (branch disabled) | 6.59 | 12.98 | 6.53 | 12.81 | 7.6134 | 29.93 |
| GUARD | 6.63 | 13.00 | 6.55 | 12.84 | 7.6142 | 27.08 |
| FIX | 6.84 | 12.90 | 6.66 | 12.70 | 7.5435 | 27.08 |
| GATE2 | 5.74 | 11.20 | 5.46 | 10.69 | 7.1485 | 43.31 |
| FIX2 | 6.13 | 11.91 | 5.98 | 11.52 | 8.1562 | 39.20 |

**+0.43 pp ALL@1 and +0.40 pp NOVEL@1 over CONTROL — a 1.075× relative gain.** It does **not**
beat GUARD (6.63) or FIX (6.84): the cache fix is a real but *smaller* lever than the training
stabilisation already found (`--margin-ramp` + `--grad-warmup` + `--logit-scale ln`).
The aligned-cache + FIX-flags combination arm (`ALIGNED_FIX`) is **staged and queued**; it is the
run that would test whether the two gains stack past 6.84 %. It could not start because another
agent took the GPU with a 120 000-step `curve_lm.py` run, and I do not preempt other agents.

### 1.6 Inference-side causal isolation

Same head weights, same frozen backbone, same windows, **only the prefix stream swapped**:

| head | cache fed | next-word ALL@1 |
|---|---|---|
| CONTROL | legacy (its training cache) | 5.718 % |
| ALIGNED | aligned (its training cache) | **6.15 %** |
| ALIGNED | **legacy** | **5.448 %** |

A head trained on correct prefixes loses **0.70 pp** when fed the misaligned stream — a direct
causal measurement of the defect at inference time, with no confound.

## 2. The real ceiling is ~53 %, not 6.5 % — the "4-gram ~33 %" claim was an understatement

Order-k root n-gram fitted on the full train root stream, evaluated on the **same** 18 869 val
radical positions / 15 046 NOVEL, same masks:

| order k | ALL@1 | NOVEL@1 | `nrmt_arch.py` docstring |
|---|---|---|---|
| 1 | 1.74 % | 1.62 % | |
| 2 | 11.81 % | 11.43 % | |
| 3 | 38.85 % | 37.13 % | ← this is the "~33 %" |
| **4** | **53.13 %** | **49.51 %** | ← exactly the context `build_features` is given |
| 5 | 57.13 % | 52.62 % | |
| 6 | 57.95 % | 53.22 % | |

For reference on the same positions: free linear readout of ONE frozen `h_t` → `root_{t+1}`
= **6.476 % / 13.006 %**; trained head = 5.72 %; majority = 3.567 %; train-set 6-gram = 94.08 %.
53 % on positions whose 12-root context is novel is genuine generalisation, not memorisation.

So the head is at **88 %** of the single-position *linear* ceiling but at **~11 %** of the
order-4 *root-history* ceiling that its own conditioning branch is handed.

### 2.1 Why: the conditioning branch's form cannot express the lookup

`NRMTHead.build_features` gives the root head one `Linear(320→896)` + `LayerNorm` over the
concatenated embeddings of `r_{t-1..t-3}` (plus op/prev-morph), **added** to `h_t`. I measured the
ceiling of exactly that form with a closed-form ridge, no backbone, no training loop:

| k | linear on `emb(r_{t-1..t-k})` | order-k n-gram lookup |
|---|---|---|
| 1 | **1.41 %** / 3.99 % | 1.74 % |
| 2 | **1.58 %** / 3.97 % | 11.81 % |
| 3 | **1.65 %** / 4.24 % | 38.85 % |
| 4 | **1.69 %** / 4.41 % | 53.13 % |
| 5 | **1.81 %** / 4.50 % | 57.13 % |

A linear map of root embeddings stays at chance (below the 3.567 % marginal) no matter how much
history it is given, while the lookup climbs to 53 %. This is the mechanism, isolated from the
backbone: **the branch's inductive bias, not the backbone and not the readout, is the limiter.**
Note that concatenating one-hot ids would not help either — a linear map over concatenated
one-hots is additive and cannot express the interaction; only the (sparse) tensor product, i.e. a
lookup, can.

Corroboration from the existing arms: **NOFEAT (`--no-features`, branch fully disabled) = 6.59 %**
is statistically indistinguishable from GUARD (6.63 %, branch enabled + gated). The 12.8 M-param
conditioning branch adds ~0.04 pp. The controlled comparison (both arms carry
`--margin-ramp 10 --grad-warmup 10`; `feat_scale` 29.93 vs 27.08) is the cleanest available
statement that the branch is inert.

## 3. Suspect list from the brief — disposition of each

| suspect | disposition |
|---|---|
| position/alignment bug (head reads `t`, label `t+1`) | **NOT present in the head.** `rl = root_logits[:, :-1]` is compared against `target_roots[:, 1:]`; both length `T-1`; `h = H[:, :-1]`, so logits at `t` predict the root at `t+1`. Verified index-aligned. The real alignment bug was in the **cache's prefix stream** (§1). |
| root-stream side channel interfering (the clamp) | **Dead.** `extract_backbone` calls the inner Qwen2 model directly, so `active_root_ids` is never set; instrumented: 456 self-attention calls, **0** with `root_ids`, **0** with `active_root_ids`. Forcing it on moved next-word 5.718 % → 5.702 %. Confirmed abandoned. |
| prefix ablation 6.48 → 4.50 means the prefix path carries signal | Consistent: the prefix feature does carry signal (~1.2 pp when ablated), and the fix restores its *correct* value, which is where the +0.43 pp comes from. |
| masking `keep = ~isin(tgt_r, spec_t)` removing real roots / misaligning labels | **Not misaligned** (`keep` masks `root_logits[:, :-1]` against `target_roots[:, 1:]`, both `T-1`). Composition on the val windows (38 100 target positions): `<P:…>` particle tags 29.5 %, `<PARTICLE>` catch-all 20.3 %, UNK 0.7 %, **radical 18 869 = 49.5 %**; **zero** PAD/BOS/EOS. Masking particle tags is the documented intent ("forced to learn real radical roots") and the metric is computed only on kept positions, so it neither leaks nor misaligns. |

## 4. Seven-suite regression (after the module edits)

All seven re-run on the pod after patching both writers; log `artifacts/suites_after.log`:

| suite | result |
|---|---|
| `test_grammar_impl.py` | **20/20**, 0 FAIL |
| `verify_v2.py` | all PASS, exit 0 |
| `andalusian_realizer.py` | **67/67**, 0 FAIL |
| `test_sibawayh_governor.py` | **15/15** |
| `ibn_malik_automaton.py` | **46/46** |
| `test_khalil_orbits.py` | **30/30**, 0 FAIL |
| `test_awzan_order.py` | **24/24**, 0 FAIL |

## 5. The identity/echo gap

It was a misdiagnosis and it is closed as a question. The head has **one** readout aimed at `t+1`;
asking it for `t` is asking the wrong question. On the aligned arm: identity 1.888 % (CONTROL
1.739 %), echo 3.605 % (CONTROL 3.390 %) — both still at the marginal, because the readout
direction is aimed at `t+1`, exactly as expected. The meaningful ratio is head-vs-linear-ceiling on
the head's *own* task: CONTROL 5.718 / 6.476 = **88.3 %**; the ALIGNED arm's own-cache ceiling was
not measured (GPU-blocked), so 6.15 % is quoted raw rather than as a ratio.

## 6. Resource accounting

* `ALIGNED` 20k run: **wall 900 s**, **peak VRAM 8848 MiB**, 25–29 it/s, no NaN/Inf.
* Cache rebuilds (CPU): 147 s (legacy) / 155 s (aligned), each 4 equal streams.
* Multi-position / n-gram / feature probes: CPU, 0 GPU. Model-load 91 s dominates the GPU evals.
* GPU serialisation respected throughout: FIX2 was allowed to finish; my run started the moment
  the GPU freed and not before.

## 7. Still queued (GPU-blocked, scripts staged and syntax-checked)

1. `chain_fix.sh` → `run_aligned_fix.sh` — the **aligned cache + FIX flags** arm. This is the run
   that answers "does the cache fix stack past 6.84 %". Waiting on another agent's 120 k-step
   `curve_lm.py`.
2. `eval_aligned_head.py --cache /workspace/head_fix/nrmp_cache_9490_aligned` — the aligned-cache
   linear ceiling (the k=1 number needed to quote item 5 as a ratio).
3. `multipos_probe.py` — ridge on `h_t‖…‖h_{t-k+1}`, k=1..8 (one design bug found and fixed:
   it dropped one extra position per window). Theoretically low-yield given §2.1, since a linear
   map over concatenated hidden states is still additive.

## 8. Honest answer: is the head fixed?

**There was no broken head to fix, and the one real defect I found is fixed and verified — but it
is not the main story.** The prefix-stream misalignment was real, is now impossible to reproduce
silently (both writers default to `aligned`, `legacy` is md5-verified reproducible), and is worth
+0.43 pp acc@1 on the CONTROL configuration, with a clean 0.70 pp causal effect measured at
inference. What remains **unexplained as a design gap, not a bug**: every head variant sits at
~6–7 % while an order-4 root n-gram reaches 53 %. The conditioning branch that was supposed to
capture root history is measurably inert — a linear map of root embeddings reaches 1.7 % — and the
h-only arm matches the fully-conditioned arms. The next lever is therefore a **discrete / nonlinear
root-history pathway** (an explicit n-gram or hashed-context lookup over root ids, or a proper
tensor-product/MLP interaction), not more linear context, a bigger readout, or unfreezing the
backbone. Nothing in the "broken prediction head" framing survives contact with the measurements;
the reproducible defect was in the data pipeline, and the large remaining gap is architectural and
localised to one identified module (`NRMTHead.build_features`).
