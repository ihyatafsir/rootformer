# Rootformer

Neuro-symbolic Arabic language model. A word is factored as

```
word = (P, R, W, S) = (Prefix, Root, Wazn, Suffix)
```

and the model is trained on **NRMT** — next root / morph-token prediction — with an explicit
root-history channel. Every linguistic rule is traced to a primary text (al-Khalīl, Sībawayh,
Ibn Jinnī, al-Zajjājī, Ibn Fāris, Ibn Manẓūr, al-Rāzī, al-Ghazālī); rules that could not be sourced
are labelled as such rather than presented as sourced.

> **Note on الله**: left untouched by product decision.

---

## Status — 2026-10-02

### Headline result

```
RCA_UNFREEZE_A2     20,000 steps   acc@1  19.529 %   acc@5 36.414 %   CE_z 8.1519
                    NOVEL_only     acc@1  19.899 %   acc@5 37.326 %
                    majority-class marginal        3.551 %      →  5.50x
                    FIX baseline                    6.837 %      →  +12.69 pp
                    RCA gates forced to 0           2.825 %      →  BELOW the marginal
                    NOVEL >= ALL at EVERY eval — generalisation over root histories, not memorisation
```

**Caveat, stated plainly:** `CE_z` does **not** improve (8.152 vs FIX's 7.543). The arm is a much
better top-1/top-5 *ranker* and a no-better-calibrated distribution. The held-out metric also ran
with RCA dropout active (bounded at −0.13 pp, one-line fix in `evaluate`: `rca_stack.eval()`), and
the trunk memorises its 381k-position training set (train root CE 0.02–0.2).

### The mechanism

`h ← h + gate · LayerNorm(o_proj(softmax(QKᵀ/√d) · V))`, with `Q = W_q·h` and
`K = V = W_{k,v}·E_root(r_{0..t})`, attached as a **post-layer hook** on trunk layers 20–23
(9.64 M params). Gate zero-init, so it is a **bitwise no-op at step 0**
(`max|h(gate0) − h(no RCA)| = 0.000e+00`).

`K`/`V` read `morphemic_embed.root_embed`, shape **(9490, 448)** — i.e. **448 of 896 = 50 %** of the
model width. (The `(896,64)` / `(128,64)` tensors belong to `IshtiqaqAttentionV12.root_q_proj` /
`root_k_proj`, which are **dead code**: 0 of 24 attention calls carry `root_ids`.)

### Lever table — what moves the number, trunk frozen unless stated

| lever | Δ acc@1 | Δ acc@5 | note |
|---|---|---|---|
| root cross-attention itself | 6.84 → 16.5 % | | +9.7 pp |
| **residual bound** (`--rca-out-norm`) | **+2.54** | +5.30 | the real lever |
| trunk unfreeze, top 4 @ 0.01× LR | +3.0 (16.5 → 19.53) | | 0.1× **destroys** the trunk |
| width 896 → 1792 (bound ON) | +0.22 | +0.71 | small but real; best acc@5/CE_z of any arm |
| depth 4 → 8 RCA layers | +0.13 | +1.56 | layers are redundant, not dead |

**Conclusion: every lever that works fixes *how* we inject — none changes *what the trunk
represents*. The frozen trunk is the ceiling.** The RCA pathway carries essentially the whole
capability (gate-off → 3.05 %, below the 3.551 % marginal).

### The residual-bound mechanism

The injected residual ran at **RMS 14–20 against `h`'s own RMS of 1.00**. `--rca-out-norm` throttles
it (`h_drift` 21.81 → 14.47) — but **throttled, not eliminated**: it still moves `h` by ~14× its RMS,
and the gates grow *larger* (`|mean|` 3.21 vs 0.70) because LayerNorm removes the magnitude penalty.

### Computed rule > training: al-qiyās

```
neural compositional decoder, held-out root, top-1/top-5     0 / 3,588   (0.00 %)
computed qiyās, realisation given (root, wazn)            3,023 / 3,588  = 84.25 %
computed qiyās, inverse root ID from the surface, top-1   3,031 / 3,588  = 84.48 %   (chance 2.70 %)
```

A **computed rule beats 25 M words of training** at the project's central claim. Caveats: all 90
realisation fixes belong to **one root (`بدا`) and are orthographic** (hamza vs bare alif); over 8
random 37-root folds the computed analogy is marginally **worse** than the hand-coded realiser
(0.8908 vs 0.8957); and the decoder had to *infer* the root from context where qiyās is *given* it.

The ʿilla conditions were measured, and two were **rejected**: `wazn_only` (three conditions) and
**`identity` — the literal "ʿilla = the root"** — by **al-Rāzī's *mutaʿaddī*** condition
(*al-Maḥṣūl* 15719–15721): «العلة المتعدية هي التي توجد في غير مورد النص». A root is its own locus.
The ʿilla actually used is the root's **phonological class**.

### Preservation: the capability Fisher fixes the cliff

```
arm Q: A2 config at 0.1x trunk LR + F_head penalty (κ=10)
  own acc@1                        → 18.96 % @ 6k
  HEAD-INDEPENDENT (FIX head on ITS trunk)   6.656 → 6.672 %
  untouched released trunk, same measure     6.641 %   ce_z 8.0998  →  IDENTICAL
  penalty contribution RISING (0.0015 → 0.0732) — not being dodged

arm P: same but with the TEXT Fisher (F_ret)
  own acc@1 16.34 %  ← looks healthy
  HEAD-INDEPENDENT 2.676 %  ← the trunk was DAMAGED
```

**Arm P's healthy-looking self-report hid a destroyed trunk, because the head and RCA co-evolved
with the damage.** The head-independent control is what separates a real preservation from a dodge.

**And the essential/accidental split as specified is empty:** `corr(F_ret, F_syn) = 0.9959`. The
real split is *"the loss we measure the capability with"* vs *"the text distribution we train on"* —
and the text-Fisher proxy is **anti-correlated** with the damage (ratio −87, wrong sign), because the
trunk's tied LM readout is **miscalibrated** (CE 15.99 vs `ln(10052) = 9.22` for uniform).

The penalty **redirects** rather than brakes: λ=0 and κ=1 move the trunk by nearly the same norm,
while Fisher-alignment ρ falls **8.434 → 0.279**.

---

## Layout

```
src/            shipped source (see VERSIONS.md for md5s)
models/         architecture modules (ishtiqaq_attention_v12, unified_rootformer_v12, ...)
build/          per-experiment working dirs, mirrors of every pod run
corpus/         primary texts (Lisān al-ʿArab, Asās al-Balāghah, heritage_foundations/)
release/        the canonical release tree
results/        results JSONs
tools/          utilities
.secrets/       SSH keys — NEVER COMMITTED
```

`build/` holds one directory per experiment, each self-contained with its report, results JSON,
trace, and logs:

```
build/root_attn/          the RCA mechanism          REPORT_root_attn.md
build/root_attn_ctl/      frozen-trunk control       (results JSON never written — SIGTERM @15k)
build/root_attn_norm/     residual bound             → 19.17 % 
build/root_attn_depth/    4 vs 8 layers              → +0.13 pp, depth is NOT the lever
build/root_attn_width/    width 896 vs 1792          → +0.22 pp, bandwith is NOT the limit
build/ghazali_forget/     EWC / capability Fisher    RESULTS.md
build/qiyas/              al-qiyās                  QIYAS_REPORT.md
build/grammar_audit/      citation verification     AUDIT_REPORT.md
build/discrete_path/      discrete n-gram pathway
build/vocab_shrink/       cardinality ablation      9,490 → 500 classes: lift stayed 1.86x
build/echo_test/          identity/echo probes       92.34 % current root linearly decodable
build/rca_prompt/         qualitative prompt read    RCA on/off, 30 uncurated prompts
build/ishtiqaq_check/     native root-path audit     never received a single gradient
build/alt_base/           root-aware on a fresh base (in progress)
build/lexicon_rebuild/    grounded lexicon + rejected_claims (anti-fabrication filter)
build/scratch_comp/       scratch LM comparison
```

## Key facts

```
tokenizer round-trip          82.1065 % token  /  57.5884 % type
root inventory                9,490   (from 9,114; +177 Lisān roots, incl. all 69 ي-roots)
total vocab                   10,052  ·  num_awzan 142
lexicon union coverage        98.2 %
current root decodability     92.34 % (83.92 % unseen words)
next root from a single h_t   capped at 6.48 % — the information is NOT in h_t
order-4 root n-gram lookup    53.13 %   (order 1/2/3 = 1.74 / 11.81 / 38.85)
```

**The tokenizer defect is on the truth side, not the output side.** 12.04 % of radical targets fail
the exact round trip (10.81 % for genuine morphological reasons); 50.54 % of held-out targets are
control/particle roots that the metric excludes. The model's own emissions were **18869/18869
well-formed** with zero control-token leaks.

## Open problems

1. **The thesis is untested.** The claim — one root embedding shared across كتب / كاتب / مكتوب /
   كتابة generalises to unseen derivational forms — has never been demonstrated. Attempt one was
   structurally impossible (a softmax cannot emit an unseen root); attempt two was under-trained.
2. **A trunk trained for this objective.** Every lever that worked fixed injection, not
   representation. Root-awareness cannot be retrofitted into a synthesis-trained trunk whose
   synthesis and linguistic structure are 99.6 % Fisher-correlated.
3. **The derivational holdout**, properly designed, with the full 9,220-root inventory (the 84.48 %
   was measured on a candidate set restricted to 37 roots).
4. **The distinctive classical machinery mostly does not reach the gradient** — the orbit loss is
   provably inert (`max |dgrad| = 0.000e+00` over 1,000 steps), the Sībawayh governor never touches
   the loss.
5. **A hand-annotated gold morphology set** to replace the pipeline's own analyses as the aṣl basis.

## Known defects to fix

- `--eval-every 0` → `ZeroDivisionError` at `nrmt_train.py:574` (pre-existing, unfixed).
- `evaluate` never calls `rca_stack.eval()` — RCA dropout stays active at eval (−0.13 pp, conservative).
- `models/unified_rootformer_v12.py:80,91` hardcodes `num_roots=9015` — a **third** root-id space.
- `models/ishtiqaq_attention_v12.py:169,208` — `torch.clamp(root_ids, 0, 9014)` against a
  `root_embed` of shape `(9015, 64)`; 475 ids collapse and 5.01 % have no row at all.
- The standard GPU guard `pgrep -f "nrmt_train[.]py --checkpoint"` **does not match**
  `nrmt_train_width.py` or `nrmt_train_ewc.py` — it under-counts by 1–2. Use a `ps`-based counter.
- `repro_eval.py` builds `RootformerNRMT(...)` without `feat_gate=`, silently dropping the trained
  scalar `feat_gate = 0.2198` and reporting 11.14 % instead of 19.58 %.

## Reproducing a run

```
python src/nrmt_train.py \
    --checkpoint checkpoints/rootformer_v19_2_synthesis_...roots9490.tok10052.safetensors \
    --cache /workspace/head_fix/nrmp_cache_9490_aligned \
    --head-init remap --steps 20000 --batch-size 32 --eval-every 1000 --tag <TAG> \
    --feat-gate --margin-ramp 10 --grad-warmup 10 --logit-scale ln \
    --root-cross-attn top4 --unfreeze-last 0 --rca-lr-scale 1.0 --rca-out-norm
```

Checkpoint to use: `…roots9490.tok10052.safetensors`, md5 `3335a3d39091535d0fdd047dca842715`,
795,064,504 B. The checkpoint chain is **append-only** — rows 0..9113 are bit-identical across all
three root extensions.

Shared flags for the FIX/RCA lineage:
`--feat-gate --margin-ramp 10 --grad-warmup 10 --logit-scale ln`.
Note `--logit-scale ln` stops PPL inflation but **degrades `CE_z`** (7.238 → 8.156) — a top-1 /
calibration trade.

## Method notes earned the hard way

- **Aggregate metrics hide what a model emits.** The from-scratch model emitted **one tuple 80.14 %**
  of the time, rendering as the *empty string*; no score revealed that. Prompt-read anything before
  believing its number.
- **A self-reported metric can hide its own damage.** Arm P looked healthy at 16.34 % while its trunk
  was destroyed — the head and RCA co-evolved with the injury. Always keep a **head-independent**
  control.
- **Checks that cannot fail are worse than no checks.** Five were found in this codebase, including a
  verifier with the constant it was checking embedded in its own regex.
- **Never `|| true` on a transfer.** One upload "succeeded" while transferring nothing.
- Related: a classifier emits and is believed; a class of bugs was found only by a *deterministic*
  auditor after an LLM-based one **hallucinated its input file** and graded it.

## Provenance

Every rule in the grammar modules traces to a primary text with `file:line`. An independent audit
(`build/grammar_audit/AUDIT_REPORT.md`) verified **284 citation claims**: 74.3 % exact-or-near,
**VERBATIM pass 60/60 = 100 %**, locators 56/63 exact, **0 fabricated**, 0 contradicting —
rules **FAITHFUL 6 · OVERREACH 1 · UNSOURCED 0 · UNCHECKABLE 2**.

Citations that could **not** be verified from the held corpus are labelled UNCHECKABLE rather than
presented as sourced. This includes **Ibn ʿUsfūr's «غير مأخوذ به؛ لعدم اطراده»** — no Ibn ʿUsfūr
text is held; the only occurrence of his name in `heritage_foundations/` is a bibliography line.
