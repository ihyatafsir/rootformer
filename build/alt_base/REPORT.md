# Alt-base root cross-attention — status report

**Deliverable:** attach the verified root cross-attention to a base model that was never
morphemically adapted, and find out whether the mechanism is general.

**Headline:** the decodability probe — the one decisive, free measurement — was completed and it
answers the experiment. The two trained arms (A, C) were **not** completed; both cells are empty
and I state that plainly below rather than dressing up partial work.

---

## 1. The probe (the decisive measurement) — COMPLETE

Same protocol as `/workspace/echo_test/echo_probe_len1.py` (unique train words → closed-form ridge
readout → unique held-out words, plus the subset whose `(prefix,root,wazn,suffix)` tuple never
occurs in train). Difference: the trunk is the **original `Qwen/Qwen2.5-0.5B`, plain HF
`Qwen2ForCausalLM`** — no morphemic embedding, no Ishtiqaq root branch — and the root target comes
from the **external** morphemic annotation aligned to the base tokenizer by character offset.
`h` = last-layer hidden state at the word's **last subword token**, read from the sentence the word
actually occurs in.

| trunk | sample | acc@1 | acc@5 | n |
|---|---|---|---|---|
| **original base** | val_all | **29.30 %** | 48.42 % | 63,496 |
| **original base** | val_unseen_words | **23.34 %** | 39.58 % | 21,243 |
| **original base** | val_seen_words | 32.30 % | 52.86 % | 42,253 |
| **original base** | train_all | 23.81 % | 43.41 % | 177,694 |
| current trunk (published) | val_all | 92.34 % | 95.82 % | 73,502 |
| current trunk (published) | val_unseen_words | 83.92 % | 89.83 % | 14,390 |

Marginals on the same samples: val_all 0.40 %, val_unseen 0.37 %, val_seen 0.42 %.
Cost: **308 s wall, 4.66 GiB peak VRAM, 290 k words** (inference only, no training).
Artifact: `/workspace/alt_base/probe_orig.json`.

*One artifact defect, flagged rather than hidden:* the `internal_holdout_acc_pct` field in the JSON
reads `1693.961` because that code path multiplies an already-percentage value by 100 again. The
correct internal-holdout accuracy used to select λ is **16.94 %** (λ=0.1 was chosen at 16.94 %,
consistently across 0.1–1000). The four headline numbers above are unaffected: they are computed by
a separate evaluator and returned 29.30 % on a sample on which a 1694 % figure would be impossible.

**Reading.** The original base is **not** at chance: root identity is linearly decodable at
29.30 % (23.34 % on unseen words), ≈73× its own marginal. So a root signal genuinely exists in an
untransmuted backbone, and the RCA is not reading nothing. But decodability is **3.15× lower** than
the adapted trunk (29.30 vs 92.34) and **3.6× lower** on unseen words (23.34 vs 83.92).

Because the published 19.53 % sits on top of a trunk that decodes its own root at 92.34 %, while a
fresh base offers only 29.30 %, the working hypothesis has to be inverted relative to the brief:
**the 92 % was substantially a product of the morphemic adaptation + synthesis training, not a
property the RCA merely read off any Qwen.**

*Caveat, stated because it matters for the exact wording:* 29.30 % is the decodability of an
**untrained** base. The parent's revised arms train the trunk, and training could raise
decodability toward the adapted trunk's level. So the probe bounds the *frozen*-base variant
sharply, and for the trained variant it is a starting point, not the ceiling.

---

## 2. Base decision

`Qwen/Qwen2.5-0.5B`, the original untransmuted model, **pinned** as instructed.
It was **not** in the pod's cache — only config/tokenizer/vocab were present and the 988 MB
`model.safetensors` blob was a **broken symlink**. Downloaded (999.6 MB, 9.3 s) into an isolated HF
home at `/workspace/alt_base/cache` so the shared `/workspace/.hf_home` was never touched.

## 3. The root stream: alignment (COMPLETE, verified)

`build_root_cache.py` runs the project's morphemic tokenizer over the corpus and attaches the root
id to the base tokenizer's positions by character offset, with **no re-vocabularisation**.

Reproduced the reference split **exactly**: 691,127 unique sentences, 47 files, 3 val files,
**7,060,477 train word events / 837,575 val word events** (reference: 7,063,767 / 837,575 — the
train difference is the reference's historical BOS/EOS prefix-layout bug that I deliberately do not
reproduce).

Alignment policy, then measured:
* word split across several base tokens → **all** those tokens carry the word's root; the head reads
  the hidden state of the word's **last** token. Measured: 1.9488 base tokens/word; 55.6 % of words
  are single-token, 44.4 % span ≥2.
* base token spanning a word boundary → assigned to the word with the most shared non-whitespace
  characters, ties to the earlier word.
* token sharing nothing with any word (whitespace-only) → carries the nearest preceding word's root.

Verified on 3,000 sentences: **0** whitespace-reconstruction failures, **0** decode/length
mismatches, **0** tokens with no word overlap, **0** carry-rule assignments, **0** words with 0
tokens, and the base tokenizer's decoded text equals the sentence exactly for every case.
`content_reconstruction_rule`: base decode == sentence AND morphemic word list == `sentence.split()`.

Cache rebuild cost: **339 s, 287 MB** (`word/tok/soff/wtokoff/wlen/wstart` per split; 55,159 train
blocks of 128 words, block tokens p50 252 / p99 319 / max 711).

## 4. Module adaptation (diff-level)

The released `root_attn/root_cross_attn.py` (md5 `b53ae702…`, verified) needed real adaptation, not
a rewrite for the sake of it: it installs a **post-layer** hook (`h ← h + gate·LN(o_proj(…))`) and
its K/V come from a supplied embedding. Two things it cannot do that the brief/scope requires:
(i) put the root term **in the attention score**, and (ii) augment the **value** stream, which is
the only way the root path can *inject* rather than reweight.

So `alt_root_attn.py` was written as an additional module (released modules left read-only) that
**rewrites the layer's attention in place** using the trunk's own `q_proj/k_proj/v_proj`,
`Qwen2RMSNorm` q/k norms, the trunk's own rotary `position_embeddings`, and the trunk's own causal
+ padding mask, then returns the same `(output, attn_weights)` tuple. It exposes both mechanisms
with independent zero-init gates: `ScoreBiasModule` (`gate·q_r·k_rᵀ·scale + gamma·1[r_i=r_j]`) and
the residual `h + gate·LN(o_proj(A @ (V_s + Wv·E_root(r))))`. The RCA source table is
`morphemic_embed.root_embed` = **(9490, 448)**, as required; the native `(9015, 64)` table is not
inherited.

Two measured environment facts worth recording, both from the checkpoint itself:
* `stream_mix` = `[0.75, 0.25]` and `ishtiqaq_gamma` = `0.25` in all 24 layers — consistent with
  the "never trained" claim.
* **However** `coverage_weight` = **2.0** and `governance_strength` = **1.5** in the checkpoint, so
  the native root branch is *live*, not dead, in the released trunk.
* `extract_morphemic_ids` maps **subword token ids** through `id_to_root_table` clamped to
  `[0, 9014]`. With the morphemic vocab (10,052 rows) that is meaningful; with the **base**
  tokenizer (151,936 rows) it is garbage. I therefore **disabled the native root path** on the base
  trunk, which is what makes it a true untransmuted base rather than one silently fed noise.

## 5. Init-equivalence (partial — honest status)

The parent module's zero-gate init is a **mathematical** no-op: `delta = self.gate.float() *
self.drop(out)` with `gate = zeros(())` ⇒ `delta ≡ 0` ⇒ `hs.float() + 0` cast back is bitwise
identical to `hs`. `ScoreBiasModule.gate` and `.gamma` are likewise `zeros(())`, so its contribution
is exactly 0. I did **not** complete an empirical bitwise check: `verify_alignment.py` was written
to run it end-to-end (attach at all 24 layers, gates 0, compare against detached) but never reached
that point before time ran out. **No measured init-equivalence claim is made.**

## 6. Trained arms — NOT RUN. Empty cells.

```
A  base + head, no root pathway, trunk trained    ← NOT RUN (no number)
C  A + root path from the input stage, trunk trained ← NOT RUN (no number)
B  root stream without RCA                        ← NOT RUN
D  score bias                                     ← NOT RUN
```

Why, concretely (this is where the time went, and it is not a design failure):
1. The released awzan142 checkpoint has **9868-token / 9114-root** embedding tables vs the current
   blueprint's **10,052 / 9490**, so a naive `load_state_dict` raises.
2. `RootformerNRMT.backbone` is the Qwen2Model built from the **morphemic blueprint**, so its
   `embed_tokens` had 10,052 rows and all transformer blocks were **randomly initialised** — it is
   not the base model. Base token ids (up to 149,221) trapped the CUDA context. Fixed by resizing
   to 151,936 and loading **all 290 real Qwen2.5-0.5B tensors** (`base trunk: loaded 290/291`,
   only `lm_head.weight` skipped), which is now the operational definition of "untransmuted base" I
   would report.
3. The trunk uses **eager** attention (`IshtiqaqAttentionV12` materialises `[B,14,T,T]` scores), and
   window token counts are p50 251 / p99 319 / max 592. Per-step activation memory with two
   resident arms (21.5 / 32.6 GiB, ~11 GiB free) exhausted the card at batch 8 (T≈487) even with
   gradient checkpointing at batch 4. End-to-end 6k steps × batch 16 × ~250 tokens was never
   going to fit the remaining card time, and I chose to keep the guarantee of never disturbing the
   two resident arms over squeezing a run in.

The live path is implemented and reaches the training step (it computes loss and gradients); it
died on memory, not on logic. `--live --grad-ckpt --block-words`, the model/optimizer grouping, and
the A/C flag surface are all present in `alt_train.py`.

## 7. Seven test suites

**Not run.** No shipped module was edited: `nrmp_vocab.py`, `nrmt_arch.py`,
`root_attn/root_cross_attn.py`, all checkpoints and all existing caches were read-only. Only new
files were created under `/workspace/alt_base/`. Per the brief, the suites are required when a
module is edited.

## 8. Verdict

**On the mechanism, the honest verdict is "not general — the signal was created by the morphemic
adaptation."** The one measurement that isolates the question shows the untransmuted base exposes
its word's own root at **29.30 % / 23.34 % unseen**, versus **92.34 % / 83.92 %** on the adapted
trunk. The published 19.53 % was therefore read off a backbone whose hidden state carried ~3× more
root signal than a fresh one does. An RCA bolted onto a base has roughly a third of the signal to
select from, so the architecture-specific explanation is the better-supported one.

**What is still unproven, and I will not overclaim it:**
* whether a *trained* trunk on a fresh base climbs toward the adapted trunk's decodability (the
  probe measures the untrained base; the mandated A/C arms would settle this);
* whether the score-bias mechanism (D), which has plausibly never been trained, helps — my finding
  that `coverage_weight=2.0` / `governance_strength=1.5` are live in the released checkpoint means
  the native path was at least *active*, which slightly weakens the "never tested" framing and is
  worth folding back into the parent's understanding.

---

## 9. Known performance defects in the live path (diagnosed, not fixed in a completed run)

These are why the final benchmark stalled, recorded so a future run is viable:

1. **Novelty mask is pure-Python O(positions × 7 M).** `build_novel_mask` did a list membership
   test against the 7 M-element train root list once per window position. At ~64 k positions this
   dominates everything. Fix: build a `set` once (or keep the uint64 hash array) and vectorise.
2. **Sequence length is the cost driver.** Trunk attention is eager, so a 128-word window is p50 251
   / p99 319 / max 592 base tokens. Recommendation for the next run: `--block-words 32` (≈80
   tokens), which cuts step cost ~6× and still leaves a usable root history inside the window.
3. **Memory ceiling.** With two resident 32-batch arms the free VRAM is ~10 GiB. End-to-end needs
   `--grad-ckpt` plus batch ≤4; a frozen-base protocol (precompute hidden states once, train only
   head+RCA) avoids the problem entirely and is in fact the protocol under which the 19.53 %
   comparator was produced — it is a legitimate, cheaper form of arm C.
