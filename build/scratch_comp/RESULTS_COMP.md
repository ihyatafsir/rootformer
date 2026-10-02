# scratch_lm_comp.py — compositional root decoder: redesigning the derivational-holdout test

Pod `6rnaag3vvrfpr` (213.173.108.47:22515), GPU RTX PRO 4500 Blackwell 32,623 MiB,
`/workspace/venvs/rootformer/bin/python` (torch 2.8.0+cu128).

**The original baseline is untouched**: `/workspace/hf_v19_2_release/scratch_lm.py`,
md5 `71db165f8f9d09f8725402834b793a3f` (re-verified after every run). Nothing was edited in
place; the new work lives in `/workspace/scratch_comp/scratch_lm_comp.py` and writes only to
`/workspace/scratch_comp/`.

---

## 0. What was wrong, and what the fix is

`scratch_lm.py` predicts the next word's tuple (prefix, root, wazn, suffix) with four heads,
the root head being a softmax over 9,114 root **types**. Its derivational holdout scored
0.000000 top-1 **and** top-5 at 6,000 and 80,000 steps. The cause is structural: a held-out
root *type* occurs 0× in training, so its output row is only ever trained as a **negative**
class (its norm grew 5×, `|w| 0.0220 → 0.1087`) and its input row receives zero gradient
(only weight decay). A categorical softmax over root types can never emit an unseen type, so
the holdout measured the head design, not the morphemic claim.

**Fix.** The root head becomes a compositional decoder over the root's radicals:

```
P(root | h) = P(gate = CONTENT | h) · Π_s P(radical_s | radical_<s, h)
```

* `gate` is a 271-way classifier: the **270** closed-class root tokens (`<UNK>`, `<PARTICLE>`,
  `<P:…>`) plus one CONTENT class. Specials genuinely are a label set — they have no radicals —
  so they stay a label set. `<PAD>` is the ignore index.
* the CONTENT part is a small **GRU decoder** over radical slots: initial state
  `tanh(Linear(h))` (d_dec = 256), input = embedding (128) of the previous radical (BOS for
  slot 1), output = 32-way classifier (31 alphabet symbols + EOS). EOS is illegal at slot 1
  (min length 2) and a 5-letter sequence is terminal (max length 5). Everything else is the
  original backbone: same corpora, same word-level context, same four-stream input, same
  prefix/wazn/suffix heads, same loss weights (0.5 / 0.25 / 0.25 for wazn / prefix / suffix),
  same bits/word definition, same split-by-work, same holdout construction.

**An unseen root is now expressible by construction**: every 2–5 symbol sequence over the
alphabet has non-zero probability whether or not it is a vocabulary type. Bits/word stays
comparable because the root term is still exactly −log₂ P(next root), now evaluated by the
chain rule over the radical sequence. Root loss during training is the mean over valid slots
(scaled like one CE term); **reported bits are the sum**, i.e. the true NLL.

**Why the GRU / AR design rather than a fixed-slot multi-label head.** The actual data is
`{2: 1562, 3: 12450377, 4: 145976, 5: 0}` root-slot tokens in the training stream, so length
must be *generated*, not assumed: an AR decoder expresses the length distribution through EOS
and gives a properly normalised joint P(string), which is what the bits/word comparison needs.
The 5th slot is retained because 198 vocabulary roots are 5-consonantal, so keeping it costs
nothing and keeps every vocabulary root expressible. A per-slot independent head was rejected
because it cannot represent the joint length/letter dependency and its "distribution" is only
normalised if EOS is modelled per slot anyway.

**Arms.** `--root-head comp` (the fix) and `--root-head type` (**the original softmax head,
verbatim**) re-run on the *same new data* so every comparison is apples-to-apples.

---

## 1. Re-preparation against the current release (yes, done)

The existing `/workspace/sf_data` was built against the **previous** blueprint revision
(`n_roots=9114, n_awzan=130`). It was **not** reused. A fresh dataset was prepared with the
**current** blueprint (md5 `5bd3e828b41fb1a6a6039d47c2f359fc`):

| | current release |
|---|---|
| roots | **9,490** (270 special, 9,220 content) |
| awzan | **142** |
| prefixes / suffixes | 26 / 22 |
| split-by-work (basename keys / source paths) | 165 / 250 → test 16, val 9, train 140 |

New dataset `/workspace/scratch_comp/sf_data_9490/` (the old `/workspace/sf_data` was left
**untouched** and is now a labelled, older-revision comparability artefact only):

| split | sentences | words |
|---|---|---|
| train | 2,718,865 | **25,000,009** (capped at 25 M) |
| val | 2,000 | 21,163 |
| test_gen | 4,000 | 33,957 |
| test_deriv | 3,627 | 35,820 |

*Derivational holdout*: **37** content roots held out, selected by the **identical** procedure
(sentence containment over a 120 k-sentence probe, least-containing first, ~10 % coverage,
non-special, 3 consonants, containment ≥ 20):
`بدا بدل بني بوب ثان جاب ختم دال دبب دبر دفع دول ذهن زاد سعع سما سير شخص شرر شقق صبح صرح طعع
عبس عرق عزل عنا غفل قبح قلد كرر كره كلف لغا مهم نفي وهم`.

*Radical alphabet*: **31 symbols** = the 28 Arabic letters (a fixed a-priori prior) ∪
`ء أ ى`, i.e. extra symbols were admitted **only if observed in a training-stream root**
(leak-free). Coverage: every training root and **every held-out root is expressible** (0
held-out roots inexpressible). Exactly 2 content roots in the whole vocabulary are inexpressible
— the Latin-script `end`, `start`, which are junk in the inventory — and neither is held out.

**The holdout as built is a real test: an unseen root has finite, non-zero probability.**

---

## 2. Leak check (independent, full pass — `audit_comp.py`)

| split | sentences | sentences with a held-out root | held-out root occurrences | distinct held-out roots |
|---|---|---|---|---|
| train | 2,718,865 | **0** | **0 / 25,000,009** | 0 |
| val | 2,000 | **0** | 0 / 21,163 | 0 |
| test_gen | 4,000 | **0** | 0 / 33,957 | 0 |
| test_deriv | 3,627 | 3,627 | **3,987 / 35,820** | **37 / 37** |

`held-out ∩ special_ids = ∅`. The audit also round-tripped `root_letters` for all 9,218
expressible content roots. **The holdout is leak-free.**

---

## 3. Evaluation is exact, and was validated against brute force

Root top-k for the comp arm is computed by **best-first search over letter sequences** merged
with the gate's closed-class candidates. Because every factor is a probability ≤ 1, a prefix is
a non-increasing upper bound on its extensions, so the heap top is always the global maximum and
emissions are in exact descending order. `fanout=8` nodes are expanded per position per round
purely for GPU batching, with emission always evaluated against the fully-updated heap, so
exactness is preserved.

* Validated on CPU against brute-force enumeration of all sequences of length 2–5 over a
  3-symbol alphabet: **identical top-5 sequences, identical order, scores equal to 1e-6**
  (`smoke_comp.py`).
* The teacher-forced NLL was validated against a hand-written chain rule to 1e-6.
* Every reported search completed exactly: `n_bfs_inexact_joint = n_bfs_inexact_free = 0`
  everywhere.

For each position the reported **joint** top-k merges the gate's 5 best specials with the
decoder's exact best content sequences using the threshold θ = 5th-best special; nothing below θ
can enter a top-5, so the threshold is lossless. A second, θ = −∞ **content-only** search gives
the decoder's own best root strings (this is where partial credit is definable). All 3,588
held-out-root target positions are always included in both searches. `real_seen`/`special`
columns come from the budgeted sample; `n` is always reported.

---

## 4. Bits/word — against the 12.99 / 14.74 reference and the unigrams

All numbers bits/word, unweighted, same definition as `scratch_lm.py`.

| split | **comp 6 k** | **comp 24 k** | **type 6 k** (new data) | unigram (root TYPE) | unigram (compositional) | OLD type 6 k (9114/130) |
|---|---|---|---|---|---|---|
| train | 13.2848 | **12.9229** | 13.1758 | — | — | — |
| val | 12.9950 | **12.6718** | 12.9192 | 13.6556 | 9.5870 | 12.9304 |
| test_gen | 13.2548 | **12.8638** | 13.1441 | 14.0307 | 9.9001 | 12.9855 |
| test_deriv | 14.3087 | **14.1457** | 14.9141 | 16.0658 | 10.1573 | 14.7404 |

Per-head root term:

| split | comp 6 k root | comp 24 k root | type 6 k root |
|---|---|---|---|
| val | 8.0078 | 7.7610 | 7.9205 |
| test_gen | 8.2034 | 7.8843 | 8.0648 |
| test_deriv | **9.0112** | **8.9164** | **9.5936** |

* The 12.99 / 14.74 reference is the **old-vocabulary** type-head 6 k run; on the new data the
  same head gives 13.1441 / 14.9141. Use the `type 6 k` column for apples-to-apples.
* comp beats the root-**type** unigram on every split (by 0.66–1.76 bits at 6 k; 0.98–1.92 at 24 k).
* comp is 3.4–4.2 bits **worse** than the compositional (per-slot letter) unigram: a decoder that
  spells roots is held to a much stronger context-free reference, and this model does not beat it.
* On the **derivational holdout the compositional head is better than the type head on bits**
  (root term 9.0112/8.9164 vs 9.5936): the type head cannot buy probability for a type it has
  only ever seen as a negative, so its held-out NLL is much worse. At 24 k comp beats the type
  head on *every* split.

---

## 5. Root accuracy — seen roots, joint and content-only, with the fair baselines

Seen real roots (`real_seen`), top-1 / top-5 %.

| split | comp 6 k joint | comp 24 k joint | type 6 k joint | **fair: majority real root (train)** | fair: majority real root in-slice |
|---|---|---|---|---|---|
| train | 1.009 / 3.872 | 1.822 / 5.630 | 1.734 / 5.201 | **1.872 / 6.389** | 1.872 / 6.389 |
| val | 1.186 / 3.972 | 1.677 / 5.391 | 1.677 / 5.236 | **2.698 / 7.827** | 2.698 / 8.117 |
| test_gen | 2.501 / 5.882 | 3.453 / 7.835 | 3.286 / 7.526 | **2.710 / 8.585** | 2.710 / 8.965 |
| test_deriv | 1.971 / 4.754 | 2.609 / 6.348 | 2.348 / 6.203 | **2.475 / 6.541** | 2.475 / 7.322 |

Content-only (comp arm — "if the model had to emit a *root string*, not a special"):

| split | comp 6 k | comp 24 k | n |
|---|---|---|---|
| train | 3.401 / 10.204 | 4.592 / 12.245 | 588 |
| val | 4.785 / 11.221 | 5.941 / 12.376 | 606 |
| test_gen | 5.213 / 12.480 | **6.951 / 15.008** | 633 |
| test_deriv | 3.925 / 9.386 | **6.143 / 12.457** | 586 |

*The script's own degenerate reference is kept for continuity:* the unigram top-1/top-5 ids are
all special tokens (id 4; ids 3,4,5,6,20), so its `real` column is 0.000 by construction — the
fair columns above replace it.
*Seen/unseen gap:* at 24 k the model puts the correct **seen** root first 3.45 % of the time
(test_gen) but the correct **held-out** root first/last-in-top-5 **0.00 %**.

Joint accuracy on all positions: test_gen 16.4 %/29.2 % (6 k) → 17.3 %/31.5 % (24 k); test_deriv
10.0 %/19.6 % → 10.6 %/20.9 % — the `all` column is carried by the ~46 % special positions.

---

## 6. The held-out roots — exact top-1/top-5, partial credit, examples

`test_deriv`, all **3,588** held-out-root target positions (held-out roots are 3-consonantal):

| metric | comp 6 k | comp 24 k | type 6 k | chance |
|---|---|---|---|---|
| **held-out root top-1 (joint)** | **0.00000** | **0.00000** | 0.00000 | — |
| **held-out root top-5 (joint)** | **0.00000** | **0.00000** | 0.00000 | — |
| **held-out root top-1, content-only** | **0.00000** | **0.00000** | n/a (emits specials) | — |
| **held-out root top-5, content-only** | **0.00000** | **0.00000** | n/a | — |
| anagram (right radicals, wrong order) | 0.00000 | 0.00000 | 0.00000 | 0.02 % |
| slot accuracy (top-1 content string) | **0.0641** | **0.0604** | 0.1989 † | 3.23 % |
| ≥ 1 true radical present | **0.3930** | **0.4030** | 0.0502 | 26.4 % |
| mean radical overlap | **0.1508** | **0.1536** | 0.0265 | 9.68 % |
| radical recall anywhere in content top-5 | **0.4447** | **0.4788** | 0.3329 | — |
| mean edit distance to the true root | 2.80 | 2.81 | 9.43 | — |
| same-length predictions | 1.000 | 0.998 | 0.052 † | — |
| root right & wazn wrong | 0.0 | 0.0 | 0.0 | — |
| wazn right & root wrong | 0.093 | 0.088 | 0.094 | — |
| top-1 is a special token | 0.989 | 0.948 | 0.948 | — |

† The type arm's slot accuracy is computed on the 5.2 % of positions where its emitted *special
string* happens to be 3 characters long — it is not comparable and is listed only for completeness.
The comp arm's content string is a real 3-letter root at ~100 % of positions.

**The redesign changed the character of the measurement.** A softmax could give no partial
credit at all (a non-identical type is simply wrong); the decoder gives above-chance partial
credit — radicals are individually reachable (6 % of slots, ~2× chance; a true radical appears
in 39–40 % of best-content hypotheses vs 26 % chance; radical recall in the content top-5 is
44–48 %) — while exact rooting stays at zero.

**Concrete held-out predictions** (true root | joint top-1 | decoder's best content root | its content top-5 | context tail):

```
comp24k
  true=شقق  joint=<PARTICLE>  content=اله  ctop5=['اله','نفس','جمع','بيي','فعل']      ctx=[ميل,<P:في>,حفر]
  true=دال  joint=<PARTICLE>  content=كما  ctop5=['كما','قال','وقل','جمع','كون']      ctx=[<PARTICLE>,<PARTICLE>,طلب,علم,ريس,<P:أى>]
  true=نفي  joint=<PARTICLE>  content=قول  ctop5=['قول','وجه','جمع','علم','فعل']      ctx=[مثل,<P:هذا>,امر,ولذ,لحظ,<P:له>]
  true=شخص  joint=<PARTICLE>  content=ثلث  ctop5=['ثلث','رجل','وحد','وجه','جمع']      ctx=[جاز,سهم,غرض,<P:من>,عله,<P:وهو>]
  true=عرق  joint=<PARTICLE>  content=جمع  ctop5=['جمع','وحد','مثل','شيء','عين']      ctx=[روح]
  true=وهم  joint=<P:على>     content=قول  ctop5=['قول','فعل','علم','قله','وجه']      ctx=[<P:في>,نفس,<PARTICLE>,<P:بل>,أمر,فرض]
  true=قبح  joint=<PARTICLE>  content=اله  ctop5=['اله','وحد','شيء','نفس','علم']      ctx=[<PARTICLE>,<P:ولا>,<PARTICLE>,<P:أي>,<P:لا>]
  true=عنا  joint=<PARTICLE>  content=اله  ctop5=['اله','رجل','شيء','نفس','فلن']      ctx=[]

type6k, same positions
  true=شقق  joint=<PARTICLE>  content=<PARTICLE>  ctop5=['<PARTICLE>','<P:أن>','<P:ما>','اله','<P:ذلك>']
  true=شخص  joint=<PARTICLE>  content=<PARTICLE>  ctop5=['<PARTICLE>','<P:من>','<P:ما>','وحد','ثلث']
  true=عرق  joint=<PARTICLE>  content=<PARTICLE>  ctop5=['<PARTICLE>','<P:من>','<P:في>','<P:ما>','جمع']
```

The comp decoder **is** spelling roots (اله، كما، قول، ثلث، جمع) — just the high-frequency ones,
largely irrespective of context. The type head's candidates are closed-class tokens.

---

## 7. The decisive measurement: root NLL on held-out targets

`test_deriv`, per-position root NLL in **bits** (all 3,588 held-out positions; uniform over the
9,490 types would be **log₂ 9490 = 13.212 bits**):

| target category | n | comp 6 k | comp 24 k | type 6 k |
|---|---|---|---|---|
| special | 14,530 | 5.827 | 5.630 | 5.821 |
| seen real root | 14,075 | 10.485 | **10.121** | 10.213 |
| **held-out real root** | **3,588** | **16.130** | **17.510** | **22.447** |

* The type head's held-out NLL (22.45 bits) is **9.2 bits worse than uniform** — a factor of ~600
  *below* uniform mass. That is the "trained only as a negative" mechanism, now measured. Its
  held-out rows are not merely untrained, they are actively suppressed.
* The compositional head is **4.9–6.3 bits better** (16.13 / 17.51 bits): the unseen root now
  receives real, finite probability. **The head design is fixed** — this is the property the old
  experiment could not even express.
* But it is still **2.9–4.3 bits worse than uniform**, because the letter decoder is confidently
  emitting frequent roots.
* **More steps made this worse, not better** (16.13 → 17.51 bits) while improving seen roots
  (10.49 → 10.12): training sharpens the decoder onto frequent roots.

---

## 8. Train-vs-held-out per head — overfitting or undertraining?

Per-head top-1 (%) at 24 k:

| head | train | val | test_gen | test_deriv |
|---|---|---|---|---|
| wazn | 49.60 | 52.20 | 49.58 | 47.42 |
| prefix | 75.50 | 77.22 | 76.17 | 74.57 |
| suffix | 82.59 | 83.01 | 81.55 | 80.13 |
| root (joint, real_seen) | **1.822** | 1.677 | **3.453** | 2.609 |
| root (content-only) | 4.592 | 5.941 | 6.951 | 6.143 |

bits/word train vs test at 24 k: 12.9229 / 12.8638 (test_gen) — test is *not* worse.

**There is no overfitting anywhere.** The prefix/wazn/suffix heads are identical on train and
held-out works, and bits/word is flat-to-better on held-out data. The root head's train accuracy
is **1.82 %, i.e. still at (in fact slightly below) the context-free majority-real-root baseline
of 1.872 % on the training stream.** The model has not fit the training marginal for roots at
either 6 k or 24 k steps.

**The reason is visible in the budget.** The training stream averages 9.195 words per sentence
and the batch is 16 sentences, so each step sees ~147 word-positions:

| steps | sentences seen | words seen | epochs over the 25 M-word stream |
|---|---|---|---|
| 6,000 | 96,000 | 0.88 M | **0.035** |
| 24,000 | 384,000 | 3.53 M | **0.141** |

The 24 k run is **14 % of a single epoch**. A low root/held-out number here therefore means
**insufficient training / weak context conditioning**, not overfitting and not a property of the
compositional head.

*(The small train/test inversion on `real_seen` is a distribution effect, not a leak: the
majority real root occurs 1.87 % of the time in the training stream but 2.71 % in `test_gen`, so
a near-marginal predictor scores higher on the test works. Both numbers are reported.)*

---

## 9. Residual confound (measured, not assumed)

The redesign changes the **output** head; the root **input** embedding is still a lookup table,
and a held-out root's input row never received gradient. That can only matter when a held-out
root appears *earlier in the same sentence* (inside the model's context). `context_diag.py`:

| split | held-out target positions | with a held-out root earlier in the sentence |
|---|---|---|
| test_deriv | 3,588 | **360 (10.0 %)** |
| test_gen / val / train | 0 | 0 |

**90 % of held-out target positions have entirely trained input context.** The untrained-input
confound is real but affects only a tenth of the test, so it cannot explain 0/3,588. All
held-out targets are length 3.

---

## 10. Wall-clock, VRAM, NaN

| job | wall | rate | peak VRAM |
|---|---|---|---|
| re-prepare (CPU, 112 cores) | ~35 min (collection 72 s, holdout+filter ~20 min, encode 25 M words ~12 min ≈ 36.8 k words/s) | — | — |
| comp 6 k (28,740,096 params) | 175.1 s | 34.3 it/s | **870 MiB** |
| type 6 k (33,020,160 params) | 113.5 s | 52.9 it/s | 1,009 MiB |
| comp 24 k | 565.0 s | 42.5 it/s | **1,074 MiB** |

* **No NaN/Inf anywhere** — every logged loss is finite, every bits/word and accuracy is finite,
  no traceback in any log.
* Evaluation wall-clock: comp 6 k 23 m 22 s, comp 24 k 23 m 54 s (dominated by the exact search
  and by GPU contention — at times three other agents' runs shared the card), type 6 k 1 m 53 s.
* **GPU contention (honest disclosure):** the pod was shared with other agents' 20 k/120 k-step
  runs throughout. A queue waited for idle windows and took them when they appeared
  (`[02:05:04] GPU IDLE …`, `[02:28:34] …`), but after 300–900 s without a window it proceeded
  on the shared GPU and logged `CONTENDED` (once, at 00:16:37 and 00:42:55). Nothing was killed;
  concurrency affects only speed — each run is seeded (`torch.manual_seed(0)`,
  `random.Random(0)`) and deterministic in its own process, so no result depends on it.

---

## 11. Verdict

1. **The experiment now tests what it claims to test.** An unseen root is expressible by
   construction, the search over that space is exact (brute-force-verified), the holdout is
   leak-free (0 / 25,000,009), and all 3,588 held-out positions are scored. The 0 % the old
   experiment returned is no longer structurally forced.

2. **The head-design diagnosis is confirmed and quantified.** On held-out-root targets the type
   softmax costs **22.45 bits** — 9.2 bits *worse than uniform over 9,490 types* — because those
   rows were trained only as negatives. The compositional head costs **16.13 bits** (6 k) /
   **17.51 bits** (24 k). The fix is real: unseen roots went from ~600× below uniform mass to
   within ~3–4 bits of uniform. The compositional head also beats the type head on
   **bits/word on the derivational holdout at both 6 k (9.011 vs 9.594 root bits) and 24 k**,
   and at 24 k on every split.

3. **Exact generalisation to unseen roots is still not achieved.** 0 / 3,588 top-1 *and* top-5,
   joint and content-only, at 6 k and at 24 k. No anagram (right radicals, wrong order) either.

4. **But it is not refuted, and the redesigned test now says why.** The model gives above-chance
   partial credit (slot accuracy 6.0–6.4 % vs 3.2 % chance; a true radical appears in 39–40 % of
   its best content roots vs 26 % chance; 44–48 % radical recall in the content top-5), and the
   train-vs-held-out evidence says it is **undertrained, not overfit**: its root top-1 on the
   *training* stream (1.82 %) is still at the context-free majority-root baseline (1.87 %).
   4× more steps improved seen roots and bits but left the exact held-out number at zero and
   *reduced* the probability mass on unseen roots (16.13 → 17.51 bits) — i.e. it sharpens onto
   frequent roots.

5. **So: "the morphemic scheme generalises to unseen roots" is neither demonstrated nor
   falsified.** The experiment *can* tell us something now, and what it tells us is that the
   blockade was the head, that an unseen root is reachable but under-weighted relative to a plain
   uniform-over-types code, and that adding steps alone pushes the holdout in the wrong
   direction. What would settle it: train until the root head beats the context-free
   majority-root baseline **on the training stream** (currently 1.82 % vs 1.87 %), and consider
   a compositional *input* embedding (only 10 % of held-out targets are context-contaminated, so
   that is a second-order effect). Until the train-side number is clearly above the baseline,
   "0 / 3,588" cannot be read as evidence against the hypothesis.

---

## 12. Artifacts

Pod `/workspace/scratch_comp/`, mirrored to `rootformer/build/scratch_comp/`:

* `scratch_lm_comp.py` (new script; modes `prepare` / `train` / `eval`; arms `comp` / `type`)
* `sf_data_9490/{meta.json,streams.pt}` — new dataset; `{comp6k,comp24k,type6k}.pt` checkpoints
* `{comp6k,comp24k,type6k}_{train,eval}.json` — full results (eval JSONs carry the per-position
  held-out records, partial credit and examples)
* `audit_comp.{py,log,json}` (leak check + fair references), `heldout_nll.{py,json}`,
  `context_diag.{py,json}`, `inspect_vocab.py`, `smoke_comp.py` (brute-force validation),
  `cpu_sanity.py`, `cpu_partial_eval.py`, `fixup_eval_json.py`, `fixup2_train_ref.py`,
  `queue*.sh`, and all run logs.
* `/workspace/scratch_lm_run/`, `/workspace/sf_data` (old 9114/130 revision, `streams.pt`
  300,653,741 bytes, mtime Oct 1 07:19) and the original
  `/workspace/hf_v19_2_release/scratch_lm.py` (md5 `71db165f8f9d09f8725402834b793a3f`,
  re-verified) are **unmodified**; the old dataset must be labelled as the older revision if
  reused. `/workspace/sf_morph.json` is a shared path that other agents write to — it was only
  ever **read** by this work, never written (all writes went to `/workspace/scratch_comp/`).
* No log contains a traceback or an error; the only `NaN` string anywhere is the literal
  `0 NaN-lines` line printed by the queue's own check.
