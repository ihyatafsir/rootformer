# Architecture fixes for NRMT — and what they bought

**Question asked:** is the neural-layer architecture OK, are the algorithms implemented correctly,
and does it need training on Arabic / bilingual data?

**Answer:** the architecture was *not* OK for Next-Root-Morph-Token prediction. It has now been
changed, and the change is the first one in this project that improves accuracy on genuinely
novel contexts.

---

## The four architectural defects, and the fixes

The pristine model files were left byte-identical (`rootformer_v18_nrmp_model.py` md5 `5d5ef264…`,
`deepseek_v4_1_flash_model.py` md5 `59aaf833…`). Fixes live in `nrmt_arch.py` /
`rootformer_v20_nrmp_model.py`, which subclass them.

| # | Defect | Fix |
|---|---|---|
| **A** | The head was `root_logits = W·h_t` — the root stream was only *implicit* in the hidden state. A 4-gram over roots reaches ~33 % acc@1 against its ~7.5 %. | `h'_t = h_t + W_h[E(r₋₁);E(r₋₂);E(r₋₃)]`, **zero-initialised** so an existing checkpoint warm-starts unchanged, then learns the term. |
| **B** | `RootformerV18_NRMP` set `self.backbone = base.model.backbone.model`, bypassing the only code that sets `current_root_ids`. The layer-1/layer-14 hooks therefore got `None` and `DeepSeekEngramModule` fell back to `torch.arange(seq_len)` — the "radical root memory" was addressed by **sequence position**, never by root. | Keep an unregistered reference to the flash wrapper (`object.__setattr__`) and set `current_root_ids`/`current_wazn_ids` from the morphemic streams before every backbone call. |
| **C** | `cond_proj` conditioned on the **gold** root in training but the **argmax** at inference. With root acc@1 ~7 %, the affix heads were trained on gold and deployed on mostly-wrong roots. | Scheduled sampling (`ss_prob`). |
| **D** | *Al-Taqālīb*: permutation-equivalent roots share one semantic core, but were independent one-hot ids with zero parameter sharing. | Orbit-consistency loss over the **7,727** S³ permutation pairs found in the inventory. |
| **E** | *new* — the ʿāmil was only ever a decoding mask. | The head now also **reads** Sībawayh's operator state as a feature (31 operator-bearing roots: JARR×10, INNA×9, KANA×5, NASB×3, JAZM×2, FUTURE×2), vectorised via a root→operator table. |
| **F** | *new* — the previous morphological tuple was not an input at all, and the trainer could silently pass the *current* tuple. | The head rolls history and the previous `(wazn, prefix, suffix)` **internally**, so callers cannot misalign it. |

Two bugs found while validating the fixes, both real:

* assigning the flash wrapper with `self.flash = …` **registered it as a submodule**, duplicating
  every flash parameter (581 spurious `missing` keys) and handing them gradients — fixed with
  `object.__setattr__`, now `missing = 4` (exactly the four new tensors).
* an unnormalised additive feature term blew held-out PPL up to **3.0 × 10⁴**; a `LayerNorm` on the
  feature projection plus dropout fixed it (PPL 735–800).

---

## The ablation: does the architecture actually help?

Identical data (12,000 train windows = 1.52 M positions), identical protocol, identical
hyper-parameters (lr 1e-3, dropout 0.1, batch 64, 20 k steps). `use_features=False` disables
history + operator + morph conditioning and reproduces the plain `W·h_t` head.

Metric: next-root **acc@1** on the file-level held-out stream, split into ALL positions and
**context-novel** positions (12-root context never seen in train: **77.9 %** of val positions).

| step | control (h-only) ALL / NOVEL | **full NRMT** ALL / NOVEL |
|---|---|---|
| 2000 | 5.83 / 5.67 | **6.60 / 6.38** |
| 4000 | 6.46 / 6.29 | **6.85 / 6.63** |
| 6000 | 6.53 / 6.39 | **7.17 / 6.90** |
| 12000 | 6.85 / 6.65 | **7.40 / 7.08** |
| 16000 | *plateaued* | **7.43 / 7.12** |
| 20000 | *plateaued* | **7.57 / 7.25** |

**Result: the architecture wins at every matched step, by +0.5 to +0.8 pp, on both ALL and NOVEL
positions — and it was still improving when the control had plateaued.** Root-train loss at 20 k
steps: 4.13 (full) vs 5.12 (control).

**Honest caveats.**

1. The gain is real but **modest**, not transformative.
2. **PPL is worse for the full model** (1296 vs 808) and rising with the feature norm (0 → 39.5).
   It is over-confident; it needs label smoothing / temperature / a lower LR on the feature
   projection before its probabilities can be trusted. Rank metrics (acc@1/@5) are the trustworthy
   signal here.
3. **A count-based interpolated n-gram still beats both** on novel contexts: **12.1 %** vs our
   7.25 %. The head extracts far less transition structure than a lookup table does.
4. Still unimplemented: Sībawayh's valency/constituent stack (*Inqiṭāʿ al-ʿAmal*), al-Iktifāʾ clitic
   saturation, full jussive apocope in surface realisation, and *Marātib al-Maʿārif* enforcement at
   decode time.

---

## What this implies for the translation goal

Root prediction is the substrate for root-aligned translation, and it is still weak. The clearest
next move is not more of the same training: it is to **stop competing with the n-gram and feed it
in**. The count-based transition distribution is available, cheap and better than the learned head;
exposing its log-probabilities as input features to the NRMT head (rather than tuning a post-hoc
fusion weight) is the obvious way to close the 7.25 % → 12.1 % gap, and then to scale.

Only after that is joint NRMT + bilingual training worth running — and it must use a
**contamination-free** split, because the existing Grand-100 benchmark is 100 % in training data and
the `grand_scholastic_bilingual` corpus is randomly paired.

---

## Artefacts

| file | contents |
|---|---|
| `nrmt_arch.py` | NRMT head + `RootformerNRMT` wrapper; all six fixes; `build_operator_table` |
| `nrmt_train.py` | trainer with leak-free context-novel evaluation and the `--no-features` ablation switch |
| `rootformer_v20_nrmp_model.py` | earlier patch-style variant of fixes A–D |
| `nrmt_head_control.pt`, `nrmt_head_full.pt` | the two trained heads (49 MB each) |
| `nrmt_control.json`, `nrmt_full.json`, `nrmt_ablation.log` | full metric history |

All of the above is mirrored to `gdrive:rootformer_v20_backup/analysis/` (125 objects, 1.98 GiB).

---

# CORRECTION (added after the governed re-evaluation)

**The claim above — that the NRMT architecture fixes improved next-root accuracy — does not survive
evaluation on genuinely held-out *books*.**

The ablation in this document was evaluated on **val windows drawn from the same file pool as
training** (a file-level split, with ~15% of val 13-grams also present in train). It was therefore an
easier test than it appeared. Re-running the governed tiered evaluation on **150 held-out sentences
from different works** (`scholastic_sanitized` + `andalusian_canon_sanitized`), with identical tiers
and identical masking code:

| tier | v20 head acc@1 | v20 acc@5 | **new NRMT acc@1** | **new acc@5** |
|---|---|---|---|---|
| T0 raw | 7.53% | 18.59% | **4.25%** | **10.26%** |
| T1 + Al-Khalīl | 7.54% | 18.60% | 4.25% | 10.26% |
| T2 + Sībawayh | 7.59% | 18.43% | 3.84% | 10.09% |
| T3 + register prior (top-1000) | **8.03%** | **19.51%** | 3.71% | 9.97% |

| tier | v20 PPL | **new PPL** |
|---|---|---|
| T0 raw | 302 | **357,579** |

**The new architecture is roughly half as good as the shipped v20 head on this harder test, and its
probabilities are catastrophically miscalibrated (PPL 3.6e5 vs 302).** On the same test, gating again
contributes approximately nothing to top-5 (new: 10.26% → 9.97%; v20: 18.59% → 19.51%, and the ~1 pp
that exists comes from the attested-register prior, not from the grammatical masks).

**Confound to state plainly:** the two rows do not use the identical backbone checkpoint. The v20 head
was evaluated on the v20 retrained model (`nrmp_trained_final.safetensors`); the new head on
`rootformer_v19_2_synthesis_ar_backbone.safetensors`. So this comparison conflates architecture with
backbone calibration and cannot by itself attribute the gap. What it *does* establish is that the
v20.1 head, as trained, is not better end-to-end on held-out books, and that gating does not lift it
toward 20%.

To isolate architecture from backbone, the new head must be retrained on hidden states extracted from
the *same* checkpoint the v20 head was evaluated on. That has not been done.

**Lesson:** the window-based ablation was self-consistent (the head was scored on the same cached
hidden states it trained on) but not predictive of held-out-book performance. Any future NRMT claim
must be reported on held-out works.
