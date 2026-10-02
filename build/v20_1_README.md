---
language:
- ar
- en
license: apache-2.0
tags:
- rootformer
- nrmt
- nrmp
- next-root-morph-token
- classical-arabic
- morphology
- farahidi
- sibawayh
- ibn-malik
- basran
- andalusian-grammar
- grammar-constrained-decoding
library_name: transformers
pipeline_tag: text-generation
metrics:
- accuracy
- perplexity
---

# Rootformer v20.1 — NRMT: an architecture fitted for Next-Root-Morph-Token prediction

v20.1 is the **architecture** release. It changes the model so that predicting the next root is a
first-class autoregressive act instead of a byproduct of a hidden state, and it measures the change
against a matched control.

Supersedes v20 (weights-only retrain). Weights: the NRMT heads here (`nrmt_head_full.pt`) sit on the
v20 backbone checkpoint, which is unchanged and remains in
[`enver/rootformer-v20`](https://huggingface.co/enver/rootformer-v20).

---

## The problem this release fixes

The shipped head computed `root_logits = W · h_t` — a single linear map from the final hidden state.
But next-root prediction is mostly a function of the **recent root stream**: a count-based 4-gram over
roots reaches ≈ 33 % acc@1 where the head reached ≈ 7.5 %. The root history was never an explicit
input. Six defects were identified and fixed:

| # | Defect | Fix |
|---|---|---|
| **A** | Root stream only *implicit* in `h_t` | `h' = h + W_h[E(r₋₁);E(r₋₂);E(r₋₃)]`, **zero-initialised** so an existing checkpoint warm-starts numerically unchanged, then learns the term |
| **B** | The engram "radical root memory" was addressed by **sequence position**, never by root: `RootformerV18_NRMP` bypassed the only code that sets `current_root_ids`, so `DeepSeekEngramModule` fell back to `torch.arange(seq_len)` | Keep an unregistered reference to the flash wrapper and set `current_root_ids`/`current_wazn_ids` from the morphemic streams before every backbone call |
| **C** | `cond_proj` conditioned on the **gold** root in training but the **argmax** at inference — with root acc@1 ≈ 7 %, the affix heads were trained on gold and deployed on wrong roots | Scheduled sampling (`ss_prob`) |
| **D** | *Al-Taqālīb*: permutation-equivalent roots were independent one-hot ids with zero parameter sharing | Orbit-consistency loss over the **7,727** S³ permutation pairs in the inventory |
| **E** | Sībawayh's operator (*al-ʿāmil*) was only a decoding mask | The head now **reads** the operator state as a feature (31 operator-bearing roots: JARR×10, INNA×9, KANA×5, NASB×3, JAZM×2, FUTURE×2) |
| **F** | The previous morphological tuple was not an input, and the trainer could silently pass the *current* tuple | The head rolls history and the previous `(wazn, prefix, suffix)` **internally** |

Two further bugs were caught while validating: assigning the flash wrapper with `self.flash = …`
silently **registered it as a submodule**, duplicating every flash parameter (581 phantom keys) and
handing them gradients — fixed, now exactly 4 new tensors; and an unnormalised additive feature term
drove held-out PPL to **3.0 × 10⁴**, fixed with LayerNorm + dropout.

---

## The ablation — does the architecture help?

Identical data (12,000 windows = 1.52 M positions), identical protocol and hyper-parameters.
`use_features=False` disables history + operator + morph conditioning and reproduces the plain
`W · h_t` head.

Metric: next-root **acc@1** on the file-level held-out stream, on ALL positions and on
**context-novel** positions (12-root context never seen in train: **77.9 %** of val positions).

| step | control (h-only) ALL / NOVEL | **full NRMT** ALL / NOVEL |
|---|---|---|
| 2000 | 5.83 / 5.67 | **6.60 / 6.38** |
| 6000 | 6.53 / 6.39 | **7.17 / 6.90** |
| 12000 | 6.85 / 6.65 | **7.40 / 7.08** |
| 16000 | *plateaued* | **7.43 / 7.12** |
| 20000 | *plateaued* | **7.57 / 7.25** |

**The architecture wins at every matched step (+0.5 to +0.8 pp), on both ALL and novel contexts, and
was still improving when the control had plateaued.** Root train loss at 20 k steps: 4.13 vs 5.12.

### Honest caveats

1. The gain is real but **modest**, not transformative.
2. **PPL is worse for the full model** (1296 vs 808) and rises with the feature norm (0 → 39.5). It is
   over-confident and needs label smoothing / temperature before its probabilities are trusted. Rank
   metrics (acc@1/@5) are the reliable signal here.
3. **A count-based interpolated n-gram still beats both** on novel contexts: **12.1 %** vs 7.25 %.
   The learned head still extracts less transition structure than a lookup table. The next step is to
   feed the n-gram in as features rather than compete with it.
4. Not implemented: Sībawayh's valency/constituent stack (*Inqiṭāʿ al-ʿAmal*), *al-Iktifāʾ* clitic
   saturation, full jussive apocope in surface realisation, *Marātib al-Maʿārif* at decode time.

---

## Correction notice — please read before citing earlier numbers

**The v19.2 "transmutation" was a hardcoded dictionary + sentence regexes, not model output.**
Verified: the engine returns **byte-identical** text under real hidden states, zero hidden states,
`None`, and hidden states from a *different* sentence.

**The Grand-100 LaBSE score of 0.9037 is memorisation, not translation.** All **100/100** Arabic
sentences *and* **100/100** English references occur verbatim in the training corpora; the "unseen"
half is not unseen. See `BENCHMARK_CONTAMINATION.md`.

**One bilingual corpus is randomly paired.** An independent-MT agreement audit with a shuffled-pair
null gives: `pure_gold` +17, `sovereign_classical_transmute` +21, `unified_basran_andalusian` +20
(genuinely aligned) but **`grand_scholastic_bilingual` +4.7 — effectively random pairs**. The
`mujtahid_*` and `grand_scholastic_train` sets are Arabic-only (100 % empty English).

---

## Classical algorithms — correctness audit

Ten of twenty implementation checks failed against the primary texts and are now fixed. All primary
sources are bundled under `corpus/`.

| Algorithm | Source | Finding |
|---|---|---|
| Al-Khalīl phonotactics | *Kitāb al-ʿAyn* | The verbatim pair rule «القاف والكاف لا يجتمعان في كلمة واحدة … وكذلك الجيم مع القاف» was **missing**; the inventory violates it once (`جقق`). Also, every bare-alif root was banned — wrongly killing **249 real roots** (`اصل`, `ارض`, `اخذ`, `احد`, `اسس`), 2.7 % of the root space |
| Sībawayh operators | *Al-Kitāb* | His list is «لم، ولما، **واللام التي في الأمر**، ولا في النهي». `إن` was wrongly in it; **`لام الأمر` was missing entirely** |
| Sībawayh government | *Al-Kitāb* | Was a flat 1-step memory: `في الأجسام الشفافة` → `[JARR, NONE, NONE]`. Now `[NONE, JARR, JARR]` |
| Ibn Mālik *Alfiyyah* | v. 55 | Six definiteness ranks «كهم وذي … وهند وابني والغلام والذي»; we had four (missing الموصول, المضاف) |
| Ibn Mālik POS automaton | *Alfiyyah* | Blocked Fiʿl→Fiʿl unconditionally, so valid `قام وقعد` was rejected — and it **contradicted the wāw classifier in the same codebase** |
| Ibn Mālik *Lāmiyyat al-Afʿāl* | the verses | Already **correct** (v. 42, 43, 45, 47-48, 51, 60 all verified) |

---

## Files

| file | contents |
|---|---|
| `nrmt_arch.py` | the NRMT head + `RootformerNRMT`; fixes A–F; `build_operator_table` |
| `nrmt_train.py` | trainer with leak-free context-novel evaluation and the `--no-features` ablation switch |
| `nrmt_head_full.pt` | the trained NRMT head (12.38 M params, float32) |
| `nrmt_head_control.pt` | the matched h-only control head |
| `nrmt_full.json`, `nrmt_control.json`, `nrmt_ablation.log` | complete metric history for both arms |
| `classical_governance_v2.py` | corrected Al-Khalīl / Sībawayh / Ibn Mālik / Al-Shātibī constraints |
| `test_grammar_impl.py`, `verify_v2.py` | the 20-check correctness audits (original vs corrected) |
| `probe_layers.py` | per-layer root probe + weight sanity sweep |
| `nrmp_governed_eval.py`, `nrmp_eval_clean.py`, `ngram_baseline.py` | constrained evaluation, memorisation-controlled evaluation, n-gram references |
| `ARCHITECTURE_FIX.md`, `CLASSICAL_SOURCE_VERIFICATION.md`, `BENCHMARK_CONTAMINATION.md`, `NRMP_REPORT.md` | the full write-ups |
| `corpus/basran/`, `corpus/andalusian/` | the primary grammatical texts the audit was made against |

## Usage

```python
from nrmt_arch import RootformerNRMT            # + the v20 backbone
# load checkpoints/rootformer_v20_nrmp_master.safetensors from enver/rootformer-v20,
# then: model.nrmt_head.load_state_dict(torch.load("nrmt_head_full.pt"))
```

```bash
python nrmt_train.py --checkpoint <v20 backbone> --train-windows 12000 --steps 20000   # full
python nrmt_train.py ... --no-features                                                 # control
python test_grammar_impl.py     # 10/20 PASS on the original implementation
python verify_v2.py             # ALL PASS on the corrected implementation
```

## Citation

```bibtex
@software{rootformer_v20_1_2026,
  author = {Enver at Aynengine and the Farāhīdian Research Circle},
  title  = {Rootformer v20.1: An Architecture Fitted for Next-Root-Morph-Token Prediction},
  year   = {2026},
  url    = {https://huggingface.co/enver/rootformer-v20.1-nrmt}
}
```
