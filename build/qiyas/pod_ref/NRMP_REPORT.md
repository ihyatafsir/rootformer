# Rootformer NRMP — Make It Work

**Scope:** get Next-Root/Morph Prediction (NRMP) actually running, retrained, and governed by
the project's own Khalīl / Basran / Andalusian algorithms — on the RunPod RTX PRO 4500 pod.

**Pod:** `r42pwkqp0uplbw` · NVIDIA RTX PRO 4500 Blackwell (32,623 MiB, CC 12.0) · driver 580.173.02 ·
PyTorch 2.8.0+cu128 · volume `/workspace` (50 GB quota) · container disk `/` (50 GB).
SSH key injected via `PATCH /pods/{id}` (`env.PUBLIC_KEY`), pod restarted, target now
`213.173.107.14:22231`. Your original `absolut7@…` keys were preserved.

---

## 1. What was actually broken

The v19.2 release (`enver/rootformer-v19.2-sovereign-synthesis`) shipped the NRMP *architecture*
but never exercised it. Three independent defects:

**(a) The trained heads were never called.** `transmute_quickstart.py:85-88` runs one encoder pass,
discards the hidden states, and calls `MasterSovereignTransmuter.transmute_proposition(all_hs, text)`.
That method's `hidden_states` parameter is **never read** — its output is a pure function of the input
string plus the `SURFACE_SCHOLASTIC_LEXICON` / `COMPREHENSIVE_SCHOLASTIC_MAP` dicts and sentence-specific
regexes (`khalil_students_andalusian_master_engine.py:211-252, 254-272, 328-332, 365-368, 371-385, 398-408`).
`generate_words()` — the real autoregressive NRMP decoder (`rootformer_v18_nrmp_model.py:261-375`) —
is never invoked anywhere in the release.

**Verified, not inferred.** `transmute_control.py` calls the shipped engine with (i) real hidden
states, (ii) all-zero hidden states of identical shape, (iii) `hidden_states=None`, and (iv) hidden
states from a *different* sentence. For all six inputs the four English strings are **byte-identical**:

```
العلم نور يضيء العقل ويهدي إلى الحق
  all four -> "the knowledge is light illuminates the intellect and guides to truth"
القط يشرب الحليب في الصباح            (out-of-distribution)
  all four -> "the قطط is شرب حلب in صبح"
تويتر منصة اجتماعية حديثة              (out-of-distribution)
  all four -> "<PARTICLE is <PARTICLE جمع حدث"
```

The three fluent rows are exactly the README benchmark rows — they are memorised strings served by
sentence-specific regexes. OOD input degrades to raw Arabic and literal `<PARTICLE>` leakage.

**(b) The classical grammar engines were not importable, so they silently became `None`.**
`rootformer_v18_nrmp_model.py:158-169` guards `farahidian_transmutation_engine` and
`sibawayh_governance_engine` in bare `try/except` and sets the attributes to `None`. Every advertised
Sībawayh exclusion mask was therefore dead code. (After staging the modules all three construct
correctly — see §3.)

**(c) Decoder defects** in the shipped `generate_words()`:
- `<PARTICLE>` (root id 4) is never masked → can emit the literal string `<PARTICLE>`;
- wazn ids **3 `end`** and **4 `start`** are control strings that leaked into the wazn vocabulary and
  are treated as valid morphological patterns;
- the prefix head can emit particles as if they were prefixes (`الفي`, `بفي`);
- the suffix head can attach `ة` to particles (`علىة`, `منة`);
- pure greedy argmax with no repetition control → collapse into `في قلاه في قلاه في قلاه …`.

Not a defect: **checkpoint integrity is perfect.** Load report, all 568 tensors:

```
NRMP tensors found in checkpoint : 12
... loaded with matching shapes : 12
... shape mismatches            : 0
total missing keys              : 0     total unexpected keys: 0
```

`morphemic_embed.{root,wazn,prefix,suffix}_embed` and `nrmp_head.{root,wazn,prefix,suffix,cond_proj}`
all load cleanly and match the code's vocabulary exactly (9114 roots / 130 awzān / 26 prefixes /
22 suffixes). The v19.2 weights themselves are sound.

---

## 2. Retraining the NRMP heads

~30 % of analyzed tokens fall back to a catch-all (`<PARTICLE>` 19.3 % + `<UNK>` 4.8 %). Predicting
that constant inflates every naive metric, so both classes (and all `<P:…>` particles) are **masked
out of the root loss and out of the reported metric**.

- **Data:** 691,127 unique sentences from 6 sanitized corpora → **7.06 M train / 838 k val word events**,
  split **by source file** (no leakage).
- **Objective:** the model's own factorized next-word-event loss —
  `L_root + 0.5·L_wazn + 0.25·L_prefix + 0.25·L_suffix`.
- **Phase 1** (heads only, 14.45 M params, 6000 steps @ batch 32, lr 3e-4): plateau ≈ 9.6 % val acc@1.
- **Phase 2** (unfreeze upper 8 layers, 138.96 M params, lr 1e-4 / 5e-6): also plateau ≈ 9.5 % —
  so the ceiling is the task/data, not capacity.
- Best checkpoint = phase 1 (`/workspace/nrmp_trained_final.safetensors`, 793 MB).

### Held-out evaluation (150 sentences/corpus, batch=1, true radicals only)

| metric | andalusian (original → **trained**) | scholastic (original → **trained**) |
|---|---|---|
| root acc@1 | 1.86 % → **3.85 %** | 1.15 % → **6.11 %** |
| root acc@5 | 7.70 % → **9.81 %** | 6.20 % → **10.21 %** |
| root PPL | 1073 → **756** | 1156 → **653** |
| context gain (acc@1 vs shuffled context) | +1.86 pp → **+3.61 pp** | +0.77 pp → **+4.15 pp** |
| *reference:* unigram baseline | 3.35 % | 3.05 % |
| *reference:* oracle bigram | 38.63 % | 39.50 % |
| uniform over 9114 roots | 0.011 % | 0.011 % |

**Readings.**

1. The **shipped** v19.2 heads sat *below* a context-free unigram root-frequency predictor
   (1.86 % < 3.35 %; 1.15 % < 3.05 %). They were not usable for prediction.
2. The retrained heads **beat the unigram baseline** on both corpora and roughly **doubled/tripled
   the context contribution**, so they now genuinely condition on context.
3. **Large headroom remains:** an oracle bigram over the previous root reaches ~39 % vs the model's
   3.85–6.11 %. The root head is far from exploiting even first-order statistics. (The bigram is fit
   and scored on the same positions, so it is an upper bound for a first-order model, not a fair
   competitor — but the gap is the honest measure of remaining work.)
4. These numbers are **in-domain** (the corpora overlap v19.x training material); treat them as upper
   bounds, not held-out generalisation.

---

## 3. Classical grammar engines — wired in and measured

The engines already existed in `rootformer_v12/v18_next_root_morph/` and are now staged into the
release and driven by the decoder:

| Engine | Source algorithm | Role |
|---|---|---|
| `SibawayhNRMPGovernance` | Al-Khalīl ibn Aḥmad, *Kitāb al-ʿAyn* | 311 phonotactically impossible roots excluded (C₁=C₂, deep-guttural incompatibility, bare alif) |
| `SibawayhNRMPGovernance` | Sībawayh, *Al-Kitāb* — *Naẓariyyat al-ʿĀmil* | operator states `HARF_JARR / HARF_JAZM / HARF_NASB / INNA / KANA / FUTURE` drive hard exclusion masks on root, wazn, prefix, suffix heads |
| `IbnMalikVerbTransmuter` | Ibn Mālik, *Lāmiyyat al-Afʿāl* | pharyngeal / assimilated / hollow decision tree picks the imperfect wazn |
| `IbnMalikPOSAutomaton` | Ibn Mālik, *Al-Alfiyyah* | `كلامنا لفظ مفيد كاستقم واسم وفعل ثم حرف الكلم` — forbids Harf→Harf and Fiʿl→Fiʿl |
| `IbnMadaRealismFilter` | Ibn Maḍāʾ, *Kitāb al-Radd* | phantom-token elimination on the English actuator |
| `BasranSyntacticRealizer` | Baṣran surface realization | *al-Ṣarf* + *al-Naḥw* realisation via `decode_sentence(apply_syntax=True)` |

Active configuration at runtime: 311 Khalīl-forbidden roots, 67 particle roots, 81 nominal /
44 verbal awzān, 12 preposition prefixes.

### Ablation — same prompts, same weights, engines on vs off (42 generated words each)

| violation of the classical rules | **governed** | ablation |
|---|---|---|
| bare `<NONE>` wazn on a radical root | **0** | 8 |
| definite article on a verb (`اليعلم`) | **0** | 3 |
| Harf → Harf (Alfiyyah-illegal particle chain) | **0** | 7 |

Grammar firings in the governed run: `HARF_JARR`×5, `HARF_NASB`×3, `HARF_JAZM`×3, Ibn Mālik wazn
applied ×6, Sībawayh suffix masks ×15, Alfiyyah blocks Harf→Ṣifah ×32 and Fiʿl→Fiʿl ×2.
Degeneration is also gone: stutter 0.024, 13.7/14 distinct roots, 0 literal-token leaks.

---

## 4. Deliverables

| File | Purpose |
|---|---|
| `nrmp_run.py` | loads v19.2 + reports load fidelity (no silent `strict=False`); evaluates next-root acc@1/@5, radical-only + all-class, shuffled-context control, unigram/bigram references |
| `nrmp_train.py` | `--prepare` builds the file-split dataset; `--train` trains heads, `--unfreeze-last N` fine-tunes the upper backbone |
| `nrmp_generate.py` | NRMP decoder governed by the classical engines; `--no-grammar` gives the ablation |
| `transmute_control.py` | falsification test: proves the shipped transmutation ignores the model |
| `/workspace/nrmp_trained_final.safetensors` | retrained NRMP checkpoint (793 MB) |
| `results/*.json` | raw evaluation + generation records |

```bash
cd /workspace/hf_v19_2_release
export HF_HOME=/workspace/.hf_home
PY=/workspace/venvs/rootformer/bin/python          # venv, reuses system torch 2.8+cu128

$PY nrmp_run.py --checkpoint /workspace/nrmp_trained_final.safetensors --eval
$PY nrmp_generate.py "العلم نور يضيء العقل ويهدي إلى الحق"
$PY nrmp_generate.py --no-grammar "…"               # ablation
$PY nrmp_train.py --train --steps 6000 --batch-size 32
```

---

## 5. Environment work performed

- Installed `transformers / safetensors / huggingface_hub / accelerate` into
  `/workspace/venvs/rootformer` — the system interpreter is PEP-668 externally managed, which is
  why `pip install` had silently failed (`huggingface_hub` was absent).
- Cached `Qwen/Qwen2.5-0.5B` (config only is needed; weights come from the checkpoint).
- **`/workspace` hit its 50 GB quota** — this is what crashed the first checkpoint save
  (`Disk quota exceeded`). With your authorisation I deleted the **68 intermediate
  `*_step*.safetensors`** files (21.2 GB) across `v16_basran_training`, `v17_deepseek_flash` and
  `v18_next_root_morph`. **All 26 `*_master*` / final checkpoints were preserved.** Free space 22 GB.

---

## 6. The analyzer gap is structural, not a missing rule

Because ~21–30 % of tokens carry no root, I built `rootformer_analyzer_v2.py`: a strict fallback that
strips candidate affixes, reduces the residual stem (drop weak radicals, collapse gemination,
normalise hamza) and matches against the blueprint roots — a **weak-normalised index** so that
`تعالى`→`علو`, `معنى`→`عني`, `قيل`→`قول`, `صلى`→`صلو` resolve. It is consulted *only* when the original
analyzer returns no root, and it may only return roots **already in `roots_set`** (adding roots would
invalidate the checkpoint's fixed 9114-way `root_embed`/`root_head`).

Measured on 20,000 tokens (6 corpora):

| | value |
|---|---|
| junk-root rate before | 21.12 % |
| junk-root rate after | 20.75 % |
| words the base analyzer dropped | 4,224 |
| loose fallback hits on those | 1,047 |
| …of which reconstruct the surface word exactly | 118 (**11.3 %** precision) |
| strict fallback hits (reconstruction enforced) | 123 (**2.9 % of failures repaired**) |

**Conclusion: affix-stripping heuristics repair only ~3 % of the gap.** The failures are dominated by
*muʿtall* (weak-radical) morphology and by words whose roots are simply **not in the blueprint**
(proper nouns, rare roots, and particles such as `يا` / `لو` / `أما` / `ذا`). Since the root inventory
is frozen by the checkpoint, this cannot be fixed by decoding tricks. The two real options are:

1. accept ~21 % catch-all rate and make the model robust to it (masked-target training, as done here), or
2. **extend the root inventory and retrain `morphemic_embed` + `nrmp_head` from scratch** — a
   vocabulary-breaking change worth doing deliberately, not incidentally.

The recoveries that *are* produced are correct, e.g. `قالوا`→قال+وا, `كانوا`→كان+وا, `شيئا`→شيء+ا,
`الفرزدق`→ال+فرزدق, `الإله`→ال+ءله.

---

## 7. Honest limitations

1. **Analyzer coverage is the hard ceiling** (~21–30 % unanalysed catch-alls), and §6 shows it is
   *not* cheaply fixable — see the two strategic options there.
2. **Root prediction is weak in absolute terms** — 3.85 %/6.11 % acc@1 against a ~39 % oracle bigram.
3. **The English transmutation path is still the hardcoded dictionary.** It was not repaired here;
   the neural English decoder (`neural_transmuter_head.py`) remains unexercised, and
   `root_prior_bias` is never loaded.
4. **Morphophonology is incomplete.** `realize_root_and_wazn` is an `if/elif` chain; hollow,
   defective and hamzated roots are partly wrong (e.g. سبب+فَاعِل+ة → `سابة`). The decoder repairs
   hamza+alif (`أا`→`آ`) and enforces wazn/affix agreement, but full *al-Ṣarf* is not implemented.
5. **Evaluation is in-domain**, and the bigram reference is an oracle fit on the eval set.
6. Two root vocabularies coexist: 9015 (per-attention `IshtiqaqAttention` + layer-14
   `morphemic_heads`) and 9114 (the NRMP head). Both are present in the checkpoint; they are
   different spaces and should not be conflated.
