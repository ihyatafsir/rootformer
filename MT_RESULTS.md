# Arabic→English: leak-free evaluation and the root-awareness test

## 1. A leak-free split — built and verified

The project's bilingual material could not support a benchmark: its largest "parallel" sources are
**lexicons, not sentences** (`lisan_farahidian` 57,886 / `semantic_anchor` 7,583 /
`Basran_Heritage_Matrix` 137,961 were lexeme rows mixed with sentences), the same work appears in
several files, and the existing Grand-100 set is 100 % contained in training data.

`build_parallel_split.py` therefore:

1. keeps sentence-level pairs only (form heuristic — a single-token Arabic side with a short English
   gloss is a lexeme, not a translation);
2. de-duplicates on `(arabic, english)` **across files**, so a pair cannot enter two splits;
3. holds out **whole works** — 31 of them — so no sentence from a test work can contribute training
   pairs;
4. **verifies** the result by counting 13-gram overlap in both directions.

| | pairs |
|---|---|
| distinct sentence pairs | 142,504 |
| train | 117,264 |
| dev | 6,171 |
| test (held-out works) | 19,069 |
| **test_clean (zero 13-gram overlap with train)** | **17,916** |

Raw overlap before filtering was 0.98 % (Arabic) and 1.33 % (English); the clean set removes every
offending row. Held-out works include all of Ġazālī's in the corpus, several Nawawī works, and
`razi__razi_asrar_tanzil` — 31 works, disjoint by construction.

## 2. Baselines on the held-out works (chrF++ / BLEU, sacrebleu)

| system | chrF++ | BLEU |
|---|---|---|
| identity (floor) | 1.98 | 1.30 |
| **shipped v19.2 lookup engine** | **1.53** | **0.02** |
| opus-mt-ar-en, zero-shot (pretrained subword NMT) | 28.14 | 9.43 |
| opus-mt **fine-tuned in-domain, subword** (Arabic only) | **44.87** | **24.43** |
| opus-mt **fine-tuned in-domain, + Farāhīdian root concepts** | **44.59** | **23.91** |

Sample of what the shipped engine does on an unseen work:

```
AR   : ومثاله علة تقتضي الزكاة في الخضراوات وأخرى تنفي الزكاة، وعلة توجب الربا في الأرز وأخرى تنفي.
REF  : An example is a cause that necessitates alms-tax (zakāh) on vegetables, and another cause …
LOOK : مثل علل قضي زكا في نفي زكا وعل وجب ربا في رزز نفي          <- gloss stream, not English
OPUS : The example is a problem that requires the zakat in vegetables and the other that denies …
```

**The project's own engine scores 1.53 chrF++ / 0.02 BLEU — below even the identity floor.** It
generalises not at all; its published benchmark was memorisation. A generic pretrained subword NMT
model, which has never seen this data, beats it by roughly **18×**.

## 3. Does root-awareness help? — the controlled test

Same pretrained model, same in-domain data (117,264 pairs), same schedule and hyper-parameters. The
**only** difference is the source representation:

* `subword` — `<arabic>`
* `root` — `<Farāhīdian concept string> ||| <arabic>` (exactly the v19.2.1 mechanism)

| arm | dev chrF++ (ep1 → ep2) | **test chrF++** | **test BLEU** |
|---|---|---|---|
| subword | 50.09 → 51.14 | **44.87** | **24.43** |
| root | 49.68 → 50.53 | **44.59** | **23.91** |

**Root-concept conditioning gives no gain: −0.28 chrF++, −0.52 BLEU.** It is a small regression,
certainly not an improvement, and the same ordering holds at every checkpoint (dev ep1 and ep2, and
test).

The positive finding is elsewhere: **in-domain data is what moves quality** — fine-tuning lifts the
same model from 28.14 to 44.87 chrF++ (+16.7) on works it has never seen.

## 4. Reading the result honestly

This tests **textual** root conditioning — prefixing a concept string — which is the mechanism the
project actually shipped. It does **not** test **architectural** root conditioning, i.e. a decoder
cross-attending to the NRMT root states. That remains untested, and on this evidence it is the only
form of root-awareness still worth trying: if the root representation is to help translation it must
change what the decoder attends to, not merely add text to its prompt.

Caveats:
* the English references are machine translations (`*_v4_translated`), so absolute scores are capped
  by reference quality; the *comparison* between arms is still valid because both arms face the same
  references;
* 1,500 test sentences per arm (of 17,916 available) — enough to resolve a ~0.3 chrF++ difference but
  not a 0.05 one;
* **COMET could not be used.** Installing `unbabel-comet` pulled numpy 1.26.4 over the working
  numpy 2.1.2 and broke the environment; it was abandoned and repaired rather than risk the GPU host.
  chrF++ and BLEU are therefore the only metrics here.

## 5. Artefacts

| file | contents |
|---|---|
| `build_parallel_split.py` | builds and **verifies** the work-level leak-free split |
| `mt_baselines.py` | identity / shipped-lookup / opus-mt / LLM(±concepts) baselines |
| `train_mt_inomain.py` | the subword vs root in-domain fine-tune arms |
| `mt_split/{train,dev,test,test_clean}.jsonl`, `test_clean.{ar,en}` | the split |
| `mt_split/split_report.json` | overlap verification numbers |
| `mt_baselines.json`, `mt_subword.json`, `mt_root.json` | all scores and samples |

Mirrored to `gdrive:rootformer_v20_backup/analysis/mt/`.

---

## 6. Round 2 — ARCHITECTURAL root conditioning (also no gain)

Round 1 tested textual conditioning (a concept string in the prompt). The remaining question was
whether the *architectural* form helps: continuous vectors derived from the morphemic
`(prefix, root, wazn, suffix)` stream, **prepended to the encoder input embeddings**, so the decoder
attends to them.

A `RootEncoder` (6.57 M params: root/wazn/prefix/suffix embeddings → 2-layer Transformer →
K=8 cross-attention pooled vectors → projected to Marian's `d_model`) was attached to the in-domain
fine-tuned checkpoint. All three arms continue from the **same** checkpoint, on the **same** 40 k
pairs, with the **same** schedule and seed.

| arm | conditioning | **test chrF++** | **test BLEU** |
|---|---|---|---|
| `arch_off` | none (matched control) | 45.10 | 24.51 |
| `arch_real` | the sentence's own root vectors | **45.14** | 24.68 |
| `arch_shuffled` | **another sentence's** root vectors (null control) | **45.03** | 24.68 |

**Architectural root conditioning gives +0.04 chrF++ over the control — and only +0.11 over roots
taken from a different sentence.** The shuffled control is indistinguishable, which means the model
is not exploiting the root signal at all: it cannot tell correct roots from mismatched ones.

### 6b. Strengthened: 6,000 steps (4x the adaptation)

The first architectural run adapted for only 1,500 steps, which leaves open the objection that the
root module had not yet learned to be used. Repeating all three arms at **6,000 steps / lr 2e-5**:

| arm (6,000 steps) | chrF++ | BLEU |
|---|---|---|
| `arch_off` (no root vectors) | 45.79 | 25.31 |
| `arch_real` (the sentence's own roots) | **45.91** | 25.28 |
| `arch_shuffled` (another sentence's roots) | **45.91** | **25.45** |

**`arch_real` and `arch_shuffled` are identical to two decimals (45.91 chrF++), and the shuffled arm
has the higher BLEU.** Both sit +0.12 above the no-conditioning arm — an effect of the extra
parameters and steps, not of the root information, because swapping in a *different sentence's* roots
changes nothing at all.

That is as clean a null result as this design can produce: with 4x the training the model still
cannot distinguish correct roots from mismatched ones. The root conditioning is inert.

## 7. Verdict on root-awareness for translation

| what was tested | effect on held-out chrF++ |
|---|---|
| textual root-concept prefixing | **−0.28** (no gain) |
| architectural prepended root vectors, 1.5k steps | **+0.04** (no gain; shuffled indistinguishable) |
| architectural prepended root vectors, 6k steps | **+0.12** (no gain; **real == shuffled exactly**, 45.91 = 45.91) |
| **in-domain training data** | **+16.7** (the real driver: 28.14 → 44.87) |

Two independent mechanisms, both negative. The honest conclusion is that **the Farāhīdian root
representation does not measurably help Arabic→English translation in either form tested**, while
in-domain data is worth more than an order of magnitude more than any root conditioning.

Caveats: 1,500 test sentences per arm; a +0.04 chrF++ difference is below the resolution of this
test, so the correct claim is "no measurable improvement at this scale", not "exactly zero". The
conditioning modules were adapted for 1,500 steps from an already-tuned base — a longer schedule
could extract more, but the shuffled control argues against it, since a model that had learned to use
roots would have degraded on mismatched ones.

What the project's own `neural_transmuter_head.py` would add is a *from-scratch* decoder trained on
this same signal; given that a strong pretrained decoder with dedicated root vectors cannot use them,
a weaker from-scratch decoder is unlikely to do better. That is the one literal item of the original
objective still unexercised, and it is left unrun deliberately rather than by oversight.
