# Prompt-test: from-scratch Arabic morphemic LM `sf_morph.pt`

**Checkpoint** `/workspace/scratch_lm_run/sf_morph.pt` — `scratch_lm.py --arm morph --steps 6000`, 32.72 M params, 25,000,003 training words, no Qwen transplant, no BPE, one position per word carrying `(prefix, root, wazn, suffix)`.

**Verdict up front:** the model never learned the task. It learned the *frequency marginal* of each head and produces a near-constant output regardless of context. Train accuracy and held-out accuracy are the same number, so this is **not overfitting**; and the number is barely above a context-free unigram, so it is **not generalisation** either. It is 6,000 steps of a 32.7 M-parameter transformer over a 9,114-way root vocabulary — the run is far too short to have an opinion. Details and the evidence below.

## 0. Harness fidelity

`repro_check.py` reimplements `heldout_root_eval.py`'s identity split on top of the harness primitives and reproduces **all six published numbers exactly**, including `n`:

| split | bucket | published | harness |
|---|---|---|---|
| test_gen | real_seen_root | 1.490% / 5.228%, n=19,126 | 1.490% / 5.228%, n=19,126 |
| test_gen | special | 41.471% / 68.348%, n=16,631 | 41.471% / 68.348%, n=16,631 |
| test_gen | heldout_root | 0.000% / 0.000%, n=0 | 0.000% / 0.000%, n=0 |
| test_deriv | real_seen_root | 1.468% / 4.916%, n=19,753 | 1.468% / 4.916%, n=19,753 |
| test_deriv | special | 39.202% / 66.607%, n=17,818 | 39.202% / 66.607%, n=17,818 |
| test_deriv | heldout_root | 0.000% / 0.000%, n=4,117 | 0.000% / 0.000%, n=4,117 |

Everything below uses `scratch_lm.root_accuracy`'s convention: **every** position is scored, inputs `x[:-1]`, targets `x[1:]`, `pad(...,0)`, and the model's own `(root, wazn, prefix, suffix)` return order. Context is truncated to 48 words; the model is causally masked with absolute position embeddings, so right-padding cannot change any real position's logits. Raw readable output comes from the project's own `nrmp_vocab.decode_word`, not a substitute realiser.

## 1. Train vs held-out accuracy, per head

| split | positions | prefix top1/top5 | root top1/top5 | wazn top1/top5 | suffix top1/top5 |
|---|---|---|---|---|---|
| train | 22,262,495 | 72.19% / 94.14% | 19.07% / 34.29% | 47.47% / 82.66% | 83.01% / 95.21% |
| train_seen | 824,456 | 72.20% / 94.17% | 19.19% / 34.65% | 47.63% / 82.80% | 83.07% / 95.26% |
| val | 17,330 | 72.54% / 94.46% | 20.79% / 34.81% | 46.15% / 81.47% | 82.91% / 94.91% |
| test_gen | 35,757 | 73.43% / 94.31% | 20.18% / 34.74% | 47.02% / 82.51% | 83.21% / 95.23% |
| test_deriv | 41,688 | 71.55% / 93.98% | 17.50% / 30.88% | 44.10% / 81.86% | 82.21% / 94.91% |

`train_seen` = the 94,237 distinct train sentences that `scratch_lm.train`'s sampler actually fed to the optimiser, replayed from `random.Random(0)` with the script's `bs=16, max_sent=64` — not a sample, not an estimate.

**There is no overfitting.** Root top-1 is 19.07% on all of train, 19.19% on the exact sentences seen, 20.79% on val, 20.18% on test_gen, 17.50% on test_deriv. The model's best in-sample performance is the same as its out-of-sample performance to within a point. It has not fitted the training set, so it cannot be overfitting it.

### Against a context-free unigram on the identical positions

| split | head | model top1 | unigram top1 | model top5 | unigram top5 |
|---|---|---|---|---|---|
| train | prefix | 72.19% | 72.23% | 94.14% | 92.61% |
| train | root | 19.07% | 18.34% | 34.29% | 29.79% |
| train | wazn | 47.47% | 47.46% | 82.66% | 81.11% |
| train | suffix | 83.01% | 83.75% | 95.21% | 94.31% |
| test_gen | prefix | 73.43% | 73.01% | 94.31% | 92.52% |
| test_gen | root | 20.18% | 18.91% | 34.74% | 30.03% |
| test_gen | wazn | 47.02% | 46.77% | 82.51% | 80.38% |
| test_gen | suffix | 83.21% | 83.75% | 95.23% | 94.32% |
| test_deriv | prefix | 71.55% | 71.47% | 93.98% | 92.55% |
| test_deriv | root | 17.50% | 16.58% | 30.88% | 26.99% |
| test_deriv | wazn | 44.10% | 42.66% | 81.86% | 78.95% |
| test_deriv | suffix | 82.21% | 82.12% | 94.91% | 93.48% |

The unigram's root top-5 is `['<PARTICLE>', '<UNK>', '<P:من>', '<P:في>', '<P:لا>']` — five *special* tokens, carrying 29.2% of all training positions.

**Most of the model is indistinguishable from that unigram.** `vs_unigram.py` looks up the train marginal probability of the model's own predictions:

| split | head | model top1 | mean marginal prob of its own top-1 | model top5 | mean marginal prob of its own top-5 |
|---|---|---|---|---|---|
| train | prefix | 72.19% | 70.99% | 94.14% | 90.83% |
| train | root | 19.07% | 16.21% | 34.29% | 26.31% |
| train | wazn | 47.47% | 42.73% | 82.66% | 78.93% |
| train | suffix | 83.01% | 83.08% | 95.21% | 93.37% |
| test_gen | prefix | 73.43% | 71.14% | 94.31% | 90.89% |
| test_gen | root | 20.18% | 16.33% | 34.74% | 26.40% |
| test_gen | wazn | 47.02% | 42.81% | 82.51% | 79.01% |
| test_gen | suffix | 83.21% | 83.08% | 95.23% | 93.39% |
| test_deriv | prefix | 71.55% | 71.15% | 93.98% | 90.99% |
| test_deriv | root | 17.50% | 16.42% | 30.88% | 26.44% |
| test_deriv | wazn | 44.10% | 42.71% | 81.86% | 79.07% |
| test_deriv | suffix | 82.21% | 83.08% | 94.91% | 93.43% |

The model's top-1 predictions are items whose *marginal* probability alone already accounts for ~equal accuracy. The suffix head is literally at the marginal (83.01% model vs 83.08% marginal mass of its own picks); prefix is 72.19% vs 70.99%.

The only head with real (if small) headroom over the unigram is **root**: +0.7 pt top-1 on train, +1.3 pt on test_gen, +0.9 pt on test_deriv, and +4.5/+4.7/+3.9 pt top-5. And `collapse_analysis.py` shows ~90% of those correct root predictions are `<PARTICLE>`.

## 2. What it actually predicts: collapse

| split | positions | distinct top-1 tuples emitted | distinct true tuples | most frequent tuple | top-5 tuples | adjacent-position argmax changes |
|---|---|---|---|---|---|---|
| train | 22,519 | 50 | 6,574 | 79.24% | 93.17% | 17.0% |
| test_gen | 23,111 | 48 | 5,281 | 80.14% | 94.01% | 15.8% |
| test_deriv | 26,629 | 45 | 5,812 | 80.25% | 94.25% | 14.4% |

The dominant output is the empty string, because `<PARTICLE>` (and `<UNK>`) realise to nothing — the encoder could not represent the word, so the tuple carries no surface:

| split | share | surface | tuple | root class |
|---|---|---|---|---|
| test_gen | 80.14% | `` | `<NONE>+<PARTICLE>+<NONE>+<NONE>` | SPECIAL |
| test_gen | 8.18% | `` | `<NONE>+<PARTICLE>+فَعَلَ+<NONE>` | SPECIAL |
| test_gen | 2.70% | `` | `<NONE>+<UNK>+<NONE>+<NONE>` | SPECIAL |
| test_gen | 1.51% | `` | `<NONE>+<PARTICLE>+يَفْعَلُ+<NONE>` | SPECIAL |
| test_gen | 1.47% | `أن` | `<NONE>+<P:أن>+<NONE>+<NONE>` | SPECIAL |
| test_gen | 0.74% | `ال` | `ال+<PARTICLE>+فَعَلَ+<NONE>` | SPECIAL |
| test_deriv | 80.25% | `` | `<NONE>+<PARTICLE>+<NONE>+<NONE>` | SPECIAL |
| test_deriv | 9.46% | `` | `<NONE>+<PARTICLE>+فَعَلَ+<NONE>` | SPECIAL |
| test_deriv | 2.38% | `` | `<NONE>+<UNK>+<NONE>+<NONE>` | SPECIAL |
| test_deriv | 1.09% | `` | `<NONE>+<PARTICLE>+يَفْعَلُ+<NONE>` | SPECIAL |
| test_deriv | 1.07% | `أن` | `<NONE>+<P:أن>+<NONE>+<NONE>` | SPECIAL |
| test_deriv | 0.80% | `ال` | `ال+<PARTICLE>+فَعَلَ+<NONE>` | SPECIAL |

**Eighty percent of every position on every split gets the same tuple.** The model emits 45-50 distinct top-1 tuples where the truth has 5,000-6,600. It does have a little context sensitivity — the adjacent-position root argmax changes 15-17% of the time — but it stays inside a handful of high-frequency special tokens.

## 3. The `special` bucket: legitimate, not a collapse

`special_root_ids` is ids 0-70: 5 control tokens plus the **66** `<P:...>` function words of `COMMON_PARTICLES` (`في`, `من`, `إلى`, `على`, `لا`, `أن`, `كان`, …). These are real closed-class vocabulary items that the project's encoder genuinely emits, not a dumping ground. Composition of the bucket as **targets**:

| root | share of specials | share of all positions |
|---|---|---|
| `<PARTICLE>` | 40.3% | 18.76% |
| `<UNK>` | 9.2% | 4.29% |
| `<P:في>` | 5.7% | 2.63% |
| `<P:من>` | 5.4% | 2.50% |
| `<P:لا>` | 3.9% | 1.83% |
| `<P:أن>` | 3.3% | 1.53% |
| `<P:على>` | 3.3% | 1.52% |
| `<P:ما>` | 2.7% | 1.24% |

`<PARTICLE>` is 40.3% of the bucket, and `<PARTICLE> + <UNK>` are 49.5%. On the root head the model scores **41.471%** on this bucket (test_gen). A context-free predictor that always answers `<PARTICLE>` scores **18.76%** overall; the train unigram scores **40.4%** *within the special bucket*. So the model's special-bucket accuracy is essentially the frequency of the specials themselves. `<UNK>` is a legitimate token here: the corpus genuinely contains words the analyser could not decompose, and the model is right to predict it 4.29% of the time. This is **not** a degenerate collapse; it is the model reproducing a skewed marginal.

## 4. Held-out roots: what it predicts instead

The 38 held-out roots (checkpoint ids and the live vocabulary's reading of each) received zero training sentences. They are mostly triconsonantal content roots, but note that the filter selected on *root-string shape*, not on word class, so the set also contains function-word material the analyser happened to file as a 3-consonant root — `اذا` (id 148), `انه` (319), `بده` (454), `وان` (8759), `موه` (7930):

```
148=اذا, 319=انه, 443=بدا, 452=بدل, 454=بده, 497=برر, 533=برك, 1017=ثان, 2037=حين, 2078=ختم, 2439=دبب, 2445=دبر, 2673=دفع, 3353=زاد, 3688=ستت, 3726=سحق, 3747=سدد, 3878=سقط, 4074=سير, 4133=شجر, 4598=صغر, 4773=ضبط, 4894=ضمن, 5064=طعع, 5773=عهد, 5872=غرض, 5947=غفل, 6058=فتر, 6430=قتل, 7078=كره, 7392=لحق, 7700=مدن, 7744=مره, 7907=مهم, 7930=موه, 8119=نسخ, 8759=وان, 8848=ودع
```

**`test_gen/heldout_root` n=0 is structural, not a bug to fix.** `test_gen` is *defined* as the test sentences containing **no** held-out root (`scratch_lm.prepare`: `if roots_of(s) & hold_roots: test_deriv.append(s) else: test_gen.append(s)`). Counting targets whose root is in the hold set inside `test_gen` is therefore identically zero — verified directly: 0 held-out-root tokens in `test_gen`, 0 in `train`, 4,602 in `test_deriv`. `heldout_root_eval.py` already reports `test_deriv/heldout_root`, and that arm is the salvageable one; the `test_gen` arm can never be non-empty and should be dropped from the report rather than "fixed".

For held-out-root targets the model predicts a special token, essentially always. Below are raw prompts from `test_deriv`; each target is a word from a held-out root.

```
--- test_deriv sentence 2830 (held-out-root targets)
RAW SENTENCE (test_deriv #2830, 7 words): نفست وأولعت بالشيء وسقط في
ENCODED TUPLES: [(1, 8232, 45, 13), (1, 4, 1, 1), (4, 9047, 8, 13), (9, 4477, 45, 1), (4, 3878, 45, 1), (1, 5, 1, 1), (1, 4, 1, 1)]
  [pos 4] CONTEXT: نفست وأولعت بالشيء
           TRUE NEXT: 'وسقط'  (4,3878,45,1) [و+سقط+فَعَلَ+<NONE>] root_tag=HELD-OUT
           PRED top1: ''  (1,4,1,1) [<NONE>+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL   miss (NOT in joint top5)
             prefix: '<NONE>':-0.40  'و':-2.34  'ال':-3.16  'ف':-3.16  'وال':-3.36  'ب':-3.54
             root  : '<PARTICLE>':-1.77  '<P:من>':-2.96  '<UNK>':-3.40  '<P:في>':-3.53  '<P:لا>':-3.90  '<P:هو>':-4.02
             wazn  : '<NONE>':-0.68  'فَعَلَ':-1.70  'فَاعِل':-3.24  'فَعِيل':-3.35  'فِعَال':-3.39  'يَفْعَلُ':-3.47
             suffix: '<NONE>':-0.21  'ا':-3.06  'ه':-3.40  'ة':-3.46  'ي':-4.58  'ها':-4.68
             joint top5: ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'و'[و+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ف'[ف+<PARTICLE>+<NONE>+<NONE>]<SPECIAL>
```
```
--- test_deriv sentence 3121 (held-out-root targets)
RAW SENTENCE (test_deriv #3121, 12 words): هذه الطاعات تماثل الصدقات في الأجور وسماها صدقة على طريق القبلة وتجنيس
ENCODED TUPLES: [(1, 35, 1, 1), (3, 5064, 42, 19), (1, 7649, 31, 1), (3, 4539, 45, 19), (1, 5, 1, 1), (3, 1562, 8, 1), (4, 3952, 45, 4), (1, 4539, 45, 20), (1, 8, 1, 1), (1, 5026, 55, 1), (3, 58, 90, 20), (4, 1517, 37, 1)]
  [pos 1] CONTEXT: هذه
           TRUE NEXT: 'الطاعات'  (3,5064,42,19) [ال+طعع+فَاعِل+ات] root_tag=HELD-OUT
           PRED top1: 'ال'  (3,4,1,1) [ال+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL   miss (NOT in joint top5)
             prefix: 'ال':-0.67  '<NONE>':-0.82  'و':-4.65  'ب':-5.43  'ل':-5.53  'ف':-5.58
             root  : '<PARTICLE>':-1.50  '<P:التي>':-3.92  'علم':-4.27  '<P:ما>':-4.70  'شيء':-4.72  '<P:من>':-4.80
             wazn  : '<NONE>':-1.17  'فَعَلَ':-1.43  'أَفْعَال':-2.30  'فِعَال':-2.88  'مَفَاعِل':-2.90  'فُعُول':-3.16
             suffix: '<NONE>':-0.55  'ة':-1.38  'ات':-3.00  'ية':-3.20  'ها':-3.77  'ين':-4.64
             joint top5: 'ال'[ال+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'الة'[ال+<PARTICLE>+<NONE>+ة]<SPECIAL>
```
```
--- test_deriv sentence 2755 (held-out-root targets)
RAW SENTENCE (test_deriv #2755, 14 words): حرب حدثنا بن حدثنا سعيد هو في جميع نسخ بلادنا في
ENCODED TUPLES: [(1, 1702, 45, 1), (1, 1676, 45, 11), (1, 4, 1, 1), (1, 744, 45, 1), (1, 4, 1, 1), (1, 1676, 45, 11), (1, 3829, 55, 1), (1, 4, 1, 1), (1, 26, 1, 1), (1, 5, 1, 1), (1, 1488, 55, 1), (1, 8119, 45, 1), (1, 691, 84, 11), (1, 5, 1, 1)]
  [pos 11] CONTEXT: حرب حدثنا بن حدثنا سعيد هو في جميع
           TRUE NEXT: 'نسخ'  (1,8119,45,1) [<NONE>+نسخ+فَعَلَ+<NONE>] root_tag=HELD-OUT
           PRED top1: ''  (1,4,45,1) [<NONE>+<PARTICLE>+فَعَلَ+<NONE>] root_tag=SPECIAL   miss (NOT in joint top5)
             prefix: '<NONE>':-0.34  'ال':-1.48  'و':-3.99  'أن':-4.32  'ل':-5.50  'ف':-5.67
             root  : '<PARTICLE>':-1.85  'بيي':-2.63  'ابن':-3.18  'اله':-3.41  'بنن':-3.73  '<P:ما>':-3.90
             wazn  : 'فَعَلَ':-1.19  '<NONE>':-1.21  'فِعَال':-2.71  'يَفْعُلُ':-3.11  'فَاعِل':-3.14  'فَعِيل':-3.28
             suffix: '<NONE>':-0.24  'ة':-2.92  'ي':-3.35  'ه':-3.39  'هم':-4.18  'ين':-4.19
             joint top5: ''[<NONE>+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ة'[<NONE>+<PARTICLE>+فَعَلَ+ة]<SPECIAL>
```
```
--- test_deriv sentence 3226 (held-out-root targets)
RAW SENTENCE (test_deriv #3226, 13 words): العالم غير بشيء من الأوقات البتة بل هو ثابت زللا وأبدأ
ENCODED TUPLES: [(3, 5660, 42, 1), (1, 52, 1, 1), (1, 4, 1, 1), (8, 4477, 45, 1), (1, 6, 1, 1), (3, 9002, 6, 1), (3, 376, 45, 20), (1, 4, 1, 1), (1, 45, 1, 1), (1, 26, 1, 1), (1, 1021, 42, 1), (1, 3535, 120, 15), (4, 443, 8, 1)]
  [pos 12] CONTEXT: العالم غير بشيء من الأوقات البتة بل هو ثابت زللا
           TRUE NEXT: 'وأبدأ'  (4,443,8,1) [و+بدا+أَفْعَلَ+<NONE>] root_tag=HELD-OUT
           PRED top1: ''  (1,4,1,1) [<NONE>+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL   miss (NOT in joint top5)
             prefix: '<NONE>':-0.37  'و':-2.62  'ب':-3.01  'ل':-3.34  'ال':-3.39  'ف':-3.40
             root  : '<PARTICLE>':-1.92  '<UNK>':-2.76  '<P:من>':-3.03  '<P:في>':-3.07  '<P:لا>':-3.62  '<P:على>':-3.70
             wazn  : '<NONE>':-0.53  'فَعَلَ':-1.78  'فَاعِل':-3.70  'فَعِيل':-3.72  'فِعَال':-3.72  'يَفْعَلُ':-3.80
             suffix: '<NONE>':-0.19  'ة':-3.20  'ا':-3.30  'ه':-3.68  'ها':-4.32  'ك':-4.86
             joint top5: ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'و'[و+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'ب'[ب+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ل'[ل+<PARTICLE>+<NONE>+<NONE>]<SPECIAL>
```
```
--- test_deriv sentence 2997 (held-out-root targets)
RAW SENTENCE (test_deriv #2997, 13 words): زاد على الثلاثة منها وكما يشمل الحكم ما لم
ENCODED TUPLES: [(1, 4, 1, 1), (1, 4, 1, 1), (1, 4, 1, 1), (1, 3353, 45, 1), (1, 8, 1, 1), (3, 1112, 84, 20), (16, 8375, 1, 1), (1, 4, 1, 1), (1, 9035, 45, 15), (1, 4391, 118, 1), (3, 1891, 45, 1), (1, 21, 1, 1), (1, 22, 1, 1)]
  [pos 3] CONTEXT: 
           TRUE NEXT: 'زاد'  (1,3353,45,1) [<NONE>+زاد+فَعَلَ+<NONE>] root_tag=HELD-OUT
           PRED top1: ''  (1,4,1,1) [<NONE>+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL   miss (NOT in joint top5)
             prefix: '<NONE>':-0.33  'ال':-2.29  'و':-3.03  'ف':-3.76  'ب':-3.89  'وال':-3.93
             root  : '<PARTICLE>':-1.44  '<UNK>':-3.26  '<P:من>':-3.82  '<P:في>':-3.87  '<P:على>':-4.13  '<P:لا>':-4.29
             wazn  : '<NONE>':-0.70  'فَعَلَ':-1.50  'فَاعِل':-3.35  'فِعَال':-3.40  'فَعِيل':-3.54  'فُعُول':-3.98
             suffix: '<NONE>':-0.20  'ة':-3.08  'ه':-3.74  'ا':-3.82  'ت':-4.18  'ي':-4.51
             joint top5: ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'و'[و+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ة'[<NONE>+<PARTICLE>+<NONE>+ة]<SPECIAL>
```
```
--- test_deriv sentence 1167 (held-out-root targets)
RAW SENTENCE (test_deriv #1167, 13 words): وثم مذهب ثالث أن العامل هو المشتمل على البدل أن العامل
ENCODED TUPLES: [(1, 8800, 45, 1), (1, 2965, 92, 1), (1, 1112, 42, 1), (1, 15, 1, 1), (3, 5692, 42, 1), (1, 26, 1, 1), (3, 4391, 109, 1), (1, 8, 1, 1), (3, 452, 45, 1), (1, 4, 1, 1), (1, 15, 1, 1), (1, 4, 1, 1), (3, 5692, 42, 1)]
  [pos 8] CONTEXT: وثم مذهب ثالث أن العامل هو المشتمل على
           TRUE NEXT: 'البدل'  (3,452,45,1) [ال+بدل+فَعَلَ+<NONE>] root_tag=HELD-OUT
           PRED top1: ''  (1,4,1,1) [<NONE>+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL   miss (NOT in joint top5)
             prefix: '<NONE>':-0.33  'ال':-1.39  'أن':-3.86  'ب':-6.76  'بال':-6.92  'من':-7.01
             root  : '<PARTICLE>':-1.85  '<P:ما>':-2.84  '<P:أن>':-3.18  '<P:ذلك>':-3.93  '<P:هذا>':-4.16  'وجه':-4.47
             wazn  : '<NONE>':-1.03  'فَعَلَ':-1.23  'فِعَال':-2.99  'فَاعِل':-3.25  'فَعِيل':-3.28  'فُعُول':-3.39
             suffix: '<NONE>':-0.20  'ة':-2.53  'ه':-3.48  'ين':-4.05  'ها':-4.37  'هم':-4.98
             joint top5: ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'ة'[<NONE>+<PARTICLE>+<NONE>+ة]<SPECIAL>
```
```
--- test_deriv sentence 1272 (held-out-root targets)
RAW SENTENCE (test_deriv #1272, 5 words): وعمرة وقتيلة وأله
ENCODED TUPLES: [(1, 4, 1, 1), (4, 5681, 45, 20), (1, 4, 1, 1), (4, 6430, 55, 20), (4, 283, 45, 1)]
  [pos 3] CONTEXT: وعمرة
           TRUE NEXT: 'وقتيلة'  (4,6430,55,20) [و+قتل+فَعِيل+ة] root_tag=HELD-OUT
           PRED top1: ''  (1,4,1,1) [<NONE>+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL   miss (NOT in joint top5)
             prefix: '<NONE>':-0.28  'ال':-2.62  'و':-2.82  'وال':-3.76  'ف':-3.87  'ب':-4.25
             root  : '<PARTICLE>':-1.52  '<UNK>':-3.71  '<P:من>':-3.95  '<P:في>':-3.99  'بنن':-4.24  '<P:على>':-4.64
             wazn  : '<NONE>':-0.90  'فَعَلَ':-1.21  'فِعَال':-3.18  'فَاعِل':-3.27  'فَعِيل':-3.32  'أَفْعَلَ':-3.87
             suffix: '<NONE>':-0.23  'ة':-2.72  'ه':-3.77  'ا':-3.89  'ها':-4.12  'ت':-4.21
             joint top5: ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ة'[<NONE>+<PARTICLE>+<NONE>+ة]<SPECIAL> || 'و'[و+<PARTICLE>+<NONE>+<NONE>]<SPECIAL>
```
```
--- test_deriv sentence 1587 (held-out-root targets)
RAW SENTENCE (test_deriv #1587, 13 words): إن عرف بذلك قتل عزر وفي هذا الحديث دلالة ظاهرة لمذهب
ENCODED TUPLES: [(1, 4, 1, 1), (1, 14, 1, 1), (1, 5386, 45, 1), (1, 468, 45, 8), (1, 6430, 45, 1), (1, 4, 1, 1), (1, 5415, 45, 1), (1, 9000, 45, 1), (1, 34, 1, 1), (3, 1676, 55, 1), (1, 2730, 84, 20), (1, 5218, 42, 20), (10, 2965, 92, 1)]
  [pos 4] CONTEXT: إن عرف بذلك
           TRUE NEXT: 'قتل'  (1,6430,45,1) [<NONE>+قتل+فَعَلَ+<NONE>] root_tag=HELD-OUT
           PRED top1: ''  (1,4,1,1) [<NONE>+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL   miss (NOT in joint top5)
             prefix: '<NONE>':-0.32  'ال':-2.44  'و':-3.15  'ف':-3.21  'ب':-3.63  'ل':-3.84
             root  : '<PARTICLE>':-1.66  '<UNK>':-3.33  '<P:من>':-3.64  '<P:في>':-3.71  '<P:على>':-3.75  '<P:أو>':-3.78
             wazn  : '<NONE>':-0.74  'فَعَلَ':-1.41  'فِعَال':-3.34  'فَاعِل':-3.41  'فَعِيل':-3.42  'أَفْعَلَ':-3.94
             suffix: '<NONE>':-0.23  'ا':-2.96  'ه':-3.23  'ة':-3.41  'ت':-4.48  'ي':-4.50
             joint top5: ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ا'[<NONE>+<PARTICLE>+<NONE>+ا]<SPECIAL> || 'و'[و+<PARTICLE>+<NONE>+<NONE>]<SPECIAL>
```
```
--- test_deriv sentence 2261 (held-out-root targets)
RAW SENTENCE (test_deriv #2261, 7 words): في النسخة الغرق والحرق العامين مطموسة
ENCODED TUPLES: [(1, 5, 1, 1), (3, 8119, 45, 20), (1, 3, 1, 1), (3, 5876, 45, 1), (5, 1738, 45, 1), (3, 5697, 42, 17), (1, 5122, 94, 20)]
  [pos 1] CONTEXT: في
           TRUE NEXT: 'النسخة'  (3,8119,45,20) [ال+نسخ+فَعَلَ+ة] root_tag=HELD-OUT
           PRED top1: ''  (1,4,45,1) [<NONE>+<PARTICLE>+فَعَلَ+<NONE>] root_tag=SPECIAL   miss (NOT in joint top5)
             prefix: '<NONE>':-0.45  'ال':-1.11  'أن':-5.18  'و':-5.86  'على':-6.00  'ب':-6.05
             root  : '<PARTICLE>':-2.00  '<P:هذا>':-3.87  '<P:ذلك>':-3.90  '<P:هذه>':-4.01  'حقق':-4.08  'كتب':-4.32
             wazn  : 'فَعَلَ':-1.14  '<NONE>':-1.34  'فِعَال':-2.64  'فُعُول':-3.20  'فَعِيل':-3.26  'فَاعِل':-3.46
             suffix: '<NONE>':-0.27  'ة':-2.35  'ه':-3.34  'ها':-3.92  'ية':-4.08  'هم':-4.47
             joint top5: ''[<NONE>+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ة'[<NONE>+<PARTICLE>+فَعَلَ+ة]<SPECIAL>
```
```
--- test_deriv sentence 1204 (held-out-root targets)
RAW SENTENCE (test_deriv #1204, 11 words): وكمل المنقوص في التصغير ما لم غير ثالثا كما
ENCODED TUPLES: [(4, 7221, 45, 1), (3, 8261, 94, 1), (1, 5, 1, 1), (3, 4598, 37, 1), (1, 21, 1, 1), (1, 22, 1, 1), (1, 4, 1, 1), (1, 52, 1, 1), (1, 4, 1, 1), (1, 1112, 42, 15), (1, 7206, 1, 1)]
  [pos 3] CONTEXT: وكمل المنقوص في
           TRUE NEXT: 'التصغير'  (3,4598,37,1) [ال+صغر+تَفْعِيل+<NONE>] root_tag=HELD-OUT
           PRED top1: ''  (1,4,45,1) [<NONE>+<PARTICLE>+فَعَلَ+<NONE>] root_tag=SPECIAL   miss (NOT in joint top5)
             prefix: '<NONE>':-0.56  'ال':-0.91  'أن':-5.61  'على':-5.85  'و':-6.07  'بال':-6.38
             root  : '<PARTICLE>':-2.03  '<P:هذا>':-3.89  '<P:هذه>':-3.98  'حقق':-3.99  '<P:ذلك>':-4.00  'كتب':-4.21
             wazn  : 'فَعَلَ':-1.15  '<NONE>':-1.44  'فِعَال':-2.61  'فَعِيل':-3.06  'فُعُول':-3.14  'فَاعِل':-3.35
             suffix: '<NONE>':-0.27  'ة':-2.38  'ه':-3.15  'ية':-4.04  'ها':-4.11  'ين':-4.36
             joint top5: ''[<NONE>+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ة'[<NONE>+<PARTICLE>+فَعَلَ+ة]<SPECIAL>
```
```
--- test_deriv sentence 53 (held-out-root targets)
RAW SENTENCE (test_deriv #53, 15 words): وابن حذار حكم بني سدد وهو أحد بني سعد بن ثعلبة بن ذودان يقول في
ENCODED TUPLES: [(1, 8774, 42, 1), (1, 1692, 84, 1), (1, 1891, 45, 1), (1, 746, 45, 1), (1, 3747, 120, 1), (4, 26, 1, 1), (1, 74, 45, 1), (1, 746, 45, 1), (1, 3829, 45, 1), (1, 744, 45, 1), (1, 1086, 61, 20), (1, 744, 45, 1), (1, 2974, 78, 1), (1, 6917, 118, 1), (15, 3, 1, 1)]
  [pos 4] CONTEXT: وابن حذار حكم بني
           TRUE NEXT: 'سدد'  (1,3747,120,1) [<NONE>+سدد+يَفْعُلُ+<NONE>] root_tag=HELD-OUT
           PRED top1: ''  (1,4,1,1) [<NONE>+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL   miss (NOT in joint top5)
             prefix: '<NONE>':-0.42  'ال':-1.98  'و':-2.76  'ب':-3.76  'ل':-3.96  'ف':-4.01
             root  : '<PARTICLE>':-1.78  '<UNK>':-3.22  '<P:من>':-3.69  '<P:في>':-3.72  'بنن':-3.85  '<P:على>':-4.07
             wazn  : '<NONE>':-0.93  'فَعَلَ':-1.26  'فِعَال':-3.09  'فَعِيل':-3.28  'فَاعِل':-3.28  'فُعُول':-3.89
             suffix: '<NONE>':-0.24  'ة':-2.80  'ه':-3.41  'ا':-3.74  'ي':-4.06  'ت':-4.63
             joint top5: ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'و'[و+<PARTICLE>+<NONE>+<NONE>]<SPECIAL>
```
```
--- test_deriv sentence 906 (held-out-root targets)
RAW SENTENCE (test_deriv #906, 12 words): رب لا حبها أبدأ ويرحم أله عبدا قال وقال
ENCODED TUPLES: [(1, 4, 1, 1), (1, 3008, 45, 1), (1, 20, 1, 1), (1, 4, 1, 1), (1, 1596, 45, 4), (1, 443, 8, 1), (4, 3070, 118, 1), (1, 283, 45, 1), (1, 5236, 45, 15), (1, 6397, 45, 1), (1, 4, 1, 1), (1, 9016, 84, 1)]
  [pos 5] CONTEXT: رب لا حبها
           TRUE NEXT: 'أبدأ'  (1,443,8,1) [<NONE>+بدا+أَفْعَلَ+<NONE>] root_tag=HELD-OUT
           PRED top1: ''  (1,4,1,1) [<NONE>+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL   miss (NOT in joint top5)
             prefix: '<NONE>':-0.37  'و':-2.25  'ال':-2.91  'ل':-3.34  'ف':-3.41  'ب':-3.94
             root  : '<PARTICLE>':-1.71  '<UNK>':-2.74  '<P:لا>':-2.85  '<P:في>':-3.41  '<P:من>':-3.62  '<P:على>':-3.93
             wazn  : '<NONE>':-0.58  'فَعَلَ':-1.63  'فِعَال':-3.54  'فَاعِل':-3.66  'أَفْعَلَ':-3.73  'فَعِيل':-3.74
             suffix: '<NONE>':-0.18  'ة':-3.33  'ه':-3.56  'ا':-3.67  'ي':-4.47  'ها':-4.68
             joint top5: ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'و'[و+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ل'[ل+<PARTICLE>+<NONE>+<NONE>]<SPECIAL>
```
```
--- test_deriv sentence 1074 (held-out-root targets)
RAW SENTENCE (test_deriv #1074, 12 words): ماهية العلة غيرة لموهية المعلول ونحن علم صحة تعقل كل واحدة من
ENCODED TUPLES: [(1, 7930, 43, 1), (3, 5659, 45, 20), (1, 52, 90, 20), (10, 7930, 2, 21), (3, 5659, 94, 1), (4, 33, 45, 1), (1, 5660, 2, 1), (1, 4517, 45, 20), (1, 5593, 33, 1), (1, 50, 1, 1), (1, 8819, 42, 20), (1, 6, 1, 1)]
  [pos 3] CONTEXT: ماهية العلة غيرة
           TRUE NEXT: 'لموهية'  (10,7930,2,21) [ل+موه+<UNK>+ية] root_tag=HELD-OUT
           PRED top1: ''  (1,4,1,1) [<NONE>+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL   miss (NOT in joint top5)
             prefix: '<NONE>':-0.34  'ال':-2.26  'و':-3.03  'ف':-3.65  'ل':-3.73  'لل':-4.03
             root  : '<PARTICLE>':-1.64  '<P:في>':-3.45  '<P:من>':-3.64  '<UNK>':-3.98  '<P:لا>':-4.04  '<P:على>':-4.09
             wazn  : '<NONE>':-0.87  'فَعَلَ':-1.52  'فَاعِل':-3.05  'فِعَال':-3.24  'مَفْعَل':-3.54  'فَعِيل':-3.56
             suffix: '<NONE>':-0.34  'ة':-1.93  'ا':-3.56  'ها':-3.66  'ية':-3.75  'ت':-4.53
             joint top5: ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'ة'[<NONE>+<PARTICLE>+<NONE>+ة]<SPECIAL> || 'ال'[ال+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'و'[و+<PARTICLE>+<NONE>+<NONE>]<SPECIAL>
```
```
--- test_deriv sentence 2490 (held-out-root targets)
RAW SENTENCE (test_deriv #2490, 6 words): إلى قرب العهد قوله
ENCODED TUPLES: [(1, 7, 1, 1), (1, 6493, 45, 1), (3, 5773, 45, 1), (1, 4, 1, 1), (1, 4, 1, 1), (1, 6782, 65, 1)]
  [pos 2] CONTEXT: إلى قرب
           TRUE NEXT: 'العهد'  (3,5773,45,1) [ال+عهد+فَعَلَ+<NONE>] root_tag=HELD-OUT
           PRED top1: ''  (1,4,1,1) [<NONE>+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL   miss (NOT in joint top5)
             prefix: '<NONE>':-0.46  'ال':-1.85  'و':-2.96  'ب':-3.40  'ف':-3.70  'ل':-3.80
             root  : '<PARTICLE>':-1.81  '<P:من>':-3.09  '<UNK>':-3.16  '<P:ما>':-3.79  '<P:في>':-3.79  '<P:أن>':-3.87
             wazn  : '<NONE>':-0.74  'فَعَلَ':-1.45  'فِعَال':-3.15  'فَاعِل':-3.61  'فَعِيل':-3.63  'فُعُول':-3.71
             suffix: '<NONE>':-0.20  'ة':-3.03  'ه':-3.52  'ا':-3.95  'ين':-4.40  'هم':-4.66
             joint top5: ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'و'[و+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL>
```

**Error taxonomy on held-out roots:** across the sampled positions (and across the 20 held-out-root prompts below, plus the aggregate 4,117-position `test_deriv/heldout_root` arm at 0.00% top-1 / 0.00% top-5), the prediction came back as `<PARTICLE>`, `<UNK>`, or a `<P:...>` function word. **No word from the held-out root was ever emitted, at any rank in the joint top-5.** There is **no partial credit** to speak of: the model does not produce `كتب/كاتب/مكتوب`-style root-preserving confusions, it does not enter the derivational space at all. `<PARTICLE>` at roughly -1.6 to -1.9 nats beats the true root by 2-4 nats at every sampled position. Caveat: this is a qualitative reading of ~20 sentences, not a counted taxonomy over all 4,117 positions; what the counts do establish is the 0.00%/0.00% ceiling.

## 5. Verbatim prompts from held-out works

**Sampling.** Sentences are drawn from the project's own `test_gen` split of `/workspace/sf_data/streams.pt` (held-out works, never trained on). A fixed seed picks them; **no curation, no filtering for good or bad examples**. Every position is scored in one forward pass. The context shown is exactly what the project's encoder produced, so the prompt is literally the model input. Where the printed context looks odd (`سبته وسبه سبا`) the *source sentence* is odd — these come from a lexicon of rare words, and the analyser segments them aggressively.

```
--- test_gen sentence 1326 (held-out WORK, never trained on)
RAW SENTENCE (test_gen #1326, 15 words): سبته وسبه سبا طعنه في سبته قالت بعض العرب لبييها وكان مجروحا بتت
ENCODED TUPLES: [(1, 3657, 45, 3), (1, 8900, 45, 3), (1, 3655, 45, 1), (1, 5067, 45, 3), (1, 5, 1, 1), (1, 3657, 45, 3), (1, 6743, 42, 1), (1, 51, 1, 1), (1, 4, 1, 1), (3, 5349, 45, 1), (10, 820, 120, 4), (1, 9036, 84, 1), (1, 1284, 94, 15), (1, 4, 1, 1), (1, 376, 120, 1)]
  [pos 1] CONTEXT: سبته
           TRUE NEXT: 'وسبه'  (1,8900,45,3) [<NONE>+وسب+فَعَلَ+ه] root_tag=real
           PRED top1: ''  (1,4,1,1) [<NONE>+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL   miss (NOT in joint top5)
             prefix: '<NONE>':-0.34  'و':-2.56  'ال':-2.92  'ف':-3.31  'ب':-3.70  'ل':-3.75
             root  : '<PARTICLE>':-1.60  '<UNK>':-3.21  '<P:من>':-3.38  '<P:في>':-3.44  '<P:لا>':-4.01  '<P:على>':-4.01
             wazn  : '<NONE>':-0.66  'فَعَلَ':-1.54  'فِعَال':-3.42  'فَعِيل':-3.48  'فَاعِل':-3.55  'يَفْعَلُ':-3.79
             suffix: '<NONE>':-0.21  'ا':-3.38  'ه':-3.41  'ة':-3.45  'ي':-4.40  'ت':-4.45
             joint top5: ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'و'[و+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ف'[ف+<PARTICLE>+<NONE>+<NONE>]<SPECIAL>
  [pos 2] CONTEXT: سبته وسبه
           TRUE NEXT: 'سبا'  (1,3655,45,1) [<NONE>+سبا+فَعَلَ+<NONE>] root_tag=real
           PRED top1: ''  (1,4,1,1) [<NONE>+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL   miss (NOT in joint top5)
             prefix: '<NONE>':-0.31  'و':-2.44  'ال':-3.25  'ف':-3.32  'ل':-3.75  'وال':-3.89
             root  : '<PARTICLE>':-1.37  '<UNK>':-3.55  '<P:في>':-3.61  '<P:من>':-4.04  '<P:لا>':-4.18  '<P:إن>':-4.35
             wazn  : '<NONE>':-0.73  'فَعَلَ':-1.42  'فَعِيل':-3.26  'فَاعِل':-3.39  'فِعَال':-3.40  'أَفْعَلَ':-3.67
             suffix: '<NONE>':-0.22  'ه':-3.25  'ا':-3.37  'ة':-3.43  'ي':-3.98  'ت':-4.50
             joint top5: ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'و'[و+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ف'[ف+<PARTICLE>+<NONE>+<NONE>]<SPECIAL>
  [pos 3] CONTEXT: سبته وسبه سبا
           TRUE NEXT: 'طعنه'  (1,5067,45,3) [<NONE>+طعن+فَعَلَ+ه] root_tag=real
           PRED top1: ''  (1,4,45,1) [<NONE>+<PARTICLE>+فَعَلَ+<NONE>] root_tag=SPECIAL   miss (NOT in joint top5)
             prefix: '<NONE>':-0.43  'ال':-1.99  'و':-2.48  'ف':-3.77  'ل':-3.89  'ب':-3.96
             root  : '<PARTICLE>':-1.68  '<UNK>':-3.71  '<P:من>':-4.06  'اله':-4.26  'بنن':-4.40  '<P:في>':-4.42
             wazn  : 'فَعَلَ':-0.94  '<NONE>':-1.21  'فِعَال':-3.04  'فَعِيل':-3.21  'فَاعِل':-3.32  'أَفْعَلَ':-3.83
             suffix: '<NONE>':-0.23  'ة':-2.97  'ه':-3.30  'ا':-3.88  'ي':-4.15  'ت':-4.49
             joint top5: ''[<NONE>+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'و'[و+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+<NONE>+<NONE>]<SPECIAL>
  [pos 4] CONTEXT: سبته وسبه سبا طعنه
           TRUE NEXT: 'في'  (1,5,1,1) [<NONE>+<P:في>+<NONE>+<NONE>] root_tag=SPECIAL
           PRED top1: ''  (1,4,1,1) [<NONE>+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL   miss (NOT in joint top5)
             prefix: '<NONE>':-0.30  'و':-2.48  'ال':-3.13  'ف':-3.44  'ل':-3.84  'ب':-4.07
             root  : '<PARTICLE>':-1.58  '<P:في>':-3.40  '<UNK>':-3.49  '<P:من>':-3.54  '<P:عن>':-4.29  '<P:لا>':-4.41
             wazn  : '<NONE>':-0.83  'فَعَلَ':-1.23  'فَعِيل':-3.25  'فِعَال':-3.27  'فَاعِل':-3.41  'أَفْعَلَ':-3.86
             suffix: '<NONE>':-0.20  'ه':-3.28  'ا':-3.40  'ة':-3.67  'ي':-3.90  'هم':-4.74
             joint top5: ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'و'[و+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'و'[و+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL>
  [pos 5] CONTEXT: سبته وسبه سبا طعنه في
           TRUE NEXT: 'سبته'  (1,3657,45,3) [<NONE>+سبت+فَعَلَ+ه] root_tag=real
           PRED top1: ''  (1,4,45,1) [<NONE>+<PARTICLE>+فَعَلَ+<NONE>] root_tag=SPECIAL   miss (NOT in joint top5)
             prefix: '<NONE>':-0.36  'ال':-1.28  'أن':-5.51  'على':-5.67  'عن':-6.21  'و':-6.46
             root  : '<PARTICLE>':-1.89  'كتب':-3.62  '<P:ذلك>':-3.94  '<P:هذا>':-4.04  'حقق':-4.14  '<P:هذه>':-4.39
             wazn  : 'فَعَلَ':-1.02  '<NONE>':-1.38  'فِعَال':-2.55  'فَعِيل':-2.96  'فَاعِل':-3.55  'فُعُول':-3.62
             suffix: '<NONE>':-0.27  'ة':-2.65  'ه':-2.87  'ها':-3.95  'هم':-4.04  'ين':-4.37
             joint top5: ''[<NONE>+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ة'[<NONE>+<PARTICLE>+فَعَلَ+ة]<SPECIAL>
  [pos 6] CONTEXT: سبته وسبه سبا طعنه في سبته
           TRUE NEXT: 'قالت'  (1,6743,42,1) [<NONE>+قلت+فَاعِل+<NONE>] root_tag=real
           PRED top1: ''  (1,4,1,1) [<NONE>+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL   miss (NOT in joint top5)
             prefix: '<NONE>':-0.36  'و':-2.30  'ال':-2.88  'ف':-3.17  'وال':-3.64  'ل':-4.04
             root  : '<PARTICLE>':-1.64  '<P:في>':-3.51  '<UNK>':-3.65  '<P:من>':-3.86  '<P:لا>':-4.02  '<P:إن>':-4.13
             wazn  : '<NONE>':-0.82  'فَعَلَ':-1.32  'فِعَال':-3.04  'فَعِيل':-3.23  'فَاعِل':-3.59  'أَفْعَلَ':-3.79
             suffix: '<NONE>':-0.22  'ة':-3.15  'ه':-3.38  'ا':-3.62  'ي':-4.24  'ت':-4.34
             joint top5: ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'و'[و+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ف'[ف+<PARTICLE>+<NONE>+<NONE>]<SPECIAL>
  [pos 7] CONTEXT: سبته وسبه سبا طعنه في سبته قالت
           TRUE NEXT: 'بعض'  (1,51,1,1) [<NONE>+<P:بعض>+<NONE>+<NONE>] root_tag=SPECIAL
           PRED top1: ''  (1,4,1,1) [<NONE>+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL   miss (NOT in joint top5)
             prefix: '<NONE>':-0.31  'و':-2.54  'ال':-2.69  'ف':-3.61  'وال':-3.89  'ل':-4.01
             root  : '<PARTICLE>':-1.68  '<P:في>':-3.49  '<UNK>':-3.51  '<P:من>':-3.55  'بنن':-4.08  'قال':-4.23
             wazn  : '<NONE>':-0.90  'فَعَلَ':-1.15  'فِعَال':-3.00  'فَعِيل':-3.29  'فَاعِل':-3.62  'أَفْعَلَ':-3.83
             suffix: '<NONE>':-0.20  'ه':-3.21  'ة':-3.34  'ا':-3.73  'ي':-4.20  'ها':-4.61
             joint top5: ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'و'[و+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'و'[و+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL>
  [pos 8] CONTEXT: سبته وسبه سبا طعنه في سبته قالت بعض
           TRUE NEXT: ''  (1,4,1,1) [<NONE>+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL
           PRED top1: ''  (1,4,45,1) [<NONE>+<PARTICLE>+فَعَلَ+<NONE>] root_tag=SPECIAL   root-hit (in joint top5)
             prefix: '<NONE>':-0.45  'ال':-1.10  'و':-5.40  'ب':-5.98  'ل':-6.18  'أن':-6.20
             root  : '<PARTICLE>':-1.60  'اله':-3.50  'بيي':-4.07  'عبد':-4.61  'نسس':-4.69  'ابن':-4.70
             wazn  : 'فَعَلَ':-1.19  '<NONE>':-1.24  'فِعَال':-2.91  'يَفْعُلُ':-3.04  'أَفْعَال':-3.07  'فَاعِل':-3.29
             suffix: '<NONE>':-0.30  'ة':-2.41  'ه':-3.34  'ين':-3.87  'ات':-3.88  'ها':-4.08
             joint top5: ''[<NONE>+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ة'[<NONE>+<PARTICLE>+فَعَلَ+ة]<SPECIAL>
  [pos 9] CONTEXT: سبته وسبه سبا طعنه في سبته قالت بعض
           TRUE NEXT: 'العرب'  (3,5349,45,1) [ال+عرب+فَعَلَ+<NONE>] root_tag=real
           PRED top1: ''  (1,4,1,1) [<NONE>+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL   miss (NOT in joint top5)
             prefix: '<NONE>':-0.37  'و':-2.53  'ال':-2.56  'ف':-3.31  'وال':-3.59  'ب':-3.93
             root  : '<PARTICLE>':-1.68  '<UNK>':-3.29  '<P:في>':-3.78  '<P:من>':-3.86  '<P:لا>':-4.21  '<P:إن>':-4.37
             wazn  : '<NONE>':-0.84  'فَعَلَ':-1.28  'فِعَال':-3.08  'فَعِيل':-3.33  'فَاعِل':-3.45  'أَفْعَلَ':-3.75
             suffix: '<NONE>':-0.22  'ة':-3.20  'ه':-3.33  'ا':-3.75  'ت':-4.29  'ي':-4.29
             joint top5: ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'و'[و+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ف'[ف+<PARTICLE>+<NONE>+<NONE>]<SPECIAL>
  [pos 10] CONTEXT: سبته وسبه سبا طعنه في سبته قالت بعض العرب
           TRUE NEXT: 'لبييها'  (10,820,120,4) [ل+بيي+يَفْعُلُ+ها] root_tag=real
           PRED top1: ''  (1,4,1,1) [<NONE>+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL   miss (NOT in joint top5)
             prefix: '<NONE>':-0.38  'و':-2.41  'وال':-2.69  'ال':-3.08  'ف':-3.31  'ب':-4.23
             root  : '<PARTICLE>':-1.67  '<P:في>':-3.46  '<UNK>':-3.68  '<P:من>':-3.78  'قال':-4.03  '<P:لا>':-4.17
             wazn  : '<NONE>':-0.79  'فَعَلَ':-1.41  'فِعَال':-3.05  'فَعِيل':-3.21  'فَاعِل':-3.33  'أَفْعَلَ':-3.60
             suffix: '<NONE>':-0.19  'ه':-3.22  'ة':-3.53  'ا':-3.63  'ي':-4.13  'ت':-4.71
             joint top5: ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'و'[و+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'وال'[وال+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+<NONE>+<NONE>]<SPECIAL>
  [pos 11] CONTEXT: سبته وسبه سبا طعنه في سبته قالت بعض العرب لبييها
           TRUE NEXT: 'وكان'  (1,9036,84,1) [<NONE>+وكن+فِعَال+<NONE>] root_tag=real
           PRED top1: ''  (1,4,1,1) [<NONE>+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL   miss (NOT in joint top5)
             prefix: '<NONE>':-0.12  'و':-3.67  'ال':-3.79  'ف':-4.28  'على':-4.57  'ل':-4.67
             root  : '<PARTICLE>':-0.82  '<UNK>':-3.34  '<P:في>':-4.12  'بنن':-4.40  '<P:عن>':-4.41  'قال':-4.55
             wazn  : '<NONE>':-0.46  'فَعَلَ':-1.67  'فَعِيل':-3.28  'فَاعِل':-3.67  'فِعَال':-3.88  'يَفْعَلُ':-4.02
             suffix: '<NONE>':-0.18  'ة':-2.44  'ه':-4.03  'ا':-4.51  'ي':-4.78  'ت':-5.15
             joint top5: ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ة'[<NONE>+<PARTICLE>+<NONE>+ة]<SPECIAL> || ''[<NONE>+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'و'[و+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+<NONE>+<NONE>]<SPECIAL>
```
```
--- test_gen sentence 3882 (held-out WORK, never trained on)
RAW SENTENCE (test_gen #3882, 12 words): من الوجود فالمفعولات عنده أربعة لا خمسة ورد قصدت ليس معناه
ENCODED TUPLES: [(1, 6, 1, 1), (3, 8807, 74, 1), (7, 6260, 94, 19), (1, 54, 45, 3), (1, 3019, 8, 20), (1, 20, 1, 1), (1, 2316, 45, 20), (1, 8871, 45, 1), (1, 4, 1, 1), (1, 6634, 45, 13), (1, 24, 1, 1), (1, 5768, 113, 1)]
  [pos 1] CONTEXT: من
           TRUE NEXT: 'الوجود'  (3,8807,74,1) [ال+وجد+فُعُول+<NONE>] root_tag=real
           PRED top1: ''  (1,4,1,1) [<NONE>+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL   miss (NOT in joint top5)
             prefix: '<NONE>':-0.32  'ال':-1.40  'أن':-5.65  'ل':-6.03  'ب':-6.03  'في':-6.46
             root  : '<PARTICLE>':-1.74  '<P:غير>':-3.87  '<P:ذلك>':-4.01  'اله':-4.23  '<P:كل>':-4.27  '<P:قبل>':-4.42
             wazn  : '<NONE>':-1.09  'فَعَلَ':-1.20  'فِعَال':-3.00  'فَاعِل':-3.44  'أَفْعَال':-3.58  'يَفْعُلُ':-3.59
             suffix: '<NONE>':-0.22  'ة':-2.87  'ه':-3.42  'ات':-4.13  'ين':-4.17  'ها':-4.29
             joint top5: ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'ة'[<NONE>+<PARTICLE>+<NONE>+ة]<SPECIAL>
  [pos 2] CONTEXT: من الوجود
           TRUE NEXT: 'فالمفعولات'  (7,6260,94,19) [فال+فعل+مَفْعُول+ات] root_tag=real
           PRED top1: ''  (1,4,1,1) [<NONE>+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL   miss (NOT in joint top5)
             prefix: '<NONE>':-0.38  'ال':-2.08  'و':-3.51  'ب':-3.53  'ل':-3.64  'وال':-3.79
             root  : '<PARTICLE>':-1.65  '<P:في>':-3.05  '<UNK>':-3.16  '<P:لا>':-3.45  '<P:من>':-3.52  'ها':-3.83
             wazn  : '<NONE>':-0.65  'فَعَلَ':-1.81  'فَاعِل':-2.85  'مَفْعُول':-3.37  'مَفْعَل':-3.53  'فُعُول':-3.72
             suffix: '<NONE>':-0.21  'ة':-2.40  'ا':-3.65  'ية':-3.93  'ها':-4.44  'ه':-4.47
             joint top5: ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ة'[<NONE>+<PARTICLE>+<NONE>+ة]<SPECIAL> || ''[<NONE>+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'و'[و+<PARTICLE>+<NONE>+<NONE>]<SPECIAL>
  [pos 3] CONTEXT: من الوجود فالمفعولات
           TRUE NEXT: 'عنده'  (1,54,45,3) [<NONE>+<P:عند>+فَعَلَ+ه] root_tag=SPECIAL
           PRED top1: ''  (1,4,1,1) [<NONE>+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL   miss (NOT in joint top5)
             prefix: '<NONE>':-0.35  'ال':-2.80  'ب':-3.02  'و':-3.27  'ل':-3.52  'ف':-3.75
             root  : '<PARTICLE>':-1.85  '<UNK>':-2.65  '<P:في>':-3.07  '<P:من>':-3.10  '<P:لا>':-3.41  'ها':-3.70
             wazn  : '<NONE>':-0.56  'فَعَلَ':-1.92  'فَاعِل':-3.02  'مَفْعُول':-3.50  'مَفْعَل':-3.58  'فَعِيل':-3.76
             suffix: '<NONE>':-0.15  'ة':-3.03  'ا':-3.44  'ه':-4.25  'ية':-4.65  'ها':-4.77
             joint top5: ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ب'[ب+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'ة'[<NONE>+<PARTICLE>+<NONE>+ة]<SPECIAL>
  [pos 4] CONTEXT: من الوجود فالمفعولات عنده
           TRUE NEXT: 'أربعة'  (1,3019,8,20) [<NONE>+ربع+أَفْعَلَ+ة] root_tag=real
           PRED top1: ''  (1,4,1,1) [<NONE>+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL   miss (NOT in joint top5)
             prefix: '<NONE>':-0.34  'و':-2.57  'ال':-3.00  'ف':-3.32  'ب':-3.62  'ل':-3.67
             root  : '<PARTICLE>':-1.85  '<P:من>':-3.08  '<UNK>':-3.36  '<P:في>':-3.44  '<P:لا>':-3.48  '<P:هو>':-3.68
             wazn  : '<NONE>':-0.65  'فَعَلَ':-1.64  'فَاعِل':-3.44  'فَعِيل':-3.47  'فِعَال':-3.76  'يَفْعَلُ':-3.84
             suffix: '<NONE>':-0.15  'ا':-3.33  'ة':-3.52  'ه':-3.73  'ين':-4.91  'ها':-5.01
             joint top5: ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'و'[و+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ف'[ف+<PARTICLE>+<NONE>+<NONE>]<SPECIAL>
  [pos 5] CONTEXT: من الوجود فالمفعولات عنده أربعة
           TRUE NEXT: 'لا'  (1,20,1,1) [<NONE>+<P:لا>+<NONE>+<NONE>] root_tag=SPECIAL
           PRED top1: ''  (1,4,1,1) [<NONE>+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL   miss (NOT in joint top5)
             prefix: '<NONE>':-0.24  'و':-2.52  'ال':-2.98  'ف':-3.72  'من':-4.22  'ل':-4.40
             root  : '<PARTICLE>':-1.99  '<P:من>':-2.64  '<P:أو>':-3.54  'وحد':-3.62  'عشر':-3.67  '<P:ما>':-3.75
             wazn  : '<NONE>':-0.88  'فَعَلَ':-1.37  'فَاعِل':-3.09  'مَفْعَل':-3.33  'فِعَال':-3.34  'فَعِيل':-3.54
             suffix: '<NONE>':-0.25  'ة':-2.44  'ها':-3.77  'ا':-3.79  'ين':-4.01  'ه':-4.06
             joint top5: ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'ة'[<NONE>+<PARTICLE>+<NONE>+ة]<SPECIAL> || 'و'[و+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'من'[<NONE>+<P:من>+<NONE>+<NONE>]<SPECIAL>
  [pos 6] CONTEXT: من الوجود فالمفعولات عنده أربعة لا
           TRUE NEXT: 'خمسة'  (1,2316,45,20) [<NONE>+خمس+فَعَلَ+ة] root_tag=real
           PRED top1: ''  (1,4,118,1) [<NONE>+<PARTICLE>+يَفْعَلُ+<NONE>] root_tag=SPECIAL   miss (NOT in joint top5)
             prefix: '<NONE>':-0.05  'ب':-4.59  'ل':-4.65  'ال':-4.99  'بال':-5.87  'و':-5.97
             root  : '<PARTICLE>':-1.77  'كون':-2.83  'جوز':-3.57  'بدد':-3.70  'مكن':-3.95  '<P:من>':-4.03
             wazn  : 'يَفْعَلُ':-1.21  '<NONE>':-1.50  'فَعَلَ':-2.03  'تَفَعَّلَ':-2.68  'يَفْعُلُ':-3.15  'يَفْتَعِلُ':-3.35
             suffix: '<NONE>':-0.16  'ة':-2.89  'ه':-3.74  'ها':-4.35  'ون':-4.46  'ا':-4.54
             joint top5: ''[<NONE>+<PARTICLE>+يَفْعَلُ+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'ة'[<NONE>+<PARTICLE>+يَفْعَلُ+ة]<SPECIAL> || ''[<NONE>+<PARTICLE>+تَفَعَّلَ+<NONE>]<SPECIAL>
  [pos 7] CONTEXT: من الوجود فالمفعولات عنده أربعة لا خمسة
           TRUE NEXT: 'ورد'  (1,8871,45,1) [<NONE>+ورد+فَعَلَ+<NONE>] root_tag=real
           PRED top1: ''  (1,4,1,1) [<NONE>+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL   miss (NOT in joint top5)
             prefix: '<NONE>':-0.41  'و':-2.02  'ال':-2.21  'ل':-3.44  'ف':-3.78  'لل':-4.69
             root  : '<PARTICLE>':-1.90  '<P:لا>':-2.92  'وحد':-3.23  '<P:من>':-3.91  '<P:أو>':-3.93  'عشر':-3.93
             wazn  : '<NONE>':-1.01  'فَعَلَ':-1.14  'فَاعِل':-2.86  'فِعَال':-3.37  'مَفْعَل':-3.49  'أَفْعَلَ':-3.80
             suffix: '<NONE>':-0.28  'ة':-1.93  'ية':-3.87  'ها':-4.06  'ين':-4.31  'ا':-4.64
             joint top5: ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'و'[و+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ة'[<NONE>+<PARTICLE>+<NONE>+ة]<SPECIAL> || 'ال'[ال+<PARTICLE>+<NONE>+<NONE>]<SPECIAL>
  [pos 8] CONTEXT: من الوجود فالمفعولات عنده أربعة لا خمسة ورد
           TRUE NEXT: ''  (1,4,1,1) [<NONE>+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL
           PRED top1: ''  (1,4,1,1) [<NONE>+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL   EXACT (in joint top5)
             prefix: '<NONE>':-0.49  'ال':-1.53  'و':-3.13  'ل':-3.21  'ب':-4.20  'في':-4.20
             root  : '<PARTICLE>':-1.85  '<UNK>':-2.28  '<P:لا>':-3.21  'ها':-3.46  '<P:في>':-3.61  '<P:من>':-3.89
             wazn  : '<NONE>':-0.74  'فَعَلَ':-1.33  'فَاعِل':-3.05  'فِعَال':-3.41  'فُعُول':-3.80  'فَعِيل':-3.97
             suffix: '<NONE>':-0.17  'ة':-2.54  'ه':-4.04  'ها':-4.40  'ية':-4.60  'ين':-4.81
             joint top5: ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || ''[<NONE>+<UNK>+<NONE>+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL>
  [pos 9] CONTEXT: من الوجود فالمفعولات عنده أربعة لا خمسة ورد
           TRUE NEXT: 'قصدت'  (1,6634,45,13) [<NONE>+قصد+فَعَلَ+ت] root_tag=real
           PRED top1: ''  (1,4,1,1) [<NONE>+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL   miss (NOT in joint top5)
             prefix: '<NONE>':-0.40  'و':-2.49  'ال':-2.51  'ل':-3.28  'ف':-3.52  'ب':-3.76
             root  : '<PARTICLE>':-1.82  '<UNK>':-2.61  '<P:لا>':-2.99  'ها':-3.40  '<P:في>':-3.69  '<P:من>':-3.79
             wazn  : '<NONE>':-0.63  'فَعَلَ':-1.47  'فَاعِل':-3.34  'فِعَال':-3.68  'فَعِيل':-3.91  'يَفْعَلُ':-4.07
             suffix: '<NONE>':-0.17  'ة':-2.90  'ه':-3.85  'ها':-4.13  'ا':-4.37  'ية':-4.86
             joint top5: ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'و'[و+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ة'[<NONE>+<PARTICLE>+<NONE>+ة]<SPECIAL>
  [pos 10] CONTEXT: من الوجود فالمفعولات عنده أربعة لا خمسة ورد قصدت
           TRUE NEXT: 'ليس'  (1,24,1,1) [<NONE>+<P:ليس>+<NONE>+<NONE>] root_tag=SPECIAL
           PRED top1: ''  (1,4,1,1) [<NONE>+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL   miss (NOT in joint top5)
             prefix: '<NONE>':-0.44  'ال':-1.69  'ل':-3.25  'و':-3.35  'ب':-3.85  'في':-4.14
             root  : '<PARTICLE>':-1.74  '<UNK>':-2.32  'ها':-3.38  '<P:لا>':-3.67  '<P:في>':-3.68  '<P:من>':-4.14
             wazn  : '<NONE>':-0.70  'فَعَلَ':-1.39  'فَاعِل':-3.27  'فِعَال':-3.37  'فُعُول':-3.69  'فَعِيل':-3.95
             suffix: '<NONE>':-0.26  'ة':-2.09  'ها':-3.91  'ه':-4.08  'ية':-4.31  'ا':-4.71
             joint top5: ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'ة'[<NONE>+<PARTICLE>+<NONE>+ة]<SPECIAL> || ''[<NONE>+<UNK>+<NONE>+<NONE>]<SPECIAL>
  [pos 11] CONTEXT: من الوجود فالمفعولات عنده أربعة لا خمسة ورد قصدت ليس
           TRUE NEXT: 'معناه'  (1,5768,113,1) [<NONE>+عنه+مِفْعَال+<NONE>] root_tag=real
           PRED top1: ''  (1,3,1,1) [<NONE>+<UNK>+<NONE>+<NONE>] root_tag=SPECIAL   miss (NOT in joint top5)
             prefix: '<NONE>':-0.45  'ب':-2.16  'ل':-2.27  'في':-3.42  'ال':-3.67  'من':-4.20
             root  : '<UNK>':-1.97  '<PARTICLE>':-2.16  '<P:في>':-2.60  'ها':-2.77  '<P:من>':-2.98  '<P:هو>':-4.20
             wazn  : '<NONE>':-0.63  'فَعَلَ':-1.59  'فَاعِل':-3.27  'يَفْعَلُ':-3.38  'فِعَال':-3.67  'مَفْعَل':-3.88
             suffix: '<NONE>':-0.18  'ا':-2.84  'ه':-3.44  'ة':-3.55  'ها':-4.38  'هما':-5.10
             joint top5: ''[<NONE>+<UNK>+<NONE>+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ب'[ب+<UNK>+<NONE>+<NONE>]<SPECIAL> || 'ل'[ل+<UNK>+<NONE>+<NONE>]<SPECIAL> || ''[<NONE>+<UNK>+فَعَلَ+<NONE>]<SPECIAL>
```
```
--- test_gen sentence 617 (held-out WORK, never trained on)
RAW SENTENCE (test_gen #617, 10 words): العجل والعبث والعثي قربان جذب جبذ يقال
ENCODED TUPLES: [(3, 5302, 45, 1), (5, 5231, 45, 1), (5, 5277, 45, 12), (1, 6493, 114, 18), (1, 4, 1, 1), (1, 1263, 45, 1), (1, 3, 1, 1), (1, 1178, 45, 1), (1, 4, 1, 1), (1, 6397, 118, 1)]
  [pos 1] CONTEXT: العجل
           TRUE NEXT: 'والعبث'  (5,5231,45,1) [وال+عبث+فَعَلَ+<NONE>] root_tag=real
           PRED top1: ''  (1,4,1,1) [<NONE>+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL   miss (NOT in joint top5)
             prefix: '<NONE>':-0.41  'وال':-2.43  'ال':-2.52  'و':-2.86  'ف':-3.65  'ب':-4.13
             root  : '<PARTICLE>':-1.80  '<P:في>':-3.12  '<UNK>':-3.20  '<P:لا>':-3.52  '<P:من>':-3.57  '<P:الذي>':-3.61
             wazn  : '<NONE>':-0.63  'فَعَلَ':-1.77  'فَاعِل':-2.97  'فَعِيل':-3.37  'فِعَال':-3.53  'يَفْعَلُ':-3.63
             suffix: '<NONE>':-0.13  'ة':-3.34  'ه':-3.96  'ا':-4.10  'ي':-4.49  'ية':-5.08
             joint top5: ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'وال'[وال+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'و'[و+<PARTICLE>+<NONE>+<NONE>]<SPECIAL>
  [pos 2] CONTEXT: العجل والعبث
           TRUE NEXT: 'والعثي'  (5,5277,45,12) [وال+عثث+فَعَلَ+ي] root_tag=real
           PRED top1: ''  (1,4,1,1) [<NONE>+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL   miss (NOT in joint top5)
             prefix: '<NONE>':-0.48  'وال':-2.25  'ال':-2.28  'و':-2.97  'ب':-3.91  'ف':-3.98
             root  : '<PARTICLE>':-1.85  '<UNK>':-2.84  '<P:في>':-2.90  '<P:على>':-3.61  '<P:الذي>':-3.62  '<P:من>':-3.79
             wazn  : '<NONE>':-0.64  'فَعَلَ':-1.68  'فَاعِل':-2.86  'فَعِيل':-3.41  'فِعَال':-3.48  'يَفْعَلُ':-3.98
             suffix: '<NONE>':-0.12  'ة':-3.62  'ه':-3.80  'ي':-4.18  'ا':-4.27  'ية':-5.25
             joint top5: ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'وال'[وال+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'و'[و+<PARTICLE>+<NONE>+<NONE>]<SPECIAL>
  [pos 3] CONTEXT: العجل والعبث والعثي
           TRUE NEXT: 'قربان'  (1,6493,114,18) [<NONE>+قرب+يَتَفَاعَلُ+ان] root_tag=real
           PRED top1: ''  (1,4,1,1) [<NONE>+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL   miss (NOT in joint top5)
             prefix: '<NONE>':-0.51  'وال':-1.99  'ال':-2.30  'و':-2.73  'ف':-3.85  'ب':-4.32
             root  : '<PARTICLE>':-1.70  '<P:في>':-3.50  '<UNK>':-3.76  '<P:من>':-3.99  '<P:على>':-4.24  '<P:هو>':-4.26
             wazn  : '<NONE>':-0.84  'فَعَلَ':-1.53  'فَاعِل':-2.93  'فَعِيل':-3.16  'فِعَال':-3.25  'يَفْعَلُ':-3.69
             suffix: '<NONE>':-0.21  'ة':-2.78  'ه':-3.69  'ا':-3.86  'ية':-4.42  'ي':-4.44
             joint top5: ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'وال'[وال+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'و'[و+<PARTICLE>+<NONE>+<NONE>]<SPECIAL>
  [pos 4] CONTEXT: العجل والعبث والعثي قربان
           TRUE NEXT: ''  (1,4,1,1) [<NONE>+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL
           PRED top1: ''  (1,4,1,1) [<NONE>+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL   EXACT (in joint top5)
             prefix: '<NONE>':-0.52  'و':-2.47  'ال':-2.72  'ب':-2.87  'ل':-3.24  'بال':-3.48
             root  : '<PARTICLE>':-1.95  '<P:في>':-2.51  '<UNK>':-2.86  '<P:من>':-3.17  '<P:عن>':-3.58  '<P:على>':-3.62
             wazn  : '<NONE>':-0.61  'فَعَلَ':-1.59  'فِعَال':-3.45  'فَاعِل':-3.78  'فُعُول':-3.81  'فَعِيل':-3.81
             suffix: '<NONE>':-0.21  'ة':-2.95  'ا':-3.25  'ه':-3.82  'ها':-4.04  'ية':-4.66
             joint top5: ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'و'[و+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'في'[<NONE>+<P:في>+<NONE>+<NONE>]<SPECIAL>
  [pos 5] CONTEXT: العجل والعبث والعثي قربان
           TRUE NEXT: 'جذب'  (1,1263,45,1) [<NONE>+جذب+فَعَلَ+<NONE>] root_tag=real
           PRED top1: ''  (1,4,1,1) [<NONE>+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL   miss (NOT in joint top5)
             prefix: '<NONE>':-0.43  'ال':-2.03  'و':-2.78  'وال':-3.41  'ف':-3.58  'ب':-3.94
             root  : '<PARTICLE>':-1.68  '<UNK>':-3.41  '<P:في>':-3.65  '<P:من>':-3.97  '<P:لا>':-4.01  '<P:على>':-4.02
             wazn  : '<NONE>':-0.78  'فَعَلَ':-1.45  'فَاعِل':-3.23  'فِعَال':-3.34  'فَعِيل':-3.54  'يَفْعَلُ':-3.88
             suffix: '<NONE>':-0.22  'ة':-2.74  'ا':-3.78  'ه':-3.87  'ها':-4.09  'ت':-4.26
             joint top5: ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'و'[و+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ة'[<NONE>+<PARTICLE>+<NONE>+ة]<SPECIAL>
  [pos 6] CONTEXT: العجل والعبث والعثي قربان جذب
           TRUE NEXT: ''  (1,3,1,1) [<NONE>+<UNK>+<NONE>+<NONE>] root_tag=SPECIAL
           PRED top1: ''  (1,4,1,1) [<NONE>+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL   miss (NOT in joint top5)
             prefix: '<NONE>':-0.45  'ال':-1.83  'و':-2.62  'ف':-3.78  'وال':-3.86  'ب':-4.15
             root  : '<PARTICLE>':-1.94  '<P:من>':-3.88  '<UNK>':-4.22  'عشر':-4.58  '<P:أو>':-4.67  '<P:ما>':-4.69
             wazn  : '<NONE>':-1.14  'فَعَلَ':-1.18  'فَاعِل':-3.01  'فِعَال':-3.12  'فَعِيل':-3.25  'يَفْعَلُ':-3.82
             suffix: '<NONE>':-0.29  'ة':-2.57  'ه':-3.52  'ا':-3.68  'ها':-3.93  'ين':-4.17
             joint top5: ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'و'[و+<PARTICLE>+<NONE>+<NONE>]<SPECIAL>
  [pos 7] CONTEXT: العجل والعبث والعثي قربان جذب
           TRUE NEXT: 'جبذ'  (1,1178,45,1) [<NONE>+جبذ+فَعَلَ+<NONE>] root_tag=real
           PRED top1: ''  (1,4,1,1) [<NONE>+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL   miss (NOT in joint top5)
             prefix: '<NONE>':-0.21  'ال':-2.90  'و':-3.62  'ب':-3.70  'ل':-4.12  'ف':-4.45
             root  : '<PARTICLE>':-1.73  '<UNK>':-2.78  '<P:من>':-3.96  '<P:في>':-4.03  '<P:أن>':-4.14  '<P:لا>':-4.25
             wazn  : '<NONE>':-0.76  'فَعَلَ':-1.36  'يَفْعَلُ':-3.32  'فَاعِل':-3.51  'فِعَال':-3.75  'فَعِيل':-3.83
             suffix: '<NONE>':-0.18  'ة':-3.35  'ه':-3.65  'ا':-3.89  'ت':-4.10  'ها':-4.20
             joint top5: ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ة'[<NONE>+<PARTICLE>+<NONE>+ة]<SPECIAL> || 'و'[و+<PARTICLE>+<NONE>+<NONE>]<SPECIAL>
  [pos 8] CONTEXT: العجل والعبث والعثي قربان جذب جبذ
           TRUE NEXT: ''  (1,4,1,1) [<NONE>+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL
           PRED top1: ''  (1,4,1,1) [<NONE>+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL   EXACT (in joint top5)
             prefix: '<NONE>':-0.41  'ال':-2.34  'و':-2.37  'ف':-3.52  'ب':-3.91  'وال':-4.05
             root  : '<PARTICLE>':-1.89  '<P:من>':-3.52  '<UNK>':-3.96  'عشر':-4.16  '<P:أو>':-4.27  'ثلث':-4.70
             wazn  : '<NONE>':-1.08  'فَعَلَ':-1.14  'فَاعِل':-3.05  'فِعَال':-3.21  'فَعِيل':-3.29  'أَفْعَلَ':-3.77
             suffix: '<NONE>':-0.26  'ة':-2.82  'ه':-3.37  'ا':-3.50  'ها':-4.11  'ين':-4.28
             joint top5: ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'و'[و+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL>
  [pos 9] CONTEXT: العجل والعبث والعثي قربان جذب جبذ
           TRUE NEXT: 'يقال'  (1,6397,118,1) [<NONE>+قال+يَفْعَلُ+<NONE>] root_tag=real
           PRED top1: ''  (1,4,1,1) [<NONE>+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL   miss (NOT in joint top5)
             prefix: '<NONE>':-0.35  'ال':-2.59  'و':-2.66  'ف':-3.49  'وال':-3.55  'ب':-3.97
             root  : '<PARTICLE>':-1.71  '<UNK>':-3.62  '<P:من>':-3.91  '<P:في>':-4.24  '<P:هو>':-4.53  '<P:أو>':-4.57
             wazn  : '<NONE>':-0.91  'فَعَلَ':-1.27  'فَاعِل':-3.14  'فِعَال':-3.32  'فَعِيل':-3.33  'يَفْعَلُ':-3.65
             suffix: '<NONE>':-0.24  'ة':-2.93  'ه':-3.45  'ا':-3.52  'ها':-4.10  'ت':-4.22
             joint top5: ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'و'[و+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ة'[<NONE>+<PARTICLE>+<NONE>+ة]<SPECIAL>
```
```
--- test_gen sentence 1617 (held-out WORK, never trained on)
RAW SENTENCE (test_gen #1617, 10 words): عن وقد أله ذمهم بحكمين حق غيهم أحدهما قوله
ENCODED TUPLES: [(18, 3, 1, 1), (1, 9004, 45, 1), (1, 4, 1, 1), (1, 283, 45, 1), (1, 2959, 45, 5), (8, 1891, 45, 17), (1, 1878, 45, 1), (1, 6043, 45, 5), (1, 74, 45, 6), (1, 6782, 65, 1)]
  [pos 1] CONTEXT: عن
           TRUE NEXT: 'وقد'  (1,9004,45,1) [<NONE>+وقد+فَعَلَ+<NONE>] root_tag=real
           PRED top1: ''  (1,4,1,1) [<NONE>+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL   miss (NOT in joint top5)
             prefix: '<NONE>':-0.22  'ال':-2.86  'و':-3.35  'ف':-3.74  'ب':-3.99  'ل':-4.12
             root  : '<PARTICLE>':-1.71  '<UNK>':-3.10  '<P:لا>':-3.14  '<P:من>':-3.55  '<P:في>':-3.68  '<P:أن>':-3.87
             wazn  : '<NONE>':-0.62  'فَعَلَ':-1.69  'يَفْعَلُ':-3.42  'فَاعِل':-3.49  'فِعَال':-3.52  'فَعِيل':-3.76
             suffix: '<NONE>':-0.15  'ة':-3.25  'ه':-3.90  'ا':-4.12  'ت':-4.55  'ي':-4.61
             joint top5: ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ة'[<NONE>+<PARTICLE>+<NONE>+ة]<SPECIAL> || 'و'[و+<PARTICLE>+<NONE>+<NONE>]<SPECIAL>
  [pos 2] CONTEXT: عن وقد
           TRUE NEXT: ''  (1,4,1,1) [<NONE>+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL
           PRED top1: ''  (1,4,45,1) [<NONE>+<PARTICLE>+فَعَلَ+<NONE>] root_tag=SPECIAL   root-hit (in joint top5)
             prefix: '<NONE>':-0.06  'ال':-4.73  'في':-5.08  'ل':-5.23  'لل':-5.56  'على':-5.64
             root  : '<PARTICLE>':-1.79  'تقدم':-3.14  '<P:بين>':-3.22  'ذكر':-3.43  'كون':-3.45  '<P:كان>':-3.63
             wazn  : 'فَعَلَ':-0.94  '<NONE>':-1.49  'يَفْعَلُ':-2.53  'أَفْعَلَ':-2.95  'تَفَعَّلَ':-3.22  'فَعْلَلَ':-3.32
             suffix: '<NONE>':-0.26  'ت':-2.47  'نا':-3.34  'ه':-3.50  'ي':-4.31  'ها':-4.55
             joint top5: ''[<NONE>+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ت'[<NONE>+<PARTICLE>+فَعَلَ+ت]<SPECIAL> || 'نا'[<NONE>+<PARTICLE>+فَعَلَ+نا]<SPECIAL> || ''[<NONE>+<PARTICLE>+يَفْعَلُ+<NONE>]<SPECIAL>
  [pos 3] CONTEXT: عن وقد
           TRUE NEXT: 'أله'  (1,283,45,1) [<NONE>+اله+فَعَلَ+<NONE>] root_tag=real
           PRED top1: ''  (1,4,1,1) [<NONE>+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL   miss (NOT in joint top5)
             prefix: '<NONE>':-0.34  'ال':-2.26  'و':-3.12  'ب':-3.47  'ل':-4.00  'ف':-4.15
             root  : '<PARTICLE>':-1.68  '<UNK>':-2.75  '<P:في>':-3.20  '<P:من>':-3.39  '<P:أن>':-3.81  '<P:على>':-3.90
             wazn  : '<NONE>':-0.66  'فَعَلَ':-1.52  'فِعَال':-3.28  'فَاعِل':-3.47  'فَعِيل':-3.55  'فُعُول':-4.05
             suffix: '<NONE>':-0.19  'ة':-3.16  'ا':-3.67  'ه':-3.70  'ي':-4.37  'ت':-4.68
             joint top5: ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'و'[و+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ة'[<NONE>+<PARTICLE>+<NONE>+ة]<SPECIAL>
  [pos 4] CONTEXT: عن وقد أله
           TRUE NEXT: 'ذمهم'  (1,2959,45,5) [<NONE>+ذمم+فَعَلَ+هم] root_tag=real
           PRED top1: ''  (1,4,1,1) [<NONE>+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL   miss (NOT in joint top5)
             prefix: '<NONE>':-0.35  'على':-1.91  'و':-3.37  'ب':-3.63  'ال':-3.90  'ل':-4.34
             root  : '<PARTICLE>':-1.21  '<UNK>':-1.62  'علم':-3.13  '<P:من>':-3.84  '<P:على>':-3.97  'عزز':-4.25
             wazn  : '<NONE>':-0.38  'فَعَلَ':-2.08  'أَفْعَلَ':-2.83  'فَعِيل':-3.95  'فِعَال':-4.07  'فَاعِل':-4.10
             suffix: '<NONE>':-0.11  'ه':-3.68  'ا':-4.13  'هم':-4.44  'ي':-4.48  'ة':-4.59
             joint top5: ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'على'[على+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || ''[<NONE>+<UNK>+<NONE>+<NONE>]<SPECIAL> || 'و'[و+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'على'[على+<UNK>+<NONE>+<NONE>]<SPECIAL>
  [pos 5] CONTEXT: عن وقد أله ذمهم
           TRUE NEXT: 'بحكمين'  (8,1891,45,17) [ب+حكم+فَعَلَ+ين] root_tag=real
           PRED top1: ''  (1,4,1,1) [<NONE>+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL   miss (NOT in joint top5)
             prefix: '<NONE>':-0.25  'و':-2.99  'ال':-3.36  'ب':-3.41  'ل':-3.83  'أن':-3.95
             root  : '<PARTICLE>':-1.82  '<P:من>':-2.48  '<UNK>':-2.69  '<P:أن>':-2.95  '<P:في>':-3.03  '<P:عن>':-3.59
             wazn  : '<NONE>':-0.53  'فَعَلَ':-1.62  'فِعَال':-3.45  'فَعِيل':-3.62  'أَفْعَلَ':-3.97  'فَاعِل':-3.99
             suffix: '<NONE>':-0.17  'ا':-3.36  'ه':-3.40  'ة':-3.86  'هم':-4.40  'ي':-4.42
             joint top5: ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'من'[<NONE>+<P:من>+<NONE>+<NONE>]<SPECIAL> || 'و'[و+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+<NONE>+<NONE>]<SPECIAL>
  [pos 6] CONTEXT: عن وقد أله ذمهم بحكمين
           TRUE NEXT: 'حق'  (1,1878,45,1) [<NONE>+حقق+فَعَلَ+<NONE>] root_tag=real
           PRED top1: ''  (1,4,1,1) [<NONE>+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL   miss (NOT in joint top5)
             prefix: '<NONE>':-0.31  'و':-2.61  'ال':-2.68  'ف':-3.72  'ب':-3.73  'ل':-4.03
             root  : '<PARTICLE>':-1.62  '<P:من>':-2.94  '<P:في>':-3.13  '<UNK>':-3.19  '<P:أن>':-3.59  '<P:على>':-3.75
             wazn  : '<NONE>':-0.65  'فَعَلَ':-1.51  'فِعَال':-3.13  'فَعِيل':-3.45  'فَاعِل':-3.78  'أَفْعَلَ':-3.91
             suffix: '<NONE>':-0.19  'ة':-3.16  'ه':-3.47  'ا':-3.73  'هم':-4.47  'ي':-4.52
             joint top5: ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'و'[و+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ة'[<NONE>+<PARTICLE>+<NONE>+ة]<SPECIAL>
  [pos 7] CONTEXT: عن وقد أله ذمهم بحكمين حق
           TRUE NEXT: 'غيهم'  (1,6043,45,5) [<NONE>+غيي+فَعَلَ+هم] root_tag=real
           PRED top1: ''  (1,4,1,1) [<NONE>+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL   miss (NOT in joint top5)
             prefix: '<NONE>':-0.46  'ال':-1.44  'و':-3.63  'ب':-3.71  'ل':-3.92  'ف':-4.74
             root  : '<PARTICLE>':-2.01  'اله':-2.52  '<UNK>':-3.07  '<P:من>':-3.13  '<P:ما>':-3.51  '<P:ذلك>':-3.71
             wazn  : '<NONE>':-0.87  'فَعَلَ':-1.32  'فِعَال':-2.91  'فَاعِل':-3.35  'فَعِيل':-3.46  'يَفْعُلُ':-3.55
             suffix: '<NONE>':-0.21  'ة':-3.08  'ه':-3.18  'هم':-4.11  'ا':-4.29  'ي':-4.29
             joint top5: ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'اله'[<NONE>+اله+<NONE>+<NONE>]<real>
  [pos 8] CONTEXT: عن وقد أله ذمهم بحكمين حق غيهم
           TRUE NEXT: 'أحدهما'  (1,74,45,6) [<NONE>+أحد+فَعَلَ+هما] root_tag=real
           PRED top1: ''  (1,4,1,1) [<NONE>+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL   miss (NOT in joint top5)
             prefix: '<NONE>':-0.30  'و':-2.46  'ال':-3.15  'ف':-3.58  'ب':-3.71  'ل':-3.85
             root  : '<PARTICLE>':-1.64  '<P:في>':-2.99  '<P:من>':-3.10  '<UNK>':-3.15  '<P:أن>':-3.72  '<P:على>':-3.77
             wazn  : '<NONE>':-0.68  'فَعَلَ':-1.44  'فِعَال':-3.22  'فَعِيل':-3.33  'فَاعِل':-3.65  'أَفْعَلَ':-3.88
             suffix: '<NONE>':-0.19  'ه':-3.30  'ا':-3.42  'ة':-3.45  'ي':-4.35  'هم':-4.54
             joint top5: ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'و'[و+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ه'[<NONE>+<PARTICLE>+<NONE>+ه]<SPECIAL>
  [pos 9] CONTEXT: عن وقد أله ذمهم بحكمين حق غيهم أحدهما
           TRUE NEXT: 'قوله'  (1,6782,65,1) [<NONE>+قله+فَوْعَل+<NONE>] root_tag=real
           PRED top1: ''  (1,4,1,1) [<NONE>+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL   miss (NOT in joint top5)
             prefix: '<NONE>':-0.33  'و':-2.36  'ف':-3.32  'ب':-3.48  'ال':-3.60  'ل':-3.75
             root  : '<PARTICLE>':-1.84  '<P:من>':-2.71  '<P:في>':-2.81  '<P:على>':-3.10  '<UNK>':-3.25  '<P:عن>':-3.67
             wazn  : '<NONE>':-0.60  'فَعَلَ':-1.64  'فِعَال':-3.27  'فَعِيل':-3.37  'أَفْعَلَ':-3.82  'فَاعِل':-3.83
             suffix: '<NONE>':-0.22  'ا':-3.01  'ة':-3.31  'ه':-3.32  'هم':-4.55  'ي':-4.58
             joint top5: ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'و'[و+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'ا'[<NONE>+<PARTICLE>+<NONE>+ا]<SPECIAL> || 'ف'[ف+<PARTICLE>+<NONE>+<NONE>]<SPECIAL>
```
```
--- test_gen sentence 2666 (held-out WORK, never trained on)
RAW SENTENCE (test_gen #2666, 15 words): لم يوجد منما يملك الميت نمرة كان ستر جميع البدن واجبا لوجب على
ENCODED TUPLES: [(1, 22, 1, 1), (1, 8807, 118, 1), (16, 21, 1, 1), (1, 7881, 118, 1), (3, 7933, 45, 1), (1, 3, 1, 1), (1, 8294, 45, 20), (1, 4, 1, 1), (1, 25, 1, 1), (1, 3690, 45, 1), (1, 1488, 55, 1), (3, 453, 45, 1), (1, 8804, 42, 15), (1, 7371, 65, 1), (1, 8, 1, 1)]
  [pos 1] CONTEXT: لم
           TRUE NEXT: 'يوجد'  (1,8807,118,1) [<NONE>+وجد+يَفْعَلُ+<NONE>] root_tag=real
           PRED top1: ''  (1,4,120,1) [<NONE>+<PARTICLE>+يَفْعُلُ+<NONE>] root_tag=SPECIAL   miss (NOT in joint top5)
             prefix: '<NONE>':-0.02  'ال':-6.01  'على':-6.28  'ف':-6.50  'ل':-6.65  'من':-6.73
             root  : '<PARTICLE>':-1.74  'كنن':-1.82  '<P:لا>':-3.99  'وجد':-4.19  'علم':-4.29  'صحح':-4.30
             wazn  : 'يَفْعُلُ':-1.21  'يَفْعَلُ':-1.43  '<NONE>':-1.57  '<UNK>':-2.98  'فَعَلَ':-3.01  'تَفَعَّلَ':-3.30
             suffix: '<NONE>':-0.07  'ه':-3.40  'ا':-4.94  'ها':-5.22  'ة':-5.72  'هم':-5.83
             joint top5: ''[<NONE>+<PARTICLE>+يَفْعُلُ+<NONE>]<SPECIAL> || 'كنن'[<NONE>+كنن+يَفْعُلُ+<NONE>]<real> || ''[<NONE>+<PARTICLE>+يَفْعَلُ+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'يكنن'[<NONE>+كنن+يَفْعَلُ+<NONE>]<real>
  [pos 2] CONTEXT: لم يوجد
           TRUE NEXT: 'منما'  (16,21,1,1) [من+<P:ما>+<NONE>+<NONE>] root_tag=SPECIAL
           PRED top1: ''  (1,4,1,1) [<NONE>+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL   miss (NOT in joint top5)
             prefix: '<NONE>':-0.37  'ال':-2.12  'ل':-2.90  'ب':-3.23  'في':-3.76  'لل':-4.42
             root  : '<PARTICLE>':-1.99  '<UNK>':-2.19  '<P:في>':-2.78  '<P:ذلك>':-3.46  'ها':-3.47  '<P:من>':-3.55
             wazn  : '<NONE>':-0.65  'فَعَلَ':-1.48  'فَاعِل':-3.15  'فِعَال':-3.46  'مَفْعَل':-3.76  'فَعِيل':-3.84
             suffix: '<NONE>':-0.20  'ا':-2.60  'ة':-3.45  'ه':-3.50  'ها':-4.96  'ي':-5.25
             joint top5: ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || ''[<NONE>+<UNK>+<NONE>+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ا'[<NONE>+<PARTICLE>+<NONE>+ا]<SPECIAL>
  [pos 3] CONTEXT: لم يوجد منما
           TRUE NEXT: 'يملك'  (1,7881,118,1) [<NONE>+ملك+يَفْعَلُ+<NONE>] root_tag=real
           PRED top1: ''  (1,4,1,1) [<NONE>+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL   miss (NOT in joint top5)
             prefix: '<NONE>':-0.05  'في':-4.39  'ب':-4.95  'ل':-5.22  'على':-5.48  'ف':-5.82
             root  : '<PARTICLE>':-1.91  '<P:لا>':-3.00  '<P:هو>':-3.00  '<P:كان>':-3.04  'ذكر':-3.52  '<UNK>':-3.63
             wazn  : '<NONE>':-0.88  'فَعَلَ':-1.47  'يَفْعَلُ':-2.22  'يَفْعُلُ':-3.25  'أَفْعَلَ':-3.29  'تَفَعَّلَ':-3.60
             suffix: '<NONE>':-0.19  'ه':-2.53  'ت':-3.63  'ا':-4.27  'نا':-4.43  'ها':-4.61
             joint top5: ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'ه'[<NONE>+<PARTICLE>+<NONE>+ه]<SPECIAL> || ''[<NONE>+<PARTICLE>+يَفْعَلُ+<NONE>]<SPECIAL> || 'ت'[<NONE>+<PARTICLE>+<NONE>+ت]<SPECIAL>
  [pos 4] CONTEXT: لم يوجد منما يملك
           TRUE NEXT: 'الميت'  (3,7933,45,1) [ال+ميت+فَعَلَ+<NONE>] root_tag=real
           PRED top1: ''  (1,3,1,1) [<NONE>+<UNK>+<NONE>+<NONE>] root_tag=SPECIAL   miss (NOT in joint top5)
             prefix: '<NONE>':-0.43  'ال':-2.07  'ب':-2.94  'ل':-3.07  'في':-3.84  'من':-3.92
             root  : '<UNK>':-1.89  '<PARTICLE>':-2.12  '<P:من>':-3.24  '<P:في>':-3.32  '<P:أن>':-3.40  '<P:ذلك>':-3.81
             wazn  : '<NONE>':-0.66  'فَعَلَ':-1.37  'فِعَال':-3.34  'فَاعِل':-3.59  'فَعِيل':-3.88  'فُعُول':-4.13
             suffix: '<NONE>':-0.19  'ه':-2.95  'ا':-3.35  'ة':-3.57  'ي':-4.75  'هم':-5.02
             joint top5: ''[<NONE>+<UNK>+<NONE>+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || ''[<NONE>+<UNK>+فَعَلَ+<NONE>]<SPECIAL> || 'ال'[ال+<UNK>+<NONE>+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL>
  [pos 5] CONTEXT: لم يوجد منما يملك الميت
           TRUE NEXT: ''  (1,3,1,1) [<NONE>+<UNK>+<NONE>+<NONE>] root_tag=SPECIAL
           PRED top1: ''  (1,4,1,1) [<NONE>+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL   miss (NOT in joint top5)
             prefix: '<NONE>':-0.45  'ال':-2.55  'و':-2.78  'وال':-2.98  'ف':-3.11  'ب':-3.51
             root  : '<PARTICLE>':-1.90  '<UNK>':-2.96  '<P:في>':-3.45  '<P:من>':-3.57  '<P:لا>':-3.72  '<P:لم>':-3.84
             wazn  : '<NONE>':-0.68  'فَعَلَ':-1.60  'فَاعِل':-3.16  'فِعَال':-3.31  'فَعِيل':-3.31  'يَفْعَلُ':-3.77
             suffix: '<NONE>':-0.21  'ة':-2.84  'ا':-3.33  'ه':-3.57  'ها':-4.67  'ية':-4.67
             joint top5: ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'و'[و+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'وال'[وال+<PARTICLE>+<NONE>+<NONE>]<SPECIAL>
  [pos 6] CONTEXT: لم يوجد منما يملك الميت
           TRUE NEXT: 'نمرة'  (1,8294,45,20) [<NONE>+نمر+فَعَلَ+ة] root_tag=real
           PRED top1: ''  (1,4,1,1) [<NONE>+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL   miss (NOT in joint top5)
             prefix: '<NONE>':-0.23  'ال':-2.71  'ب':-3.47  'و':-3.84  'ل':-3.99  'ف':-4.47
             root  : '<PARTICLE>':-1.95  '<UNK>':-2.71  '<P:أن>':-3.36  '<P:من>':-3.44  '<P:في>':-3.58  '<P:كان>':-3.84
             wazn  : '<NONE>':-0.71  'فَعَلَ':-1.40  'فَاعِل':-3.44  'يَفْعَلُ':-3.54  'فِعَال':-3.55  'فَعِيل':-3.73
             suffix: '<NONE>':-0.17  'ة':-3.21  'ا':-3.48  'ه':-3.65  'ت':-4.42  'ها':-4.73
             joint top5: ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || ''[<NONE>+<UNK>+<NONE>+<NONE>]<SPECIAL> || 'ة'[<NONE>+<PARTICLE>+<NONE>+ة]<SPECIAL>
  [pos 7] CONTEXT: لم يوجد منما يملك الميت نمرة
           TRUE NEXT: ''  (1,4,1,1) [<NONE>+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL
           PRED top1: ''  (1,4,1,1) [<NONE>+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL   EXACT (in joint top5)
             prefix: '<NONE>':-0.45  'ال':-1.73  'و':-2.52  'ف':-3.38  'ل':-4.01  'ب':-4.62
             root  : '<PARTICLE>':-1.86  '<P:من>':-3.72  '<P:لا>':-3.75  '<P:أو>':-3.81  '<UNK>':-3.84  'وحد':-4.15
             wazn  : '<NONE>':-0.98  'فَعَلَ':-1.16  'فِعَال':-3.07  'فَاعِل':-3.08  'فَعِيل':-3.43  'أَفْعَلَ':-3.83
             suffix: '<NONE>':-0.26  'ة':-2.13  'ية':-4.05  'ه':-4.16  'ا':-4.30  'ها':-4.39
             joint top5: ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'ة'[<NONE>+<PARTICLE>+<NONE>+ة]<SPECIAL>
  [pos 8] CONTEXT: لم يوجد منما يملك الميت نمرة
           TRUE NEXT: 'كان'  (1,25,1,1) [<NONE>+<P:كان>+<NONE>+<NONE>] root_tag=SPECIAL
           PRED top1: ''  (1,4,1,1) [<NONE>+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL   miss (NOT in joint top5)
             prefix: '<NONE>':-0.37  'ال':-2.42  'و':-2.81  'ف':-3.28  'ب':-3.68  'ل':-3.79
             root  : '<PARTICLE>':-1.86  '<UNK>':-3.11  '<P:في>':-3.57  '<P:من>':-3.57  '<P:لا>':-3.89  '<P:لم>':-3.91
             wazn  : '<NONE>':-0.73  'فَعَلَ':-1.42  'فِعَال':-3.22  'فَاعِل':-3.34  'فَعِيل':-3.48  'أَفْعَلَ':-3.96
             suffix: '<NONE>':-0.22  'ة':-2.75  'ا':-3.59  'ه':-3.65  'ت':-4.26  'ها':-4.33
             joint top5: ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'و'[و+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ة'[<NONE>+<PARTICLE>+<NONE>+ة]<SPECIAL>
  [pos 9] CONTEXT: لم يوجد منما يملك الميت نمرة كان
           TRUE NEXT: 'ستر'  (1,3690,45,1) [<NONE>+ستر+فَعَلَ+<NONE>] root_tag=real
           PRED top1: ''  (1,4,1,1) [<NONE>+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL   miss (NOT in joint top5)
             prefix: '<NONE>':-0.25  'ال':-2.32  'ل':-3.20  'ب':-3.90  'في':-4.41  'لل':-4.43
             root  : '<PARTICLE>':-1.92  '<UNK>':-2.71  '<P:في>':-3.11  '<P:من>':-3.61  '<P:ذلك>':-3.62  'ها':-4.22
             wazn  : '<NONE>':-0.78  'فَعَلَ':-1.55  'فَاعِل':-2.84  'فِعَال':-3.46  'مَفْعَل':-3.50  'فَعِيل':-3.53
             suffix: '<NONE>':-0.29  'ا':-2.00  'ه':-3.46  'ة':-3.55  'ها':-4.52  'ي':-4.86
             joint top5: ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'ا'[<NONE>+<PARTICLE>+<NONE>+ا]<SPECIAL> || 'ال'[ال+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ل'[ل+<PARTICLE>+<NONE>+<NONE>]<SPECIAL>
  [pos 10] CONTEXT: لم يوجد منما يملك الميت نمرة كان ستر
           TRUE NEXT: 'جميع'  (1,1488,55,1) [<NONE>+جمع+فَعِيل+<NONE>] root_tag=real
           PRED top1: ''  (1,4,1,1) [<NONE>+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL   miss (NOT in joint top5)
             prefix: '<NONE>':-0.35  'ال':-1.78  'و':-3.27  'ف':-4.12  'ل':-4.29  'ب':-4.38
             root  : '<PARTICLE>':-1.83  '<P:من>':-3.39  '<UNK>':-3.63  '<P:لا>':-4.01  '<P:أو>':-4.12  '<P:ما>':-4.27
             wazn  : '<NONE>':-0.92  'فَعَلَ':-1.26  'فَاعِل':-2.90  'فِعَال':-3.18  'فَعِيل':-3.42  'مَفْعَل':-3.80
             suffix: '<NONE>':-0.21  'ة':-2.58  'ه':-3.74  'ا':-4.13  'ين':-4.45  'ها':-4.58
             joint top5: ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'ة'[<NONE>+<PARTICLE>+<NONE>+ة]<SPECIAL>
  [pos 11] CONTEXT: لم يوجد منما يملك الميت نمرة كان ستر جميع
           TRUE NEXT: 'البدن'  (3,453,45,1) [ال+بدن+فَعَلَ+<NONE>] root_tag=real
           PRED top1: ''  (1,4,1,1) [<NONE>+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL   miss (NOT in joint top5)
             prefix: '<NONE>':-0.51  'ال':-1.01  'و':-5.01  'ل':-5.54  'ف':-5.71  'ب':-5.79
             root  : '<PARTICLE>':-1.69  'اله':-3.31  '<P:ما>':-3.72  '<P:من>':-4.55  '<P:ذلك>':-4.60  '<P:غير>':-4.68
             wazn  : '<NONE>':-1.17  'فَعَلَ':-1.17  'فِعَال':-2.95  'فَاعِل':-2.98  'أَفْعَال':-3.30  'فَعِيل':-3.35
             suffix: '<NONE>':-0.23  'ة':-2.41  'ه':-3.82  'ين':-3.94  'ات':-4.28  'ية':-4.51
             joint top5: ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'ة'[<NONE>+<PARTICLE>+<NONE>+ة]<SPECIAL>
```
```
--- test_gen sentence 197 (held-out WORK, never trained on)
RAW SENTENCE (test_gen #197, 10 words): فإن تك هامة بهراة فقد هما وزقية
ENCODED TUPLES: [(6, 14, 1, 1), (1, 936, 45, 1), (1, 8667, 42, 20), (8, 8478, 45, 20), (1, 4, 1, 1), (1, 6268, 45, 1), (1, 4, 1, 1), (1, 4, 1, 1), (1, 28, 42, 1), (4, 3507, 45, 21)]
  [pos 1] CONTEXT: فإن
           TRUE NEXT: 'تك'  (1,936,45,1) [<NONE>+تكك+فَعَلَ+<NONE>] root_tag=real
           PRED top1: ''  (1,4,1,1) [<NONE>+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL   miss (NOT in joint top5)
             prefix: '<NONE>':-0.20  'ال':-1.87  'ل':-5.63  'ب':-5.90  'في':-6.10  'ف':-6.14
             root  : '<PARTICLE>':-1.85  '<P:كان>':-1.91  '<P:لم>':-3.23  'اله':-3.41  '<P:ذلك>':-3.62  '<P:كل>':-4.30
             wazn  : '<NONE>':-0.77  'فَعَلَ':-1.23  'فَاعِل':-3.39  'أَفْعَلَ':-3.74  'فِعَال':-3.91  'يَفْعُلُ':-4.05
             suffix: '<NONE>':-0.19  'ت':-2.91  'ة':-3.68  'ا':-3.77  'ه':-4.17  'نا':-4.73
             joint top5: ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'كان'[<NONE>+<P:كان>+<NONE>+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'كان'[<NONE>+<P:كان>+فَعَلَ+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+<NONE>+<NONE>]<SPECIAL>
  [pos 2] CONTEXT: فإن تك
           TRUE NEXT: 'هامة'  (1,8667,42,20) [<NONE>+همم+فَاعِل+ة] root_tag=real
           PRED top1: ''  (1,4,1,1) [<NONE>+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL   miss (NOT in joint top5)
             prefix: '<NONE>':-0.47  'ال':-1.56  'ب':-3.48  'ل':-3.79  'و':-3.86  'ف':-3.96
             root  : '<PARTICLE>':-2.09  '<P:أن>':-2.84  '<UNK>':-3.04  '<P:في>':-3.20  '<P:على>':-3.45  '<P:من>':-3.70
             wazn  : '<NONE>':-0.74  'فَعَلَ':-1.45  'فِعَال':-3.10  'فَاعِل':-3.51  'فُعُول':-3.57  'فَعِيل':-3.76
             suffix: '<NONE>':-0.19  'ة':-2.90  'ه':-3.60  'ا':-3.94  'ي':-4.46  'ها':-4.69
             joint top5: ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'ة'[<NONE>+<PARTICLE>+<NONE>+ة]<SPECIAL>
  [pos 3] CONTEXT: فإن تك هامة
           TRUE NEXT: 'بهراة'  (8,8478,45,20) [ب+هرا+فَعَلَ+ة] root_tag=real
           PRED top1: ''  (1,4,1,1) [<NONE>+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL   miss (NOT in joint top5)
             prefix: '<NONE>':-0.37  'ال':-2.07  'و':-3.06  'ف':-3.28  'ل':-3.84  'ب':-3.87
             root  : '<PARTICLE>':-1.87  '<P:في>':-3.12  '<P:إلى>':-3.49  '<P:على>':-3.56  '<UNK>':-3.63  '<P:من>':-3.65
             wazn  : '<NONE>':-0.79  'فَعَلَ':-1.29  'فِعَال':-3.13  'فَاعِل':-3.50  'فَعِيل':-3.67  'مَفْعَل':-4.09
             suffix: '<NONE>':-0.21  'ة':-2.75  'ا':-3.52  'ه':-3.82  'ها':-4.38  'ت':-4.42
             joint top5: ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ة'[<NONE>+<PARTICLE>+<NONE>+ة]<SPECIAL> || 'و'[و+<PARTICLE>+<NONE>+<NONE>]<SPECIAL>
  [pos 4] CONTEXT: فإن تك هامة بهراة
           TRUE NEXT: ''  (1,4,1,1) [<NONE>+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL
           PRED top1: ''  (1,4,1,1) [<NONE>+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL   EXACT (in joint top5)
             prefix: '<NONE>':-0.44  'ال':-1.39  'و':-3.38  'ف':-3.81  'ل':-4.54  'ب':-4.92
             root  : '<PARTICLE>':-1.68  '<P:أو>':-4.28  '<P:من>':-4.29  '<P:ما>':-4.30  '<P:لا>':-4.43  '<P:في>':-4.58
             wazn  : '<NONE>':-1.09  'فَعَلَ':-1.26  'فِعَال':-2.92  'فَاعِل':-3.04  'فُعُول':-3.42  'فَعِيل':-3.49
             suffix: '<NONE>':-0.39  'ة':-1.61  'ية':-3.62  'ها':-3.94  'ا':-4.24  'ين':-4.43
             joint top5: ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ة'[<NONE>+<PARTICLE>+<NONE>+ة]<SPECIAL> || 'ال'[ال+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL>
  [pos 5] CONTEXT: فإن تك هامة بهراة
           TRUE NEXT: 'فقد'  (1,6268,45,1) [<NONE>+فقد+فَعَلَ+<NONE>] root_tag=real
           PRED top1: ''  (1,4,1,1) [<NONE>+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL   miss (NOT in joint top5)
             prefix: '<NONE>':-0.34  'ال':-2.39  'و':-2.93  'ف':-3.08  'ب':-3.91  'ل':-4.04
             root  : '<PARTICLE>':-1.82  '<P:في>':-3.53  '<P:على>':-3.60  '<P:لا>':-3.64  '<UNK>':-3.65  '<P:أو>':-3.89
             wazn  : '<NONE>':-0.77  'فَعَلَ':-1.41  'فِعَال':-3.26  'فَاعِل':-3.28  'فَعِيل':-3.60  'فُعُول':-4.02
             suffix: '<NONE>':-0.23  'ة':-2.65  'ا':-3.57  'ه':-3.89  'ت':-3.94  'ها':-4.13
             joint top5: ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ة'[<NONE>+<PARTICLE>+<NONE>+ة]<SPECIAL> || 'و'[و+<PARTICLE>+<NONE>+<NONE>]<SPECIAL>
  [pos 6] CONTEXT: فإن تك هامة بهراة فقد
           TRUE NEXT: ''  (1,4,1,1) [<NONE>+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL
           PRED top1: ''  (1,4,45,1) [<NONE>+<PARTICLE>+فَعَلَ+<NONE>] root_tag=SPECIAL   root-hit (in joint top5)
             prefix: '<NONE>':-0.05  'ال':-4.39  'في':-5.36  'ل':-5.42  'و':-5.84  'ك':-5.89
             root  : '<PARTICLE>':-1.69  'قال':-2.99  '<P:كان>':-3.32  '<P:بين>':-3.33  'تقدم':-3.81  'كون':-3.86
             wazn  : 'فَعَلَ':-0.90  '<NONE>':-1.37  'يَفْعَلُ':-2.63  'تَفَعَّلَ':-3.14  'أَفْعَلَ':-3.27  'فَعْلَلَ':-3.66
             suffix: '<NONE>':-0.25  'ت':-2.37  'ه':-3.66  'نا':-3.79  'ها':-4.17  'ا':-4.74
             joint top5: ''[<NONE>+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ت'[<NONE>+<PARTICLE>+فَعَلَ+ت]<SPECIAL> || 'ت'[<NONE>+<PARTICLE>+<NONE>+ت]<SPECIAL> || 'ه'[<NONE>+<PARTICLE>+فَعَلَ+ه]<SPECIAL>
  [pos 7] CONTEXT: فإن تك هامة بهراة فقد
           TRUE NEXT: ''  (1,4,1,1) [<NONE>+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL
           PRED top1: ''  (1,4,1,1) [<NONE>+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL   EXACT (in joint top5)
             prefix: '<NONE>':-0.32  'ال':-2.39  'و':-3.03  'ب':-3.63  'ف':-3.64  'وال':-4.07
             root  : '<PARTICLE>':-1.67  '<UNK>':-3.12  '<P:في>':-3.19  '<P:من>':-3.47  '<P:على>':-3.75  '<P:لا>':-4.16
             wazn  : '<NONE>':-0.68  'فَعَلَ':-1.48  'فِعَال':-3.25  'فَاعِل':-3.45  'فَعِيل':-3.61  'يَفْعَلُ':-4.06
             suffix: '<NONE>':-0.20  'ة':-2.79  'ا':-3.75  'ه':-3.90  'ت':-4.23  'ها':-4.27
             joint top5: ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ة'[<NONE>+<PARTICLE>+<NONE>+ة]<SPECIAL> || 'و'[و+<PARTICLE>+<NONE>+<NONE>]<SPECIAL>
  [pos 8] CONTEXT: فإن تك هامة بهراة فقد
           TRUE NEXT: 'هما'  (1,28,42,1) [<NONE>+<P:هما>+فَاعِل+<NONE>] root_tag=SPECIAL
           PRED top1: ''  (1,4,1,1) [<NONE>+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL   miss (NOT in joint top5)
             prefix: '<NONE>':-0.31  'ال':-2.44  'و':-3.02  'ب':-3.70  'ف':-3.71  'وال':-4.00
             root  : '<PARTICLE>':-1.59  '<UNK>':-3.18  '<P:في>':-3.27  '<P:من>':-3.51  '<P:على>':-3.90  '<P:لا>':-4.23
             wazn  : '<NONE>':-0.69  'فَعَلَ':-1.47  'فِعَال':-3.28  'فَاعِل':-3.38  'فَعِيل':-3.53  'فُعُول':-4.06
             suffix: '<NONE>':-0.21  'ة':-2.75  'ا':-3.71  'ه':-3.85  'ها':-4.29  'ت':-4.31
             joint top5: ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ة'[<NONE>+<PARTICLE>+<NONE>+ة]<SPECIAL> || 'و'[و+<PARTICLE>+<NONE>+<NONE>]<SPECIAL>
  [pos 9] CONTEXT: فإن تك هامة بهراة فقد هما
           TRUE NEXT: 'وزقية'  (4,3507,45,21) [و+زقق+فَعَلَ+ية] root_tag=real
           PRED top1: ''  (1,4,1,1) [<NONE>+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL   miss (NOT in joint top5)
             prefix: '<NONE>':-0.31  'ال':-2.37  'و':-3.11  'ف':-3.74  'ب':-3.74  'ل':-3.94
             root  : '<PARTICLE>':-1.60  '<P:في>':-3.16  '<UNK>':-3.25  '<P:من>':-3.41  '<P:على>':-3.86  '<P:لا>':-4.11
             wazn  : '<NONE>':-0.72  'فَعَلَ':-1.49  'فِعَال':-3.28  'فَاعِل':-3.37  'فَعِيل':-3.50  'يَفْعَلُ':-3.93
             suffix: '<NONE>':-0.24  'ة':-2.66  'ا':-3.35  'ه':-3.80  'ها':-4.17  'ت':-4.33
             joint top5: ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ة'[<NONE>+<PARTICLE>+<NONE>+ة]<SPECIAL> || 'و'[و+<PARTICLE>+<NONE>+<NONE>]<SPECIAL>
```
```
--- test_gen sentence 296 (held-out WORK, never trained on)
RAW SENTENCE (test_gen #296, 11 words): وسقمت أجسامهم فأمرهم البيي أله على بالخروج إلى
ENCODED TUPLES: [(1, 4, 1, 1), (4, 3890, 45, 13), (1, 1349, 6, 5), (6, 77, 45, 5), (3, 820, 120, 1), (1, 4, 1, 1), (1, 283, 45, 1), (17, 3, 1, 1), (1, 4, 1, 1), (9, 2144, 74, 1), (1, 7, 1, 1)]
  [pos 1] CONTEXT: 
           TRUE NEXT: 'وسقمت'  (4,3890,45,13) [و+سقم+فَعَلَ+ت] root_tag=real
           PRED top1: ''  (1,4,1,1) [<NONE>+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL   miss (NOT in joint top5)
             prefix: '<NONE>':-0.30  'ال':-2.26  'و':-3.35  'ف':-3.91  'ب':-3.95  'ل':-4.11
             root  : '<PARTICLE>':-1.61  '<UNK>':-3.13  '<P:في>':-3.68  '<P:من>':-3.71  '<P:أن>':-3.91  '<P:لا>':-4.07
             wazn  : '<NONE>':-0.67  'فَعَلَ':-1.54  'فَاعِل':-3.32  'فِعَال':-3.46  'فَعِيل':-3.68  'يَفْعَلُ':-3.92
             suffix: '<NONE>':-0.18  'ة':-3.10  'ه':-3.96  'ا':-3.99  'ت':-4.18  'ي':-4.54
             joint top5: ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ة'[<NONE>+<PARTICLE>+<NONE>+ة]<SPECIAL> || 'و'[و+<PARTICLE>+<NONE>+<NONE>]<SPECIAL>
  [pos 2] CONTEXT: وسقمت
           TRUE NEXT: 'أجسامهم'  (1,1349,6,5) [<NONE>+جسم+أَفْعَال+هم] root_tag=real
           PRED top1: ''  (1,4,1,1) [<NONE>+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL   miss (NOT in joint top5)
             prefix: '<NONE>':-0.43  'ال':-1.75  'و':-3.18  'ب':-3.66  'ل':-3.88  'ف':-4.20
             root  : '<PARTICLE>':-1.63  '<UNK>':-3.26  '<P:من>':-3.59  '<P:في>':-3.85  '<P:على>':-4.00  '<P:ما>':-4.51
             wazn  : '<NONE>':-0.80  'فَعَلَ':-1.44  'فِعَال':-3.15  'فَاعِل':-3.39  'فَعِيل':-3.59  'فُعُول':-3.76
             suffix: '<NONE>':-0.26  'ة':-2.60  'ا':-3.70  'ه':-3.84  'ت':-4.26  'ها':-4.29
             joint top5: ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ة'[<NONE>+<PARTICLE>+<NONE>+ة]<SPECIAL> || 'ال'[ال+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL>
  [pos 3] CONTEXT: وسقمت أجسامهم
           TRUE NEXT: 'فأمرهم'  (6,77,45,5) [ف+أمر+فَعَلَ+هم] root_tag=real
           PRED top1: ''  (1,4,1,1) [<NONE>+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL   miss (NOT in joint top5)
             prefix: '<NONE>':-0.39  'و':-2.26  'ال':-2.91  'ف':-3.18  'ل':-3.66  'ب':-3.81
             root  : '<PARTICLE>':-1.64  '<P:من>':-3.40  '<P:في>':-3.49  '<UNK>':-3.66  '<P:على>':-3.83  '<P:لا>':-3.95
             wazn  : '<NONE>':-0.71  'فَعَلَ':-1.64  'فِعَال':-3.23  'فَعِيل':-3.59  'فَاعِل':-3.62  'فُعُول':-3.76
             suffix: '<NONE>':-0.30  'ة':-2.53  'ا':-3.32  'ها':-3.70  'ت':-3.91  'ه':-3.93
             joint top5: ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'و'[و+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ة'[<NONE>+<PARTICLE>+<NONE>+ة]<SPECIAL> || 'ال'[ال+<PARTICLE>+<NONE>+<NONE>]<SPECIAL>
  [pos 4] CONTEXT: وسقمت أجسامهم فأمرهم
           TRUE NEXT: 'البيي'  (3,820,120,1) [ال+بيي+يَفْعُلُ+<NONE>] root_tag=real
           PRED top1: ''  (1,4,1,1) [<NONE>+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL   miss (NOT in joint top5)
             prefix: '<NONE>':-0.31  'و':-2.83  'ال':-3.16  'ب':-3.35  'ف':-3.53  'ل':-3.66
             root  : '<PARTICLE>':-1.73  '<UNK>':-2.97  '<P:في>':-3.25  '<P:من>':-3.33  '<P:على>':-3.47  '<P:لا>':-3.93
             wazn  : '<NONE>':-0.64  'فَعَلَ':-1.60  'فَاعِل':-3.48  'فِعَال':-3.50  'فَعِيل':-3.64  'يَفْعَلُ':-3.94
             suffix: '<NONE>':-0.22  'ة':-3.12  'ا':-3.32  'ه':-3.79  'ها':-4.20  'ت':-4.29
             joint top5: ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'و'[و+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ة'[<NONE>+<PARTICLE>+<NONE>+ة]<SPECIAL>
  [pos 5] CONTEXT: وسقمت أجسامهم فأمرهم البيي
           TRUE NEXT: ''  (1,4,1,1) [<NONE>+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL
           PRED top1: ''  (1,4,1,1) [<NONE>+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL   EXACT (in joint top5)
             prefix: '<NONE>':-0.14  'على':-2.35  'ال':-4.49  'و':-5.17  'ف':-5.56  'وال':-5.60
             root  : '<PARTICLE>':-0.17  '<UNK>':-2.75  'ها':-5.61  '<P:لا>':-5.67  'علم':-5.70  '<P:على>':-5.79
             wazn  : '<NONE>':-0.10  'فَعَلَ':-2.95  'فَاعِل':-4.73  'فَعِيل':-4.85  'يَفْعَلُ':-5.21  'أَفْعَلَ':-5.40
             suffix: '<NONE>':-0.05  'ة':-3.53  'ه':-5.09  'ا':-5.80  'ية':-6.76  'ها':-6.77
             joint top5: ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'على'[على+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ة'[<NONE>+<PARTICLE>+<NONE>+ة]<SPECIAL> || 'ال'[ال+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'و'[و+<PARTICLE>+<NONE>+<NONE>]<SPECIAL>
  [pos 6] CONTEXT: وسقمت أجسامهم فأمرهم البيي
           TRUE NEXT: 'أله'  (1,283,45,1) [<NONE>+اله+فَعَلَ+<NONE>] root_tag=real
           PRED top1: 'أله'  (1,283,45,1) [<NONE>+اله+فَعَلَ+<NONE>] root_tag=real   EXACT (in joint top5)
             prefix: '<NONE>':-0.03  'ال':-4.86  'و':-4.91  'ب':-5.03  'ل':-5.57  'ف':-6.34
             root  : 'اله':-0.05  '<PARTICLE>':-4.06  '<UNK>':-5.88  '<P:في>':-6.48  '<P:من>':-6.49  'رسل':-6.68
             wazn  : 'فَعَلَ':-0.06  '<NONE>':-3.23  'فِعَال':-5.62  'فَعِيل':-5.75  'أَفْعَلَ':-5.90  'فَاعِل':-5.94
             suffix: '<NONE>':-0.02  'ه':-5.07  'ة':-5.93  'ا':-6.31  'ك':-6.56  'ت':-6.76
             joint top5: 'أله'[<NONE>+اله+فَعَلَ+<NONE>]<real> || 'الأله'[ال+اله+فَعَلَ+<NONE>]<real> || 'وأله'[و+اله+فَعَلَ+<NONE>]<real> || 'بأله'[ب+اله+فَعَلَ+<NONE>]<real> || 'ألهه'[<NONE>+اله+فَعَلَ+ه]<real>
  [pos 7] CONTEXT: وسقمت أجسامهم فأمرهم البيي أله
           TRUE NEXT: 'على'  (17,3,1,1) [على+<UNK>+<NONE>+<NONE>] root_tag=SPECIAL
           PRED top1: 'على'  (17,3,1,1) [على+<UNK>+<NONE>+<NONE>] root_tag=SPECIAL   EXACT (in joint top5)
             prefix: 'على':-0.22  'عن':-1.97  '<NONE>':-2.98  'ب':-5.58  'و':-6.58  'ل':-6.81
             root  : '<UNK>':-0.14  '<PARTICLE>':-3.07  'عنن':-3.43  'ها':-3.87  '<P:أن>':-5.72  '<P:على>':-5.98
             wazn  : '<NONE>':-0.07  'فَعَلَ':-3.39  'يَفْعَلُ':-5.25  'أَفْعَلَ':-5.36  'فَاعِل':-5.49  'فَعِيل':-5.91
             suffix: '<NONE>':-0.08  'هما':-4.04  'ه':-4.05  'هم':-4.45  'ا':-4.96  'ة':-5.08
             joint top5: 'على'[على+<UNK>+<NONE>+<NONE>]<SPECIAL> || 'عن'[عن+<UNK>+<NONE>+<NONE>]<SPECIAL> || ''[<NONE>+<UNK>+<NONE>+<NONE>]<SPECIAL> || 'علىهما'[على+<UNK>+<NONE>+هما]<SPECIAL> || 'علىه'[على+<UNK>+<NONE>+ه]<SPECIAL>
  [pos 8] CONTEXT: وسقمت أجسامهم فأمرهم البيي أله على
           TRUE NEXT: ''  (1,4,1,1) [<NONE>+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL
           PRED top1: ''  (1,4,1,1) [<NONE>+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL   EXACT (in joint top5)
             prefix: '<NONE>':-0.01  'ال':-4.90  'و':-6.65  'على':-7.70  'ف':-7.87  'ب':-8.52
             root  : '<PARTICLE>':-0.01  'صلا':-5.57  '<UNK>':-6.16  'قال':-7.35  '<P:من>':-7.62  '<P:في>':-7.99
             wazn  : '<NONE>':-0.03  'فَعَلَ':-3.76  'فِعَال':-6.52  'يَفْعُلُ':-7.10  'فَاعِل':-7.32  'يَفْعَلُ':-7.56
             suffix: '<NONE>':-0.01  'ة':-5.01  'ه':-7.37  'ا':-8.05  'ها':-8.82  'ية':-9.55
             joint top5: ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ة'[<NONE>+<PARTICLE>+<NONE>+ة]<SPECIAL> || 'و'[و+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ه'[<NONE>+<PARTICLE>+<NONE>+ه]<SPECIAL>
  [pos 9] CONTEXT: وسقمت أجسامهم فأمرهم البيي أله على
           TRUE NEXT: 'بالخروج'  (9,2144,74,1) [بال+خرج+فُعُول+<NONE>] root_tag=real
           PRED top1: ''  (1,4,1,1) [<NONE>+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL   miss (NOT in joint top5)
             prefix: '<NONE>':-0.27  'و':-2.84  'ال':-3.03  'ف':-3.55  'ب':-3.84  'ل':-3.90
             root  : '<PARTICLE>':-1.73  '<UNK>':-3.01  '<P:في>':-3.16  '<P:على>':-3.59  'قال':-3.72  '<P:من>':-3.88
             wazn  : '<NONE>':-0.67  'فَعَلَ':-1.42  'فِعَال':-3.07  'فَاعِل':-3.69  'أَفْعَلَ':-3.77  'فَعِيل':-3.80
             suffix: '<NONE>':-0.17  'ة':-3.25  'ه':-3.69  'ا':-3.97  'ت':-4.23  'ها':-4.59
             joint top5: ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'و'[و+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ة'[<NONE>+<PARTICLE>+<NONE>+ة]<SPECIAL>
  [pos 10] CONTEXT: وسقمت أجسامهم فأمرهم البيي أله على بالخروج
           TRUE NEXT: 'إلى'  (1,7,1,1) [<NONE>+<P:إلى>+<NONE>+<NONE>] root_tag=SPECIAL
           PRED top1: ''  (1,4,1,1) [<NONE>+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL   miss (NOT in joint top5)
             prefix: '<NONE>':-0.37  'و':-2.55  'ال':-3.07  'ف':-3.24  'وال':-3.43  'ب':-3.58
             root  : '<PARTICLE>':-1.92  '<P:في>':-2.78  '<UNK>':-3.02  '<P:من>':-3.38  '<P:إلى>':-3.48  '<P:عن>':-3.48
             wazn  : '<NONE>':-0.59  'فَعَلَ':-1.59  'فِعَال':-2.99  'فَاعِل':-3.81  'فَعِيل':-3.83  'يَفْعَلُ':-4.03
             suffix: '<NONE>':-0.17  'ة':-3.28  'ه':-3.58  'ا':-4.02  'ت':-4.60  'ها':-4.63
             joint top5: ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'و'[و+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ف'[ف+<PARTICLE>+<NONE>+<NONE>]<SPECIAL>
```
```
--- test_gen sentence 2194 (held-out WORK, never trained on)
RAW SENTENCE (test_gen #2194, 11 words): واسعة وضرع سجيل طويل وناقة عظيمة الضرع وساجل الرجل
ENCODED TUPLES: [(1, 8906, 42, 20), (4, 4819, 45, 1), (1, 3707, 55, 1), (1, 5173, 55, 1), (1, 4, 1, 1), (4, 8266, 42, 20), (1, 4, 1, 1), (1, 5536, 55, 20), (3, 4819, 45, 1), (4, 3707, 42, 1), (3, 3060, 45, 1)]
  [pos 1] CONTEXT: واسعة
           TRUE NEXT: 'وضرع'  (4,4819,45,1) [و+ضرع+فَعَلَ+<NONE>] root_tag=real
           PRED top1: ''  (1,4,1,1) [<NONE>+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL   miss (NOT in joint top5)
             prefix: '<NONE>':-0.33  'و':-2.54  'ال':-2.57  'ف':-3.44  'ل':-3.88  'ب':-4.26
             root  : '<PARTICLE>':-1.74  '<P:من>':-3.28  '<P:في>':-3.64  '<UNK>':-3.79  '<P:لا>':-3.99  '<P:على>':-4.28
             wazn  : '<NONE>':-0.80  'فَعَلَ':-1.40  'فِعَال':-3.25  'فَاعِل':-3.39  'فَعِيل':-3.52  'أَفْعَلَ':-3.76
             suffix: '<NONE>':-0.26  'ة':-2.68  'ا':-3.54  'ها':-3.86  'ه':-3.87  'ت':-4.00
             joint top5: ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'و'[و+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ة'[<NONE>+<PARTICLE>+<NONE>+ة]<SPECIAL>
  [pos 2] CONTEXT: واسعة وضرع
           TRUE NEXT: 'سجيل'  (1,3707,55,1) [<NONE>+سجل+فَعِيل+<NONE>] root_tag=real
           PRED top1: ''  (1,4,1,1) [<NONE>+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL   miss (NOT in joint top5)
             prefix: '<NONE>':-0.37  'ال':-2.12  'و':-2.93  'ل':-3.88  'ب':-3.88  'ف':-4.06
             root  : '<PARTICLE>':-1.84  '<UNK>':-4.01  '<P:من>':-4.13  '<P:في>':-4.37  'بنن':-4.48  'رجل':-4.82
             wazn  : '<NONE>':-1.11  'فَعَلَ':-1.23  'فِعَال':-3.03  'فَاعِل':-3.09  'فَعِيل':-3.24  'أَفْعَلَ':-3.40
             suffix: '<NONE>':-0.30  'ة':-2.79  'ا':-3.38  'ه':-3.50  'ت':-4.01  'ها':-4.09
             joint top5: ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'ة'[<NONE>+<PARTICLE>+<NONE>+ة]<SPECIAL>
  [pos 3] CONTEXT: واسعة وضرع سجيل
           TRUE NEXT: 'طويل'  (1,5173,55,1) [<NONE>+طول+فَعِيل+<NONE>] root_tag=real
           PRED top1: ''  (1,4,1,1) [<NONE>+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL   miss (NOT in joint top5)
             prefix: '<NONE>':-0.35  'ال':-2.41  'و':-2.66  'وال':-3.76  'ف':-3.82  'ب':-4.03
             root  : '<PARTICLE>':-1.81  '<P:من>':-3.58  'بنن':-3.76  '<UNK>':-3.95  '<P:في>':-4.22  'قال':-4.70
             wazn  : '<NONE>':-1.05  'فَعَلَ':-1.18  'فِعَال':-3.05  'فَعِيل':-3.15  'فَاعِل':-3.32  'يَفْعَلُ':-3.64
             suffix: '<NONE>':-0.24  'ة':-3.05  'ه':-3.29  'ا':-3.58  'ي':-4.24  'ها':-4.39
             joint top5: ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'و'[و+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL>
  [pos 4] CONTEXT: واسعة وضرع سجيل طويل
           TRUE NEXT: ''  (1,4,1,1) [<NONE>+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL
           PRED top1: ''  (1,4,1,1) [<NONE>+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL   EXACT (in joint top5)
             prefix: '<NONE>':-0.39  'ال':-2.25  'و':-2.66  'ف':-3.71  'ب':-3.83  'وال':-3.89
             root  : '<PARTICLE>':-1.77  '<P:من>':-3.37  '<UNK>':-3.66  '<P:في>':-4.19  'بنن':-4.38  '<P:هو>':-4.61
             wazn  : '<NONE>':-0.98  'فَعَلَ':-1.27  'فِعَال':-3.11  'فَعِيل':-3.13  'فَاعِل':-3.25  'يَفْعَلُ':-3.68
             suffix: '<NONE>':-0.25  'ة':-3.05  'ه':-3.18  'ا':-3.50  'ي':-4.28  'ها':-4.35
             joint top5: ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'و'[و+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL>
  [pos 5] CONTEXT: واسعة وضرع سجيل طويل
           TRUE NEXT: 'وناقة'  (4,8266,42,20) [و+نقق+فَاعِل+ة] root_tag=real
           PRED top1: ''  (1,4,1,1) [<NONE>+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL   miss (NOT in joint top5)
             prefix: '<NONE>':-0.38  'ال':-2.32  'و':-2.70  'ف':-3.62  'وال':-3.66  'ب':-3.87
             root  : '<PARTICLE>':-1.66  '<UNK>':-3.59  '<P:من>':-3.85  '<P:في>':-4.10  '<P:لا>':-4.59  '<P:على>':-4.60
             wazn  : '<NONE>':-0.92  'فَعَلَ':-1.31  'فَاعِل':-3.13  'فِعَال':-3.26  'فَعِيل':-3.27  'يَفْعَلُ':-3.75
             suffix: '<NONE>':-0.26  'ة':-2.93  'ه':-3.42  'ا':-3.51  'ت':-4.08  'ها':-4.20
             joint top5: ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'و'[و+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ة'[<NONE>+<PARTICLE>+<NONE>+ة]<SPECIAL>
  [pos 6] CONTEXT: واسعة وضرع سجيل طويل وناقة
           TRUE NEXT: ''  (1,4,1,1) [<NONE>+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL
           PRED top1: ''  (1,4,1,1) [<NONE>+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL   EXACT (in joint top5)
             prefix: '<NONE>':-0.34  'ال':-2.38  'و':-2.61  'ف':-3.69  'وال':-3.77  'ل':-4.17
             root  : '<PARTICLE>':-1.70  '<P:من>':-3.92  '<UNK>':-4.14  'بنن':-4.30  '<P:في>':-4.33  'عشر':-4.59
             wazn  : '<NONE>':-1.04  'فَعَلَ':-1.21  'فِعَال':-3.01  'فَعِيل':-3.22  'فَاعِل':-3.23  'أَفْعَلَ':-3.68
             suffix: '<NONE>':-0.27  'ة':-2.59  'ه':-3.58  'ا':-3.73  'ها':-4.02  'ت':-4.19
             joint top5: ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'و'[و+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ة'[<NONE>+<PARTICLE>+<NONE>+ة]<SPECIAL>
  [pos 7] CONTEXT: واسعة وضرع سجيل طويل وناقة
           TRUE NEXT: 'عظيمة'  (1,5536,55,20) [<NONE>+عظم+فَعِيل+ة] root_tag=real
           PRED top1: ''  (1,4,1,1) [<NONE>+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL   miss (NOT in joint top5)
             prefix: '<NONE>':-0.35  'ال':-2.36  'و':-2.74  'ف':-3.66  'وال':-3.67  'ب':-4.07
             root  : '<PARTICLE>':-1.61  '<UNK>':-3.74  '<P:من>':-4.00  '<P:في>':-4.19  '<P:لا>':-4.61  '<P:على>':-4.73
             wazn  : '<NONE>':-0.93  'فَعَلَ':-1.29  'فَاعِل':-3.14  'فِعَال':-3.21  'فَعِيل':-3.30  'أَفْعَلَ':-3.73
             suffix: '<NONE>':-0.27  'ة':-2.77  'ه':-3.55  'ا':-3.60  'ت':-3.94  'ها':-4.00
             joint top5: ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'و'[و+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ة'[<NONE>+<PARTICLE>+<NONE>+ة]<SPECIAL>
  [pos 8] CONTEXT: واسعة وضرع سجيل طويل وناقة عظيمة
           TRUE NEXT: 'الضرع'  (3,4819,45,1) [ال+ضرع+فَعَلَ+<NONE>] root_tag=real
           PRED top1: ''  (1,4,1,1) [<NONE>+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL   miss (NOT in joint top5)
             prefix: '<NONE>':-0.37  'ال':-2.24  'و':-2.49  'ف':-3.68  'وال':-3.84  'ل':-4.17
             root  : '<PARTICLE>':-1.73  '<P:من>':-3.82  '<UNK>':-4.34  'عشر':-4.57  'بنن':-4.67  '<P:في>':-4.69
             wazn  : '<NONE>':-1.11  'فَعَلَ':-1.22  'فِعَال':-3.02  'فَاعِل':-3.14  'فَعِيل':-3.14  'أَفْعَلَ':-3.60
             suffix: '<NONE>':-0.31  'ة':-2.46  'ه':-3.58  'ا':-3.72  'ها':-3.82  'ت':-4.08
             joint top5: ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'و'[و+<PARTICLE>+<NONE>+<NONE>]<SPECIAL>
  [pos 9] CONTEXT: واسعة وضرع سجيل طويل وناقة عظيمة الضرع
           TRUE NEXT: 'وساجل'  (4,3707,42,1) [و+سجل+فَاعِل+<NONE>] root_tag=real
           PRED top1: ''  (1,4,1,1) [<NONE>+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL   miss (NOT in joint top5)
             prefix: '<NONE>':-0.41  'و':-2.59  'ال':-2.65  'وال':-2.66  'ف':-3.58  'ب':-4.06
             root  : '<PARTICLE>':-1.83  '<P:من>':-4.05  '<P:في>':-4.24  '<UNK>':-4.30  'بنن':-4.66  'قال':-4.70
             wazn  : '<NONE>':-1.00  'فَعَلَ':-1.41  'فِعَال':-3.02  'فَاعِل':-3.07  'فَعِيل':-3.10  'أَفْعَلَ':-3.52
             suffix: '<NONE>':-0.31  'ة':-2.66  'ه':-3.42  'ا':-3.45  'ها':-3.86  'ية':-4.18
             joint top5: ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'و'[و+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'وال'[وال+<PARTICLE>+<NONE>+<NONE>]<SPECIAL>
  [pos 10] CONTEXT: واسعة وضرع سجيل طويل وناقة عظيمة الضرع وساجل
           TRUE NEXT: 'الرجل'  (3,3060,45,1) [ال+رجل+فَعَلَ+<NONE>] root_tag=real
           PRED top1: ''  (1,4,1,1) [<NONE>+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL   miss (NOT in joint top5)
             prefix: '<NONE>':-0.35  'ال':-2.44  'و':-2.67  'وال':-3.61  'ف':-3.82  'ب':-4.09
             root  : '<PARTICLE>':-1.94  '<P:من>':-3.64  '<UNK>':-4.18  'عشر':-4.31  'بنن':-4.33  '<P:في>':-4.39
             wazn  : '<NONE>':-1.10  'فَعَلَ':-1.16  'فِعَال':-2.99  'فَعِيل':-3.20  'فَاعِل':-3.25  'أَفْعَلَ':-3.60
             suffix: '<NONE>':-0.26  'ة':-2.96  'ه':-3.38  'ا':-3.47  'ها':-4.00  'ت':-4.36
             joint top5: ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'و'[و+<PARTICLE>+<NONE>+<NONE>]<SPECIAL>
```
```
--- test_gen sentence 385 (held-out WORK, never trained on)
RAW SENTENCE (test_gen #385, 12 words): بين العلم بالوضع وبين العلم بالغلبة فإن العلم بالوضع وضع
ENCODED TUPLES: [(1, 55, 1, 1), (3, 5660, 45, 1), (9, 8946, 45, 1), (1, 8774, 55, 1), (3, 5660, 45, 1), (9, 5952, 45, 20), (6, 14, 1, 1), (3, 5660, 45, 1), (9, 8946, 45, 1), (1, 3, 1, 1), (1, 8946, 45, 1), (1, 4, 1, 1)]
  [pos 1] CONTEXT: بين
           TRUE NEXT: 'العلم'  (3,5660,45,1) [ال+علم+فَعَلَ+<NONE>] root_tag=real
           PRED top1: ''  (1,4,1,1) [<NONE>+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL   miss (NOT in joint top5)
             prefix: '<NONE>':-0.41  'ال':-1.35  'أن':-3.08  'و':-5.30  'ب':-5.55  'ف':-5.65
             root  : '<PARTICLE>':-1.64  '<P:أن>':-2.33  '<P:ما>':-3.62  'اله':-3.99  '<UNK>':-4.07  '<P:هذا>':-4.07
             wazn  : '<NONE>':-0.94  'فَعَلَ':-1.27  'فِعَال':-3.07  'فَعِيل':-3.31  'فُعُول':-3.41  'يَفْعُلُ':-3.42
             suffix: '<NONE>':-0.22  'ة':-2.90  'ه':-3.19  'ين':-3.78  'ي':-4.12  'هم':-4.33
             joint top5: ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'أن'[أن+<PARTICLE>+<NONE>+<NONE>]<SPECIAL>
  [pos 2] CONTEXT: بين العلم
           TRUE NEXT: 'بالوضع'  (9,8946,45,1) [بال+وضع+فَعَلَ+<NONE>] root_tag=real
           PRED top1: ''  (1,4,1,1) [<NONE>+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL   miss (NOT in joint top5)
             prefix: '<NONE>':-0.51  'و':-2.44  'وال':-2.56  'ال':-2.59  'ف':-3.12  'ب':-3.47
             root  : '<PARTICLE>':-1.69  '<UNK>':-2.96  '<P:لا>':-3.35  '<P:من>':-3.66  '<P:في>':-3.68  '<P:هو>':-3.74
             wazn  : '<NONE>':-0.64  'فَعَلَ':-1.71  'فَاعِل':-3.33  'فِعَال':-3.38  'فَعِيل':-3.44  'فُعُول':-3.70
             suffix: '<NONE>':-0.14  'ة':-3.28  'ه':-3.59  'ا':-4.18  'ي':-4.73  'ية':-4.87
             joint top5: ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'و'[و+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'وال'[وال+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL>
  [pos 3] CONTEXT: بين العلم بالوضع
           TRUE NEXT: 'وبين'  (1,8774,55,1) [<NONE>+وبن+فَعِيل+<NONE>] root_tag=real
           PRED top1: ''  (1,4,1,1) [<NONE>+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL   miss (NOT in joint top5)
             prefix: '<NONE>':-0.55  'وال':-2.16  'ال':-2.23  'و':-2.54  'ف':-3.40  'ب':-3.87
             root  : '<PARTICLE>':-1.69  '<UNK>':-3.42  '<P:لا>':-3.56  '<P:في>':-3.66  '<P:من>':-3.69  '<P:الذي>':-3.86
             wazn  : '<NONE>':-0.70  'فَعَلَ':-1.67  'فَاعِل':-2.97  'فِعَال':-3.27  'فَعِيل':-3.33  'مَفْعَل':-3.82
             suffix: '<NONE>':-0.16  'ة':-2.84  'ه':-3.94  'ا':-4.23  'ية':-4.46  'ي':-4.88
             joint top5: ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'وال'[وال+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'و'[و+<PARTICLE>+<NONE>+<NONE>]<SPECIAL>
  [pos 4] CONTEXT: بين العلم بالوضع وبين
           TRUE NEXT: 'العلم'  (3,5660,45,1) [ال+علم+فَعَلَ+<NONE>] root_tag=real
           PRED top1: ''  (1,4,45,1) [<NONE>+<PARTICLE>+فَعَلَ+<NONE>] root_tag=SPECIAL   miss (NOT in joint top5)
             prefix: '<NONE>':-0.36  'ال':-1.30  'أن':-4.41  'و':-6.49  'ب':-6.52  'بال':-6.61
             root  : '<PARTICLE>':-1.75  '<P:أن>':-3.54  '<P:ما>':-3.59  '<P:ذلك>':-4.20  'بيي':-4.22  '<P:غير>':-4.26
             wazn  : 'فَعَلَ':-1.13  '<NONE>':-1.25  'فِعَال':-3.00  'فَاعِل':-3.01  'فَعِيل':-3.09  'فُعُول':-3.34
             suffix: '<NONE>':-0.17  'ة':-2.78  'ه':-3.42  'ين':-4.29  'ي':-4.62  'ها':-4.91
             joint top5: ''[<NONE>+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ة'[<NONE>+<PARTICLE>+فَعَلَ+ة]<SPECIAL>
  [pos 5] CONTEXT: بين العلم بالوضع وبين العلم
           TRUE NEXT: 'بالغلبة'  (9,5952,45,20) [بال+غلب+فَعَلَ+ة] root_tag=real
           PRED top1: ''  (1,4,1,1) [<NONE>+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL   miss (NOT in joint top5)
             prefix: '<NONE>':-0.60  'و':-2.28  'ال':-2.29  'وال':-2.48  'ف':-3.21  'ب':-3.43
             root  : '<PARTICLE>':-1.69  '<UNK>':-3.18  '<P:أن>':-3.57  '<P:من>':-3.58  '<P:لا>':-3.74  '<P:هو>':-3.75
             wazn  : '<NONE>':-0.77  'فَعَلَ':-1.55  'فَعِيل':-3.18  'فِعَال':-3.21  'فَاعِل':-3.28  'فُعُول':-3.56
             suffix: '<NONE>':-0.16  'ة':-3.14  'ه':-3.46  'ا':-4.15  'ي':-4.86  'ين':-4.92
             joint top5: ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'و'[و+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'وال'[وال+<PARTICLE>+<NONE>+<NONE>]<SPECIAL>
  [pos 6] CONTEXT: بين العلم بالوضع وبين العلم بالغلبة
           TRUE NEXT: 'فإن'  (6,14,1,1) [ف+<P:إن>+<NONE>+<NONE>] root_tag=SPECIAL
           PRED top1: ''  (1,4,1,1) [<NONE>+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL   miss (NOT in joint top5)
             prefix: '<NONE>':-0.54  'وال':-2.21  'و':-2.31  'ال':-2.62  'ف':-3.10  'ل':-3.98
             root  : '<PARTICLE>':-1.66  '<UNK>':-3.31  '<P:لا>':-3.45  '<P:على>':-3.48  '<P:في>':-3.55  '<P:من>':-3.92
             wazn  : '<NONE>':-0.64  'فَعَلَ':-1.71  'فَاعِل':-3.20  'فِعَال':-3.33  'فَعِيل':-3.46  'فُعُول':-3.79
             suffix: '<NONE>':-0.19  'ة':-2.65  'ه':-3.87  'ية':-4.19  'ا':-4.22  'ي':-4.74
             joint top5: ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'وال'[وال+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'و'[و+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL>
  [pos 7] CONTEXT: بين العلم بالوضع وبين العلم بالغلبة فإن
           TRUE NEXT: 'العلم'  (3,5660,45,1) [ال+علم+فَعَلَ+<NONE>] root_tag=real
           PRED top1: ''  (1,4,1,1) [<NONE>+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL   miss (NOT in joint top5)
             prefix: '<NONE>':-0.14  'ال':-2.23  'ل':-5.46  'ب':-5.90  'في':-6.26  'على':-6.29
             root  : '<PARTICLE>':-1.58  '<P:كان>':-1.69  '<P:لم>':-2.74  'اله':-3.51  '<P:ذلك>':-4.13  'قلن':-4.19
             wazn  : '<NONE>':-0.63  'فَعَلَ':-1.23  'فَاعِل':-3.54  'أَفْعَلَ':-3.74  'فُعُول':-4.38  'فِعَال':-4.46
             suffix: '<NONE>':-0.15  'ت':-3.15  'ا':-3.68  'ه':-4.00  'ة':-4.12  'نا':-5.12
             joint top5: ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'كان'[<NONE>+<P:كان>+<NONE>+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'كان'[<NONE>+<P:كان>+فَعَلَ+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+<NONE>+<NONE>]<SPECIAL>
  [pos 8] CONTEXT: بين العلم بالوضع وبين العلم بالغلبة فإن العلم
           TRUE NEXT: 'بالوضع'  (9,8946,45,1) [بال+وضع+فَعَلَ+<NONE>] root_tag=real
           PRED top1: ''  (1,4,1,1) [<NONE>+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL   miss (NOT in joint top5)
             prefix: '<NONE>':-0.46  'ال':-2.40  'ف':-2.93  'و':-2.97  'ب':-3.13  'وال':-3.55
             root  : '<PARTICLE>':-1.78  '<UNK>':-2.84  '<P:لا>':-3.44  '<P:أن>':-3.66  '<P:في>':-3.76  '<P:على>':-3.91
             wazn  : '<NONE>':-0.69  'فَعَلَ':-1.54  'فَاعِل':-3.08  'فَعِيل':-3.54  'فِعَال':-3.61  'فُعُول':-3.69
             suffix: '<NONE>':-0.18  'ة':-3.02  'ه':-3.58  'ا':-3.63  'ية':-4.80  'ي':-4.88
             joint top5: ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ف'[ف+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'و'[و+<PARTICLE>+<NONE>+<NONE>]<SPECIAL>
  [pos 9] CONTEXT: بين العلم بالوضع وبين العلم بالغلبة فإن العلم بالوضع
           TRUE NEXT: ''  (1,3,1,1) [<NONE>+<UNK>+<NONE>+<NONE>] root_tag=SPECIAL
           PRED top1: ''  (1,4,1,1) [<NONE>+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL   miss (NOT in joint top5)
             prefix: '<NONE>':-0.39  'ال':-2.37  'وال':-3.03  'و':-3.07  'ف':-3.07  'ب':-3.81
             root  : '<PARTICLE>':-1.73  '<P:لا>':-3.32  '<UNK>':-3.40  '<P:في>':-3.55  '<P:الذي>':-3.68  '<P:هو>':-3.79
             wazn  : '<NONE>':-0.67  'فَعَلَ':-1.66  'فَاعِل':-2.84  'فَعِيل':-3.48  'فِعَال':-3.54  'مَفْعَل':-3.76
             suffix: '<NONE>':-0.16  'ة':-2.97  'ا':-3.74  'ه':-3.88  'ية':-4.77  'ي':-4.97
             joint top5: ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'وال'[وال+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'و'[و+<PARTICLE>+<NONE>+<NONE>]<SPECIAL>
  [pos 10] CONTEXT: بين العلم بالوضع وبين العلم بالغلبة فإن العلم بالوضع
           TRUE NEXT: 'وضع'  (1,8946,45,1) [<NONE>+وضع+فَعَلَ+<NONE>] root_tag=real
           PRED top1: ''  (1,3,1,1) [<NONE>+<UNK>+<NONE>+<NONE>] root_tag=SPECIAL   miss (NOT in joint top5)
             prefix: '<NONE>':-0.20  'ال':-3.21  'ب':-3.29  'ل':-3.74  'أن':-3.97  'و':-4.33
             root  : '<UNK>':-2.05  '<PARTICLE>':-2.34  '<P:أن>':-3.09  '<P:لا>':-3.40  '<P:هو>':-3.50  '<P:في>':-3.53
             wazn  : '<NONE>':-0.62  'فَعَلَ':-1.52  'يَفْعَلُ':-3.42  'فَاعِل':-3.52  'فَعِيل':-4.07  'فِعَال':-4.11
             suffix: '<NONE>':-0.12  'ه':-3.62  'ة':-3.65  'ا':-4.19  'ت':-4.60  'ها':-4.91
             joint top5: ''[<NONE>+<UNK>+<NONE>+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || ''[<NONE>+<UNK>+فَعَلَ+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'ال'[ال+<UNK>+<NONE>+<NONE>]<SPECIAL>
  [pos 11] CONTEXT: بين العلم بالوضع وبين العلم بالغلبة فإن العلم بالوضع وضع
           TRUE NEXT: ''  (1,4,1,1) [<NONE>+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL
           PRED top1: ''  (1,4,1,1) [<NONE>+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL   EXACT (in joint top5)
             prefix: '<NONE>':-0.58  'ال':-1.28  'و':-3.54  'ل':-3.63  'ب':-3.63  'ف':-3.90
             root  : '<PARTICLE>':-1.93  '<UNK>':-2.88  '<P:من>':-3.64  '<P:في>':-3.78  '<P:أو>':-3.98  '<P:لا>':-4.12
             wazn  : '<NONE>':-0.88  'فَعَلَ':-1.20  'فَاعِل':-3.02  'فِعَال':-3.45  'فَعِيل':-3.63  'فُعُول':-3.82
             suffix: '<NONE>':-0.18  'ة':-2.96  'ه':-3.30  'ا':-3.72  'ين':-4.83  'ية':-4.92
             joint top5: ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'ة'[<NONE>+<PARTICLE>+<NONE>+ة]<SPECIAL>
```
```
--- test_gen sentence 1497 (held-out WORK, never trained on)
RAW SENTENCE (test_gen #1497, 9 words): لم وجد ولا ترابا الثاني عند الثالث
ENCODED TUPLES: [(1, 22, 1, 1), (1, 8807, 2, 1), (1, 4, 1, 1), (4, 20, 1, 1), (1, 3007, 31, 1), (3, 1140, 42, 1), (1, 54, 1, 1), (1, 4, 1, 1), (3, 1112, 42, 1)]
  [pos 1] CONTEXT: لم
           TRUE NEXT: 'وجد'  (1,8807,2,1) [<NONE>+وجد+<UNK>+<NONE>] root_tag=real
           PRED top1: ''  (1,4,120,1) [<NONE>+<PARTICLE>+يَفْعُلُ+<NONE>] root_tag=SPECIAL   miss (NOT in joint top5)
             prefix: '<NONE>':-0.02  'ال':-6.01  'على':-6.28  'ف':-6.50  'ل':-6.65  'من':-6.73
             root  : '<PARTICLE>':-1.74  'كنن':-1.82  '<P:لا>':-3.99  'وجد':-4.19  'علم':-4.29  'صحح':-4.30
             wazn  : 'يَفْعُلُ':-1.21  'يَفْعَلُ':-1.43  '<NONE>':-1.57  '<UNK>':-2.98  'فَعَلَ':-3.01  'تَفَعَّلَ':-3.30
             suffix: '<NONE>':-0.07  'ه':-3.40  'ا':-4.94  'ها':-5.22  'ة':-5.72  'هم':-5.83
             joint top5: ''[<NONE>+<PARTICLE>+يَفْعُلُ+<NONE>]<SPECIAL> || 'كنن'[<NONE>+كنن+يَفْعُلُ+<NONE>]<real> || ''[<NONE>+<PARTICLE>+يَفْعَلُ+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'يكنن'[<NONE>+كنن+يَفْعَلُ+<NONE>]<real>
  [pos 2] CONTEXT: لم وجد
           TRUE NEXT: ''  (1,4,1,1) [<NONE>+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL
           PRED top1: ''  (1,4,1,1) [<NONE>+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL   EXACT (in joint top5)
             prefix: '<NONE>':-0.36  'ال':-2.08  'ل':-2.97  'ب':-3.35  'في':-3.87  'لل':-4.32
             root  : '<PARTICLE>':-1.90  '<UNK>':-2.22  '<P:في>':-2.55  'ها':-3.41  '<P:من>':-3.58  '<P:ذلك>':-3.95
             wazn  : '<NONE>':-0.60  'فَعَلَ':-1.57  'فَاعِل':-3.15  'فِعَال':-3.53  'مَفْعَل':-3.78  'فَعِيل':-3.96
             suffix: '<NONE>':-0.17  'ا':-2.91  'ة':-3.13  'ه':-3.86  'ها':-4.89  'ية':-5.34
             joint top5: ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || ''[<NONE>+<UNK>+<NONE>+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'في'[<NONE>+<P:في>+<NONE>+<NONE>]<SPECIAL>
  [pos 3] CONTEXT: لم وجد
           TRUE NEXT: 'ولا'  (4,20,1,1) [و+<P:لا>+<NONE>+<NONE>] root_tag=SPECIAL
           PRED top1: ''  (1,4,1,1) [<NONE>+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL   miss (NOT in joint top5)
             prefix: '<NONE>':-0.40  'ال':-2.20  'ل':-3.13  'و':-3.19  'ب':-3.45  'ف':-3.64
             root  : '<PARTICLE>':-1.85  '<UNK>':-2.81  '<P:في>':-3.26  '<P:لا>':-3.49  '<P:من>':-3.87  '<P:على>':-3.87
             wazn  : '<NONE>':-0.68  'فَعَلَ':-1.56  'فَاعِل':-3.11  'فِعَال':-3.53  'فَعِيل':-3.67  'مَفْعَل':-3.76
             suffix: '<NONE>':-0.21  'ة':-2.83  'ا':-3.08  'ه':-3.85  'ها':-4.59  'ت':-4.73
             joint top5: ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ة'[<NONE>+<PARTICLE>+<NONE>+ة]<SPECIAL> || 'ل'[ل+<PARTICLE>+<NONE>+<NONE>]<SPECIAL>
  [pos 4] CONTEXT: لم وجد ولا
           TRUE NEXT: 'ترابا'  (1,3007,31,1) [<NONE>+ربا+تَفَاعَلَ+<NONE>] root_tag=real
           PRED top1: ''  (1,4,1,1) [<NONE>+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL   miss (NOT in joint top5)
             prefix: '<NONE>':-0.08  'ل':-4.26  'ال':-4.41  'ب':-4.46  'بال':-5.52  'و':-5.58
             root  : '<PARTICLE>':-1.90  'كون':-3.27  'جوز':-3.48  'مكن':-3.95  'بدد':-3.99  'وجد':-4.25
             wazn  : '<NONE>':-1.42  'يَفْعَلُ':-1.63  'فَعَلَ':-1.76  'تَفَعَّلَ':-2.83  'فِعَال':-3.19  'يَفْعُلُ':-3.36
             suffix: '<NONE>':-0.20  'ة':-2.67  'ه':-3.72  'ا':-3.98  'ي':-4.37  'ون':-4.37
             joint top5: ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+يَفْعَلُ+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'ة'[<NONE>+<PARTICLE>+<NONE>+ة]<SPECIAL> || ''[<NONE>+<PARTICLE>+تَفَعَّلَ+<NONE>]<SPECIAL>
  [pos 5] CONTEXT: لم وجد ولا ترابا
           TRUE NEXT: 'الثاني'  (3,1140,42,1) [ال+ثني+فَاعِل+<NONE>] root_tag=real
           PRED top1: ''  (1,4,1,1) [<NONE>+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL   miss (NOT in joint top5)
             prefix: '<NONE>':-0.49  'ال':-2.02  'ل':-2.68  'و':-3.12  'ب':-3.22  'في':-4.02
             root  : '<PARTICLE>':-1.84  '<UNK>':-2.36  '<P:في>':-3.34  '<P:لا>':-3.36  '<P:من>':-3.71  'ها':-3.77
             wazn  : '<NONE>':-0.71  'فَعَلَ':-1.48  'فَاعِل':-3.24  'فِعَال':-3.59  'فُعُول':-3.77  'فَعِيل':-3.86
             suffix: '<NONE>':-0.21  'ة':-2.78  'ا':-3.21  'ه':-3.64  'ها':-4.52  'ية':-4.76
             joint top5: ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || ''[<NONE>+<UNK>+<NONE>+<NONE>]<SPECIAL> || 'ل'[ل+<PARTICLE>+<NONE>+<NONE>]<SPECIAL>
  [pos 6] CONTEXT: لم وجد ولا ترابا الثاني
           TRUE NEXT: 'عند'  (1,54,1,1) [<NONE>+<P:عند>+<NONE>+<NONE>] root_tag=SPECIAL
           PRED top1: ''  (1,3,1,1) [<NONE>+<UNK>+<NONE>+<NONE>] root_tag=SPECIAL   miss (NOT in joint top5)
             prefix: '<NONE>':-0.39  'و':-2.78  'ل':-2.90  'ب':-3.24  'ال':-3.35  'ف':-3.52
             root  : '<UNK>':-1.99  '<PARTICLE>':-2.28  '<P:لا>':-2.82  '<P:أن>':-3.05  '<P:في>':-3.07  '<P:من>':-3.15
             wazn  : '<NONE>':-0.45  'فَعَلَ':-1.92  'فَاعِل':-3.36  'فَعِيل':-3.84  'مَفْعَل':-3.87  'يَفْعَلُ':-3.89
             suffix: '<NONE>':-0.10  'ه':-3.58  'ا':-3.61  'ة':-4.22  'ي':-5.21  'ك':-5.37
             joint top5: ''[<NONE>+<UNK>+<NONE>+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'و'[و+<UNK>+<NONE>+<NONE>]<SPECIAL> || 'ل'[ل+<UNK>+<NONE>+<NONE>]<SPECIAL> || 'ب'[ب+<UNK>+<NONE>+<NONE>]<SPECIAL>
  [pos 7] CONTEXT: لم وجد ولا ترابا الثاني عند
           TRUE NEXT: ''  (1,4,1,1) [<NONE>+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL
           PRED top1: ''  (1,4,45,1) [<NONE>+<PARTICLE>+فَعَلَ+<NONE>] root_tag=SPECIAL   root-hit (in joint top5)
             prefix: '<NONE>':-0.41  'ال':-1.15  'ل':-5.96  'أن':-6.21  'و':-6.23  'ب':-6.33
             root  : '<PARTICLE>':-1.82  'اله':-2.92  '<P:ذلك>':-3.24  '<P:هذا>':-3.96  '<P:ما>':-4.09  'بيي':-4.22
             wazn  : 'فَعَلَ':-1.07  '<NONE>':-1.40  'فَاعِل':-2.72  'فُعُول':-3.10  'فِعَال':-3.34  'فَعِيل':-3.50
             suffix: '<NONE>':-0.16  'ة':-2.87  'ه':-3.41  'ين':-4.40  'ية':-4.73  'ي':-4.83
             joint top5: ''[<NONE>+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'ة'[<NONE>+<PARTICLE>+فَعَلَ+ة]<SPECIAL>
  [pos 8] CONTEXT: لم وجد ولا ترابا الثاني عند
           TRUE NEXT: 'الثالث'  (3,1112,42,1) [ال+ثلث+فَاعِل+<NONE>] root_tag=real
           PRED top1: ''  (1,4,1,1) [<NONE>+<PARTICLE>+<NONE>+<NONE>] root_tag=SPECIAL   miss (NOT in joint top5)
             prefix: '<NONE>':-0.40  'ال':-2.37  'و':-2.88  'ل':-3.31  'ف':-3.35  'ب':-3.54
             root  : '<PARTICLE>':-1.96  '<UNK>':-2.67  '<P:لا>':-3.42  '<P:في>':-3.60  '<P:أن>':-3.69  '<P:على>':-3.78
             wazn  : '<NONE>':-0.64  'فَعَلَ':-1.49  'فَاعِل':-3.23  'فَعِيل':-3.66  'فِعَال':-3.69  'مَفْعَل':-4.01
             suffix: '<NONE>':-0.16  'ة':-3.27  'ا':-3.42  'ه':-3.49  'ي':-5.00  'ك':-5.03
             joint top5: ''[<NONE>+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || ''[<NONE>+<PARTICLE>+فَعَلَ+<NONE>]<SPECIAL> || 'ال'[ال+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || 'و'[و+<PARTICLE>+<NONE>+<NONE>]<SPECIAL> || ''[<NONE>+<UNK>+<NONE>+<NONE>]<SPECIAL>
```

## 6. What the sample can and cannot support

* **Can**: the qualitative claim that the model's output is context-insensitive. 80% identical top-1 on 400+ sampled positions across three splits, and a per-head table over 82 M evaluated positions (22.3 M train, 824 k train_seen, 35.8 k test_gen, 41.7 k test_deriv) is far more than a handful of prompts.
* **Can**: that the failure is under-training, not generalisation. The train-vs-held-out gap is ~1 point on a ~19% base, and the model is at the unigram marginal on every head.
* **Cannot**: say anything about whether an adequate amount of training would work. This checkpoint is 6,000 steps of 32.7 M parameters over a 9,114-way root output space with ~1.5 M word positions touched (about 0.06 of one epoch of 25 M words) — the training log itself shows the loss oscillating between 6.44 and 7.35 with no downward trend. Nothing here tests the morphemic hypothesis; it tests a model that had not started learning.

## 7. Side finding: the live vocabulary no longer matches the checkpoint

`sf_morph.pt` was trained under a **9,114-root / 130-wazn** id space. The live `nrmp_vocab.py` (rebuilt 2026-10-01 18:07) now exposes **9,490 roots / 142 awzan**. The new ids are *appended*, so ids < 9114 are stable — verified by checking that no prepared stream uses any id >= the checkpoint's head sizes, and by reproducing the published numbers. But an encoder built from the live vocab can emit ids the checkpoint cannot embed: scanning 12,000 corpus sentences, **245 distinct new root ids** (`<P:فإن>`, `<P:فإما>`, …) appear in 16,991 tokens, and every prompt harness here had to trim the context at the first out-of-range id. Any future decoding of this checkpoint with the current vocab must guard for that.

## 8. Artefacts

All under `/workspace/scratch_prompt/` on the pod, mirrored to `rootformer/build/scratch_prompt/`:

| file | contents |
|---|---|
| `per_head_acc.json` | per-head top-1/top-5 by split, unigram on identical positions, unigram on real-root-only targets |
| `vs_unigram.json` | marginal probability of the model's own predictions, per head/split |
| `root_marginal.json` | exact root-stream marginals, special/held-out/real split, constant-`<PARTICLE>` and unigram baselines |
| `collapse_analysis.json` | distinct top-1 tuples, top-5 share, P(`<PARTICLE>`), adjacent argmax change rate, what the argmax picked when correct |
| `special_bucket_test_gen.json`, `special_bucket_test_deriv.json` | composition of the `special` bucket |
| `prompts_testgen.json`, `prompts_heldout_positions.json` | the raw prompt records |
| `verify_ids.py`, `repro_check.py`, `probe2.py`, `fixups.py`, `collapse_analysis.py`, `vs_unigram.py`, `root_marginal.py`, `prompt_at_positions.py`, `corpus_read.py` | the scripts |
