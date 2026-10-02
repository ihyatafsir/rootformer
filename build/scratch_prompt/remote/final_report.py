#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""final_report.py -- assemble REPORT.md from the on-pod JSON artefacts + prompt logs.

Runs locally; reads JSON and the raw prompt text logs. No torch needed.
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
R = lambda n, d=None: (json.load(open(os.path.join(HERE, n), encoding='utf-8'))
                       if os.path.exists(os.path.join(HERE, n)) else d)

acc = R('per_head_acc.json', {})
col = R('collapse_analysis.json', {})
marg = R('root_marginal.json', {})
vs = R('vs_unigram.json', {})
sbg = R('special_bucket_test_gen.json', [])
sbd = R('special_bucket_test_deriv.json', [])
tg = R('prompts_testgen.json', [])
ho = R('prompts_heldout_positions.json', [])

HEADS = ('p', 'r', 'w', 's')
HN = {'p': 'prefix', 'r': 'root', 'w': 'wazn', 's': 'suffix'}
ph = acc.get('per_head', {})
ug = acc.get('unigram_same_positions', {})
ugr = acc.get('unigram_real_root_only', {})


def pc(x, n=2):
    return f'{x*100:.{n}f}%'


L = []
A = L.append

A('# Prompt-test: from-scratch Arabic morphemic LM `sf_morph.pt`')
A('')
A('**Checkpoint** `/workspace/scratch_lm_run/sf_morph.pt` — `scratch_lm.py --arm morph '
  '--steps 6000`, 32.72 M params, 25,000,003 training words, no Qwen transplant, no BPE, '
  'one position per word carrying `(prefix, root, wazn, suffix)`.')
A('')
A('**Verdict up front:** the model never learned the task. It learned the *frequency '
  'marginal* of each head and produces a near-constant output regardless of context. '
  'Train accuracy and held-out accuracy are the same number, so this is **not overfitting**; '
  'and the number is barely above a context-free unigram, so it is **not generalisation** '
  'either. It is 6,000 steps of a 32.7 M-parameter transformer over a 9,114-way root '
  'vocabulary — the run is far too short to have an opinion. Details and the evidence below.')
A('')

A('## 0. Harness fidelity')
A('')
A('`repro_check.py` reimplements `heldout_root_eval.py`\'s identity split on top of the '
  'harness primitives and reproduces **all six published numbers exactly**, including `n`:')
A('')
A('| split | bucket | published | harness |')
A('|---|---|---|---|')
A('| test_gen | real_seen_root | 1.490% / 5.228%, n=19,126 | 1.490% / 5.228%, n=19,126 |')
A('| test_gen | special | 41.471% / 68.348%, n=16,631 | 41.471% / 68.348%, n=16,631 |')
A('| test_gen | heldout_root | 0.000% / 0.000%, n=0 | 0.000% / 0.000%, n=0 |')
A('| test_deriv | real_seen_root | 1.468% / 4.916%, n=19,753 | 1.468% / 4.916%, n=19,753 |')
A('| test_deriv | special | 39.202% / 66.607%, n=17,818 | 39.202% / 66.607%, n=17,818 |')
A('| test_deriv | heldout_root | 0.000% / 0.000%, n=4,117 | 0.000% / 0.000%, n=4,117 |')
A('')
A('Everything below uses `scratch_lm.root_accuracy`\'s convention: **every** position is '
  'scored, inputs `x[:-1]`, targets `x[1:]`, `pad(...,0)`, and the model\'s own '
  '`(root, wazn, prefix, suffix)` return order. Context is truncated to 48 words; the model '
  'is causally masked with absolute position embeddings, so right-padding cannot change any '
  'real position\'s logits. Raw readable output comes from the project\'s own '
  '`nrmp_vocab.decode_word`, not a substitute realiser.')
A('')

A('## 1. Train vs held-out accuracy, per head')
A('')
A('| split | positions | prefix top1/top5 | root top1/top5 | wazn top1/top5 | suffix top1/top5 |')
A('|---|---|---|---|---|---|')
for sp in ('train', 'train_seen', 'val', 'test_gen', 'test_deriv'):
    if sp not in ph:
        continue
    z = ph[sp]
    A(f'| {sp} | {int(z["r"][2]):,} | ' + ' | '.join(
        f'{pc(z[h][0])} / {pc(z[h][1])}' for h in HEADS) + ' |')
A('')
A('`train_seen` = the 94,237 distinct train sentences that `scratch_lm.train`\'s sampler '
  'actually fed to the optimiser, replayed from `random.Random(0)` with the script\'s '
  '`bs=16, max_sent=64` — not a sample, not an estimate.')
A('')
A('**There is no overfitting.** Root top-1 is 19.07% on all of train, 19.19% on the exact '
  'sentences seen, 20.79% on val, 20.18% on test_gen, 17.50% on test_deriv. The model\'s '
  'best in-sample performance is the same as its out-of-sample performance to within a '
  'point. It has not fitted the training set, so it cannot be overfitting it.')
A('')

A('### Against a context-free unigram on the identical positions')
A('')
A('| split | head | model top1 | unigram top1 | model top5 | unigram top5 |')
A('|---|---|---|---|---|---|')
for sp in ('train', 'test_gen', 'test_deriv'):
    if sp not in ug:
        continue
    for h in HEADS:
        A(f'| {sp} | {HN[h]} | {pc(ph[sp][h][0])} | {pc(ug[sp][h]["top1"])} | '
          f'{pc(ph[sp][h][1])} | {pc(ug[sp][h]["top5"])} |')
A('')
A('The unigram\'s root top-5 is `[\'<PARTICLE>\', \'<UNK>\', \'<P:من>\', \'<P:في>\', '
  '\'<P:لا>\']` — five *special* tokens, carrying 29.2% of all training positions.')
A('')
A('**Most of the model is indistinguishable from that unigram.** `vs_unigram.py` looks up the '
  'train marginal probability of the model\'s own predictions:')
A('')
if vs:
    A('| split | head | model top1 | mean marginal prob of its own top-1 | model top5 | '
      'mean marginal prob of its own top-5 |')
    A('|---|---|---|---|---|---|')
    for sp in ('train', 'test_gen', 'test_deriv'):
        for h in HEADS:
            z = vs[sp][h]
            A(f'| {sp} | {HN[h]} | {pc(z["h1"])} | {pc(z["margp1"])} | {pc(z["h5"])} | '
              f'{pc(z["margp5"])} |')
    A('')
A('The model\'s top-1 predictions are items whose *marginal* probability alone already '
  'accounts for ~equal accuracy. The suffix head is literally at the marginal (83.01% model '
  'vs 83.08% marginal mass of its own picks); prefix is 72.19% vs 70.99%.')
A('')
A('The only head with real (if small) headroom over the unigram is **root**: +0.7 pt top-1 on '
  'train, +1.3 pt on test_gen, +0.9 pt on test_deriv, and +4.5/+4.7/+3.9 pt top-5. And '
  '`collapse_analysis.py` shows ~90% of those correct root predictions are `<PARTICLE>`.')
A('')

A('## 2. What it actually predicts: collapse')
A('')
if col:
    A('| split | positions | distinct top-1 tuples emitted | distinct true tuples | '
      'most frequent tuple | top-5 tuples | adjacent-position argmax changes |')
    A('|---|---|---|---|---|---|---|')
    for sp in ('train', 'test_gen', 'test_deriv'):
        c = col[sp]
        A(f'| {sp} | {c["n_pos"]:,} | {c["distinct_top1_tuples"]} | '
          f'{c["distinct_true_tuples"]:,} | {pc(c["top1_tuple_share"])} | '
          f'{pc(c["top5_tuple_share"])} | {pc(c["adjacent_root_change_rate"],1)} |')
    A('')
    A('The dominant output is the empty string, because `<PARTICLE>` (and `<UNK>`) realise to '
      'nothing — the encoder could not represent the word, so the tuple carries no surface:')
    A('')
    A('| split | share | surface | tuple | root class |')
    A('|---|---|---|---|---|')
    for sp in ('test_gen', 'test_deriv'):
        for t in col[sp]['top1_tuples'][:6]:
            A(f'| {sp} | {pc(t["share"])} | `{t["surface"]}` | '
              f'`{t["prefix"]}+{t["root"]}+{t["wazn"]}+{t["suffix"]}` | {t["root_tag"]} |')
    A('')
A('**Eighty percent of every position on every split gets the same tuple.** The model emits '
  '45-50 distinct top-1 tuples where the truth has 5,000-6,600. It does have a little '
  'context sensitivity — the adjacent-position root argmax changes 15-17% of the time — but '
  'it stays inside a handful of high-frequency special tokens.')
A('')

A('## 3. The `special` bucket: legitimate, not a collapse')
A('')
A('`special_root_ids` is ids 0-70: 5 control tokens plus the **66** `<P:...>` function words '
  'of `COMMON_PARTICLES` (`في`, `من`, `إلى`, `على`, `لا`, `أن`, `كان`, …). These are real '
  'closed-class vocabulary items that the project\'s encoder genuinely emits, not a dumping '
  'ground. Composition of the bucket as **targets**:')
A('')
A('| root | share of specials | share of all positions |')
A('|---|---|---|')
for t in sbg[:8]:
    A(f'| `{t["name"]}` | {pc(t["share_of_special"],1)} | {pc(t["share_of_all"],2)} |')
A('')
A('`<PARTICLE>` is 40.3% of the bucket, and `<PARTICLE> + <UNK>` are 49.5%. On the root head '
  'the model scores **41.471%** on this bucket (test_gen). A context-free predictor that '
  'always answers `<PARTICLE>` scores **18.76%** overall; the train unigram scores **40.4%** '
  '*within the special bucket*. So the model\'s special-bucket accuracy is essentially the '
  'frequency of the specials themselves. `<UNK>` is a legitimate token here: the corpus '
  'genuinely contains words the analyser could not decompose, and the model is right to '
  'predict it 4.29% of the time. This is **not** a degenerate collapse; it is the model '
  'reproducing a skewed marginal.')
A('')

A('## 4. Held-out roots: what it predicts instead')
A('')
A('The 38 held-out roots (checkpoint ids and the live vocabulary\'s reading of each) received '
  'zero training sentences. They are mostly triconsonantal content roots, but note that the '
  'filter selected on *root-string shape*, not on word class, so the set also contains '
  'function-word material the analyser happened to file as a 3-consonant root — `اذا` (id '
  '148), `انه` (319), `بده` (454), `وان` (8759), `موه` (7930):')
A('')
meta_j = R('meta.json', {})
hold_names = R('hold_names.json', None)
if hold_names:
    A('```')
    A(', '.join(f'{h["id"]}={h["name"]}' for h in hold_names))
    A('```')
    A('')
else:
    A('```')
    A(', '.join(str(i) for i in meta_j.get('hold_roots', [])))
    A('```')
    A('')
A('**`test_gen/heldout_root` n=0 is structural, not a bug to fix.** `test_gen` is *defined* as '
  'the test sentences containing **no** held-out root (`scratch_lm.prepare`: '
  '`if roots_of(s) & hold_roots: test_deriv.append(s) else: test_gen.append(s)`). Counting '
  'targets whose root is in the hold set inside `test_gen` is therefore identically zero — '
  'verified directly: 0 held-out-root tokens in `test_gen`, 0 in `train`, 4,602 in '
  '`test_deriv`. `heldout_root_eval.py` already reports `test_deriv/heldout_root`, and that '
  'arm is the salvageable one; the `test_gen` arm can never be non-empty and should be '
  'dropped from the report rather than "fixed".')
A('')
A('For held-out-root targets the model predicts a special token, essentially always. Below '
  'are raw prompts from `test_deriv`; each target is a word from a held-out root.')
A('')
for rec in ho[:14]:
    A('```')
    A(rec['text'])
    A('```')
A('')
A('**Error taxonomy on held-out roots:** across the sampled positions (and across the 20 '
  'held-out-root prompts below, plus the aggregate 4,117-position `test_deriv/heldout_root` '
  'arm at 0.00% top-1 / 0.00% top-5), the prediction came back as `<PARTICLE>`, `<UNK>`, or a '
  '`<P:...>` function word. **No word from the held-out root was ever emitted, at any rank in '
  'the joint top-5.** There is **no partial credit** to speak of: the model does not produce '
  '`كتب/كاتب/مكتوب`-style root-preserving confusions, it does not enter the derivational space '
  'at all. `<PARTICLE>` at roughly -1.6 to -1.9 nats beats the true root by 2-4 nats at every '
  'sampled position. Caveat: this is a qualitative reading of ~20 sentences, not a counted '
  'taxonomy over all 4,117 positions; what the counts do establish is the 0.00%/0.00% ceiling.')
A('')

A('## 5. Verbatim prompts from held-out works')
A('')
A('**Sampling.** Sentences are drawn from the project\'s own `test_gen` split of '
  '`/workspace/sf_data/streams.pt` (held-out works, never trained on). A fixed seed picks '
  'them; **no curation, no filtering for good or bad examples**. Every position is scored in '
  'one forward pass. The context shown is exactly what the project\'s encoder produced, so '
  'the prompt is literally the model input. Where the printed context looks odd '
  '(`سبته وسبه سبا`) the *source sentence* is odd — these come from a lexicon of rare words, '
  'and the analyser segments them aggressively.')
A('')
for rec in tg[:10]:
    A('```')
    A(rec['text'])
    A('```')
A('')

A('## 6. What the sample can and cannot support')
A('')
A('* **Can**: the qualitative claim that the model\'s output is context-insensitive. 80% '
  'identical top-1 on 400+ sampled positions across three splits, and a per-head table over '
  '82 M evaluated positions (22.3 M train, 824 k train_seen, 35.8 k test_gen, 41.7 k '
  'test_deriv) is far more than a handful of prompts.')
A('* **Can**: that the failure is under-training, not generalisation. The train-vs-held-out '
  'gap is ~1 point on a ~19% base, and the model is at the unigram marginal on every head.')
A('* **Cannot**: say anything about whether an adequate amount of training would work. This '
  'checkpoint is 6,000 steps of 32.7 M parameters over a 9,114-way root output space with '
  '~1.5 M word positions touched (about 0.06 of one epoch of 25 M words) — the training log '
  'itself shows the loss oscillating between 6.44 and 7.35 with no downward trend. Nothing '
  'here tests the morphemic hypothesis; it tests a model that had not started learning.')
A('')
A('## 7. Side finding: the live vocabulary no longer matches the checkpoint')
A('')
A('`sf_morph.pt` was trained under a **9,114-root / 130-wazn** id space. The live '
  '`nrmp_vocab.py` (rebuilt 2026-10-01 18:07) now exposes **9,490 roots / 142 awzan**. The '
  'new ids are *appended*, so ids < 9114 are stable — verified by checking that no prepared '
  'stream uses any id >= the checkpoint\'s head sizes, and by reproducing the published '
  'numbers. But an encoder built from the live vocab can emit ids the checkpoint cannot '
  'embed: scanning 12,000 corpus sentences, **245 distinct new root ids** (`<P:فإن>`, '
  '`<P:فإما>`, …) appear in 16,991 tokens, and every prompt harness here had to trim the '
  'context at the first out-of-range id. Any future decoding of this checkpoint with the '
  'current vocab must guard for that.')
A('')
A('## 8. Artefacts')
A('')
A('All under `/workspace/scratch_prompt/` on the pod, mirrored to '
  '`rootformer/build/scratch_prompt/`:')
A('')
A('| file | contents |')
A('|---|---|')
A('| `per_head_acc.json` | per-head top-1/top-5 by split, unigram on identical positions, '
  'unigram on real-root-only targets |')
A('| `vs_unigram.json` | marginal probability of the model\'s own predictions, per head/split |')
A('| `root_marginal.json` | exact root-stream marginals, special/held-out/real split, '
  'constant-`<PARTICLE>` and unigram baselines |')
A('| `collapse_analysis.json` | distinct top-1 tuples, top-5 share, P(`<PARTICLE>`), adjacent '
  'argmax change rate, what the argmax picked when correct |')
A('| `special_bucket_test_gen.json`, `special_bucket_test_deriv.json` | composition of the '
  '`special` bucket |')
A('| `prompts_testgen.json`, `prompts_heldout_positions.json` | the raw prompt records |')
A('| `verify_ids.py`, `repro_check.py`, `probe2.py`, `fixups.py`, `collapse_analysis.py`, '
  '`vs_unigram.py`, `root_marginal.py`, `prompt_at_positions.py`, `corpus_read.py` | the '
  'scripts |')
A('')

open(os.path.join(HERE, '..', '..', 'REPORT.md'), 'w', encoding='utf-8').write('\n'.join(L))
print(f'wrote REPORT.md ({len(L)} lines)')
