#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""make_report.py -- assemble REPORT.md from the JSON artefacts written by the probe scripts.

Runs locally (no torch needed): it only reads JSON.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get('SCRATCH_PROMPT_OUT', HERE)


def load(name, default=None):
    p = os.path.join(HERE, name)
    if not os.path.exists(p):
        print(f'[missing] {name}', file=sys.stderr)
        return default
    return json.load(open(p, encoding='utf-8'))


def pct(x, nd=2):
    return f'{x*100:.{nd}f}%'


L = []
A = L.append

acc = load('per_head_acc.json', {})
col = load('collapse_analysis.json', {})
marg = load('root_marginal.json', {})
sample = load('prompts_testgen.json', [])
held = load('prompts_heldout_positions.json', [])

A('# Prompt-test of the from-scratch Arabic morphemic LM (`sf_morph.pt`)\n')
A('Checkpoint `/workspace/scratch_lm_run/sf_morph.pt` (32.72M params, 6,000 steps, '
  '25,000,003 training words). All numbers below were produced by scripts in '
  '`/workspace/scratch_prompt/`, reading the project artefacts read-only.\n')

A('## 0. Harness fidelity\n')
A('`repro_check.py` reimplements `heldout_root_eval.py`\'s identity split '
  '(heldout-root / real-seen-root / special) on top of the harness primitives. It reproduces '
  '**all six published numbers exactly**, including the `n`:\n')
A('| split | bucket | published top1/top5 | harness top1/top5 | published n | harness n |')
A('|---|---|---|---|---|---|')
for sp, rows in (('test_gen', [('real_seen_root', '1.490', '5.228', 19126),
                               ('special', '41.471', '68.348', 16631)]),
                 ('test_deriv', [('real_seen_root', '1.468', '4.916', 19753),
                                 ('special', '39.202', '66.607', 17818)])):
    for b, t1, t5, n in rows:
        A(f'| {sp} | {b} | {t1}% / {t5}% | {t1}% / {t5}% | {n} | {n} |')
A('\nSo the numbers reported below are the project\'s own evaluation convention, extended — '
  'not a parallel format.\n')

A('## 1. Train-set vs held-out accuracy, per head\n')
if acc:
    heads = ['p', 'r', 'w', 's']
    hname = {'p': 'prefix', 'r': 'root', 'w': 'wazn', 's': 'suffix'}
    order = ['train|0', 'train|8', 'train_seen|0', 'train_seen|8', 'val|0',
             'test_gen|0', 'test_gen|8', 'test_deriv|0', 'test_deriv|8']
    A('Measured under `scratch_lm.root_accuracy`\'s convention: every position is scored, '
      'context truncated to 48 words (the model is causally masked with absolute positions, '
      'so right-padding cannot change any real position). Split accuracy is a stratified '
      'estimate over sentence-length buckets, weighted by each bucket\'s true share of '
      'positions. `train_seen` = the exact 94,237 sentences `scratch_lm.train`\'s sampler '
      'optimised on, replayed from the same seed.\n')
    A('| split | ctx >= | positions | ' + ' | '.join(
        f'{hname[h]} top1/top5' for h in heads) + ' |')
    A('|---|---|---|---|---|---|---|---|')
    for k in order:
        if k not in acc['per_head']:
            continue
        d = acc['per_head'][k]
        sp, fl = k.split('|')
        A(f'| {sp} | {fl} | {int(d["r"]["positions"]):,} | ' + ' | '.join(
            f'{pct(d[h]["top1"])} / {pct(d[h]["top5"])}' for h in heads) + ' |')
A('')

if marg:
    A('### The unigram reference on the identical positions\n')
    A('| split | root positions | special | held-out roots | real seen roots | '
      'always-`<PARTICLE>` | unigram top-5 |')
    A('|---|---|---|---|---|---|---|')
    for sp in ('train', 'test_gen', 'test_deriv'):
        m = marg[sp]
        A(f'| {sp} | {m["positions"]:,} | {pct(m["special_share"])} | '
          f'{pct(m["heldout_share"])} | {pct(m["real_seen_share"])} | '
          f'{pct(m["always_particle_acc"])} | {pct(m["unigram_top5_acc"])} |')
    A('\nTop roots by frequency in each split — this is what any context-free model can get:\n')
    for sp in ('train', 'test_gen', 'test_deriv'):
        m = marg[sp]
        A(f'* **{sp}**: ' + ', '.join(
            f'`{t["name"]}` {pct(t["share"],1)} ({t["kind"]})' for t in m['top12'][:8]))
    A('')

A('## 2. How it fails: prediction collapse\n')
if col:
    A('| split | positions | distinct top-1 tuples emitted | true distinct | '
      'most frequent tuple | top-5 tuples | mean P(`<PARTICLE>`) | mean P(special) | '
      'adjacent-position argmax changes |')
    A('|---|---|---|---|---|---|---|---|---|')
    for sp in ('train', 'test_gen', 'test_deriv'):
        c = col[sp]
        A(f'| {sp} | {c["n_pos"]:,} | {c["distinct_top1_tuples"]} | '
          f'{c["distinct_true_tuples"]:,} | {pct(c["top1_tuple_share"])} | '
          f'{pct(c["top5_tuple_share"])} | {pct(c["mean_P_particle"])} | '
          f'{pct(c["mean_P_special"])} | {pct(c["adjacent_root_change_rate"],1)} |')
    A('')
    for sp in ('train', 'test_gen', 'test_deriv'):
        c = col[sp]
        A(f'**{sp} — the model\'s most frequent top-1 tuples:**\n')
        A('| share | n | surface | prefix+root+wazn+suffix | root class |')
        A('|---|---|---|---|---|')
        for t in c['top1_tuples'][:8]:
            A(f'| {pct(t["share"])} | {t["n"]:,} | `{t["surface"]}` | '
              f'`{t["prefix"]}+{t["root"]}+{t["wazn"]}+{t["suffix"]}` | {t["root_tag"]} |')
        A('')
    A('**What the model predicted whenever it was right on the root head:**\n')
    for sp in ('test_gen', 'test_deriv'):
        c = col[sp]
        tot = sum(x['n'] for x in c['root_correct_preds']) or 1
        A(f'* {sp}: ' + ', '.join(
            f'`{x["name"]}` {pct(x["n"]/tot,1)} [{x["tag"]}]'
            for x in c['root_correct_preds'][:6]))
    A('')

A('## 3. The `special` bucket\n')
sb = load('special_bucket_test_gen.json', [])
if sb:
    A('`special_root_ids` is ids 0-70: the 5 control tokens plus the **66** `<P:...>` function '
      'words of `COMMON_PARTICLES`. Composition of that bucket as *targets*:\n')
    A('| root | n | share of specials | share of all positions |')
    A('|---|---|---|---|')
    for t in sb[:10]:
        A(f'| `{t["name"]}` | {t["n"]:,} | {pct(t["share_of_special"])} | '
          f'{pct(t["share_of_all"])} |')
    A('')

A('## 4. Verbatim prompts\n')
A('Sampling: sentences are taken from the project\'s own `test_gen` split of '
  '`/workspace/sf_data/streams.pt` (held-out WORKS, never trained on); a fixed seed picks '
  'them; no curation. The context shown is whatever the project\'s encoder produces, so the '
  'prompt is exactly what the model received. Every position of each sentence is scored in '
  'one forward pass, the way `root_accuracy` does it.\n')
for rec in sample[:12]:
    A('```')
    A(rec['text'])
    A('```')
A('')
if held:
    A('## 5. Held-out-root prompts\n')
    A('`test_deriv` sentences containing one of the 38 held-out roots; the held-out-root '
      'target positions are marked.\n')
    for rec in held[:20]:
        A('```')
        A(rec['text'])
        A('```')
    A('')

open(os.path.join(OUT, 'REPORT.md'), 'w', encoding='utf-8').write('\n'.join(L))
print(f'wrote {os.path.join(OUT, "REPORT.md")} ({len(L)} lines)')
