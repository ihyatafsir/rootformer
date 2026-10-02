#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""collapse_analysis.py -- quantify the prediction collapse and the `special` bucket.

Measures, per split:
  * how many DISTINCT argmax tuples the model ever emits, and the top-5 share of them
  * the mean per-position probability mass the model puts on <PARTICLE>
  * context sensitivity: among the positions the model scored, how often do two positions of
    the SAME sentence get different top-1 roots?  A constant predictor -> 0.
  * the identity of the correct predictions: which root id did the argmax pick when it was right?
Batched over whole sentences, one forward pass per batch.
"""
import json
import random
import sys
from collections import Counter

import torch

sys.path.insert(0, '/workspace/hf_v19_2_release')
sys.path.insert(0, '/workspace/hf_v19_2_release/models')

D = '/workspace/sf_data'
OUT = '/workspace/scratch_prompt'
CKPT = '/workspace/scratch_lm_run/sf_morph.pt'
W = 48
BS = 64

meta = json.load(open(f'{D}/meta.json'))
NR, NW, NP, NS = meta['n_roots'], meta['n_awzan'], meta['n_prefixes'], meta['n_suffixes']
SPECIAL = set(meta['special_root_ids'])
HOLD = set(meta['hold_roots'])
HEADS = ('p', 'r', 'w', 's')
IDX = {'p': 0, 'r': 1, 'w': 2, 's': 3}

print('[*] loading ...', flush=True)
RAW = torch.load(f'{D}/streams.pt', weights_only=False)
import scratch_lm as S      # noqa: E402
import nrmp_vocab as nv     # noqa: E402

_V = next(v for k, v in vars(nv).items() if isinstance(v, type) and 'MorphemicVocab' in k)
VOCAB = _V('/workspace/hf_v19_2_release/data/rootformer_v12_arabic_blueprint.json')
device = torch.device('cpu')
model = S.MorphemicLM((NR, NW, NP, NS)).to(device)
model.load_state_dict(torch.load(CKPT, map_location=device, weights_only=True))
model.eval()

PARTICLE = VOCAB.root2id['<PARTICLE>']


@torch.no_grad()
def analyse(split, n_batches=40, seed=3):
    d = RAW[split]
    n = len(d['L'])
    rng = random.Random(seed)
    top1_tuples = Counter()
    true_tuples = Counter()
    correct_preds = {h: Counter() for h in HEADS}
    per_head_pred = {h: Counter() for h in HEADS}
    mass_particle = 0.0
    mass_special = 0.0
    n_pos = 0
    same_sent_changes = 0
    same_sent_pairs = 0
    distinct_root_argmax_per_sent = []
    for _ in range(n_batches):
        ch = rng.sample(range(n), min(BS, n))
        Ls = [min(int(d['L'][i]), W) for i in ch]
        if max(Ls) < 2:
            continue
        T = max(Ls) - 1
        sub = torch.zeros((len(ch), 4, T + 1), dtype=torch.long)
        for b, i in enumerate(ch):
            for k, key in enumerate(('P', 'R', 'W', 'S')):
                seq = d[key][i][:T + 1]
                sub[b, k, :len(seq)] = torch.tensor(seq, dtype=torch.long)
        m = torch.zeros((len(ch), T), dtype=torch.bool)
        for b, L in enumerate(Ls):
            m[b, :L - 1] = True
        lr, lw, lp, ls = model(sub[:, 0, :-1], sub[:, 1, :-1], sub[:, 2, :-1], sub[:, 3, :-1])
        lg = {'r': lr, 'w': lw, 'p': lp, 's': ls}
        pr = {h: torch.softmax(lg[h], -1) for h in HEADS}
        arg = {h: lg[h].argmax(-1) for h in HEADS}
        for h in HEADS:
            t = sub[:, IDX[h], 1:]
            ok = (arg[h] == t) & m
            correct_preds[h].update(arg[h][ok].reshape(-1).tolist())
            per_head_pred[h].update(arg[h][m].reshape(-1).tolist())
        for b in range(len(ch)):
            mass_particle += float(pr['r'][b, :, PARTICLE][m[b]].sum())
            spm = pr['r'][b][:, list(SPECIAL)].sum(-1)
            mass_special += float(spm[m[b]].sum())
            n_pos += int(m[b].sum())
            roots = arg['r'][b][m[b]].tolist()
            if len(roots) >= 2:
                same_sent_pairs += len(roots) - 1
                same_sent_changes += sum(1 for a, bb in zip(roots, roots[1:]) if a != bb)
                distinct_root_argmax_per_sent.append(len(set(roots)))
        for b in range(len(ch)):
            for j in range(T):
                if not m[b, j]:
                    continue
                top1_tuples[(int(arg['p'][b, j]), int(arg['r'][b, j]),
                             int(arg['w'][b, j]), int(arg['s'][b, j]))] += 1
                true_tuples[(int(sub[b, 0, j + 1]), int(sub[b, 1, j + 1]),
                             int(sub[b, 2, j + 1]), int(sub[b, 3, j + 1]))] += 1
    return {'top1_tuples': top1_tuples, 'true_tuples': true_tuples,
            'correct_preds': correct_preds, 'per_head_pred': per_head_pred,
            'mass_particle': mass_particle, 'mass_special': mass_special, 'n_pos': n_pos,
            'same_sent_changes': same_sent_changes, 'same_sent_pairs': same_sent_pairs,
            'distinct_per_sent': distinct_root_argmax_per_sent}


def nm(h, i):
    return (VOCAB.id2prefix.get(i, '?') if h == 'p' else VOCAB.id2root.get(i, '?') if h == 'r'
            else VOCAB.id2wazn.get(i, '?') if h == 'w' else VOCAB.id2suffix.get(i, '?'))


def tag(i):
    return 'SPECIAL' if i in SPECIAL else ('HELD-OUT' if i in HOLD else 'real')


if __name__ == '__main__':
    res = {}
    for split in ('train', 'test_gen', 'test_deriv'):
        a = analyse(split)
        res[split] = a
        n = a['n_pos']
        print('\n' + '=' * 100)
        print(f'{split}: n_pos={n}')
        print(f'  distinct top-1 tuples emitted by the model : '
              f'{len(a["top1_tuples"])}   (true: {len(a["true_tuples"])})')
        s5 = sum(v for _, v in a['top1_tuples'].most_common(5)) / max(n, 1)
        s1 = a['top1_tuples'].most_common(1)[0][1] / max(n, 1)
        print(f'  most frequent single tuple covers          : {s1*100:.2f}%')
        print(f'  the 5 most frequent tuples cover           : {s5*100:.2f}%')
        print(f'  mean P(<PARTICLE>) in the root head        : '
              f'{a["mass_particle"]/max(n,1)*100:.2f}%')
        print(f'  mean P(special root) in the root head      : '
              f'{a["mass_special"]/max(n,1)*100:.2f}%')
        print(f'  adjacent-position root argmax CHANGES      : '
              f'{a["same_sent_changes"]}/{a["same_sent_pairs"]} = '
              f'{a["same_sent_changes"]/max(a["same_sent_pairs"],1)*100:.1f}%')
        d = a['distinct_per_sent']
        print(f'  distinct root argmax within a sentence     : mean '
              f'{sum(d)/max(len(d),1):.2f} (over {len(d)} sentences)')
        print('  MODEL top-1 tuples:')
        for t, c in a['top1_tuples'].most_common(8):
            print(f'    {c/max(n,1)*100:6.2f}%  n={c:7d}  {VOCAB.decode_word(*t)!r:16s} '
                  f'[{nm("p",t[0])}+{nm("r",t[1])}+{nm("w",t[2])}+{nm("s",t[3])}] '
                  f'root={tag(t[1])}')
        print('  TRUE top-1 tuples:')
        for t, c in a['true_tuples'].most_common(8):
            print(f'    {c/max(n,1)*100:6.2f}%  n={c:7d}  {VOCAB.decode_word(*t)!r:16s} '
                  f'[{nm("p",t[0])}+{nm("r",t[1])}+{nm("w",t[2])}+{nm("s",t[3])}] '
                  f'root={tag(t[1])}')
        print('  ROOT head, where the model was CORRECT (top-1):')
        for i, c in a['correct_preds']['r'].most_common(10):
            print(f'    {c/max(a["correct_preds"]["r"].total(),1)*100:6.2f}%  n={c:7d}  '
                  f'id={i:5d} {VOCAB.id2root.get(i,"?")!r:16s} [{tag(i)}]')
        print('  ROOT head, ALL argmaxes (not just correct):')
        for i, c in a['per_head_pred']['r'].most_common(10):
            print(f'    {c/max(a["per_head_pred"]["r"].total(),1)*100:6.2f}%  n={c:7d}  '
                  f'id={i:5d} {VOCAB.id2root.get(i,"?")!r:16s} [{tag(i)}]')
    outj = {}
    for split, a in res.items():
        outj[split] = {
            'n_pos': a['n_pos'],
            'distinct_top1_tuples': len(a['top1_tuples']),
            'distinct_true_tuples': len(a['true_tuples']),
            'top1_tuple_share': a['top1_tuples'].most_common(1)[0][1] / max(a['n_pos'], 1),
            'top5_tuple_share': sum(v for _, v in a['top1_tuples'].most_common(5)) / max(a['n_pos'], 1),
            'mean_P_particle': a['mass_particle'] / max(a['n_pos'], 1),
            'mean_P_special': a['mass_special'] / max(a['n_pos'], 1),
            'adjacent_root_change_rate': a['same_sent_changes'] / max(a['same_sent_pairs'], 1),
            'mean_distinct_root_per_sentence': sum(a['distinct_per_sent']) / max(len(a['distinct_per_sent']), 1),
            'top1_tuples': [{'tuple': list(t), 'n': c, 'share': c / max(a['n_pos'], 1),
                             'surface': VOCAB.decode_word(*t),
                             'prefix': nm('p', t[0]), 'root': nm('r', t[1]),
                             'wazn': nm('w', t[2]), 'suffix': nm('s', t[3]),
                             'root_tag': tag(t[1])} for t, c in a['top1_tuples'].most_common(15)],
            'true_tuples': [{'tuple': list(t), 'n': c, 'share': c / max(a['n_pos'], 1),
                             'surface': VOCAB.decode_word(*t),
                             'prefix': nm('p', t[0]), 'root': nm('r', t[1]),
                             'wazn': nm('w', t[2]), 'suffix': nm('s', t[3]),
                             'root_tag': tag(t[1])} for t, c in a['true_tuples'].most_common(15)],
            'root_correct_preds': [{'id': i, 'name': VOCAB.id2root.get(i, '?'), 'n': c,
                                    'tag': tag(i)}
                                   for i, c in a['correct_preds']['r'].most_common(20)],
            'root_all_preds': [{'id': i, 'name': VOCAB.id2root.get(i, '?'), 'n': c,
                                'tag': tag(i)}
                               for i, c in a['per_head_pred']['r'].most_common(20)],
        }
    json.dump(outj, open(f'{OUT}/collapse_analysis.json', 'w'), ensure_ascii=False, indent=2)
    print(f'\nwrote {OUT}/collapse_analysis.json')
