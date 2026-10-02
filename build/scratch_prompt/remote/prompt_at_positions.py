#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""prompt_at_positions.py -- prompt the model at EVERY position of real sentences, batched.

Verbatim prompts: the context shown is the project's OWN encoding path applied to the real
sentence -- VOCAB.encode_sentence(sentence) for the ids, and
VOCAB.decode_word(*tuple) for the readable Arabic.  For test_gen / test_deriv the raw sentence
was never in the training stream, and for `position 0` contexts (just the decoded first word)
the prompt is cleanly out-of-sample.

Modes
  testgen  N   N random test_gen sentences, every position reported
  heldout  N   N random test_deriv sentences that contain one of the 38 held-out roots,
               with the held-out-root positions reported
"""
import json
import random
import sys
from collections import Counter

import torch

sys.path.insert(0, '/workspace/hf_v19_2_release')
sys.path.insert(0, '/workspace/hf_v19_2_release/models')
sys.path.insert(0, '/workspace/scratch_prompt')

D = '/workspace/sf_data'
OUT = '/workspace/scratch_prompt'
CKPT = '/workspace/scratch_lm_run/sf_morph.pt'
W = 128

meta = json.load(open(f'{D}/meta.json'))
NR, NW, NP, NS = meta['n_roots'], meta['n_awzan'], meta['n_prefixes'], meta['n_suffixes']
SPECIAL = set(meta['special_root_ids'])
HOLD = set(meta['hold_roots'])
HEADS = ('p', 'r', 'w', 's')
IDX = {'p': 0, 'r': 1, 'w': 2, 's': 3}
NAME = {'p': 'prefix', 'r': 'root', 'w': 'wazn', 's': 'suffix'}
LIM = {'p': NP, 'r': NR, 'w': NW, 's': NS}

import scratch_lm as S        # noqa: E402
import nrmp_vocab as nv       # noqa: E402

_V = next(v for k, v in vars(nv).items() if isinstance(v, type) and 'MorphemicVocab' in k)
VOCAB = _V('/workspace/hf_v19_2_release/data/rootformer_v12_arabic_blueprint.json')
device = torch.device('cpu')
model = S.MorphemicLM((NR, NW, NP, NS)).to(device)
model.load_state_dict(torch.load(CKPT, map_location=device, weights_only=True))
model.eval()

print('[*] loading streams.pt ...', flush=True)
RAW = torch.load(f'{D}/streams.pt', weights_only=False)

OVER = Counter()


def all_in_range(t):
    return all(t[IDX[h]] < LIM[h] for h in HEADS)


def safe(split, i):
    """Encoded sentence i, trimmed at the first token outside the checkpoint id space."""
    P, R, W_, S_ = RAW[split]['P'][i], RAW[split]['R'][i], RAW[split]['W'][i], RAW[split]['S'][i]
    out = []
    for t in zip(P, R, W_, S_):
        if not all_in_range(t):
            OVER['trimmed_tokens'] += 1
            break
        out.append(t)
    return out


def decoded_words(enc):
    return [VOCAB.decode_word(*t) for t in enc]


def tuple_desc(t):
    return (f'({t[0]},{t[1]},{t[2]},{t[3]}) '
            f'[{VOCAB.id2prefix.get(t[0],"?")}+{VOCAB.id2root.get(t[1],"?")}'
            f'+{VOCAB.id2wazn.get(t[2],"?")}+{VOCAB.id2suffix.get(t[3],"?")}]')


def root_tag(r):
    return 'SPECIAL' if r in SPECIAL else ('HELD-OUT' if r in HOLD else 'real')


@torch.no_grad()
def predict_at(enc, positions, beam=6):
    """enc: list of tuples.  Returns {pos: prediction dict} for the given next-word positions.

    Position p means: context = enc[:p], target = enc[p].  All positions are scored in ONE
    forward pass, exactly as scratch_lm.root_accuracy does.
    """
    sub = torch.tensor([[t[k] for t in enc] for k in range(4)], dtype=torch.long).unsqueeze(0)
    lr, lw, lp, ls = model(sub[:, 0, :-1], sub[:, 1, :-1], sub[:, 2, :-1], sub[:, 3, :-1])
    lg = {'r': lr, 'w': lw, 'p': lp, 's': ls}
    res = {}
    for p in positions:
        if p < 1 or p >= len(enc):
            continue
        j = p - 1                                   # logits index predicting target p
        lp_h = {h: torch.log_softmax(lg[h][0, j], -1) for h in HEADS}
        per = {}
        for h in HEADS:
            v, ii = lp_h[h].topk(beam)
            per[h] = [{'id': int(x), 'name': (VOCAB.id2prefix.get(int(x), '?') if h == 'p'
                                              else VOCAB.id2root.get(int(x), '?') if h == 'r'
                                              else VOCAB.id2wazn.get(int(x), '?') if h == 'w'
                                              else VOCAB.id2suffix.get(int(x), '?')),
                        'logp': float(y)} for y, x in zip(v, ii)]
        cand = []
        for pr in per['p']:
            for ro in per['r']:
                for wa in per['w']:
                    for su in per['s']:
                        sc = (0.25 * pr['logp'] + ro['logp'] + 0.5 * wa['logp']
                              + 0.25 * su['logp'])
                        cand.append((sc, (pr['id'], ro['id'], wa['id'], su['id'])))
        cand.sort(key=lambda x: -x[0])
        t1 = tuple(int(lg[h][0, j].argmax()) for h in HEADS)
        res[p] = {'t1': t1, 't1_surface': VOCAB.decode_word(*t1),
                  'per_head': per,
                  'joint5': [{'tuple': list(t), 'score': float(s_), 'surface': VOCAB.decode_word(*t),
                              'root': VOCAB.id2root.get(t[1], '?'), 'root_tag': root_tag(t[1]),
                              'prefix': VOCAB.id2prefix.get(t[0], '?'),
                              'wazn': VOCAB.id2wazn.get(t[2], '?'),
                              'suffix': VOCAB.id2suffix.get(t[3], '?')}
                             for s_, t in cand[:5]]}
    return res


def render(split, i, enc, positions, label):
    words = decoded_words(enc)
    preds = predict_at(enc, positions)
    lines = [f'--- {label}']
    lines.append(f'RAW SENTENCE ({split} #{i}, {len(enc)} words): '
                 f'{" ".join(w for w in words if w)}')
    lines.append(f'ENCODED TUPLES: {enc}')
    for p in positions:
        if p not in preds:
            continue
        r = preds[p]
        true_t = enc[p]
        hit1 = r['t1'] == true_t
        in5 = any(tuple(j['tuple']) == true_t for j in r['joint5'])
        lines.append(f'  [pos {p}] CONTEXT: {" ".join(w for w in words[:p] if w)}')
        lines.append(f'           TRUE NEXT: {words[p]!r}  {tuple_desc(true_t)} '
                     f'root_tag={root_tag(true_t[1])}')
        lines.append(f'           PRED top1: {r["t1_surface"]!r}  {tuple_desc(r["t1"])} '
                     f'root_tag={root_tag(r["t1"][1])}   '
                     f'{"EXACT" if hit1 else ("root-hit" if r["t1"][1] == true_t[1] else "miss")}'
                     f'{" (in joint top5)" if in5 else " (NOT in joint top5)"}')
        for h in HEADS:
            lines.append(f'             {NAME[h]:6s}: ' + '  '.join(
                f'{x["name"]!r}:{x["logp"]:.2f}' for x in r['per_head'][h]))
        lines.append('             joint top5: ' + ' || '.join(
            f'{j["surface"]!r}[{j["prefix"]}+{j["root"]}+{j["wazn"]}+{j["suffix"]}]'
            f'<{j["root_tag"]}>' for j in r['joint5']))
    return '\n'.join(lines), {'words': words, 'enc': [list(t) for t in enc],
                              'preds': {str(k): v for k, v in preds.items()}}


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'testgen'
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 7
    out = []
    if mode == 'testgen':
        N = len(RAW['test_gen']['L'])
        rng = random.Random(seed)
        order = rng.sample(range(N), N)
        got = 0
        for i in order:
            enc = safe('test_gen', i)
            if len(enc) < 8:
                continue
            positions = list(range(1, min(len(enc), 12)))
            txt, rec = render('test_gen', i, enc, positions,
                              f'test_gen sentence {i} (held-out WORK, never trained on)')
            out.append(rec); out[-1]['text'] = txt
            print(txt); print()
            got += 1
            if got >= n:
                break
        json.dump(out, open(f'{OUT}/prompts_testgen.json', 'w'), ensure_ascii=False, indent=2)
        print(f'[over-range tokens trimmed: {OVER["trimmed_tokens"]}]')
        print(f'wrote {OUT}/prompts_testgen.json')
    elif mode == 'heldout':
        d = RAW['test_deriv']
        idx = [i for i in range(len(d['L'])) if set(d['R'][i]) & HOLD]
        rng = random.Random(seed)
        rng.shuffle(idx)
        got = 0
        for i in idx:
            enc = safe('test_deriv', i)
            if len(enc) < 5:
                continue
            positions = [p for p in range(1, len(enc)) if enc[p][1] in HOLD]
            if not positions:
                continue
            positions = positions[:6]
            txt, rec = render('test_deriv', i, enc, positions,
                              f'test_deriv sentence {i} (held-out-root targets)')
            out.append(rec); out[-1]['text'] = txt
            out[-1]['heldout_positions'] = positions
            print(txt); print()
            got += 1
            if got >= n:
                break
        json.dump(out, open(f'{OUT}/prompts_heldout_positions.json', 'w'),
                  ensure_ascii=False, indent=2)
        print(f'[over-range tokens trimmed: {OVER["trimmed_tokens"]}]')
        print(f'wrote {OUT}/prompts_heldout_positions.json')


if __name__ == '__main__':
    main()
