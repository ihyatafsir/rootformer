#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
per_stream_eval.py -- score the four morphemic streams SEPARATELY.

Reason this matters: the classical grammarians (Khalil, Sibawayh, Ibn Malik, the Andalusians) are a
theory of GOVERNANCE OVER FORM -- i'rab, case, wazn, the operator's effect on the following word's
shape. They are not a theory of lexical selection, i.e. which root a speaker chooses. So their
oversight should show up in the wazn / prefix / suffix streams, NOT in the root stream.

Aggregate bits-per-word hides this. This reports, per stream:
  * bits per position (what the model spends)
  * unigram bits per position (the context-free bar)
  * top-1 / top-5 accuracy (for root and wazn; prefixes/suffixes are small inventories)

Usage: python per_stream_eval.py --model /workspace/sf_governed.pt --data /workspace/sf_data_gov \
                                  --arch governed
       python per_stream_eval.py --model /workspace/sf_morph.pt --data /workspace/sf_data \
                                  --arch plain
"""
import argparse
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, '/workspace/hf_v19_2_release')
sys.path.insert(0, '/workspace/hf_v19_2_release/models')
import scratch_lm as S


def load(arch, data_dir, model_path, device):
    meta = json.load(open(Path(data_dir) / 'meta.json'))
    if (Path(data_dir) / 'train.pt').exists():
        import governed_lm as G
        data = G.load_data(Path(data_dir))
    else:
        data = torch.load(Path(data_dir) / 'streams.pt', weights_only=False)
    if arch == 'governed':
        from governed_lm import GovernedLM
        model = GovernedLM((meta['n_roots'], meta['n_awzan'], meta['n_prefixes'],
                            meta['n_suffixes']), meta.get('n_ops', 7)).to(device)
    else:
        model = S.MorphemicLM((meta['n_roots'], meta['n_awzan'],
                               meta['n_prefixes'], meta['n_suffixes'])).to(device)
    sd = torch.load(model_path, map_location='cpu')
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f'[*] {arch} model loaded: missing={len(missing)} unexpected={len(unexpected)}')
    model.eval()
    return model, data, meta


def batches(arch, data, split, bs, rng, max_sent=200):
    if arch == 'governed':
        from governed_lm import batches_gov
        yield from batches_gov(data, split, bs, rng, max_sent=max_sent)
        return
    for b in S.batches_morph(data, split, bs, rng, max_sent=max_sent):
        b['op'] = [[0] * len(x) for x in b['r']]
        yield b


@torch.no_grad()
def run(arch, model, data, splits, device, bs=16):
    streams = ('r', 'w', 'p', 's')
    names = {'r': 'root', 'w': 'wazn', 'p': 'prefix', 's': 'suffix'}
    out = {}
    for sp in splits:
        bits = {k: 0.0 for k in streams}
        nllsum = {k: 0.0 for k in streams}
        hit1 = {k: 0 for k in streams}
        hit5 = {k: 0 for k in streams}
        n = 0
        rng = random.Random(7)
        for b in batches(arch, data, sp, bs, rng):
            tgt = {k: S.pad([x[1:] for x in b[k]], 0).to(device) for k in streams}
            inp = {k: S.pad([x[:-1] for x in b[k]], 0).to(device) for k in streams}
            o_in = S.pad([x[:-1] for x in b['op']], 0).to(device)
            m = torch.zeros_like(tgt['r'], dtype=torch.bool)
            for bi, x in enumerate(b['r']):
                m[bi, :min(len(x) - 1, tgt['r'].shape[1])] = True
            if arch == 'governed':
                lr, lw, lp, ls = model(inp['p'], inp['r'], inp['w'], inp['s'], o_in)
            else:
                lr, lw, lp, ls = model(inp['p'], inp['r'], inp['w'], inp['s'])
            logs = {'r': lr, 'w': lw, 'p': lp, 's': ls}
            for k in streams:
                lg = logs[k]
                ce = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tgt[k].reshape(-1),
                                     reduction='none').reshape(tgt[k].shape)
                nllsum[k] += float(ce[m].sum())
                bits[k] += float(ce[m].sum()) / math.log(2)
                p1 = lg.argmax(-1)
                p5 = lg.topk(min(5, lg.shape[-1]), -1).indices
                hit1[k] += int(((p1 == tgt[k]) & m).sum())
                hit5[k] += int(((p5 == tgt[k].unsqueeze(-1)).any(-1) & m).sum())
            n += int(m.sum())

        # context-free unigram per stream
        C = {}
        for k in streams:
            c = Counter()
            for seq in data['train'][k.upper() if k != 's' else 'S'][:200000]:
                c.update(seq)
            C[k] = c
        uni = {}
        for k in streams:
            c = C[k]
            tot = sum(c.values())
            V = max(c) + 2
            uni[k] = sum(-math.log2(max((c.get(x, 0) + 0.5) / (tot + 0.5 * V), 1e-12))
                         for seq in data[sp][k.upper() if k != 's' else 'S'][:2000]
                         for x in seq[1:]) / max(
                sum(len(seq) - 1 for seq in data[sp][k.upper() if k != 's' else 'S'][:2000]), 1)

        out[sp] = {}
        print(f'\n  {sp}  (n={n} positions)')
        print(f'    {"stream":<8}{"model b/pos":>13}{"unigram b/pos":>15}{"gain":>8}'
              f'{"top1":>8}{"top5":>8}{"vocab":>8}')
        for k in streams:
            bpp = bits[k] / max(n, 1)
            g = 100 * (uni[k] - bpp) / uni[k] if uni[k] else 0
            V = {'r': 9114, 'w': 130, 'p': 26, 's': 22}[k]
            out[sp][names[k]] = {'bits_per_pos': bpp, 'unigram_bits': uni[k], 'gain_pct': g,
                                 'top1': hit1[k] / max(n, 1), 'top5': hit5[k] / max(n, 1)}
            print(f'    {names[k]:<8}{bpp:>13.3f}{uni[k]:>15.3f}{g:>7.1f}%'
                  f'{100*hit1[k]/max(n,1):>7.2f}%{100*hit5[k]/max(n,1):>7.2f}%{V:>8}')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', choices=['plain', 'governed'], required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    device = torch.device('cuda')
    model, data, meta = load(args.arch, args.data, args.model, device)
    res = run(args.arch, model, data, ['test_gen', 'test_deriv'], device)
    op = args.out or f'/workspace/per_stream_{args.arch}.json'
    json.dump(res, open(op, 'w'), indent=2)
    print(f'\nwrote {op}')


if __name__ == '__main__':
    main()
