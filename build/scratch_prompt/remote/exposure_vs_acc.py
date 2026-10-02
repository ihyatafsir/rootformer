#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""exposure_vs_acc.py -- was sf_morph.pt still learning, or had it stopped?

The 120k learning curve answers this going FORWARD.  This answers it from the existing
6k checkpoint, using a signal that is already in the training data: how many times each
training sentence was actually drawn by the sampler.

scratch_lm.train draws `rng.sample(range(n), min(bs*max_sent, n))` per generator run, with
`bs=16, max_sent=64` and `random.Random(0)`; 6000 steps were taken.  Replaying that exactly
gives an exposure count per sentence.  If the model is still learning, accuracy should RISE
with exposure.  If it has saturated at the marginal, there will be no relationship -- it loses
on the sentences it has seen 12 times exactly as often as on those seen once.

Also decomposes every training position by target class (special / real root) and reports the
model's top-1 probability ON the correct class, which separates "sharp but at the marginal"
from "learning context".

Read-only.  CPU only.
"""
import json
import random
import sys
import time
from collections import defaultdict

import torch

sys.path.insert(0, '/workspace/hf_v19_2_release')
sys.path.insert(0, '/workspace/hf_v19_2_release/models')

D = '/workspace/sf_data'
OUT = '/workspace/scratch_prompt'
CKPT = '/workspace/scratch_lm_run/sf_morph.pt'
W = 48
BS = 64

meta = json.load(open(f'{D}/meta.json'))
NR = meta['n_roots']
SPECIAL = set(meta['special_root_ids'])
HOLD = set(meta['hold_roots'])
print('[*] loading ...', flush=True)
RAW = torch.load(f'{D}/streams.pt', weights_only=False)
import scratch_lm as S      # noqa: E402
import nrmp_vocab as nv     # noqa: E402
_V = next(v for k, v in vars(nv).items() if isinstance(v, type) and 'MorphemicVocab' in k)
VOCAB = _V('/workspace/hf_v19_2_release/data/rootformer_v12_arabic_blueprint.json')
dev = torch.device('cpu')
model = S.MorphemicLM((NR, meta['n_awzan'], meta['n_prefixes'], meta['n_suffixes'])).to(dev)
model.load_state_dict(torch.load(CKPT, map_location=dev, weights_only=True))
model.eval()

# ---- exact exposure counts from the training sampler -------------------------------
n = len(RAW['train']['L'])
Lens = RAW['train']['L']
rng = random.Random(0)
bs, max_sent = 16, 64
expo = [0] * n
steps = 0
t0 = time.time()
while steps < 6000:
    order = rng.sample(range(n), min(bs * max_sent, n))
    cur = []
    for i in order:
        if Lens[i] < 2:
            continue
        cur.append(i)
        if len(cur) == bs:
            for j in cur:
                expo[j] += 1
            cur = []
            steps += 1
            if steps >= 6000:
                break
print(f'[*] replayed {steps} steps in {time.time()-t0:.0f}s; '
      f'seen={sum(1 for e in expo if e)} distinct sentences', flush=True)

# ---- accuracy vs exposure bucket ---------------------------------------------------
seen = [i for i in range(n) if expo[i] > 0]
buckets = [(1, 1), (2, 2), (3, 4), (5, 8), (9, 16), (17, 10**9)]
res = {}


@torch.no_grad()
def score(idx_list):
    """Per-target aggregates for a list of sentence indices."""
    agg = {'n': 0, 'root_h1': 0, 'root_h5': 0, 'p_h1': 0, 'w_h1': 0, 's_h1': 0,
           'root_n': 0, 'root_h1_real': 0, 'root_n_real': 0, 'root_h1_spec': 0,
           'root_n_spec': 0, 'root_n_hold': 0, 'root_h1_hold': 0}
    for i in range(0, len(idx_list), BS):
        ch = idx_list[i:i + BS]
        lns = [min(int(Lens[j]), W) for j in ch]
        T = max(max(lns) - 1, 1)
        sub = torch.zeros((len(ch), 4, T + 1), dtype=torch.long)
        for b, j in enumerate(ch):
            for k, key in enumerate(('P', 'R', 'W', 'S')):
                seq = RAW['train'][key][j][:T + 1]
                sub[b, k, :len(seq)] = torch.tensor(seq, dtype=torch.long)
        m = torch.zeros((len(ch), T), dtype=torch.bool)
        for b, L in enumerate(lns):
            m[b, :L - 1] = True
        lr, lw, lp, ls = model(sub[:, 0, :-1], sub[:, 1, :-1], sub[:, 2, :-1], sub[:, 3, :-1])
        top1 = {0: lp.argmax(-1), 1: lr.argmax(-1), 2: lw.argmax(-1), 3: ls.argmax(-1)}
        top5r = lr.topk(5, -1).indices
        rt = sub[:, 1, 1:]
        agg['n'] += int(m.sum())
        agg['root_n'] += int(m.sum())
        agg['root_h1'] += int(((top1[1] == rt) & m).sum())
        agg['root_h5'] += int(((top5r == rt.unsqueeze(-1)).any(-1) & m).sum())
        agg['p_h1'] += int(((top1[0] == sub[:, 0, 1:]) & m).sum())
        agg['w_h1'] += int(((top1[2] == sub[:, 2, 1:]) & m).sum())
        agg['s_h1'] += int(((top1[3] == sub[:, 3, 1:]) & m).sum())
        sp = torch.zeros(NR, dtype=torch.bool)
        sp[list(SPECIAL)] = True
        hd = torch.zeros(NR, dtype=torch.bool)
        hd[list(HOLD)] = True
        real = (~sp[rt]) & (~hd[rt]) & m
        spec = sp[rt] & m
        agg['root_n_real'] += int(real.sum())
        agg['root_h1_real'] += int(((top1[1] == rt) & real).sum())
        agg['root_n_spec'] += int(spec.sum())
        agg['root_h1_spec'] += int(((top1[1] == rt) & spec).sum())
    return agg


print('\n' + '=' * 100)
print('ACCURACY vs EXPOSURE  (training sentences, bucketed by how many times the sampler '
      'drew them)')
print('=' * 100)
print(f"{'times seen':>12s} {'sentences':>10s} {'positions':>10s} {'root t1':>9s} "
      f"{'root t5':>9s} {'prefix':>8s} {'wazn':>8s} {'suffix':>8s} "
      f"{'root t1 real':>13s} {'root t1 spec':>13s}")
tot = {'n': 0, 'root_h1': 0, 'root_h5': 0, 'p_h1': 0, 'w_h1': 0, 's_h1': 0,
       'root_n_real': 0, 'root_h1_real': 0, 'root_n_spec': 0, 'root_h1_spec': 0}
rows = []
for lo, hi in buckets:
    idx = [i for i in seen if lo <= expo[i] <= hi]
    if not idx:
        continue
    a = score(idx)
    tot['n'] += a['n']; tot['root_h1'] += a['root_h1']; tot['root_h5'] += a['root_h5']
    tot['p_h1'] += a['p_h1']; tot['w_h1'] += a['w_h1']; tot['s_h1'] += a['s_h1']
    tot['root_n_real'] += a['root_n_real']; tot['root_h1_real'] += a['root_h1_real']
    tot['root_n_spec'] += a['root_n_spec']; tot['root_h1_spec'] += a['root_h1_spec']
    label = f'{lo}' if lo == hi else (f'{lo}+' if hi > 10**8 else f'{lo}-{hi}')
    rr = a['root_h1_real'] / max(a['root_n_real'], 1)
    rs = a['root_h1_spec'] / max(a['root_n_spec'], 1)
    print(f'{label:>12s} {len(idx):10d} {a["n"]:10d} '
          f'{a["root_h1"]/max(a["root_n"],1)*100:8.2f}% {a["root_h5"]/max(a["root_n"],1)*100:8.2f}% '
          f'{a["p_h1"]/max(a["n"],1)*100:7.2f}% {a["w_h1"]/max(a["n"],1)*100:7.2f}% '
          f'{a["s_h1"]/max(a["n"],1)*100:7.2f}% {rr*100:12.2f}% {rs*100:12.2f}%')
    rows.append({'lo': lo, 'hi': hi, 'sentences': len(idx), 'positions': a['n'],
                 'root_top1': a['root_h1'] / max(a['root_n'], 1),
                 'root_top5': a['root_h5'] / max(a['root_n'], 1),
                 'prefix_top1': a['p_h1'] / max(a['n'], 1),
                 'wazn_top1': a['w_h1'] / max(a['n'], 1),
                 'suffix_top1': a['s_h1'] / max(a['n'], 1),
                 'root_top1_real': rr, 'root_top1_special': rs})
print(f'{"ALL":>12s} {"":>10s} {tot["n"]:10d} '
      f'{tot["root_h1"]/max(tot["n"],1)*100:8.2f}% {tot["root_h5"]/max(tot["n"],1)*100:8.2f}% '
      f'{tot["p_h1"]/max(tot["n"],1)*100:7.2f}% {tot["w_h1"]/max(tot["n"],1)*100:7.2f}% '
      f'{tot["s_h1"]/max(tot["n"],1)*100:7.2f}% '
      f'{tot["root_h1_real"]/max(tot["root_n_real"],1)*100:12.2f}% '
      f'{tot["root_h1_spec"]/max(tot["root_n_spec"],1)*100:12.2f}%')

# never-seen training sentences, as a control
never = [i for i in range(n) if expo[i] == 0 and Lens[i] >= 2]
print(f'\n[control] {len(never):,} training sentences the sampler never drew '
      f'(0 exposures) -- a held-out-within-train sample of the same distribution')
# sample them to keep the cost bounded
rng2 = random.Random(4)
ctl = rng2.sample(never, min(40000, len(never)))
a = score(ctl)
print(f'  positions={a["n"]:,}  root t1/t5 {a["root_h1"]/max(a["root_n"],1)*100:.2f}%/'
      f'{a["root_h5"]/max(a["root_n"],1)*100:.2f}%   real {a["root_h1_real"]/max(a["root_n_real"],1)*100:.2f}%  '
      f'special {a["root_h1_spec"]/max(a["root_n_spec"],1)*100:.2f}%')

json.dump({'exposure_buckets': rows,
           'never_seen_control': {'sentences': len(ctl), 'positions': a['n'],
                                  'root_top1': a['root_h1'] / max(a['root_n'], 1),
                                  'root_top5': a['root_h5'] / max(a['root_n'], 1),
                                  'root_top1_real': a['root_h1_real'] / max(a['root_n_real'], 1)},
           'total_seen_sentences': len(seen), 'steps': steps},
          open(f'{OUT}/exposure_vs_acc.json', 'w'), indent=2)
print(f'\nwrote {OUT}/exposure_vs_acc.json')
