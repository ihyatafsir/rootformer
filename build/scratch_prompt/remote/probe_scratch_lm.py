#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""probe_scratch_lm.py -- prompt-test /workspace/scratch_lm_run/sf_morph.pt.

Read-only on every project artefact.  Writes only under /workspace/scratch_prompt/.

Evaluation convention -- copied from scratch_lm.py, not invented:
  * inputs  x[:-1] per stream, targets x[1:], both per word (NRMT-P: one position/word)
  * pad(seqs, 0) ; the model call is exactly scratch_lm.MorphemicLM.forward
  * the model returns (root_logits, wazn_logits, prefix_logits, suffix_logits) -- note that
    order from forward(): `return self.h_r(h), self.h_w(h), self.h_p(h), self.h_s(h)`
  * scratch_lm.root_accuracy scores EVERY position (not just the last one); that is the
    function whose published numbers (test_deriv/real_seen_root top1 1.47%) we are explaining,
    so we use the same convention.  scratch_lm.eval_bits, by contrast, scores only the final
    position per sentence; that difference is called out in the report.
"""
import json
import math
import random
import sys
from collections import Counter, defaultdict

sys.path.insert(0, '/workspace/hf_v19_2_release')
sys.path.insert(0, '/workspace/hf_v19_2_release/models')

import torch

D = '/workspace/sf_data'
OUT = '/workspace/scratch_prompt'
CKPT = '/workspace/scratch_lm_run/sf_morph.pt'
CTX = 128
BS = 64
CTX_FLOOR = 64
N_BATCHES = 1200          # identical budget per split -> comparable n

meta = json.load(open(f'{D}/meta.json'))
NR, NW, NP, NS = meta['n_roots'], meta['n_awzan'], meta['n_prefixes'], meta['n_suffixes']
SPECIAL_ROOT_IDS = set(meta['special_root_ids'])
HOLD = set(meta['hold_roots'])

print('[*] loading streams.pt (~300MB) ...', flush=True)
DATA = torch.load(f'{D}/streams.pt', weights_only=False)

import scratch_lm as S      # noqa: E402
import nrmp_vocab as nv     # noqa: E402

_V = next(v for k, v in vars(nv).items() if isinstance(v, type) and 'MorphemicVocab' in k)
VOCAB = _V('/workspace/hf_v19_2_release/data/rootformer_v12_arabic_blueprint.json')
device = torch.device('cpu')

model = S.MorphemicLM((NR, NW, NP, NS)).to(device)
model.load_state_dict(torch.load(CKPT, map_location=device, weights_only=True))
model.eval()
print(f'[*] loaded {CKPT}: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params', flush=True)

HEADS = ('p', 'r', 'w', 's')          # our canonical order
NAME = {'p': 'prefix', 'r': 'root', 'w': 'wazn', 's': 'suffix'}


def fwd(ctx):
    p_in = S.pad(ctx['p'], 0).to(device)
    r_in = S.pad(ctx['r'], 0).to(device)
    w_in = S.pad(ctx['w'], 0).to(device)
    s_in = S.pad(ctx['s'], 0).to(device)
    lr, lw, lp, ls = model(p_in, r_in, w_in, s_in)     # NOTE the project's return order
    return {'r': lr, 'w': lw, 'p': lp, 's': ls}


def make_batch(split, order, floor=0, trunc=CTX):
    """ctx (x[:-1]) / tgt (x[1:]) for a list of sentence indices, plus the valid mask."""
    d = DATA[split]
    src = {'p': d['P'], 'r': d['R'], 'w': d['W'], 's': d['S']}
    ctx, tgt = {}, {}
    for h in HEADS:
        ctx[h] = [src[h][j][:trunc][:-1] for j in order]
        tgt[h] = [src[h][j][:trunc][1:] for j in order]
    T = max(len(x) for x in ctx['r'])
    m = torch.zeros((len(order), T), dtype=torch.bool)
    for b, x in enumerate(ctx['r']):
        L = len(x)                                  # == number of targets
        keep = L
        if floor:
            keep = max(L - floor, 0)                # only targets at position >= floor
        if keep > 0:
            m[b, L - keep:L] = True
    return ctx, tgt, m


ACC = defaultdict(lambda: defaultdict(lambda: {'n': 0, 'h1': 0, 'h5': 0}))
NBATCH = Counter()
TUPLE_TOP = defaultdict(Counter)
PRED_WHEN_CORRECT = defaultdict(Counter)
MARGINAL = {h: Counter() for h in HEADS}
MAXLEN = 0


def run(split, floor=0, n_batches=N_BATCHES, collect=False, seed=11):
    d = DATA[split]
    n = len(d['L'])
    if not n:
        return
    rng = random.Random(seed)
    for _ in range(n_batches):
        order = rng.sample(range(n), min(BS, n))
        ctx, tgt, m = make_batch(split, order, floor)
        if m.sum() == 0:
            continue
        lg = fwd(ctx)
        NBATCH[(split, floor)] += 1
        for h in HEADS:
            t = S.pad(tgt[h], 0).to(device)
            logits = lg[h]
            top5 = logits.topk(5, -1).indices
            a = ACC[(split, floor)][h]
            a['n'] += int(m.sum())
            a['h1'] += int(((top5[..., 0] == t) & m).sum())
            a['h5'] += int(((top5 == t.unsqueeze(-1)).any(-1) & m).sum())
            if collect:
                hit = (top5[..., 0] == t) & m
                PRED_WHEN_CORRECT[(split, h)].update(top5[..., 0][hit].reshape(-1).tolist())
        if collect:
            B, T = m.shape
            for b in range(B):
                for tpos in range(T):
                    if not m[b, tpos]:
                        continue
                    pt = (int(lg['p'][b, tpos].argmax()), int(lg['r'][b, tpos].argmax()),
                          int(lg['w'][b, tpos].argmax()), int(lg['s'][b, tpos].argmax()))
                    tt = (int(tgt['p'][b][tpos]) if tpos < len(tgt['p'][b]) else 0,
                          int(tgt['r'][b][tpos]) if tpos < len(tgt['r'][b]) else 0,
                          int(tgt['w'][b][tpos]) if tpos < len(tgt['w'][b]) else 0,
                          int(tgt['s'][b][tpos]) if tpos < len(tgt['s'][b]) else 0)
                    TUPLE_TOP[(split, 'pred_top1')][pt] += 1
                    TUPLE_TOP[(split, 'true')][tt] += 1


def seen_sentence_indices():
    """Indices of TRAIN sentences the model actually optimised on.

    Replays scratch_lm.batches_morph exactly: rng seeded 0 in train(), then
    `order = rng.sample(range(n), min(bs*max_sent, n))` per generator instantiation, with
    batches of `bs` indices emitted until the generator is exhausted.  scratch_lm.train
    re-creates the generator (`it = batches_morph(...)`) after each run of the inner for
    loop, which happens whenever the yielded batch count is exhausted -- with
    min(bs*max_sent, n) = min(1024, n) sampled indices that is a SINGLE generator run per
    outer while iteration, i.e. 1024/16 = 64 batches = 64 optimizer steps.
    """
    d = DATA['train']
    n = len(d['L'])
    rng = random.Random(0)
    bs, max_sent = 16, 64          # scratch_lm.py defaults
    seen, steps, run = set(), 0, 0
    targets = 0
    while targets < 6000:
        order = rng.sample(range(n), min(bs * max_sent, n))
        cur = []
        for i in order:
            if len(d['L'][i]) < 2:
                continue
            cur.append(i)
            if len(cur) == bs:
                seen.update(cur)
                cur = []
                steps += 1
                targets += 1
                if targets >= 6000:
                    break
        run += 1
    return sorted(seen), steps


def run_indices(split, order, floor=0, trunc=CTX, collect=False, tag=''):
    """Score an explicit list of sentence indices (no resampling)."""
    for i in range(0, len(order), BS):
        ch = order[i:i + BS]
        ctx, tgt, m = make_batch(split, ch, floor, trunc)
        if m.sum() == 0:
            continue
        lg = fwd(ctx)
        NBATCH[(tag or split, floor)] += 1
        _score_batch(tag or split, floor, lg, tgt, m, collect)


def _score_batch(key_split, floor, lg, tgt, m, collect):
    for h in HEADS:
        t = S.pad(tgt[h], 0).to(device)
        logits = lg[h]
        top5 = logits.topk(5, -1).indices
        a = ACC[(key_split, floor)][h]
        a['n'] += int(m.sum())
        a['h1'] += int(((top5[..., 0] == t) & m).sum())
        a['h5'] += int(((top5 == t.unsqueeze(-1)).any(-1) & m).sum())
    if collect:
        B, T = m.shape
        for b in range(B):
            for tpos in range(T):
                if not m[b, tpos]:
                    continue
                pt = (int(lg['p'][b, tpos].argmax()), int(lg['r'][b, tpos].argmax()),
                      int(lg['w'][b, tpos].argmax()), int(lg['s'][b, tpos].argmax()))
                TUPLE_TOP[(key_split, 'pred_top1')][pt] += 1


def unigram_reference():
    for h, key in (('r', 'R'), ('w', 'W'), ('p', 'P'), ('s', 'S')):
        for seq in DATA['train'][key]:
            MARGINAL[h].update(seq)
    out = {}
    for h in HEADS:
        c = MARGINAL[h]
        tot = sum(c.values())
        top5 = c.most_common(5)
        out[h] = {'top1_id': top5[0][0], 'top1_p': top5[0][1] / tot,
                  'top5_ids': [x for x, _ in top5],
                  'top5_mass': sum(v for _, v in top5) / tot,
                  'n_distinct': len(c), 'n_tokens': tot,
                  'H_bits': -sum(v / tot * math.log2(v / tot) for v in c.values())}
    return out


def surface(t):
    return VOCAB.decode_word(*t)


def tbl(rows):
    print('\n' + '=' * 108)
    print('PER-HEAD ACCURACY  (top-1 / top-5 %, by head; n = target positions scored)')
    print('=' * 108)
    print(f"{'split':11s} {'ctx>=':>5s} {'n':>8s} | " + ' | '.join(
        f'{NAME[h]:>13s}' for h in HEADS))
    for split, floor, heads in rows:
        n = heads['r']['n']
        print(f'{split:11s} {floor:5d} {n:8d} | ' + ' | '.join(
            f"{heads[h]['h1']/max(heads[h]['n'],1)*100:5.2f}/{heads[h]['h5']/max(heads[h]['n'],1)*100:5.2f}"
            for h in HEADS))


def acc_rows():
    return [(sp, fl, ACC[(sp, fl)]) for sp, fl in
            sorted(ACC.keys(), key=lambda k: (k[0], k[1]))]


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'acc'
    import time
    if mode == 'acc':
        plan = [('train', 0), ('train', CTX_FLOOR), ('train', 128),
                ('val', 0),
                ('test_gen', 0), ('test_gen', CTX_FLOOR),
                ('test_deriv', 0), ('test_deriv', CTX_FLOOR), ('test_deriv', 128)]
        for split, floor in plan:
            t0 = time.time()
            run(split, floor=floor)
            print(f'  scored {split:11s} ctx>={floor:3d}: {NBATCH[(split,floor)]} batches, '
                  f'n={ACC[(split,floor)]["r"]["n"]}, {time.time()-t0:.0f}s', flush=True)
        tbl(acc_rows())
        ug = unigram_reference()
        print('\n' + '=' * 108)
        print('UNIGRAM REFERENCE -- empirical train marginal per stream (the context-free '
              'baseline the model must beat)')
        print('=' * 108)
        for h in HEADS:
            u = ug[h]
            nm = VOCAB.id2root.get(u['top1_id'], '?') if h == 'r' else (
                VOCAB.id2wazn.get(u['top1_id'], '?') if h == 'w' else (
                    VOCAB.id2prefix.get(u['top1_id'], '?') if h == 'p'
                    else VOCAB.id2suffix.get(u['top1_id'], '?')))
            print(f"  {NAME[h]:6s} top1={nm!r:14s} (id {u['top1_id']:5d}) p={u['top1_p']*100:6.2f}%"
                  f"   top5 mass={u['top5_mass']*100:6.2f}%   distinct={u['n_distinct']:6d}"
                  f"   H={u['H_bits']:.3f} bits")
        json.dump({'per_head': [{'split': sp, 'ctx_floor': fl,
                                 **{h: {'top1': heads[h]['h1'] / max(heads[h]['n'], 1),
                                        'top5': heads[h]['h5'] / max(heads[h]['n'], 1),
                                        'n': heads[h]['n']} for h in HEADS}}
                                for sp, fl, heads in acc_rows()],
                   'unigram': ug}, open(f'{OUT}/per_head_acc.json', 'w'), indent=2)
        print(f'\nwrote {OUT}/per_head_acc.json')
    elif mode == 'train_seen':
        seen, steps = seen_sentence_indices()
        print(f'[*] replaying the training sampler: {len(seen)} distinct TRAIN sentences were '
              f'optimised on in {steps} steps', flush=True)
        t0 = time.time()
        run_indices('train', seen, floor=0, tag='train_seen')
        print(f'  train_seen (ctx>=0): {time.time()-t0:.0f}s '
              f'n={ACC[("train_seen",0)]["r"]["n"]}', flush=True)
        t0 = time.time()
        run_indices('train', seen, floor=64, tag='train_seen')
        print(f'  train_seen (ctx>=64): {time.time()-t0:.0f}s '
              f'n={ACC[("train_seen",64)]["r"]["n"]}', flush=True)
        tbl(acc_rows())
        json.dump({f'{sp}|{fl}': {h: {'top1': ACC[(sp, fl)][h]['h1'] / max(ACC[(sp, fl)][h]['n'], 1),
                                       'top5': ACC[(sp, fl)][h]['h5'] / max(ACC[(sp, fl)][h]['n'], 1),
                                       'n': ACC[(sp, fl)][h]['n']} for h in HEADS}
                   for sp, fl in ACC}, open(f'{OUT}/acc_trainseen.json', 'w'), indent=2)
        print(f'wrote {OUT}/acc_trainseen.json')
    elif mode == 'top':
        for split in ('train', 'test_gen', 'test_deriv'):
            t0 = time.time()
            run(split, floor=0, n_batches=300, collect=True)
            print(f'  collected {split} ({time.time()-t0:.0f}s)', flush=True)
        res = {}
        for (split, kind), c in sorted(TUPLE_TOP.items()):
            tot = sum(c.values())
            res[f'{split}/{kind}'] = {
                'total': tot, 'distinct': len(c),
                'top': [{'tuple': list(t), 'n': n, 'share': n / max(tot, 1),
                         'prefix': VOCAB.id2prefix.get(t[0], '?'),
                         'root': VOCAB.id2root.get(t[1], '?'),
                         'wazn': VOCAB.id2wazn.get(t[2], '?'),
                         'suffix': VOCAB.id2suffix.get(t[3], '?'),
                         'surface': surface(t)} for t, n in c.most_common(20)]}
        json.dump(res, open(f'{OUT}/tuple_collapse.json', 'w'), ensure_ascii=False, indent=2)
        for split in ('train', 'test_gen', 'test_deriv'):
            c = TUPLE_TOP[(split, 'pred_top1')]
            tot = sum(c.values())
            print(f'\n### {split}: model top-1 tuple covers {len(c)} distinct tuples on '
                  f'{tot} real positions; top-5 share = '
                  f'{sum(n for _, n in c.most_common(5))/max(tot,1)*100:.2f}%')
            for t, n in c.most_common(12):
                print(f"   {n/max(tot,1)*100:6.2f}%  n={n:7d}  tuple={t}  "
                      f"{VOCAB.id2prefix.get(t[0],'?')}+{VOCAB.id2root.get(t[1],'?')}"
                      f"+{VOCAB.id2wazn.get(t[2],'?')}+{VOCAB.id2suffix.get(t[3],'?')}"
                      f"  -> {surface(t)!r}")
            tc = TUPLE_TOP[(split, 'true')]
            tt = sum(tc.values())
            print(f'   -- for contrast, the TRUE top-1 tuples:')
            for t, n in tc.most_common(6):
                print(f"   {n/max(tt,1)*100:6.2f}%  n={n:7d}  -> {surface(t)!r}  "
                      f"[{VOCAB.id2prefix.get(t[0],'?')}+{VOCAB.id2root.get(t[1],'?')}"
                      f"+{VOCAB.id2wazn.get(t[2],'?')}+{VOCAB.id2suffix.get(t[3],'?')}]")
        print(f'\nwrote {OUT}/tuple_collapse.json')
    elif mode == 'special':
        out = {}
        for split in ('train', 'test_gen', 'test_deriv'):
            run(split, floor=0, n_batches=300, collect=True)
        for (split, h), c in sorted(PRED_WHEN_CORRECT.items()):
            tot = sum(c.values())
            rec = {'n_correct_top1': tot,
                   'top': [{'id': i,
                            'name': (VOCAB.id2root.get(i, '?') if h == 'r' else
                                     VOCAB.id2wazn.get(i, '?') if h == 'w' else
                                     VOCAB.id2prefix.get(i, '?') if h == 'p' else
                                     VOCAB.id2suffix.get(i, '?')),
                            'n': n, 'share': n / max(tot, 1)} for i, n in c.most_common(12)]}
            out[f'{split}/{NAME[h]}'] = rec
            if h != 'r':
                continue
            print(f'\n### {split} / root: {tot} correct top-1 root predictions. '
                  f'What were they?')
            for i, n in c.most_common(15):
                nm = VOCAB.id2root.get(i, '?')
                tag = 'SPECIAL' if i in SPECIAL_ROOT_IDS else (
                    'HELD-OUT' if i in HOLD else 'real')
                print(f"      id={i:5d} {nm!r:18s} [{tag:8s}] n={n:7d} {n/max(tot,1)*100:6.2f}%")
        # how concentrated: entropy of the correct-prediction distribution
        json.dump(out, open(f'{OUT}/special_composition.json', 'w'), ensure_ascii=False, indent=2)
        print(f'\nwrote {OUT}/special_composition.json')
