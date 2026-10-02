#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scratch_lm.py -- from-scratch NRMT-P language modelling, with NO Qwen transplant.

The clean test of the project's thesis: train from random init on Arabic, with one position per word
carrying (prefix, root, wazn, suffix), predicting the NEXT word's tuple with four heads. No BPE
anywhere. Context is measured in WORDS, so the sequence length is the word count.

Reported on held-out WORKS:
  * BITS PER WORD, unweighted (true information content) -- reported next to the CONTEXT-FREE
    UNIGRAM bits/word, which is the number a model must beat to show it learned structure.
  * NEXT-ROOT accuracy (top-1/top-5), likewise against the unigram reference.

Plus a DERIVATIONAL-FAMILY HOLDOUT: every sentence containing a word whose root is in the held-out
root set is removed from training, and the model is scored on sentences built from exactly those
roots. Because one root embedding is shared across كتب / كاتب / مكتوب / كتابة, the morphemic scheme
can generalise to unseen derivational forms in a way a subword scheme structurally cannot.

Usage:
  python scratch_lm.py prepare --max-words 25000000
  python scratch_lm.py train --steps 6000
"""
import argparse
import glob
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, '/workspace/hf_v19_2_release')
sys.path.insert(0, '/workspace/hf_v19_2_release/models')

AR = re.compile(r'[\u0600-\u06FF]')
DIAC = re.compile(r'[\u064b-\u0652\u0670\u0640]')
SPLIT = re.compile(r'[\n.!?؟؛:]+')
OUT = Path('/workspace/sf_data')
CTX_WORDS = 128
HOLD_ROOTS_N = 200

SOURCES = [
    '/workspace/scholastic_sanitized',
    '/workspace/andalusian_canon_sanitized',
    '/workspace/heritage_foundations',
    '/workspace/rootformer_v12/v18_next_root_morph/data',
    '/workspace/rootformer_v12/raw_translations',
    '/workspace/rootformer_v12/v17_deepseek_flash/data',
]


def norm(s):
    return DIAC.sub('', s).strip()


def sentences_by_file():
    """Yield (file_key, sentence) with sentence-level text."""
    for root in SOURCES:
        if not os.path.isdir(root):
            continue
        for ext in ('*.txt', '*.jsonl', '*.json'):
            for p in sorted(glob.glob(os.path.join(root, '**', ext), recursive=True)):
                if os.path.getsize(p) > 400 * 1024 * 1024:
                    continue
                key = os.path.basename(p)
                try:
                    if p.endswith('.jsonl'):
                        chunks = []
                        for line in open(p, encoding='utf-8', errors='ignore'):
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                d = json.loads(line)
                            except Exception:
                                continue
                            for k in ('arabic', 'text', 'ar'):
                                if isinstance(d.get(k), str):
                                    chunks.append(d[k]); break
                    elif p.endswith('.json'):
                        d = json.load(open(p, encoding='utf-8', errors='ignore'))
                        chunks = []

                        def rec(x):
                            if isinstance(x, str):
                                if AR.search(x):
                                    chunks.append(x)
                            elif isinstance(x, dict):
                                hit = False
                                for k in ('arabic', 'text', 'ar', 'content'):
                                    if isinstance(x.get(k), str):
                                        chunks.append(x[k]); hit = True; break
                                if not hit:
                                    for v in x.values():
                                        rec(v)
                            elif isinstance(x, list):
                                for v in x[:5000]:
                                    rec(v)
                        rec(d)
                    else:
                        chunks = [Path(p).read_text(encoding='utf-8', errors='ignore')]
                except Exception:
                    continue
                for ch in chunks:
                    for s in SPLIT.split(ch):
                        s = s.strip()
                        if len(s) < 12 or not AR.search(s):
                            continue
                        ws = [w for w in s.split() if AR.search(w)]
                        if len(ws) < 4:
                            continue
                        yield key, ' '.join(ws)


def get_vocab():
    import nrmp_vocab as nv
    V = next(v for k, v in vars(nv).items() if isinstance(v, type) and 'MorphemicVocab' in k)
    return V('/workspace/hf_v19_2_release/data/rootformer_v12_arabic_blueprint.json')


def prepare(args):
    OUT.mkdir(parents=True, exist_ok=True)
    vocab = get_vocab()
    print(f'[*] vocab: roots={vocab.num_roots} awzan={vocab.num_awzan} '
          f'prefixes={vocab.num_prefixes} suffixes={vocab.num_suffixes}', flush=True)

    # ---- collect, dedupe, split BY FILE -------------------------------------------------
    by_file = {}
    seen = set()
    t0 = time.time()
    for key, s in sentences_by_file():
        k = norm(s)
        if k in seen:
            continue
        seen.add(k)
        by_file.setdefault(key, []).append(s)
    files = sorted(by_file, key=lambda k: -len(by_file[k]))
    print(f'[*] {len(files)} files, {sum(len(v) for v in by_file.values())} unique sentences '
          f'({time.time()-t0:.0f}s)', flush=True)

    rng = random.Random(0)
    rng.shuffle(files)
    n_test = max(4, len(files) // 10)
    test_files, rest = files[:n_test], files[n_test:]
    n_val = max(2, len(rest) // 15)
    val_files, train_files = rest[:n_val], rest[n_val:]
    print(f'[*] held-out works: test={len(test_files)} val={len(val_files)} '
          f'train={len(train_files)}', flush=True)

    # ---- derivational holdout roots -----------------------------------------------------
    sample_words = []
    for f in train_files[:400]:
        for s in by_file[f][:40]:
            sample_words.extend(s.split())
    rng.shuffle(sample_words)
    rc = Counter()
    for w in sample_words[:400000]:
        e = vocab.encode_sentence(w)
        if e:
            rc[e[0][1]] += 1
    # IMPORTANT: exclude special tokens. The raw frequency table is topped by <UNK>,
    # <PARTICLE> and <P:...> clitics -- holding those out would delete almost every sentence
    # and would test nothing derivational. Hold out real triconsonantal CONTENT roots only.
    # Selecting the most FREQUENT roots removed 90% of all sentences (200 common roots like
    # قول/علم/كون occur almost everywhere), leaving only 2.7M training words. Instead pick roots
    # by SENTENCE CONTAINMENT and stop once ~10% of sentences are held out, so the training corpus
    # survives while the held-out families are genuinely unseen.
    probe = []
    for f in train_files[:600]:
        for s in by_file[f][:60]:
            probe.append(s)
    rng.shuffle(probe)
    probe = probe[:120000]
    containment = Counter()
    for s in probe:
        for r in {t[1] for t in vocab.encode_sentence(s)}:
            containment[r] += 1
    nprobe = max(len(probe), 1)
    cand = []
    for r, c in containment.items():
        name = vocab.id2root.get(r, '')
        if name.startswith('<') or len(DIAC.sub('', name)) != 3:
            continue
        if c < 20:                      # must be learnable-from-context, not a hapax
            continue
        cand.append((c / nprobe, r))
    cand.sort()                          # least-containing first
    hold_roots, covered = set(), 0
    for frac, r in cand:
        if covered >= 0.10 * nprobe or len(hold_roots) >= 600:
            break
        hold_roots.add(r)
        covered += max(int(frac * nprobe), 1)
    print(f'[*] holdout targets ~10% of sentences; selected {len(hold_roots)} roots',
          flush=True)
    names = [vocab.id2root.get(r, '?') for r in list(hold_roots)[:8]]
    print(f'[*] derivational holdout roots: {len(hold_roots)} content roots, e.g. {names}',
          flush=True)
    spec = sum(n for r, n in rc.items() if vocab.id2root.get(r, '').startswith('<'))
    unk = rc.get(vocab.UNK_ROOT, 0)
    tot_rc = max(sum(rc.values()), 1)
    print(f'[*] task composition: specials {100*spec/tot_rc:.1f}%  '
          f'(of which <UNK> {100*unk/tot_rc:.1f}%)  real roots {100*(tot_rc-spec)/tot_rc:.1f}%',
          flush=True)

    def roots_of(s):
        e = vocab.encode_sentence(s)
        return {t[1] for t in e}

    train_sents, val_sents = [], []
    kept = dropped = 0
    for fi, f in enumerate(train_files):
        if fi % 20 == 0:
            print(f'    filtering train file {fi+1}/{len(train_files)} '
                  f'kept={kept} dropped={dropped}', flush=True)
        for s in by_file[f]:
            if roots_of(s) & hold_roots:
                dropped += 1
                continue
            train_sents.append(s); kept += 1
    for f in val_files:
        for s in by_file[f]:
            if roots_of(s) & hold_roots:
                continue
            val_sents.append(s)
    test_gen, test_deriv = [], []
    for f in test_files:
        for s in by_file[f]:
            if roots_of(s) & hold_roots:
                test_deriv.append(s)
            else:
                test_gen.append(s)
    print(f'[*] train sentences {kept} (dropped {dropped} for holding out roots)', flush=True)
    print(f'[*] val {len(val_sents)} test_gen {len(test_gen)} test_deriv {len(test_deriv)}',
          flush=True)

    # cap training words
    words = 0
    train_capped = []
    for s in train_sents:
        train_capped.append(s)
        words += len(s.split())
        if words >= args.max_words:
            break
    print(f'[*] training words (capped): {words}', flush=True)

    rng.shuffle(val_sents); rng.shuffle(test_gen); rng.shuffle(test_deriv)
    test_gen, test_deriv = test_gen[:4000], test_deriv[:4000]
    val_sents = val_sents[:2000]

    # ---- encode everything --------------------------------------------------------------
    def enc_words(sents, tag):
        """Return per-sentence id sequences for the four morphemic streams (NRMT-P: 1 per word)."""
        P, R, W, S, L = [], [], [], [], []
        t = time.time()
        for i, s in enumerate(sents):
            e = vocab.encode_sentence(s)
            if len(e) < 2:
                continue
            P.append([x[0] for x in e]); R.append([x[1] for x in e])
            W.append([x[2] for x in e]); S.append([x[3] for x in e])
            L.append(len(e))
            if (i + 1) % 200000 == 0:
                print(f'    {tag} {i+1}/{len(sents)} ({time.time()-t:.0f}s)', flush=True)
        return {'P': P, 'R': R, 'W': W, 'S': S, 'L': L}

    data = {'train': enc_words(train_capped, 'train'),
            'val': enc_words(val_sents, 'val'),
            'test_gen': enc_words(test_gen, 'test_gen'),
            'test_deriv': enc_words(test_deriv, 'test_deriv')}
    torch.save(data, OUT / 'streams.pt')
    meta = {'n_roots': vocab.num_roots, 'n_awzan': vocab.num_awzan,
            'n_prefixes': vocab.num_prefixes, 'n_suffixes': vocab.num_suffixes,
            'hold_roots': sorted(hold_roots),
            'special_root_ids': [i for i, r in enumerate(vocab.roots_list)
                                 if str(r).startswith('<')],
            'train_words': words, 'max_words': args.max_words,
            'test_files': test_files, 'val_files': val_files,
            'counts': {k: len(v['L']) for k, v in data.items()}}
    json.dump(meta, open(OUT / 'meta.json', 'w'), indent=2, ensure_ascii=False)
    print(f'[*] saved {OUT}/streams.pt  counts={meta["counts"]}')


# --------------------------------------------------------------------------------- model
class Block(nn.Module):
    def __init__(self, d, h, ffn, drop=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d); self.ln2 = nn.LayerNorm(d)
        self.att = nn.MultiheadAttention(d, h, dropout=drop, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, ffn), nn.GELU(), nn.Linear(ffn, d), nn.Dropout(drop))

    def forward(self, x, mask):
        y = self.ln1(x)
        a, _ = self.att(y, y, y, attn_mask=mask, need_weights=False)
        x = x + a
        return x + self.mlp(self.ln2(x))


class MorphemicLM(nn.Module):
    """NRMT-P: one position per word; four heads predicting the NEXT word's tuple."""
    def __init__(self, sizes, d=512, nl=8, h=8, ffn=2048, maxlen=768):
        super().__init__()
        nr, nw, np_, ns = sizes
        self.d = d
        # Allocate embedding width by difficulty, matching the project's FarahidianWordEmbedding:
        # the root is a 9,114-way decision and must not get the same width as a 22-way suffix.
        d_r, d_w = d // 2, d // 4
        d_p = d // 8
        d_s = d - d_r - d_w - d_p
        self.e_r = nn.Embedding(nr, d_r); self.e_w = nn.Embedding(nw, d_w)
        self.e_p = nn.Embedding(np_, d_p); self.e_s = nn.Embedding(ns, d_s)
        self.pos = nn.Embedding(maxlen, d)
        self.blocks = nn.ModuleList([Block(d, h, ffn) for _ in range(nl)])
        self.ln = nn.LayerNorm(d)
        self.h_r = nn.Linear(d, nr, bias=False); self.h_w = nn.Linear(d, nw, bias=False)
        self.h_p = nn.Linear(d, np_, bias=False); self.h_s = nn.Linear(d, ns, bias=False)

    def forward(self, p, r, w, s):
        T = r.shape[1]
        x = torch.cat([self.e_r(r), self.e_w(w), self.e_p(p), self.e_s(s)], -1)
        x = x + self.pos(torch.arange(T, device=r.device))[None]
        m = torch.triu(torch.full((T, T), float('-inf'), device=r.device), 1)
        for b in self.blocks:
            x = b(x, m)
        h = self.ln(x)
        return self.h_r(h), self.h_w(h), self.h_p(h), self.h_s(h)


# --------------------------------------------------------------------------------- train
def batches_morph(d, split, bs, rng, max_sent=64):
    P, R, W, S, L = d[split]['P'], d[split]['R'], d[split]['W'], d[split]['S'], d[split]['L']
    n = len(L)
    order = rng.sample(range(n), min(bs * max_sent, n))
    cur = {'p': [], 'r': [], 'w': [], 's': []}
    for i in order:
        if L[i] < 2:
            continue
        cur['p'].append(P[i][:CTX_WORDS]); cur['r'].append(R[i][:CTX_WORDS])
        cur['w'].append(W[i][:CTX_WORDS]); cur['s'].append(S[i][:CTX_WORDS])
        if len(cur['r']) == bs:
            yield cur; cur = {'p': [], 'r': [], 'w': [], 's': []}


@torch.no_grad()
def eval_bits(model, data, splits, device, bs=16):
    """bits per WORD, unweighted, on each split."""
    model.eval()
    res = {}
    for split in splits:
        total_bits, total_words = 0.0, 0
        rng = random.Random(7)
        for b in batches_morph(data, split, bs, rng, max_sent=8000):
            r = pad([x[1:] for x in b['r']], 0).to(device)
            p_in = pad([x[:-1] for x in b['p']], 0).to(device)
            r_in = pad([x[:-1] for x in b['r']], 0).to(device)
            w_in = pad([x[:-1] for x in b['w']], 0).to(device)
            s_in = pad([x[:-1] for x in b['s']], 0).to(device)
            w_tg = pad([x[1:] for x in b['w']], 0).to(device)
            p_tg = pad([x[1:] for x in b['p']], 0).to(device)
            s_tg = pad([x[1:] for x in b['s']], 0).to(device)
            # valid = real word positions only; padding must not enter the average
            lens = [len(x) - 1 for x in b['r']]
            T = r.shape[1]
            m = torch.zeros((len(lens), T), dtype=torch.bool)
            for bi, L in enumerate(lens):
                m[bi, :min(L, T)] = True
            m = m.to(device)
            lr, lw, lp, ls = model(p_in, r_in, w_in, s_in)
            nll = (F.cross_entropy(lr.reshape(-1, lr.shape[-1]), r.reshape(-1), reduction='none')
                   + F.cross_entropy(lw.reshape(-1, lw.shape[-1]), w_tg.reshape(-1),
                                     reduction='none')
                   + F.cross_entropy(lp.reshape(-1, lp.shape[-1]), p_tg.reshape(-1),
                                     reduction='none')
                   + F.cross_entropy(ls.reshape(-1, ls.shape[-1]), s_tg.reshape(-1),
                                     reduction='none')).reshape(r.shape)
            total_bits += float(nll[m].sum()) / math.log(2)
            total_words += int(m.sum())
        res[split] = total_bits / max(total_words, 1)
    return res


def pad(seqs, val):
    """Right-pad a list of id sequences to a rectangle."""
    T = max(len(x) for x in seqs)
    o = torch.full((len(seqs), T), val, dtype=torch.long)
    for i, x in enumerate(seqs):
        o[i, :len(x)] = torch.tensor(x) if not isinstance(x, torch.Tensor) else x
    return o


def _counts(data, split):
    """Unigram counts per stream from a split (no training needed)."""
    C = [Counter(), Counter(), Counter(), Counter()]
    d = data[split]
    for arr, c in zip((d['R'], d['W'], d['P'], d['S']), C):
        for seq in arr:
            c.update(seq)
    return C


def unigram_bits(data, splits):
    """Bits/word of a context-free unigram model -- the number a model must beat."""
    C = _counts(data, 'train')
    tot = [sum(c.values()) for c in C]
    # NOTE: the denominator must be the number of WORDS, not 4x that. Bits per word is the sum
    # of the four streams' NLLs for one word -- the same definition the model is scored with.
    out = {}
    for sp in splits:
        d = data[sp]
        bits, npos = 0.0, 0
        for si, (arr, c, T) in enumerate(zip((d['R'], d['W'], d['P'], d['S']), C, tot)):
            V = max(c) + 2
            for seq in arr[:-1]:
                for x in seq[1:]:
                    p = (c.get(x, 0) + 0.5) / (T + 0.5 * V)
                    bits += -math.log2(max(p, 1e-12))
                if si == 0:                       # count positions ONCE, not per stream
                    npos += max(len(seq) - 1, 0)
        out[sp] = bits / max(npos, 1)
    return out


def unigram_root_acc(data, splits, special_ids):
    C = _counts(data, 'train')
    top1 = C[0].most_common(1)[0][0]
    top5 = {x for x, _ in C[0].most_common(5)}
    sp_set = set(special_ids)
    out = {}
    for sp in splits:
        d = data[sp]
        acc = {k: {'n': 0, 'h1': 0, 'h5': 0} for k in ('all', 'real', 'special')}
        for seq in d['R'][:-1]:
            for x in seq[1:]:
                keys = ['all'] + (['special'] if x in sp_set else ['real'])
                for k in keys:
                    acc[k]['n'] += 1
                    acc[k]['h1'] += (x == top1)
                    acc[k]['h5'] += (x in top5)
        out[sp] = {k: {'top1': v['h1'] / max(v['n'], 1),
                       'top5': v['h5'] / max(v['n'], 1), 'n': v['n']}
                   for k, v in acc.items()}
    return out


@torch.no_grad()
def root_accuracy(model, data, splits, device, n_roots, special_ids, bs=16, max_sent=8000):
    """Next-root accuracy on held-out WORKS, split REAL roots vs special tokens.

    ~44% of positions in this corpus are <PARTICLE>/<P:...>/<UNK>, which are trivially
    predictable. Averaging over all positions therefore flatters the model; the real-root
    column is the one that reflects root prediction.
    """
    model.eval()
    spec = torch.zeros(n_roots, dtype=torch.bool)
    for i in special_ids:
        if 0 <= i < n_roots:
            spec[i] = True
    spec = spec.to(device)
    out = {}
    for sp in splits:
        acc = {k: {'n': 0, 'h1': 0, 'h5': 0} for k in ('all', 'real', 'special')}
        rng = random.Random(11)
        D = data[sp]
        order = rng.sample(range(len(D['L'])), min(max_sent, len(D['L'])))
        for i in range(0, len(order), bs):
            chunk = order[i:i + bs]
            P = [D['P'][j][:CTX_WORDS] for j in chunk]
            R = [D['R'][j][:CTX_WORDS] for j in chunk]
            W = [D['W'][j][:CTX_WORDS] for j in chunk]
            S = [D['S'][j][:CTX_WORDS] for j in chunk]
            if min(len(x) for x in R) < 2:
                continue
            r_tg = pad([x[1:] for x in R], 0).to(device)
            p_in = pad([x[:-1] for x in P], 0).to(device)
            r_in = pad([x[:-1] for x in R], 0).to(device)
            w_in = pad([x[:-1] for x in W], 0).to(device)
            s_in = pad([x[:-1] for x in S], 0).to(device)
            m = torch.zeros_like(r_tg, dtype=torch.bool)
            for bi, x in enumerate(R):
                m[bi, :min(len(x) - 1, r_tg.shape[1])] = True
            lr, _, _, _ = model(p_in, r_in, w_in, s_in)
            pred1 = lr.argmax(-1)
            pred5 = lr.topk(min(5, lr.shape[-1]), -1).indices
            hit1 = (pred1 == r_tg) & m
            hit5 = (pred5 == r_tg.unsqueeze(-1)).any(-1) & m
            is_real = (~spec[r_tg]) & m
            is_spec = spec[r_tg] & m
            for key, mm in (('all', m), ('real', is_real), ('special', is_spec)):
                acc[key]['n'] += int(mm.sum())
                acc[key]['h1'] += int((hit1 & mm).sum())
                acc[key]['h5'] += int((hit5 & mm).sum())
        out[sp] = {k: {'top1': v['h1'] / max(v['n'], 1),
                       'top5': v['h5'] / max(v['n'], 1), 'n': v['n']}
                   for k, v in acc.items()}
    return out


def train(args):
    device = torch.device('cuda')
    meta = json.load(open(OUT / 'meta.json'))
    data = torch.load(OUT / 'streams.pt', weights_only=False)
    rng = random.Random(0)
    torch.manual_seed(0)

    model = MorphemicLM((meta['n_roots'], meta['n_awzan'],
                         meta['n_prefixes'], meta['n_suffixes'])).to(device)
    nparam = sum(p.numel() for p in model.parameters())
    print(f'[*] ARM={args.arm} params={nparam/1e6:.2f}M train_words={meta["train_words"]}',
          flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.steps,
        pct_start=max(0.02, min(0.3, 200.0 / max(args.steps, 1))))
    t0 = time.time()
    step = 0
    while step < args.steps:
        it = batches_morph(data, 'train', args.batch, rng)
        for b in it:
            if args.arm == 'morph':
                p_in = pad([x[:-1] for x in b['p']], 0).to(device)
                r_in = pad([x[:-1] for x in b['r']], 0).to(device)
                w_in = pad([x[:-1] for x in b['w']], 0).to(device)
                s_in = pad([x[:-1] for x in b['s']], 0).to(device)
                r_tg = pad([x[1:] for x in b['r']], 0).to(device)
                w_tg = pad([x[1:] for x in b['w']], 0).to(device)
                p_tg = pad([x[1:] for x in b['p']], 0).to(device)
                s_tg = pad([x[1:] for x in b['s']], 0).to(device)
                lr, lw, lp, ls = model(p_in, r_in, w_in, s_in)
                ig = 0   # pad id; never a legitimate target
                loss = (F.cross_entropy(lr.reshape(-1, lr.shape[-1]), r_tg.reshape(-1),
                                        ignore_index=ig)
                        + 0.5 * F.cross_entropy(lw.reshape(-1, lw.shape[-1]), w_tg.reshape(-1),
                                                ignore_index=ig)
                        + 0.25 * F.cross_entropy(lp.reshape(-1, lp.shape[-1]), p_tg.reshape(-1),
                                                 ignore_index=ig)
                        + 0.25 * F.cross_entropy(ls.reshape(-1, ls.shape[-1]), s_tg.reshape(-1),
                                                 ignore_index=ig))
            else:
                x = nn.utils.rnn.pad_sequence(b, batch_first=True, padding_value=0).to(device)
                logits = model(x[:, :-1])
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                       x[:, 1:].reshape(-1), ignore_index=0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % 500 == 0:
                print(f'  step {step}/{args.steps} loss {float(loss):.4f} '
                      f'{step/(time.time()-t0):.2f} it/s', flush=True)
            if step % 10000 == 0:
                vb = eval_bits(model, data, ['val'], device)
                print(f'    [val @{step}] bits/word {vb["val"]:.3f}', flush=True)
                model.train()
            if step >= args.steps:
                break

    out = {'arm': 'morph', 'params': nparam, 'steps': args.steps}
    out['unigram_bits_per_word'] = unigram_bits(data, ['val', 'test_gen', 'test_deriv'])
    out['bits_per_word'] = eval_bits(model, data, ['val', 'test_gen', 'test_deriv'], device)
    out['root_acc'] = root_accuracy(model, data, ['test_gen', 'test_deriv'], device,
                                    meta['n_roots'], meta['special_root_ids'])
    out['unigram_root_acc'] = unigram_root_acc(data, ['test_gen', 'test_deriv'],
                                               meta['special_root_ids'])
    print(f'\n  *** {args.arm}: bits/word  ' +
          '  '.join(f'{k}={v:.3f}' for k, v in out['bits_per_word'].items()), flush=True)
    json.dump(out, open('/workspace/sf_morph.json', 'w'), indent=2)
    torch.save(model.state_dict(), '/workspace/sf_morph.pt')
    print(f'wrote /workspace/sf_morph.json')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['prepare', 'train'])
    ap.add_argument('--arm', choices=['morph'], default='morph')
    ap.add_argument('--max-words', type=int, default=25_000_000)
    ap.add_argument('--steps', type=int, default=6000)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--lr', type=float, default=6e-4)
    args = ap.parse_args()
    if args.mode == 'prepare':
        prepare(args)
    else:
        train(args)


if __name__ == '__main__':
    main()
