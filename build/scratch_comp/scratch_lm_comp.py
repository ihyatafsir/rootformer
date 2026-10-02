#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scratch_lm_comp.py -- COMPOSITIONAL root decoder for the from-scratch NRMT-P LM.

WHY THIS FILE EXISTS
--------------------
`scratch_lm.py` (md5 71db165f8f9d09f8725402834b793a3f, kept byte-identical) predicts the next
word's tuple (prefix, root, wazn, suffix) with four heads, the ROOT head being a softmax over
root *types*.  On its derivational-family holdout that head scored 0.000000 top-1 AND top-5 at
both 6,000 and 80,000 steps -- structurally forced, not an empirical finding: a held-out root
type occurs 0x in training, so its input row never receives gradient and its output row is only
ever trained as a NEGATIVE class.  A categorical softmax over root types cannot emit a type it
has never seen.  The holdout therefore measured the head design, not the morphemic claim.

WHAT CHANGES (and only this)
----------------------------
The root head becomes a COMPOSITIONAL decoder over the root's radicals:

    P(root | h) = P(gate = CONTENT | h) * PROD_s P(radical_s | radical_<s, h)

  * `gate` is a 271-way classifier: the 270 closed-class root tokens (<UNK>, <PARTICLE>,
    <P:..>) plus one CONTENT class.  Specials are a genuine label set -- they have no
    radicals -- so they stay a label set.
  * the CONTENT part is a small GRU decoder (one step per radical slot, max 5 slots, forced
    by the observed root-length distribution 2..5) over the radical alphabet implied by the
    training roots.  Initial state = tanh(Linear(h)).

An unseen root is now expressible BY CONSTRUCTION: every sequence of 2..5 alphabet symbols has
nonzero probability, whether or not it is a vocabulary type.  Nothing else is weakened:

  * same corpora, same sentence/word tokenisation, same word-level context (CTX_WORDS=128);
  * same four-stream input (one position per word, NRMT-P), same backbone (d=512, 8 layers,
    8 heads, ffn 2048) with byte-identical submodule names and shapes;
  * same prefix/wazn/suffix heads with the same loss weights (0.5/0.25/0.25);
  * same bits/word definition: bits/word = (sum of the four heads' NLL for one word)/log 2;
  * same split-by-WORK procedure, same leak-checked derivational holdout construction.

The root NLL is now the chain-rule NLL of the root string, i.e. still exactly
-log2 P(next root) for the same event, so bits/word remains comparable.

Arms
----
  --root-head comp   the fix (compositional decoder)
  --root-head type   the original 9114/9490-way root-type softmax, re-run on the SAME new data
                     so the comparison is apples-to-apples (this is the baseline, not a
                     weakened version of it: it is the original head verbatim)

Modes
-----
  prepare  build the dataset with the CURRENT release vocab (9490 roots / 142 awzan)
  train    train one arm
  eval     full evaluation of a checkpoint: bits/word + per-head top-1, root top-1/top-5 with
           exact best-first search over sequences, held-out partial credit + examples
"""
import argparse
import glob
import hashlib
import heapq
import itertools
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, '/workspace/hf_v19_2_release')
sys.path.insert(0, '/workspace/hf_v19_2_release/models')

AR = re.compile(r'[\u0600-\u06FF]')
DIAC = re.compile(r'[\u064b-\u0652\u0670\u0640]')
SPLIT = re.compile(r'[\n.!?؟؛:]+')
OUT = Path('/workspace/scratch_comp/sf_data_9490')
CTX_WORDS = 128
ARABIC28 = 'ابتثجحخدذرزسشصضطظعغفقكلمنهوي'      # a priori linguistic prior, not data-derived
assert len(ARABIC28) == 28
EOS_SENT = -1                                   # terminal marker inside a search prefix

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
    """Yield (file_key, sentence). VERBATIM from scratch_lm.py."""
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
                except Exception as exc:
                    print(f'[scratch_lm_comp] skipping corpus file {p!r}: {exc!r}', file=sys.stderr)
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


def blueprint_md5(path='/workspace/hf_v19_2_release/data/rootformer_v12_arabic_blueprint.json'):
    return hashlib.md5(open(path, 'rb').read()).hexdigest()


# ==================================================================================== prepare
def prepare(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    vocab = get_vocab()
    print(f'[*] CURRENT release vocab: roots={vocab.num_roots} awzan={vocab.num_awzan} '
          f'prefixes={vocab.num_prefixes} suffixes={vocab.num_suffixes}', flush=True)
    print(f'[*] blueprint md5={blueprint_md5()}', flush=True)

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

    # ---- derivational holdout roots (IDENTICAL procedure to scratch_lm.py) --------------
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
        if c < 20:
            continue
        cand.append((c / nprobe, r))
    cand.sort()
    hold_roots, covered = set(), 0
    for frac, r in cand:
        if covered >= 0.10 * nprobe or len(hold_roots) >= 600:
            break
        hold_roots.add(r)
        covered += max(int(frac * nprobe), 1)
    print(f'[*] holdout targets ~10% of sentences; selected {len(hold_roots)} roots', flush=True)
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
        return {t[1] for t in vocab.encode_sentence(s)}

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

    def enc_words(sents, tag):
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

    # ---- radical alphabet, from the TRAINING STREAM ONLY (leak-free) --------------------
    # The 28 standard Arabic letters are a fixed a-priori prior; any other symbol must have
    # been observed in a training root.  A held-out root is expressible iff all its symbols
    # are in this alphabet -- checked and reported below.
    observed_roots = {}
    for seq in data['train']['R']:
        for x in seq:
            observed_roots[x] = None
    chars = set(ARABIC28)
    for rid in observed_roots:
        name = vocab.id2root.get(rid, '')
        if not name.startswith('<'):
            chars.update(name)
    alphabet = sorted(chars)
    char2i = {c: i for i, c in enumerate(alphabet)}
    n_sym = len(alphabet) + 1                 # letters + EOS/BOS (shared index)
    n_alphabet = len(alphabet)

    special_root_ids = [i for i, r in enumerate(vocab.roots_list) if str(r).startswith('<')]
    spec_set = set(special_root_ids)
    content_root_ids = [i for i in range(vocab.num_roots) if i not in spec_set]

    root_letters = {}
    inexpressible = []
    for rid in content_root_ids:
        name = vocab.id2root.get(rid, '')
        if any(c not in char2i for c in name):
            inexpressible.append(rid)
            continue
        root_letters[str(rid)] = [char2i[c] for c in name]

    missing_from_train = [c for c in ARABIC28 if c not in set(chars)]
    print(f'[*] radical alphabet: {n_alphabet} symbols '
          f'(28 a-priori Arabic + {n_alphabet-28} observed extra)', flush=True)
    print(f'[*]   standard letters not observed in training roots: {missing_from_train}',
          flush=True)
    print(f'[*]   content roots NOT expressible with this alphabet: {len(inexpressible)}',
          flush=True)
    hold_inexpr = sorted(set(inexpressible) & hold_roots)
    print(f'[*]   of which held-out roots: {len(hold_inexpr)} {hold_inexpr}', flush=True)

    tc = Counter()
    for seq in data['train']['R']:
        tc.update(seq)
    len_dist = Counter()
    for rid, n in tc.items():
        if rid in spec_set or rid == 0:
            continue
        len_dist[len(vocab.id2root.get(rid, ''))] += n
    print(f'[*] ROOT LENGTH distribution in the training STREAM (tokens): '
          f'{dict(sorted(len_dist.items()))}', flush=True)
    print(f'[*]   held-out root strings: '
          f'{[vocab.id2root.get(r, "?") for r in sorted(hold_roots)]}', flush=True)

    torch.save(data, out / 'streams.pt')
    meta = {'n_roots': vocab.num_roots, 'n_awzan': vocab.num_awzan,
            'n_prefixes': vocab.num_prefixes, 'n_suffixes': vocab.num_suffixes,
            'hold_roots': sorted(hold_roots),
            'special_root_ids': special_root_ids,
            'content_root_ids': content_root_ids,
            'roots_list': [str(r) for r in vocab.roots_list],
            'alphabet': alphabet, 'n_alphabet': n_alphabet, 'n_sym': n_sym,
            'root_letters': root_letters,
            'inexpressible_content_roots': inexpressible,
            'hold_roots_inexpressible': hold_inexpr,
            'train_words': words, 'max_words': args.max_words,
            'test_files': test_files, 'val_files': val_files,
            'blueprint_md5': blueprint_md5(),
            'root_len_dist_train_tokens': {str(k): v for k, v in sorted(len_dist.items())},
            'counts': {k: len(v['L']) for k, v in data.items()}}
    json.dump(meta, open(out / 'meta.json', 'w'), indent=2, ensure_ascii=False)
    print(f'[*] saved {out}/streams.pt  counts={meta["counts"]}', flush=True)


# ==================================================================================== tables
class Tables:
    """Root-id <-> (gate class, radical sequence) tables, device-independent."""

    def __init__(self, meta):
        self.meta = meta
        self.id2root = {i: r for i, r in enumerate(meta['roots_list'])}
        self.special_ids = list(meta['special_root_ids'])
        self.n_special = len(self.special_ids)
        self.n_gate = self.n_special + 1
        self.content_class = self.n_special
        self.gate_to_root = list(self.special_ids) + [None]
        n_roots = meta['n_roots']
        gate_tab = torch.full((n_roots,), -100, dtype=torch.long)
        for j, rid in enumerate(self.special_ids):
            gate_tab[rid] = j
        for rid in meta['content_root_ids']:
            gate_tab[rid] = self.content_class
        gate_tab[0] = -100                                   # <PAD> is never a target
        self.gate_tab = gate_tab
        letters = torch.zeros((n_roots, 5), dtype=torch.long)
        lens = torch.zeros((n_roots,), dtype=torch.long)
        for k, v in meta['root_letters'].items():
            rid = int(k)
            letters[rid, :len(v)] = torch.tensor(v, dtype=torch.long)
            lens[rid] = len(v)
        self.letters = letters
        self.lens = lens
        self.alphabet = list(meta['alphabet'])
        self.n_alphabet = len(self.alphabet)
        self.eos = self.n_alphabet
        self.bos = self.n_alphabet
        self.special_set = set(self.special_ids)
        self.hold = set(meta['hold_roots'])
        self.hold_inexpr = set(meta['hold_roots_inexpressible'])

    def root_str(self, rid):
        return self.id2root.get(rid, '?')

    def letters_str(self, letters):
        return ''.join(self.alphabet[c] for c in letters)


# ==================================================================================== model
class Block(nn.Module):
    """VERBATIM from scratch_lm.py."""
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


class Backbone(nn.Module):
    """The scratch_lm.MorphemicLM body: byte-identical module names, shapes and forward."""

    def __init__(self, sizes, d=512, nl=8, h=8, ffn=2048, maxlen=768):
        super().__init__()
        nr, nw, np_, ns = sizes
        self.d = d
        d_r, d_w = d // 2, d // 4
        d_p = d // 8
        d_s = d - d_r - d_w - d_p
        self.e_r = nn.Embedding(nr, d_r); self.e_w = nn.Embedding(nw, d_w)
        self.e_p = nn.Embedding(np_, d_p); self.e_s = nn.Embedding(ns, d_s)
        self.pos = nn.Embedding(maxlen, d)
        self.blocks = nn.ModuleList([Block(d, h, ffn) for _ in range(nl)])
        self.ln = nn.LayerNorm(d)
        self.h_w = nn.Linear(d, nw, bias=False)
        self.h_p = nn.Linear(d, np_, bias=False)
        self.h_s = nn.Linear(d, ns, bias=False)

    def encode(self, p, r, w, s):
        T = r.shape[1]
        x = torch.cat([self.e_r(r), self.e_w(w), self.e_p(p), self.e_s(s)], -1)
        x = x + self.pos(torch.arange(T, device=r.device))[None]
        m = torch.triu(torch.full((T, T), float('-inf'), device=r.device), 1)
        for b in self.blocks:
            x = b(x, m)
        return self.ln(x)


class CompRootHead(nn.Module):
    """P(root|h) = gate(h) over closed-class specials + CONTENT, then a GRU over radicals."""

    def __init__(self, d, n_gate, n_sym, d_dec=256, d_emb=128, max_slots=5):
        super().__init__()
        self.max_slots = max_slots
        self.gate = nn.Linear(d, n_gate, bias=False)
        self.inp = nn.Embedding(n_sym, d_emb)
        self.h0 = nn.Linear(d, d_dec)
        self.gru = nn.GRUCell(d_emb, d_dec)
        self.out = nn.Linear(d_dec, n_sym, bias=False)

    def teacher_forced(self, h, letters, lens):
        """h (B,T,d); letters (B,T,5) ids; lens (B,T) in 2..5 (0 = special/inexpressible).

        Returns (slot_logp (B,T,5), slot_valid (B,T,5), eos_index)."""
        B, T = h.shape[:2]
        n_sym = self.out.out_features
        eos = n_sym - 1
        dev = h.device
        hf = h.reshape(B * T, -1)
        lf = letters.reshape(B * T, self.max_slots)
        lenf = lens.reshape(B * T)
        state = torch.tanh(self.h0(hf))
        inp = torch.full((B * T,), n_sym - 1, dtype=torch.long, device=dev)      # BOS
        logps, valids = [], []
        ar = torch.arange(self.max_slots, device=dev)
        for s in range(self.max_slots):
            state = self.gru(self.inp(inp), state)
            logp = F.log_softmax(self.out(state), -1)
            tgt = torch.where(ar[s] < lenf, lf[:, s], torch.full_like(lf[:, s], eos))
            valid = (ar[s] < lenf) | ((ar[s] == lenf) & (lenf >= 2))
            sel = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            logps.append(torch.where(valid, sel, torch.zeros_like(sel)).view(B, T))
            valids.append(valid.view(B, T))
            inp = lf[:, s]
        return (torch.stack(logps, -1), torch.stack(valids, -1), eos)


class LMComp(nn.Module):
    def __init__(self, sizes, n_gate, n_sym, **kw):
        super().__init__()
        self.bb = Backbone(sizes, **kw)
        self.root = CompRootHead(self.bb.d, n_gate, n_sym)

    def forward(self, p, r, w, s):
        h = self.bb.encode(p, r, w, s)
        return h, self.root.gate(h), self.bb.h_w(h), self.bb.h_p(h), self.bb.h_s(h)


class LMType(nn.Module):
    """The ORIGINAL scratch_lm.py head, verbatim (softmax over root types)."""

    def __init__(self, sizes, **kw):
        super().__init__()
        self.bb = Backbone(sizes, **kw)
        self.h_r = nn.Linear(self.bb.d, sizes[0], bias=False)

    def forward(self, p, r, w, s):
        h = self.bb.encode(p, r, w, s)
        return h, self.h_r(h), self.bb.h_w(h), self.bb.h_p(h), self.bb.h_s(h)


# ==================================================================================== utils
def pad(seqs, val):
    """VERBATIM from scratch_lm.py."""
    T = max(len(x) for x in seqs)
    o = torch.full((len(seqs), T), val, dtype=torch.long)
    for i, x in enumerate(seqs):
        o[i, :len(x)] = torch.tensor(x) if not isinstance(x, torch.Tensor) else x
    return o


def batches_morph(d, split, bs, rng, max_sent=64):
    """VERBATIM from scratch_lm.py."""
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


def _counts(data, split):
    """VERBATIM from scratch_lm.py."""
    C = [Counter(), Counter(), Counter(), Counter()]
    d = data[split]
    for arr, c in zip((d['R'], d['W'], d['P'], d['S']), C):
        for seq in arr:
            c.update(seq)
    return C


def unigram_bits(data, splits):
    """VERBATIM from scratch_lm.py (context-free root-TYPE unigram, the 12.99/14.74 comparator)."""
    C = _counts(data, 'train')
    tot = [sum(c.values()) for c in C]
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
                if si == 0:
                    npos += max(len(seq) - 1, 0)
        out[sp] = bits / max(npos, 1)
    return out


def unigram_symbol_bits(data, splits, tables):
    """Context-free reference in the COMPOSITIONAL parameterisation, on the same event and the
    same per-word denominator as the type unigram: a content root costs the sum of its
    per-slot symbol unigram NLLs (letters for slot s, EOS from slot 2), and a special root
    costs its root-TYPE unigram NLL -- otherwise specials would be free and this reference
    would not be comparable to bits/word.  This is the fair context-free comparator for a
    decoder that spells the root."""
    C0 = _counts(data, 'train')[0]
    tot0 = max(sum(C0.values()), 1)
    V0 = max(C0) + 2
    counts = [Counter() for _ in range(5)]
    for seq in data['train']['R']:
        for rid in seq:
            if rid in tables.special_set or rid == 0:
                continue
            ls = tables.letters[rid].tolist()
            L = int(tables.lens[rid])
            if L < 2:
                continue
            for s in range(5):
                if s < L:
                    counts[s][ls[s]] += 1
                elif s == L:
                    counts[s]['EOS'] += 1
    tots = [max(sum(c.values()), 1) for c in counts]
    Vs = len(tables.alphabet) + 2
    out = {}
    for sp in splits:
        bits, n = 0.0, 0
        for seq in data[sp]['R'][:-1]:
            for rid in seq[1:]:
                n += 1
                if rid in tables.special_set or rid == 0:
                    p = (C0.get(rid, 0) + 0.5) / (tot0 + 0.5 * V0)
                    bits += -math.log2(max(p, 1e-12))
                    continue
                ls = tables.letters[rid].tolist()
                L = int(tables.lens[rid])
                if L < 2:
                    continue
                for s in range(5):
                    if s < L:
                        sym = ls[s]
                    elif s == L:
                        sym = 'EOS'
                    else:
                        continue
                    p = (counts[s].get(sym, 0) + 0.5) / (tots[s] + 0.5 * Vs)
                    bits += -math.log2(max(p, 1e-12))
        out[sp] = bits / max(n, 1)
    return out


def unigram_root_acc(data, splits, tables):
    """The script's degenerate reference PLUS the two FAIR references the task asks for:
    (a) most frequent REAL (content) root over TRAIN counts, (b) most frequent real root
    WITHIN THE EVAL SLICE itself."""
    C = _counts(data, 'train')
    all_top1 = C[0].most_common(1)[0][0]
    all_top5 = {x for x, _ in C[0].most_common(5)}
    content = {r: n for r, n in C[0].items() if r not in tables.special_set and r != 0}
    ctop1 = max(content, key=content.get)
    ctop5 = {r for r, _ in Counter(content).most_common(5)}
    out = {}
    for sp in splits:
        d = data[sp]['R']
        slice_c = Counter()
        for seq in d[:-1]:
            for x in seq[1:]:
                if x not in tables.special_set and x != 0:
                    slice_c[x] += 1
        stop1 = max(slice_c, key=slice_c.get)
        stop5 = {r for r, _ in slice_c.most_common(5)}
        acc = {k: {'n': 0, 'h1': 0, 'h5': 0} for k in
               ('all', 'real', 'special', 'real_majority_train', 'real_majority_slice')}
        for seq in d[:-1]:
            for x in seq[1:]:
                is_spec = x in tables.special_set
                keys = ['all', 'special' if is_spec else 'real']
                if not is_spec and x != 0:
                    keys += ['real_majority_train', 'real_majority_slice']
                for k in keys:
                    acc[k]['n'] += 1
                    if k == 'real_majority_train':
                        acc[k]['h1'] += (x == ctop1); acc[k]['h5'] += (x in ctop5)
                    elif k == 'real_majority_slice':
                        acc[k]['h1'] += (x == stop1); acc[k]['h5'] += (x in stop5)
                    else:
                        acc[k]['h1'] += (x == all_top1); acc[k]['h5'] += (x in all_top5)
        out[sp] = {k: {'top1': v['h1'] / max(v['n'], 1),
                       'top5': v['h5'] / max(v['n'], 1), 'n': v['n']} for k, v in acc.items()}
    out['_reference_ids'] = {'unigram_all_top1': all_top1,
                             'majority_real_train_top1': ctop1,
                             'majority_real_train_top5': sorted(ctop5),
                             'majority_real_train_str': tables.root_str(ctop1)}
    return out


# ------------------------------------------------------------------ exact best-first search
@torch.no_grad()
def best_first_content(head, h, theta, k=5, max_rounds=200, fanout=8):
    """EXACT top-k over ALL letter sequences of length 2..5 under the compositional decoder,
    limited to sequences with log-prob >= theta (nothing below theta can enter the joint top-k
    once theta comes from the competing special classes).

    Exactness: every factor is a probability <= 1, so a prefix's score is a non-increasing
    upper bound on every extension of it.  A heap therefore holds the current global maximum
    of all reachable nodes, and a sequence is emitted only when it IS that maximum, so the
    emission order is exactly descending.  `fanout` nodes are popped and expanded per position
    per round purely for GPU batching (all popped nodes are expanded before any further heap
    inspection, and emission happens only against the updated heap), which does not change the
    emitted order -- it only cuts the number of round-trips.

    Returns (found, exact): found[i] is a descending list of (logprob, letters_tuple)."""
    N = h.shape[0]
    dev = h.device
    n_alpha = head.out.out_features - 1
    eos = n_alpha
    h0_all = torch.tanh(head.h0(h))
    heaps = [[(0.0, 0, ())] for _ in range(N)]   # (-score, counter, prefix); EOS_SENT terminal
    found = [[] for _ in range(N)]
    exact = [True] * N
    ctr = itertools.count(1)

    def thr_of(i):
        if len(found[i]) < k:
            return theta[i]
        return max(theta[i], found[i][k - 1][0])

    def emit(i):
        hp = heaps[i]
        while True:
            t = thr_of(i)
            while hp and -hp[0][0] < t:
                heapq.heappop(hp)
            if not hp:
                return
            p0 = hp[0][2]
            if not ((len(p0) > 0 and p0[-1] == EOS_SENT) or len(p0) == 5):
                return
            negs, c0, p0 = heapq.heappop(hp)
            letters = p0[:-1] if (len(p0) > 0 and p0[-1] == EOS_SENT) else p0
            found[i].append((-negs, letters))
            if len(found[i]) >= k:
                return

    rounds = 0
    while rounds < max_rounds:
        rounds += 1
        for i in range(N):
            emit(i)
        groups = {}
        for i in range(N):
            hp = heaps[i]
            npop = 0
            while hp and npop < fanout:
                if -hp[0][0] < thr_of(i):
                    heapq.heappop(hp)
                    continue
                p0 = hp[0][2]
                if (len(p0) > 0 and p0[-1] == EOS_SENT) or len(p0) == 5:
                    break
                negs, c0, p0 = heapq.heappop(hp)
                groups.setdefault(len(p0), []).append((i, -negs, p0))
                npop += 1
        if not groups:
            break
        for L, items in groups.items():
            state = h0_all[[it[0] for it in items]]
            for t in range(L + 1):                      # BOS, then the L prefix letters
                if t == 0:
                    sym = torch.full((len(items),), eos, dtype=torch.long, device=dev)
                else:
                    sym = torch.tensor([it[2][t - 1] for it in items],
                                       dtype=torch.long, device=dev)
                state = head.gru(head.inp(sym), state)
            logp = F.log_softmax(head.out(state), -1)
            for jj, (i, s0, p0) in enumerate(items):
                t = thr_of(i)
                lp = logp[jj]
                for c in range(n_alpha):
                    sc = s0 + float(lp[c])
                    if sc >= t:
                        heapq.heappush(heaps[i], (-sc, next(ctr), p0 + (c,)))
                if L >= 2:
                    sc = s0 + float(lp[eos])
                    if sc >= t:
                        heapq.heappush(heaps[i], (-sc, next(ctr), p0 + (EOS_SENT,)))
        for i in range(N):
            emit(i)
    for i in range(N):
        if len(found[i]) < k and heaps[i] and -heaps[i][0][0] >= theta[i]:
            exact[i] = False
    return found, exact


# ------------------------------------------------------------------ evaluation
@torch.no_grad()
def eval_bits(model, data, split, device, tables, root_head, bs=16, max_sent=8000):
    """bits per WORD, unweighted, identical definition to scratch_lm.py, plus the per-head
    breakdown.  For root_head='comp' the root term is the chain-rule NLL of the root string."""
    model.eval()
    total = {'root': 0.0, 'wazn': 0.0, 'prefix': 0.0, 'suffix': 0.0}
    nwords = 0
    rng = random.Random(7)
    for b in batches_morph(data, split, bs, rng, max_sent=max_sent):
        T = max(len(x) for x in b['r'])
        r_tg = pad([x[1:] for x in b['r']], 0).to(device)
        p_in = pad([x[:-1] for x in b['p']], 0).to(device)
        r_in = pad([x[:-1] for x in b['r']], 0).to(device)
        w_in = pad([x[:-1] for x in b['w']], 0).to(device)
        s_in = pad([x[:-1] for x in b['s']], 0).to(device)
        w_tg = pad([x[1:] for x in b['w']], 0).to(device)
        p_tg = pad([x[1:] for x in b['p']], 0).to(device)
        s_tg = pad([x[1:] for x in b['s']], 0).to(device)
        lens = [len(x) - 1 for x in b['r']]
        T = r_tg.shape[1]
        m = torch.zeros((len(lens), T), dtype=torch.bool)
        for bi, L in enumerate(lens):
            m[bi, :min(L, T)] = True
        m = m.to(device)
        h, root_logits, lw, lp, ls = model(p_in, r_in, w_in, s_in)
        if root_head == 'comp':
            gate_tg = tables.gate_tab.to(device)[r_tg]
            nll_gate = F.cross_entropy(root_logits.reshape(-1, root_logits.shape[-1]),
                                       gate_tg.reshape(-1), reduction='none', ignore_index=-100)
            gate_tg = gate_tg.reshape(r_tg.shape)
            letters = tables.letters.to(device)[r_tg]
            lenv = tables.lens.to(device)[r_tg]
            slot_logp, slot_valid, _ = model.root.teacher_forced(h, letters, lenv)
            nll_letters = -(slot_logp * slot_valid).sum(-1)
            nll_r = (nll_gate.reshape(r_tg.shape) + nll_letters).reshape(r_tg.shape)
        else:
            nll_r = F.cross_entropy(root_logits.reshape(-1, root_logits.shape[-1]),
                                    r_tg.reshape(-1), reduction='none').reshape(r_tg.shape)
        nll_w = F.cross_entropy(lw.reshape(-1, lw.shape[-1]), w_tg.reshape(-1),
                                reduction='none').reshape(r_tg.shape)
        nll_p = F.cross_entropy(lp.reshape(-1, lp.shape[-1]), p_tg.reshape(-1),
                                reduction='none').reshape(r_tg.shape)
        nll_s = F.cross_entropy(ls.reshape(-1, ls.shape[-1]), s_tg.reshape(-1),
                                reduction='none').reshape(r_tg.shape)
        for key, v in (('root', nll_r), ('wazn', nll_w), ('prefix', nll_p), ('suffix', nll_s)):
            total[key] += float(v[m].sum()) / math.log(2)
        nwords += int(m.sum())
    n = max(nwords, 1)
    out = {k: v / n for k, v in total.items()}
    out['total'] = sum(out.values())
    out['n_words'] = nwords
    return out


@torch.no_grad()
def eval_heads(model, data, split, device, tables, root_head, bs=24, max_sent=4000,
               k=5, max_rounds=200, max_rounds_free=300, free_max_pos=4000, max_pos=None,
               joint_max_pos=None):
    """Per-head top-1, and root top-1/top-5 split special / seen-real / held-out-real.

    root_head='comp' scores the root by EXACT best-first search over letter sequences merged
    with the closed-class gate candidates, so an unseen root can appear in the prediction.

    Two searches are run per batch over the same backbone pass:
      * JOINT (theta = 5th-best special log-prob): exact top-k of the joint root event over
        every valid position.  Candidates below the 5th special cannot enter a top-5, so the
        threshold is lossless and the search is cheap.  This column is directly comparable to
        the original softmax head's root_acc.
      * FREE (theta = -inf): exact top-k over content sequences regardless of the specials,
        giving content-only top-1/top-5 ('can the decoder spell the root') and the
        partial-credit material.  Run on ALL held-out-root targets plus the first
        `free_max_pos` other positions of the split (a seed-11 shuffle, so a random sample).
    """
    model.eval()
    cats = ('all', 'special', 'real_seen', 'real_hold')
    acc = {c: {'n': 0, 'r1': 0, 'r5': 0} for c in cats}
    cacc = {c: {'n': 0, 'r1': 0, 'r5': 0} for c in ('real_seen', 'real_hold')}
    head_acc = {'wazn': [0, 0], 'prefix': [0, 0], 'suffix': [0, 0]}
    records = []
    rng = random.Random(11)
    D = data[split]
    order = rng.sample(range(len(D['L'])), min(max_sent, len(D['L'])))
    npos = 0
    n_inexact = n_inexact_free = 0
    free_done = 0
    joint_done = 0
    for i in range(0, len(order), bs):
        ch = order[i:i + bs]
        P = [D['P'][j][:CTX_WORDS] for j in ch]
        R = [D['R'][j][:CTX_WORDS] for j in ch]
        W = [D['W'][j][:CTX_WORDS] for j in ch]
        Ss = [D['S'][j][:CTX_WORDS] for j in ch]
        if min(len(x) for x in R) < 2:
            continue
        r_tg = pad([x[1:] for x in R], 0).to(device)
        p_in = pad([x[:-1] for x in P], 0).to(device)
        r_in = pad([x[:-1] for x in R], 0).to(device)
        w_in = pad([x[:-1] for x in W], 0).to(device)
        s_in = pad([x[:-1] for x in Ss], 0).to(device)
        w_tg = pad([x[1:] for x in W], 0).to(device)
        p_tg = pad([x[1:] for x in P], 0).to(device)
        s_tg = pad([x[1:] for x in Ss], 0).to(device)
        m = torch.zeros_like(r_tg, dtype=torch.bool)
        for bi, x in enumerate(R):
            m[bi, :min(len(x) - 1, r_tg.shape[1])] = True
        h, root_logits, lw, lp, ls = model(p_in, r_in, w_in, s_in)
        head_acc['wazn'][0] += int(((lw.argmax(-1) == w_tg) & m).sum())
        head_acc['wazn'][1] += int(m.sum())
        head_acc['prefix'][0] += int(((lp.argmax(-1) == p_tg) & m).sum())
        head_acc['prefix'][1] += int(m.sum())
        head_acc['suffix'][0] += int(((ls.argmax(-1) == s_tg) & m).sum())
        head_acc['suffix'][1] += int(m.sum())

        idx = m.nonzero(as_tuple=False)                 # (n, 2)
        if max_pos is not None:
            if npos >= max_pos:
                break                                   # budget exhausted: stop forwarding
            if npos + idx.shape[0] > max_pos:
                idx = idx[:max_pos - npos]
        # Position budget for the JOINT search: EVERY held-out-root target is always kept
        # (that is the headline number); the rest of the split is sampled to this budget.
        rows = idx.tolist()
        tgt = [int(r_tg[bi, ti]) for bi, ti in rows]
        hold_rows = [k for k, r in enumerate(tgt) if r in tables.hold]
        if joint_max_pos is None:
            keep = list(range(len(rows)))
        else:
            others = [k for k in range(len(rows)) if tgt[k] not in tables.hold]
            budget = max(joint_max_pos - joint_done - len(hold_rows), 0)
            keep = sorted(hold_rows + others[:budget])
        joint_done += len(keep)
        idx = idx[keep]
        npos += idx.shape[0]
        if idx.shape[0] == 0:
            continue
        if root_head == 'type':
            pred1 = root_logits.argmax(-1)
            pred5 = root_logits.topk(k, -1).indices
            for bi, ti in idx.tolist():
                r = int(r_tg[bi, ti])
                p1 = int(pred1[bi, ti])
                cat = ('special' if r in tables.special_set else
                       ('real_hold' if r in tables.hold else 'real_seen'))
                hit1 = p1 == r
                hit5 = bool((pred5[bi, ti] == r).any())
                for c in ('all', cat):
                    acc[c]['n'] += 1; acc[c]['r1'] += int(hit1); acc[c]['r5'] += int(hit5)
                if cat in cacc:
                    cacc[cat]['n'] += 1; cacc[cat]['r1'] += int(hit1); cacc[cat]['r5'] += int(hit5)
                if cat == 'real_hold':
                    records.append({'split': split, 'true': tables.root_str(r),
                                    'pred': tables.root_str(p1),
                                    'top5': [tables.root_str(int(x)) for x in pred5[bi, ti].tolist()],
                                    'content_top1_ok': bool(hit1), 'content_top5_ok': bool(hit5),
                                    'content_top1': tables.root_str(p1),
                                    'content_top5': [tables.root_str(int(x))
                                                     for x in pred5[bi, ti].tolist()],
                                    'wazn_ok': int(lw[bi, ti].argmax()) == int(w_tg[bi, ti]),
                                    'root_ok': bool(hit1), 'top5_ok': bool(hit5),
                                    'pred_is_special': bool(p1 in tables.special_set),
                                    'bfs_exact': True,
                                    'ctx': [tables.root_str(int(x)) for x in r_in[bi].tolist()][:ti],
                                    'cat': cat})
            continue

        # ---- compositional arm -------------------------------------------------------
        gate_logp = F.log_softmax(root_logits, -1)
        vs = gate_logp[..., :tables.n_special]
        nshow = min(5, tables.n_special)
        top5v, top5i = vs.topk(nshow, -1)
        theta = top5v[..., -1]
        hs = h[idx[:, 0], idx[:, 1]]
        th = theta[idx[:, 0], idx[:, 1]]
        found_j, exact_j = best_first_content(model.root, hs, th, k=k, max_rounds=max_rounds)
        n_inexact += sum(1 for e in exact_j if not e)
        # free search: all held-out targets + a random sample of the rest
        hold_t = tables.hold
        targets = r_tg[idx[:, 0], idx[:, 1]].tolist()
        sel = []
        for j, r in enumerate(targets):
            if r in hold_t:
                sel.append(j)
            elif free_done < free_max_pos:
                sel.append(j)
                free_done += 1
        if sel:
            sel_t = torch.tensor(sel, dtype=torch.long, device=device)
            found_f, exact_f = best_first_content(
                model.root, hs[sel_t], torch.full((len(sel),), float('-inf'), device=device),
                k=k, max_rounds=max_rounds_free)
            n_inexact_free += sum(1 for e in exact_f if not e)
        else:
            found_f, exact_f = [], []
        fmap = {j: m_ for m_, j in enumerate(sel)}
        cbase = gate_logp[..., tables.content_class][idx[:, 0], idx[:, 1]]
        for j, (bi, ti) in enumerate(idx.tolist()):
            r = int(r_tg[bi, ti])
            cands = [(float(top5v[bi, ti, s5]), ('S', int(tables.special_ids[int(top5i[bi, ti, s5])])))
                     for s5 in range(nshow)]
            for lg, letters in found_j[j]:
                cands.append((float(cbase[j]) + lg, ('C', letters)))
            cands.sort(key=lambda x: -x[0])
            top = cands[:k]
            is_spec = r in tables.special_set
            true_letters = (tuple(tables.letters[r].tolist()[:int(tables.lens[r])])
                            if not is_spec else None)
            true = ('S', r) if is_spec else ('C', true_letters)
            hit1 = top[0][1] == true
            hit5 = any(cc[1] == true for cc in top)
            cat = 'special' if is_spec else ('real_hold' if r in hold_t else 'real_seen')
            for c in ('all', cat):
                acc[c]['n'] += 1; acc[c]['r1'] += int(hit1); acc[c]['r5'] += int(hit5)
            c_top1 = c_top5 = None
            if cat in cacc and j in fmap:
                cf = found_f[fmap[j]]
                c_top1 = bool(cf) and cf[0][1] == true_letters
                c_top5 = any(x[1] == true_letters for x in cf[:k])
                cacc[cat]['n'] += 1; cacc[cat]['r1'] += int(c_top1)
                cacc[cat]['r5'] += int(c_top5)
            if cat == 'real_hold':
                records.append({'split': split, 'true': tables.root_str(r),
                                'pred': (tables.root_str(top[0][1][1]) if top[0][1][0] == 'S'
                                         else tables.letters_str(top[0][1][1])),
                                'pred_is_special': top[0][1][0] == 'S',
                                'top5': [(tables.root_str(x[1]) if x[0] == 'S'
                                          else tables.letters_str(x[1])) for _, x in top],
                                'top5_scores': [round(s, 3) for s, _ in top],
                                'content_top1': (tables.letters_str(found_f[fmap[j]][0][1])
                                                 if (j in fmap and found_f[fmap[j]]) else None),
                                'content_top5': ([tables.letters_str(x[1])
                                                  for x in found_f[fmap[j]][:5]]
                                                 if j in fmap else None),
                                'content_top1_ok': c_top1, 'content_top5_ok': c_top5,
                                'wazn_ok': int(lw[bi, ti].argmax()) == int(w_tg[bi, ti]),
                                'root_ok': bool(hit1), 'top5_ok': bool(hit5),
                                'n_content_cands': len(found_j[j]),
                                'bfs_exact': bool(exact_j[j] and (j not in fmap or exact_f[fmap[j]])),
                                'gate_content_logp': round(float(gate_logp[bi, ti,
                                                                         tables.content_class]), 3),
                                'ctx': [tables.root_str(int(x)) for x in r_in[bi].tolist()][:ti],
                                'cat': cat})
        # Splits with NO held-out-root targets (verified by audit: train/val/test_gen have 0)
        # need no further sentences once the joint and free budgets are spent.
        if (split != 'test_deriv' and joint_max_pos is not None
                and joint_done >= joint_max_pos and free_done >= free_max_pos):
            break
    out = {'acc': {c: {'top1': v['r1'] / max(v['n'], 1), 'top5': v['r5'] / max(v['n'], 1),
                       'n': v['n']} for c, v in acc.items()},
           'content_only_acc': {c: {'top1': v['r1'] / max(v['n'], 1),
                                    'top5': v['r5'] / max(v['n'], 1), 'n': v['n']}
                                for c, v in cacc.items()},
           'per_head_top1': {k_: v[0] / max(v[1], 1) for k_, v in head_acc.items()},
           'n_pos': npos, 'n_bfs_inexact_joint': n_inexact,
           'n_bfs_inexact_free': n_inexact_free}
    return out, records


def train(args):
    device = torch.device(args.device)
    out = Path(args.out)
    meta = json.load(open(out / 'meta.json'))
    data = torch.load(out / 'streams.pt', weights_only=False)
    tables = Tables(meta)
    rng = random.Random(0)
    torch.manual_seed(0)
    sizes = (meta['n_roots'], meta['n_awzan'], meta['n_prefixes'], meta['n_suffixes'])
    if args.root_head == 'comp':
        model = LMComp(sizes, tables.n_gate, meta['n_sym']).to(device)
    else:
        model = LMType(sizes).to(device)
    nparam = sum(p.numel() for p in model.parameters())
    print(f'[*] ARM={args.tag} root_head={args.root_head} params={nparam/1e6:.2f}M '
          f'train_words={meta["train_words"]} n_roots={meta["n_roots"]} '
          f'n_gate={tables.n_gate} n_alphabet={tables.n_alphabet}', flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.steps,
        pct_start=max(0.02, min(0.3, 200.0 / max(args.steps, 1))))
    t0 = time.time()
    step = 0
    losses = []
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    while step < args.steps:
        it = batches_morph(data, 'train', args.batch, rng)
        for b in it:
            p_in = pad([x[:-1] for x in b['p']], 0).to(device)
            r_in = pad([x[:-1] for x in b['r']], 0).to(device)
            w_in = pad([x[:-1] for x in b['w']], 0).to(device)
            s_in = pad([x[:-1] for x in b['s']], 0).to(device)
            r_tg = pad([x[1:] for x in b['r']], 0).to(device)
            w_tg = pad([x[1:] for x in b['w']], 0).to(device)
            p_tg = pad([x[1:] for x in b['p']], 0).to(device)
            s_tg = pad([x[1:] for x in b['s']], 0).to(device)
            h, root_logits, lw, lp, ls = model(p_in, r_in, w_in, s_in)
            ig = 0
            loss_nonroot = (0.5 * F.cross_entropy(lw.reshape(-1, lw.shape[-1]), w_tg.reshape(-1), ignore_index=ig)
                            + 0.25 * F.cross_entropy(lp.reshape(-1, lp.shape[-1]), p_tg.reshape(-1), ignore_index=ig)
                            + 0.25 * F.cross_entropy(ls.reshape(-1, ls.shape[-1]), s_tg.reshape(-1), ignore_index=ig))
            if args.root_head == 'comp':
                gate_tg = tables.gate_tab.to(device)[r_tg]
                loss_gate = F.cross_entropy(root_logits.reshape(-1, root_logits.shape[-1]),
                                            gate_tg.reshape(-1), ignore_index=-100)
                letters = tables.letters.to(device)[r_tg]
                lenv = tables.lens.to(device)[r_tg]
                slot_logp, slot_valid, _ = model.root.teacher_forced(h, letters, lenv)
                nvalid = slot_valid.sum().clamp(min=1)
                loss_letters = -(slot_logp * slot_valid).sum() / nvalid
                loss = loss_gate + loss_letters + loss_nonroot
            else:
                loss_r = F.cross_entropy(root_logits.reshape(-1, root_logits.shape[-1]),
                                         r_tg.reshape(-1), ignore_index=ig)
                loss = loss_r + loss_nonroot
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % 500 == 0:
                losses.append((step, float(loss.detach())))
                print(f'  step {step}/{args.steps} loss {float(loss.detach()):.4f} '
                      f'{step/(time.time()-t0):.2f} it/s', flush=True)
            if step % 10000 == 0:
                vb = eval_bits(model, data, 'val', device, tables, args.root_head)
                print(f'    [val @{step}] bits/word {vb["total"]:.3f}', flush=True)
                model.train()
            if step >= args.steps:
                break
    wall = time.time() - t0
    peak = (torch.cuda.max_memory_allocated() / 2**20) if device.type == 'cuda' else float('nan')
    res = {'tag': args.tag, 'root_head': args.root_head, 'params': nparam, 'steps': args.steps,
           'batch': args.batch, 'lr': args.lr, 'wall_clock_s': wall,
           'peak_vram_mib': peak, 'it_per_s': args.steps / wall, 'losses': losses,
           'n_roots': meta['n_roots'], 'n_awzan': meta['n_awzan'],
           'n_gate': tables.n_gate, 'n_alphabet': tables.n_alphabet,
           'blueprint_md5': meta['blueprint_md5'],
           'train_words': meta['train_words']}
    ck = out / f'{args.tag}.pt'
    torch.save(model.state_dict(), ck)
    print(f'[*] saved {ck}  wall={wall:.1f}s peak_vram={peak:.0f}MiB', flush=True)
    json.dump(res, open(out / f'{args.tag}_train.json', 'w'), indent=2)
    print(f'[*] wrote {out}/{args.tag}_train.json', flush=True)


def evaluate(args):
    device = torch.device(args.device)
    out = Path(args.out)
    meta = json.load(open(out / 'meta.json'))
    data = torch.load(out / 'streams.pt', weights_only=False)
    tables = Tables(meta)
    sizes = (meta['n_roots'], meta['n_awzan'], meta['n_prefixes'], meta['n_suffixes'])
    if args.root_head == 'comp':
        model = LMComp(sizes, tables.n_gate, meta['n_sym']).to(device)
    else:
        model = LMType(sizes).to(device)
    sd = torch.load(args.ckpt, map_location=device, weights_only=True)
    model.load_state_dict(sd)
    model.eval()
    print(f'[*] loaded {args.ckpt} into {args.root_head}', flush=True)

    splits = ['train', 'val', 'test_gen', 'test_deriv']
    res = {'ckpt': args.ckpt, 'root_head': args.root_head, 'n_roots': meta['n_roots'],
           'n_awzan': meta['n_awzan'], 'n_gate': tables.n_gate,
           'n_alphabet': tables.n_alphabet,
           'hold_roots': sorted(tables.hold),
           'hold_roots_str': [tables.root_str(r) for r in sorted(tables.hold)],
           'hold_roots_inexpressible': sorted(tables.hold_inexpr),
           'blueprint_md5': meta['blueprint_md5'],
           'root_len_dist_train_tokens': meta['root_len_dist_train_tokens']}
    ub = unigram_bits(data, ['val', 'test_gen', 'test_deriv'])
    usb = unigram_symbol_bits(data, ['val', 'test_gen', 'test_deriv'], tables)
    res['unigram_bits_per_word_rootTYPE'] = ub
    res['unigram_bits_per_word_compositional'] = usb
    res['unigram_root_acc'] = unigram_root_acc(data, ['val', 'test_gen', 'test_deriv'], tables)
    print(f'[*] unigram (type) bits/word: {ub}', flush=True)
    print(f'[*] unigram (compositional slot) bits/word: {usb}', flush=True)

    res['bits_per_word'] = {}
    for sp in splits:
        ms = 8000 if sp != 'train' else (args.max_sent_eval if args.max_sent_eval else 800)
        b = eval_bits(model, data, sp, device, tables, args.root_head, max_sent=ms)
        res['bits_per_word'][sp] = b
        print(f'[*] bits/word {sp}: total={b["total"]:.4f} '
              f'(root={b["root"]:.4f} wazn={b["wazn"]:.4f} '
              f'prefix={b["prefix"]:.4f} suffix={b["suffix"]:.4f}) n={b["n_words"]}', flush=True)
    for sp in splits:
        ms = 8000 if sp != 'train' else 800
        cap = {'train': args.max_pos_train, 'val': args.max_pos_val,
               'test_gen': args.max_pos_test, 'test_deriv': args.max_pos_test}[sp]
        hres, records = eval_heads(model, data, sp, device, tables, args.root_head,
                                   max_sent=ms, k=5, max_rounds=args.max_rounds,
                                   max_rounds_free=args.max_rounds_free,
                                   free_max_pos=args.free_max_pos, max_pos=cap,
                                   joint_max_pos=args.joint_max_pos)
        res[f'heads_{sp}'] = hres
        if sp == 'test_deriv' and records:
            res['holdout_records'] = records
        print(f'[*] {sp}: per-head top1 {hres["per_head_top1"]}\n'
              f'      root joint {hres["acc"]}\n'
              f'      root content-only {hres["content_only_acc"]} '
              f'inexact_joint={hres["n_bfs_inexact_joint"]} '
              f'inexact_free={hres["n_bfs_inexact_free"]} npos={hres["n_pos"]}', flush=True)

    # ---- partial credit on held-out-root positions, computed from the records ------------
    recs = res.get('holdout_records', [])
    pc = partial_credit(recs, tables)
    res['partial_credit_holdout'] = pc
    print(f'[*] partial credit: {json.dumps({k: v for k, v in pc.items() if k != "examples"}, indent=2)}',
          flush=True)
    fn = out / f'{args.tag}_eval.json'
    json.dump(res, open(fn, 'w'), indent=2, ensure_ascii=False)
    print(f'[*] wrote {fn}', flush=True)


def _edit_dist(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def partial_credit(recs, tables):
    """Partial-credit breakdown over held-out-root target positions.

    A categorical softmax over root types cannot give ANY partial credit: a non-identical type
    is simply wrong.  A decoder produces a string, so we measure two families:

      (1) JOINT   -- what the model actually emits first (specials compete with content);
      (2) CONTENT -- what the decoder spells when only root strings are considered, i.e. the
                     best content hypothesis `content_top1` and its top-5.  This is where
                     partial credit is even definable: right radicals in the wrong order
                     (anagram), per-slot radical accuracy, edit distance, radical recall.
    """
    from collections import Counter as Ctr
    n = len(recs)
    if n == 0:
        return {'n': 0}
    out = {'n': n, 'exact_top1': 0, 'exact_top5': 0, 'anagram_top1': 0, 'same_len_top1': 0,
           'slot_acc_top1_num': 0, 'slot_acc_top1_den': 0, 'mean_overlap_top1': 0.0,
           'mean_edit_top1': 0.0, 'root_ok_wazn_wrong': 0, 'root_ok_wazn_ok': 0,
           'wazn_ok_root_wrong': 0, 'pred_is_special': 0, 'mean_edit_top1_same_len': 0.0,
           'n_same_len': 0, 'all_exact_top1_anagram': 0,
           'content_top1': 0, 'content_top5': 0, 'n_content_scored': 0,
           'n_bfs_exact': 0, 'n_bfs_inexact': 0, 'mean_pred_len': 0.0,
           # ---- content-side partial credit -------------------------------------------
           'c_n': 0, 'c_exact_top1': 0, 'c_exact_top5': 0, 'c_anagram_top1': 0,
           'c_any_radical_top1': 0, 'c_same_len': 0, 'c_slot_num': 0, 'c_slot_den': 0,
           'c_mean_overlap': 0.0, 'c_mean_edit': 0.0, 'c_special_top1': 0,
           'c_radical_recall_top5': 0.0}
    exs = []
    for r in recs:
        t = r['true']
        p = r['pred']
        out['exact_top1'] += int(r['root_ok'])
        out['exact_top5'] += int(r['top5_ok'])
        out['pred_is_special'] += int(r.get('pred_is_special', False))
        out['mean_pred_len'] += len(p)
        out['n_bfs_exact'] += int(r.get('bfs_exact', True))
        out['n_bfs_inexact'] += int(not r.get('bfs_exact', True))
        same_len = len(t) == len(p)
        if same_len:
            out['same_len_top1'] += 1
            out['n_same_len'] += 1
            out['slot_acc_top1_num'] += sum(1 for a, b in zip(t, p) if a == b)
            out['slot_acc_top1_den'] += len(t)
            out['mean_edit_top1_same_len'] += _edit_dist(t, p)
        anag = Ctr(t) == Ctr(p)
        out['anagram_top1'] += int(anag)
        out['all_exact_top1_anagram'] += int(anag and not r['root_ok'])
        out['mean_overlap_top1'] += (sum((Ctr(t) & Ctr(p)).values()) / max(len(t), 1))
        out['mean_edit_top1'] += _edit_dist(t, p)
        if r['root_ok']:
            if r['wazn_ok']:
                out['root_ok_wazn_ok'] += 1
            else:
                out['root_ok_wazn_wrong'] += 1
        elif r['wazn_ok']:
            out['wazn_ok_root_wrong'] += 1
        # ---- content-side -------------------------------------------------------------
        cp = r.get('content_top1')
        c5 = r.get('content_top5') or []
        if cp is not None:
            out['c_n'] += 1
            out['c_exact_top1'] += int(cp == t)
            out['c_exact_top5'] += int(any(x == t for x in c5))
            out['c_anagram_top1'] += int(Ctr(t) == Ctr(cp))
            out['c_special_top1'] += int(cp in tables.special_set or cp.startswith('<'))
            if len(cp) == len(t):
                out['c_same_len'] += 1
                out['c_slot_num'] += sum(1 for a, b in zip(t, cp) if a == b)
                out['c_slot_den'] += len(t)
            out['c_any_radical_top1'] += int(any(ch in cp for ch in t))
            out['c_mean_overlap'] += (sum((Ctr(t) & Ctr(cp)).values()) / max(len(t), 1))
            out['c_mean_edit'] += _edit_dist(t, cp)
            union = set(''.join(c5)) if c5 else set(cp)
            out['c_radical_recall_top5'] += (sum(1 for ch in t if ch in union) / max(len(t), 1))
        if r.get('content_top1_ok') is not None:
            out['n_content_scored'] += 1
            out['content_top1'] += int(r['content_top1_ok'])
            out['content_top5'] += int(r['content_top5_ok'])
    for k in ('exact_top1', 'exact_top5', 'anagram_top1', 'same_len_top1', 'pred_is_special',
              'root_ok_wazn_wrong', 'root_ok_wazn_ok', 'wazn_ok_root_wrong',
              'all_exact_top1_anagram', 'c_exact_top1', 'c_exact_top5', 'c_anagram_top1',
              'c_any_radical_top1', 'c_special_top1'):
        out[k + '_rate'] = out[k] / n
    out['slot_acc_top1'] = out['slot_acc_top1_num'] / max(out['slot_acc_top1_den'], 1)
    out['mean_overlap_top1'] /= n
    out['mean_edit_top1'] /= n
    out['mean_pred_len'] /= n
    out['mean_edit_top1_same_len'] /= max(out['n_same_len'], 1)
    out['content_top1_rate'] = out['content_top1'] / max(out['n_content_scored'], 1)
    out['content_top5_rate'] = out['content_top5'] / max(out['n_content_scored'], 1)
    cn = max(out['c_n'], 1)
    out['c_exact_top1_rate'] = out['c_exact_top1'] / cn
    out['c_exact_top5_rate'] = out['c_exact_top5'] / cn
    out['c_anagram_top1_rate'] = out['c_anagram_top1'] / cn
    out['c_any_radical_top1_rate'] = out['c_any_radical_top1'] / cn
    out['c_special_top1_rate'] = out['c_special_top1'] / cn
    out['c_same_len_rate'] = out['c_same_len'] / cn
    out['c_slot_acc'] = out['c_slot_num'] / max(out['c_slot_den'], 1)
    out['c_mean_overlap'] /= cn
    out['c_mean_edit'] /= cn
    out['c_radical_recall_top5'] /= cn
    # concrete examples: exact hits first, then misses
    hits = [r for r in recs if r['root_ok']]
    miss = [r for r in recs if not r['root_ok']]
    exs = [{'true': r['true'], 'pred': r['pred'], 'top5': r['top5'],
            'content_top1': r.get('content_top1'), 'content_top5': r.get('content_top5'),
            'anagram': Ctr(r['true']) == Ctr(r['pred']),
            'c_anagram': (r.get('content_top1') is not None
                          and Ctr(r['true']) == Ctr(r['content_top1'])),
            'ctx_tail': r['ctx'][-6:], 'root_ok': r['root_ok']}
           for r in (hits[:4] + miss[:12])]
    out['examples'] = exs
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['prepare', 'train', 'eval'])
    ap.add_argument('--root-head', choices=['comp', 'type'], default='comp')
    ap.add_argument('--tag', default='comp6k')
    ap.add_argument('--ckpt', default='')
    ap.add_argument('--out', default=str(OUT))
    ap.add_argument('--max-words', type=int, default=25_000_000)
    ap.add_argument('--steps', type=int, default=6000)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--lr', type=float, default=6e-4)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--max-rounds', type=int, default=200)
    ap.add_argument('--max-rounds-free', type=int, default=300)
    ap.add_argument('--free-max-pos', type=int, default=4000)
    ap.add_argument('--joint-max-pos', type=int, default=8000)
    ap.add_argument('--max-pos-train', type=int, default=8000)
    ap.add_argument('--max-pos-val', type=int, default=8000)
    ap.add_argument('--max-pos-test', type=int, default=40000)
    ap.add_argument('--max-sent-eval', type=int, default=0)
    args = ap.parse_args()
    if args.mode == 'prepare':
        prepare(args)
    elif args.mode == 'train':
        train(args)
    else:
        evaluate(args)


if __name__ == '__main__':
    main()
