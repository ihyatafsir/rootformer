#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
governed_lm.py -- the from-scratch NRMT-P model run UNDER the classical algorithms' oversight.

Three places the grammarians are put in the loop, versus the ungoverned control:

  1. ANALYSIS  -- words are decomposed by the derivational canonicalisation ladder (iʿlāl,
     ibdāl/hamza, idghām, mazīd, clitic stripping) of the Andalusian taṣrīf tradition, so a root
     the lookup misses is recovered instead of thrown away as <UNK>.
  2. TRAINING  -- Sībawayh's ʿāmil (operator state: JARR / INNA / KANA / NASB / JAZM / FUTURE) is
     fed in as an input feature, so the model conditions on government rather than only on
     surface context.
  3. DECODING  -- Al-Khalīl's phonotactic exclusion and Sībawayh's operator governance hard-mask
     root candidates. Reported separately as accuracy and as VALIDITY (violations per 1000
     positions), because a mask can only remove candidates, never add information.

Everything else -- corpus, split, architecture, steps, batch, lr -- is identical to the ungoverned
control so the comparison is matched.

Usage:
  python governed_lm.py prepare
  python governed_lm.py train --steps 80000 --batch 32 --lr 9e-4
"""
import argparse
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, '/workspace/hf_v19_2_release')
sys.path.insert(0, '/workspace/hf_v19_2_release/models')

import scratch_lm as S
from classical_analyzer import ClassicalAnalyzer

OUT = Path('/workspace/sf_data_gov')


def build_ops(vocab):
    from nrmt_arch import build_operator_table
    return build_operator_table(vocab)


# ------------------------------------------------------------------ governed analysis
class GovernedAnalyzer:
    """(P,R,W,S) plus the ʿāmil state, under the classical algorithms."""

    def __init__(self, vocab):
        self.v = vocab
        self.ca = ClassicalAnalyzer(vocab)
        self.op_table = build_ops(vocab)
        self.n_ops = int(self.op_table.max().item()) + 1

    def word(self, w):
        p, r, wz, s = self.v.encode_word(w)
        rid, how = self.ca.analyze(w)
        if rid is not None and rid != self.v.UNK_ROOT:
            r = rid
        op = int(self.op_table[r].item()) if hasattr(self.op_table[r], 'item') \
            else int(self.op_table[r])
        return p, int(r), wz, s, op, how

    def sentence(self, s):
        out = []
        for w in s.split():
            out.append(self.word(w))
        return out


def prepare(args):
    OUT.mkdir(parents=True, exist_ok=True)
    vocab = S.get_vocab()
    ga = GovernedAnalyzer(vocab)
    print(f'[*] governed analyzer ready: {ga.n_ops} operator states (Sibawayh)', flush=True)

    by_file, seen = {}, set()
    for key, s in S.sentences_by_file():
        k = S.norm(s)
        if k in seen:
            continue
        seen.add(k)
        by_file.setdefault(key, []).append(s)
    files = sorted(by_file, key=lambda k: -len(by_file[k]))
    print(f'[*] {len(files)} files, {sum(len(v) for v in by_file.values())} unique sentences',
          flush=True)

    rng = random.Random(0)
    rng.shuffle(files)
    n_test = max(4, len(files) // 10)
    test_files, rest = files[:n_test], files[n_test:]
    n_val = max(2, len(rest) // 15)
    val_files, train_files = rest[:n_val], rest[n_val:]

    # holdout roots by sentence containment (identical policy to the control)
    probe = []
    for f in train_files[:600]:
        probe += by_file[f][:60]
    rng.shuffle(probe)
    probe = probe[:120000]
    containment = Counter()
    for s in probe:
        for t in ga.sentence(s):
            containment[t[1]] += 1
    nprobe = max(len(probe), 1)
    cand = []
    for r, c in containment.items():
        name = vocab.id2root.get(r, '')
        if name.startswith('<') or len(S.DIAC.sub('', name)) != 3:
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
    print(f'[*] holdout: {len(hold_roots)} content roots (~10% of sentences)', flush=True)

    # ---- SINGLE PASS: analyse each sentence once, route it, and build its streams ----
    # The previous version called the analyzer twice per sentence (once to test the holdout,
    # once to encode), doubling a 45-minute job for nothing.
    def blank():
        return {k: [] for k in ('P', 'R', 'W', 'S', 'L', 'OP')}

    SPL = {'train': blank(), 'val': blank(), 'test_gen': blank(), 'test_deriv': blank()}
    kept = dropped = 0
    t0 = time.time()
    for fi, f in enumerate(train_files):
        if fi % 20 == 0:
            print(f'    pass1 {fi+1}/{len(train_files)} kept={kept} dropped={dropped} '
                  f'({time.time()-t0:.0f}s)', flush=True)
        for sent in by_file[f]:
            e = ga.sentence(sent)
            if len(e) < 2:
                continue
            if any(t[1] in hold_roots for t in e):
                dropped += 1
                continue
            kept += 1
            d = SPL['train']
            d['P'].append([t[0] for t in e]); d['R'].append([t[1] for t in e])
            d['W'].append([t[2] for t in e]); d['S'].append([t[3] for t in e])
            d['OP'].append([t[4] for t in e]); d['L'].append(len(e))
    for name, files_ in (('val', val_files), ('test_gen', None), ('test_deriv', None)):
        if files_ is None:
            continue
        for f in files_:
            for sent in by_file[f]:
                e = ga.sentence(sent)
                if len(e) < 2 or any(t[1] in hold_roots for t in e):
                    continue
                d = SPL[name]
                d['P'].append([t[0] for t in e]); d['R'].append([t[1] for t in e])
                d['W'].append([t[2] for t in e]); d['S'].append([t[3] for t in e])
                d['OP'].append([t[4] for t in e]); d['L'].append(len(e))
    for f in test_files:
        for sent in by_file[f]:
            e = ga.sentence(sent)
            if len(e) < 2:
                continue
            name = 'test_deriv' if any(t[1] in hold_roots for t in e) else 'test_gen'
            d = SPL[name]
            d['P'].append([t[0] for t in e]); d['R'].append([t[1] for t in e])
            d['W'].append([t[2] for t in e]); d['S'].append([t[3] for t in e])
            d['OP'].append([t[4] for t in e]); d['L'].append(len(e))
    print(f'[*] pass1 done: kept {kept} dropped {dropped} ({time.time()-t0:.0f}s)', flush=True)

    # ---- cap and save, per split, atomically -------------------------------------------
    d = SPL['train']
    words, cut = 0, len(d['L'])
    for i in range(len(d['L'])):
        words += d['L'][i]
        if words >= args.max_words:
            cut = i + 1
            break
    for k in ('P', 'R', 'W', 'S', 'OP', 'L'):
        d[k] = d[k][:cut]
    for name, lim in (('val', 2000), ('test_gen', 4000), ('test_deriv', 4000)):
        for k in ('P', 'R', 'W', 'S', 'OP', 'L'):
            SPL[name][k] = SPL[name][k][:lim]
    print(f'[*] train sentences {kept} (dropped {dropped})  words {words}', flush=True)

    data = SPL
    specials = {i for i, r in enumerate(vocab.roots_list) if str(r).startswith('<')}
    allr = [x for seq in data['train']['R'][:200000] for x in seq]
    tot = max(len(allr), 1)
    real = sum(1 for x in allr if x not in specials)
    unk = sum(1 for x in allr if x == vocab.UNK_ROOT)
    print(f'\n[*] governed analysis: real roots {100*real/tot:.1f}%  '
          f'specials {100*(tot-real)/tot:.1f}%  <UNK> {100*unk/tot:.2f}%', flush=True)

    for name, dd in data.items():
        # atomic per-split save with retries: a transient WekaFS write error must not cost
        # the whole analysis
        tmp = OUT / f'{name}.pt.tmp'
        final = OUT / f'{name}.pt'
        for attempt in range(4):
            try:
                torch.save(dd, tmp)
                tmp.replace(final)
                print(f'    saved {final.name} ({final.stat().st_size/1e6:.0f} MB)', flush=True)
                break
            except Exception as ex:
                print(f'    [retry {attempt+1}] {name}: {type(ex).__name__}: {str(ex)[:90]}',
                      flush=True)
                time.sleep(5)
        else:
            raise RuntimeError(f'could not save {name} after 4 attempts')

    json.dump({'n_roots': vocab.num_roots, 'n_awzan': vocab.num_awzan,
               'n_prefixes': vocab.num_prefixes, 'n_suffixes': vocab.num_suffixes,
               'hold_roots': sorted(hold_roots), 'n_ops': ga.n_ops,
               'special_root_ids': sorted(specials),
               'train_words': words,
               'counts': {k: len(v['L']) for k, v in data.items()}},
              open(OUT / 'meta.json', 'w'), indent=2, ensure_ascii=False)
    print(f'[*] saved {OUT}/ (per-split .pt + meta.json)')

def batches_gov(d, split, bs, rng, max_sent=64):
    """Like scratch_lm.batches_morph but also yields the ʿāmil (operator) stream."""
    D = d[split]
    L = D['L']
    n = len(L)
    order = rng.sample(range(n), min(bs * max_sent, n))
    cur = {k: [] for k in ('p', 'r', 'w', 's', 'op')}
    for i in order:
        if L[i] < 2:
            continue
        cur['p'].append(D['P'][i][:S.CTX_WORDS])
        cur['r'].append(D['R'][i][:S.CTX_WORDS])
        cur['w'].append(D['W'][i][:S.CTX_WORDS])
        cur['s'].append(D['S'][i][:S.CTX_WORDS])
        cur['op'].append(D['OP'][i][:S.CTX_WORDS])
        if len(cur['r']) == bs:
            yield cur
            cur = {k: [] for k in ('p', 'r', 'w', 's', 'op')}


@torch.no_grad()
def eval_bits_gov(model, data, splits, device, bs=16, max_sent=200):
    model.eval()
    res = {}
    for split in splits:
        total_bits, total_words = 0.0, 0
        rng = random.Random(7)
        for b in batches_gov(data, split, bs, rng, max_sent=max_sent):
            r_tg = S.pad([x[1:] for x in b['r']], 0).to(device)
            p_in = S.pad([x[:-1] for x in b['p']], 0).to(device)
            r_in = S.pad([x[:-1] for x in b['r']], 0).to(device)
            w_in = S.pad([x[:-1] for x in b['w']], 0).to(device)
            s_in = S.pad([x[:-1] for x in b['s']], 0).to(device)
            o_in = S.pad([x[:-1] for x in b['op']], 0).to(device)
            w_tg = S.pad([x[1:] for x in b['w']], 0).to(device)
            p_tg = S.pad([x[1:] for x in b['p']], 0).to(device)
            s_tg = S.pad([x[1:] for x in b['s']], 0).to(device)
            lens = [len(x) - 1 for x in b['r']]
            m = torch.zeros_like(r_tg, dtype=torch.bool)
            for bi, Ln in enumerate(lens):
                m[bi, :min(Ln, r_tg.shape[1])] = True
            lr, lw, lp, ls = model(p_in, r_in, w_in, s_in, o_in)
            nll = (F.cross_entropy(lr.reshape(-1, lr.shape[-1]), r_tg.reshape(-1),
                                   reduction='none')
                   + F.cross_entropy(lw.reshape(-1, lw.shape[-1]), w_tg.reshape(-1),
                                     reduction='none')
                   + F.cross_entropy(lp.reshape(-1, lp.shape[-1]), p_tg.reshape(-1),
                                     reduction='none')
                   + F.cross_entropy(ls.reshape(-1, ls.shape[-1]), s_tg.reshape(-1),
                                     reduction='none')).reshape(r_tg.shape)
            total_bits += float(nll[m].sum()) / math.log(2)
            total_words += int(m.sum())
        res[split] = total_bits / max(total_words, 1)
    model.train()
    return res


# ------------------------------------------------------------------ model with the ʿāmil channel
class GovernedLM(S.MorphemicLM):
    def __init__(self, sizes, n_ops, d=512, nl=8, h=8, ffn=2048, maxlen=768, d_op=64):
        super().__init__(sizes, d, nl, h, ffn, maxlen)
        self.e_op = nn.Embedding(n_ops, d_op)
        self.op_proj = nn.Linear(d_op, d)   # 64 -> d, not d -> d

    def forward(self, p, r, w, s, op):
        T = r.shape[1]
        x = torch.cat([self.e_r(r), self.e_w(w), self.e_p(p), self.e_s(s)], -1)
        x = x + self.pos(torch.arange(T, device=r.device))[None]
        x = x + self.op_proj(self.e_op(op).to(x.dtype))
        m = torch.triu(torch.full((T, T), float('-inf'), device=r.device), 1)
        for b in self.blocks:
            x = b(x, m)
        h = self.ln(x)
        return self.h_r(h), self.h_w(h), self.h_p(h), self.h_s(h)


# ------------------------------------------------------------------ masked evaluation
class Masks:
    """Al-Khalil phonotactics (hard) and Sibawayh operator governance (hard)."""

    def __init__(self, vocab):
        self.vocab = vocab
        try:
            from classical_governance_v2 import AlKhalilV2
            self.khalil = sorted(AlKhalilV2(vocab).mask())
        except Exception as e:
            print(f'  [warn] AlKhalilV2 unavailable: {e}')
            self.khalil = []
        try:
            from sibawayh_governance_engine import SibawayhNRMPGovernance
            self.sib = SibawayhNRMPGovernance(vocab)
        except Exception as e:
            print(f'  [warn] Sibawayh governance unavailable: {e}')
            self.sib = None
        print(f'  [masks] Al-Khalil forbids {len(self.khalil)} roots; '
              f'Sibawayh {"on" if self.sib else "off"}')

    def apply(self, logits, prev_tuple, roots_so_far, t, use_khalil=True, use_sib=True):
        lg = logits.clone()
        if use_khalil and self.khalil:
            lg[torch.tensor(self.khalil, device=lg.device)] = -float('inf')
        if use_sib and self.sib is not None:
            op = self.sib.get_operator_state(prev_tuple)
            lg = self.sib.apply_root_exclusion_mask(lg, op, roots_so_far, t, 0)
        return lg


@torch.no_grad()
def governed_eval(model, data, meta, vocab, splits, device, bs=16, max_sent=1500):
    """Accuracy masked vs unmasked, plus validity: how many predicted roots are IMPOSSIBLE."""
    model.eval()
    masks = Masks(vocab)
    kh = set(masks.khalil)
    out = {}
    for sp in splits:
        acc = {k: {'n': 0, 'h1': 0, 'h5': 0} for k in
               ('unmasked_real', 'masked_real', 'unmasked_all', 'masked_all')}
        viol = {'unmasked': 0, 'masked': 0}
        rng = random.Random(11)
        D = data[sp]
        order = rng.sample(range(len(D['L'])), min(max_sent, len(D['L'])))
        for i in range(0, len(order), bs):
            chunk = order[i:i + bs]
            P = [D['P'][j][:S.CTX_WORDS] for j in chunk]
            R = [D['R'][j][:S.CTX_WORDS] for j in chunk]
            W = [D['W'][j][:S.CTX_WORDS] for j in chunk]
            Sm = [D['S'][j][:S.CTX_WORDS] for j in chunk]
            OP = [D['OP'][j][:S.CTX_WORDS] for j in chunk]
            if min(len(x) for x in R) < 2:
                continue
            r_tg = S.pad([x[1:] for x in R], 0).to(device)
            p_in = S.pad([x[:-1] for x in P], 0).to(device)
            r_in = S.pad([x[:-1] for x in R], 0).to(device)
            w_in = S.pad([x[:-1] for x in W], 0).to(device)
            s_in = S.pad([x[:-1] for x in Sm], 0).to(device)
            o_in = S.pad([x[:-1] for x in OP], 0).to(device)
            m = torch.zeros_like(r_tg, dtype=torch.bool)
            for bi, x in enumerate(R):
                m[bi, :min(len(x) - 1, r_tg.shape[1])] = True
            lr, _, _, _ = model(p_in, r_in, w_in, s_in, o_in)
            spec = torch.zeros(meta['n_roots'], dtype=torch.bool, device=device)
            for j in meta['special_root_ids']:
                spec[j] = True
            real = (~spec[r_tg]) & m
            for tag, mm in (('unmasked', m), ('masked', m)):
                pred5 = lr.topk(5, -1).indices
                pred1 = lr.argmax(-1)
                if tag == 'unmasked':
                    h1 = ((pred1 == r_tg) & mm)
                    h5 = ((pred5 == r_tg.unsqueeze(-1)).any(-1) & mm)
                    viol['unmasked'] += int(((pred1.unsqueeze(-1) == torch.tensor(
                        sorted(kh), device=device)).any(-1) & mm).sum())
                else:
                    lg = lr.clone()
                    if kh:
                        lg[..., sorted(kh)] = -float('inf')
                    pred1m = lg.argmax(-1)
                    pred5m = lg.topk(5, -1).indices
                    h1 = ((pred1m == r_tg) & mm)
                    h5 = ((pred5m == r_tg.unsqueeze(-1)).any(-1) & mm)
                    viol['masked'] += int(((pred1m.unsqueeze(-1) == torch.tensor(
                        sorted(kh), device=device)).any(-1) & mm).sum())
                for key, sub in ((f'{tag}_all', mm), (f'{tag}_real', real)):
                    acc[key]['n'] += int(sub.sum())
                    acc[key]['h1'] += int((h1 & sub).sum())
                    acc[key]['h5'] += int((h5 & sub).sum())
        out[sp] = {k: {'top1': v['h1'] / max(v['n'], 1), 'top5': v['h5'] / max(v['n'], 1),
                       'n': v['n']} for k, v in acc.items()}
        out[sp]['violations_unmasked'] = viol['unmasked']
        out[sp]['violations_masked'] = viol['masked']
    return out


def load_data(out=OUT):
    merged = {}
    for name in ('train', 'val', 'test_gen', 'test_deriv'):
        f = out / f'{name}.pt'
        if not f.exists():
            raise FileNotFoundError(f)
        merged[name] = torch.load(f, weights_only=False)
    return merged


def train(args):
    device = torch.device('cuda')
    meta = json.load(open(OUT / 'meta.json'))
    data = load_data()
    import nrmp_vocab as nv
    V = next(v for k, v in vars(nv).items() if isinstance(v, type) and 'MorphemicVocab' in k)
    vocab = V('/workspace/hf_v19_2_release/data/rootformer_v12_arabic_blueprint.json')
    model = GovernedLM((meta['n_roots'], meta['n_awzan'], meta['n_prefixes'],
                        meta['n_suffixes']), meta['n_ops']).to(device)
    nparam = sum(p.numel() for p in model.parameters())
    print(f'[*] GOVERNED params={nparam/1e6:.2f}M steps={args.steps} '
          f'train_words={meta["train_words"]}', flush=True)

    rng = random.Random(0)
    torch.manual_seed(0)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.steps,
        pct_start=max(0.02, min(0.3, 200.0 / max(args.steps, 1))))
    t0 = time.time()
    step = 0
    while step < args.steps:
        for b in batches_gov(data, 'train', args.batch, rng):
            p_in = S.pad([x[:-1] for x in b['p']], 0).to(device)
            r_in = S.pad([x[:-1] for x in b['r']], 0).to(device)
            w_in = S.pad([x[:-1] for x in b['w']], 0).to(device)
            s_in = S.pad([x[:-1] for x in b['s']], 0).to(device)
            o_in = S.pad([x[:-1] for x in b['op']], 0).to(device)
            r_tg = S.pad([x[1:] for x in b['r']], 0).to(device)
            w_tg = S.pad([x[1:] for x in b['w']], 0).to(device)
            p_tg = S.pad([x[1:] for x in b['p']], 0).to(device)
            s_tg = S.pad([x[1:] for x in b['s']], 0).to(device)
            lr, lw, lp, ls = model(p_in, r_in, w_in, s_in, o_in)
            ig = 0
            loss = (F.cross_entropy(lr.reshape(-1, lr.shape[-1]), r_tg.reshape(-1),
                                    ignore_index=ig)
                    + 0.5 * F.cross_entropy(lw.reshape(-1, lw.shape[-1]), w_tg.reshape(-1),
                                            ignore_index=ig)
                    + 0.25 * F.cross_entropy(lp.reshape(-1, lp.shape[-1]), p_tg.reshape(-1),
                                             ignore_index=ig)
                    + 0.25 * F.cross_entropy(ls.reshape(-1, ls.shape[-1]), s_tg.reshape(-1),
                                             ignore_index=ig))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % 500 == 0:
                print(f'  step {step}/{args.steps} loss {float(loss):.4f} '
                      f'{step/(time.time()-t0):.2f} it/s', flush=True)
            if step >= args.steps:
                break

    res = {'arm': 'governed', 'params': nparam, 'steps': args.steps}
    res['bits_per_word'] = eval_bits_gov(model, data,
                                        ['val', 'test_gen', 'test_deriv'], device)
    res['root_acc'] = governed_eval(model, data, meta, vocab,
                                    ['test_gen', 'test_deriv'], device)
    print(f'\n  *** GOVERNED: bits/word ' +
          '  '.join(f'{k}={v:.3f}' for k, v in res['bits_per_word'].items()), flush=True)
    for sp, v in res['root_acc'].items():
        print(f'      {sp}: real top5 unmasked {100*v["unmasked_real"]["top5"]:.2f}% -> '
              f'masked {100*v["masked_real"]["top5"]:.2f}%   '
              f'violations {v["violations_unmasked"]} -> {v["violations_masked"]}', flush=True)
    json.dump(res, open('/workspace/sf_governed.json', 'w'), indent=2)
    torch.save(model.state_dict(), '/workspace/sf_governed.pt')
    print('wrote /workspace/sf_governed.json')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['prepare', 'train'])
    ap.add_argument('--max-words', type=int, default=25_000_000)
    ap.add_argument('--steps', type=int, default=80000)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--lr', type=float, default=9e-4)
    args = ap.parse_args()
    (prepare if args.mode == 'prepare' else train)(args)


if __name__ == '__main__':
    main()
