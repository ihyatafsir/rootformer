#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nrmt_train.py -- train the NRMT architecture and watch root PPL on ARABIC ONLY, leak-free.

Protocol
--------
* backbone frozen; hidden states cached once (valid because they do not depend on the head)
* the NRMT head is trained on the cached states with
      L = L_root + 0.5 L_wazn + 0.25 L_prefix + 0.25 L_suffix
          + 0.1 * impossibility hinge   (forbidden roots pushed below a margin)
          + lambda_orbit * Al-Taqalib orbit consistency
* scheduled sampling on the morphological conditioning
* evaluation reports TWO numbers on the file-level held-out stream:
      ALL val     -- the usual PPL
      NOVEL only  -- positions whose 12-root context never occurs in train
  Root PPL on NOVEL positions is the only number that proves real progress.

Usage:
  python nrmt_train.py --checkpoint checkpoints/rootformer_v20_nrmp_master.safetensors \
      --train-windows 3000 --val-windows 300 --steps 20000
"""
import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / 'models'))

WIN, STRIDE, CTX = 128, 64, 12
HASH_B = np.uint64(1000003)


def rolling_hashes(stream, n, B=HASH_B):
    a = np.asarray(stream, dtype=np.uint64)
    L = len(a) - n + 1
    if L <= 0:
        return np.zeros(0, dtype=np.uint64)
    h = np.zeros(L, dtype=np.uint64)
    for k in range(n):
        h = h * B + a[k:k + L]
    return h


def hash_ctx(ctx, B=1000003):
    # Python ints with explicit 64-bit masking: identical arithmetic to rolling_hashes
    # (uint64 wraparound) but without numpy overflow warnings.
    h = 0
    for x in ctx:
        h = (h * B + int(x)) & ((1 << 64) - 1)
    return np.uint64(h)


class Novelty:
    def __init__(self, train_stream, ctx=CTX):
        t0 = time.time()
        self.h = np.unique(rolling_hashes(train_stream, ctx))
        print(f'    novelty: {len(self.h)} distinct {ctx}-gram hashes ({time.time()-t0:.0f}s)',
              flush=True)

    def novel(self, ctx):
        h = hash_ctx(ctx)
        i = np.searchsorted(self.h, h)
        return not (i < len(self.h) and self.h[i] == h)


@torch.no_grad()
def extract_backbone(model, streams, starts, device, tag):
    P, R, W, S = streams
    hs = []
    bs = 16
    t0 = time.time()
    for i in range(0, len(starts), bs):
        idx = starts[i:i + bs]
        p = torch.stack([P[j:j + WIN] for j in idx]).to(device)
        r = torch.stack([R[j:j + WIN] for j in idx]).to(device)
        w = torch.stack([W[j:j + WIN] for j in idx]).to(device)
        s = torch.stack([S[j:j + WIN] for j in idx]).to(device)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            emb = model.morphemic_embed(p, r, w, s)
            model._set_flash(r, w)
            o = model.backbone(inputs_embeds=emb)
            h = model.final_norm(o.last_hidden_state)
        hs.append(h.float().cpu())
    print(f'    {tag}: {len(starts)} windows, hidden {torch.cat(hs).shape} in {time.time()-t0:.0f}s',
          flush=True)
    return torch.cat(hs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default=None)
    ap.add_argument('--cache', default='/workspace/nrmp_cache')
    ap.add_argument('--train-windows', type=int, default=3000)
    ap.add_argument('--val-windows', type=int, default=300)
    ap.add_argument('--steps', type=int, default=20000)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--ss-prob', type=float, default=0.3)
    ap.add_argument('--lambda-orbit', type=float, default=0.0)
    ap.add_argument('--margin', type=float, default=-2.0)
    ap.add_argument('--hist', type=int, default=3)
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--no-features', action='store_true',
                    help='CONTROL: disable root-history/operator/morph conditioning (h-only head)')
    ap.add_argument('--tag', default='full')
    ap.add_argument('--eval-every', type=int, default=2000)
    ap.add_argument('--out', default='/workspace/nrmt_results.json')
    ap.add_argument('--save', default='/workspace/nrmt_head.pt')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    import nrmp_vocab as nv
    from models.unified_rootformer_v12 import UnifiedRootformerV12
    from deepseek_v4_1_flash_model import UnifiedRootformerV17_DeepSeekFlash
    from safetensors.torch import load_file
    from nrmt_arch import RootformerNRMT, build_operator_table

    V = next(v for k, v in vars(nv).items() if isinstance(v, type) and 'MorphemicVocab' in k)
    bp = str(ROOT_DIR / 'data/rootformer_v12_arabic_blueprint.json')
    vocab = V(bp)
    base = UnifiedRootformerV12(bp, 'Qwen/Qwen2.5-0.5B', str(device), torch.bfloat16).to(device)
    flash = UnifiedRootformerV17_DeepSeekFlash(base, device, torch.bfloat16).to(device)
    model = RootformerNRMT(flash, vocab, device, torch.bfloat16, hist=args.hist,
                           dropout=args.dropout, use_features=not args.no_features).to(device)
    print(f'[*] ARM [{args.tag}] use_features={not args.no_features} dropout={args.dropout}')

    ckpt = args.checkpoint or str(ROOT_DIR / 'checkpoints/rootformer_v19_2_synthesis_ar_backbone.safetensors')
    missing, unexpected = model.load_state_dict(load_file(ckpt), strict=False)
    new_keys = [k for k in missing if k.startswith('nrmt_head.')]
    print(f'[*] checkpoint {Path(ckpt).name}: missing={len(missing)} '
          f'(of which nrmt_head new={len(new_keys)}) unexpected={len(unexpected)}')
    print(f'[*] extra-conditioning norm at init (0.0 => warm start): '
          f'{model.nrmt_head.extra_norm():.6f}')
    print(f'[*] Al-Taqalib orbit pairs: {len(model.orbit_pairs)}')

    for p in model.parameters():
        p.requires_grad = False
    # the frozen backbone's cached states are float32, so train this small head in float32 too
    model.nrmt_head.to(torch.float32)
    for p in model.nrmt_head.parameters():
        p.requires_grad = True
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'[*] trainable parameters: {n_tr/1e6:.2f}M (backbone frozen)')

    tr4 = [t.long() for t in torch.load(Path(args.cache) / 'train.pt', map_location='cpu')]
    va4 = [t.long() for t in torch.load(Path(args.cache) / 'val.pt', map_location='cpu')]
    print(f'[*] train {tr4[1].numel()} | val {va4[1].numel()} root tokens')
    nov = Novelty(tr4[1].tolist())

    def starts_for(t, n, seed):
        s = list(range(0, t.numel() - WIN - 1, STRIDE))
        np.random.default_rng(seed).shuffle(s)
        return sorted(s[:n])

    tr_st = starts_for(tr4[1], args.train_windows, 0)
    va_st = starts_for(va4[1], args.val_windows, 1)

    print('[*] caching frozen backbone hidden states', flush=True)
    Htr = extract_backbone(model, tr4, tr_st, device, 'train')
    Hva = extract_backbone(model, va4, va_st, device, 'val')

    def win_tensor(t, starts):
        return torch.stack([t[j:j + WIN] for j in starts])

    Tr, Wr, Pr, Sr = (win_tensor(tr4[i], tr_st) for i in (1, 2, 0, 3))
    Tv, Wv, Pv, Sv = (win_tensor(va4[i], va_st) for i in (1, 2, 0, 3))

    op_table = build_operator_table(vocab)

    def op_ids_of(R):
        # op state established BY the word at each position (no roll): the ʿamil of word t
        # governs what may follow at t+1.
        return op_table[R.clamp(min=0)]

    # forbidden static set: Al-Khalil phonotactics (corrected) + control strings
    try:
        from classical_governance_v2 import AlKhalilV2
        forbidden_ids = sorted(AlKhalilV2(vocab).mask())
    except Exception as e:
        print(f'[warn] {e}')
        forbidden_ids = [vocab.PAD_ROOT, vocab.BOS_ROOT, vocab.UNK_ROOT,
                         vocab.root2id['<PARTICLE>']]
    fmask_root = torch.zeros(vocab.num_roots, dtype=torch.bool)
    fmask_root[torch.tensor(forbidden_ids)] = True
    print(f'[*] static forbidden roots (impossibility hinge): {int(fmask_root.sum())}')

    specials = sorted({vocab.PAD_ROOT, vocab.BOS_ROOT, vocab.EOS_ROOT, vocab.UNK_ROOT,
                       vocab.root2id['<PARTICLE>']} |
                      {i for i, r in enumerate(vocab.roots_list) if r.startswith('<P:')})
    spec_t = torch.tensor(specials)

    # per-WINDOW tensors: inputs are positions 0..T-2, targets are positions 1..T-1.
    # (An earlier version flattened [N,T,d] -> [N*T,d], destroying the time axis.)
    Htr_w = Htr[:, :-1, :].contiguous()          # [N, T-1, d]
    Hva_w = Hva[:, :-1, :].contiguous()
    Rt_tr, Rt_va = Tr[:, 1:].contiguous(), Tv[:, 1:].contiguous()
    # operator state established by the word AT each position (Sibawayh's ʿāmil)
    Otr_w = op_ids_of(Tr)[:, :-1].contiguous()
    Ova_w = op_ids_of(Tv)[:, :-1].contiguous()
    n_win, T1 = Htr_w.shape[0], Htr_w.shape[1]
    print(f'[*] training windows: {n_win} x {T1} positions = {n_win*T1}', flush=True)

    # context-novelty mask over val INPUT positions 0..T-2 (window-major order)
    per = WIN - 1
    novel_mask = torch.zeros(Tv.shape[0] * per, dtype=torch.bool)
    for w in range(Tv.shape[0]):
        seg = Tv[w].tolist()
        for t in range(CTX, WIN):
            novel_mask[w * per + (t - 1)] = nov.novel(seg[t - CTX:t])
    keep_all = ~torch.isin(Rt_va.reshape(-1), spec_t)
    keep_novel = keep_all & novel_mask
    print(f'[*] val radical positions: all={int(keep_all.sum())} novel={int(keep_novel.sum())} '
          f'({100*int(keep_novel.sum())/max(int(keep_all.sum()),1):.1f}%)')

    head = model.nrmt_head
    opt = torch.optim.AdamW([p for p in head.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.steps,
        pct_start=max(0.03, min(0.3, 500.0 / max(args.steps, 1))))

    def step_loss(h, R, W, P, S, O, tgt_r, tgt_w, tgt_p, tgt_s):
        """h:[B,T1,d]; R/W/P/S/O:[B,T1] aligned inputs; tgt_*:[B,T1] next-position targets."""
        out = head(h, R, O, W, P, S)
        keep = ~torch.isin(tgt_r, spec_t.to(tgt_r.device))
        l_root = (F.cross_entropy(out['root_logits'][keep], tgt_r[keep]) if keep.any()
                  else out['root_logits'].sum() * 0.0)
        l_wazn = F.cross_entropy(out['wazn_logits'].reshape(-1, vocab.num_awzan),
                                 tgt_w.reshape(-1), ignore_index=vocab.PAD_WAZN)
        l_pref = F.cross_entropy(out['prefix_logits'].reshape(-1, vocab.num_prefixes),
                                 tgt_p.reshape(-1), ignore_index=vocab.PAD_PREFIX)
        l_suff = F.cross_entropy(out['suffix_logits'].reshape(-1, vocab.num_suffixes),
                                 tgt_s.reshape(-1), ignore_index=vocab.PAD_SUFFIX)
        loss = l_root + 0.5 * l_wazn + 0.25 * l_pref + 0.25 * l_suff
        info = {'root': float(l_root.detach()), 'wazn': float(l_wazn.detach()),
                'prefix': float(l_pref.detach()), 'suffix': float(l_suff.detach())}
        fm = fmask_root.to(tgt_r.device)[tgt_r]
        if fm.any() and args.margin != 0.0:
            pen = F.relu(out['root_logits'][fm] - args.margin) ** 2
            loss = loss + 0.1 * pen.mean()
            info['impossible'] = float(pen.mean().detach())
        if args.lambda_orbit > 0:
            l_orb = model.orbit_consistency_loss()
            loss = loss + args.lambda_orbit * l_orb
            info['orbit'] = float(l_orb.detach())
        return loss, info

    @torch.no_grad()
    def evaluate(H, R, W, P, S, O, B=16):
        """H:[N,T1,d]; R/W/P/S/O:[N,T1] inputs. Returns metrics on ALL and NOVEL positions."""
        head.eval()
        lg = []
        nb = H.shape[0]
        for i in range(0, nb, B):
            sl = slice(i, min(i + B, nb))
            out = head(H[sl].to(device), R[sl].to(device), O[sl].to(device),
                       W[sl].to(device), P[sl].to(device), S[sl].to(device))
            lg.append(out['root_logits'].float().cpu())
        LG = torch.cat(lg).reshape(-1, vocab.num_roots)
        tg = Rt_va.reshape(-1)
        assert LG.shape[0] == tg.shape[0], (LG.shape, tg.shape)
        res = {}
        for tag, m in (('ALL_val', keep_all), ('NOVEL_only', keep_novel)):
            l, t = LG[m], tg[m]
            if t.numel() == 0:
                continue
            ce = float(F.cross_entropy(l, t))
            res[tag] = {'n': int(t.numel()),
                        'acc@1': float((l.argmax(-1) == t).float().mean()),
                        'acc@5': float((l.topk(5, -1).indices == t.unsqueeze(-1)).any(-1)
                                       .float().mean()),
                        'ppl': math.exp(min(ce, 25.0))}
        head.train()
        return res

    hist = []
    t0 = time.time()
    head.train()
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, n_win, (args.batch_size,))
        loss, info = step_loss(
            Htr_w[idx].to(device),
            Tr[idx][:, :-1].to(device), Wr[idx][:, :-1].to(device),
            Pr[idx][:, :-1].to(device), Sr[idx][:, :-1].to(device),
            Otr_w[idx].to(device),
            Rt_tr[idx].to(device), Wr[idx][:, 1:].to(device),
            Pr[idx][:, 1:].to(device), Sr[idx][:, 1:].to(device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in head.parameters() if p.requires_grad], 1.0)
        opt.step(); sched.step()
        if step % 2000 == 0:
            print(f'  step {step}/{args.steps} loss {float(loss):.4f} '
                  f'(root {info["root"]:.3f} wazn {info["wazn"]:.3f} '
                  f'pref {info["prefix"]:.3f} suff {info["suffix"]:.3f}) '
                  f'{step/(time.time()-t0):.0f} it/s', flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            m = evaluate(Hva_w, Tv[:, :-1], Wv[:, :-1], Pv[:, :-1], Sv[:, :-1], Ova_w)
            extra = f' | extra_norm {model.nrmt_head.extra_norm():.4f}'
            print(f'  [eval @{step}] ' + ' | '.join(
                f'{k}: acc@1 {100*v["acc@1"]:.2f}% acc@5 {100*v["acc@5"]:.2f}% '
                f'ppl {v["ppl"]:.1f} (n={v["n"]})'
                for k, v in m.items()) + extra, flush=True)
            hist.append({'step': step, **m, 'extra_norm': model.nrmt_head.extra_norm()})
            torch.save({k: v.cpu() for k, v in head.state_dict().items()}, args.save)

    json.dump({'history': hist, 'args': vars(args)}, open(args.out, 'w'), indent=2)
    print(f'\nwrote {args.out} and {args.save}')


if __name__ == '__main__':
    main()
