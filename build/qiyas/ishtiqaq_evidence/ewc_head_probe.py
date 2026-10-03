#!/usr/bin/env python3
"""ewc_head_probe.py -- the HEAD-INDEPENDENT trunk-damage measure, on CPU.

This reproduces the two numbers the whole study is anchored on:

    FIX baseline's own head + released trunk          -> 6.635 %  (published)
    FIX baseline's own head + Run A's damaged trunk    -> 2.411 %  (published)

by evaluating the FIX head (head_ALIGNED_FIX.pt) on live trunk hidden states, with NO root
cross-attention attached (i.e. the RCA-ablated condition those numbers are quoted with).

It also evaluates on the frozen cache (`extract_backbone`), which is the path the 6.635 % was
originally produced by, so the two agree by construction if the pipeline is right.

Usage:
  ewc_head_probe.py --head /workspace/head_fix/head_ALIGNED_FIX.pt \
                    --cached --live [--trunk <payload.pt>] [--val-windows 300]
"""
import os, sys, json, argparse, math
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
import torch

REL = '/workspace/hf_v19_2_release'
sys.path.insert(0, REL)
sys.path.insert(0, os.path.join(REL, 'models'))
from ewc_fisher import build_model

WIN, STRIDE = 128, 64


def starts_for(t, n, seed):
    import numpy as np
    s = list(range(0, t.numel() - WIN - 1, STRIDE))
    np.random.default_rng(seed).shuffle(s)
    return sorted(s[:n])


def build_meta(model, cache, val_windows):
    import nrmt_train as T
    from nrmt_arch import build_operator_table
    va4 = [t.long() for t in torch.load(os.path.join(cache, 'val.pt'), map_location='cpu')]
    va_st = starts_for(va4[1], val_windows, 1)
    Tv = torch.stack([va4[1][j:j + WIN] for j in va_st])
    Wv = torch.stack([va4[2][j:j + WIN] for j in va_st])
    Pv = torch.stack([va4[0][j:j + WIN] for j in va_st])
    Sv = torch.stack([va4[3][j:j + WIN] for j in va_st])
    vocab = model.vocab
    op_table = build_operator_table(vocab)
    try:
        from sibawayh_governor import SibawayhGovernor
        gov = SibawayhGovernor(vocab)
        print('[*] operator stream: SibawayhGovernor (persistent)', flush=True)
    except Exception as e:
        gov = None
        print(f'[warn] governor unavailable ({e}); t-1 table', flush=True)

    def op_ids_of(P, R, W):
        if gov is None:
            return op_table[R.clamp(min=0)]
        rows = [gov.op_ids_for_ids(rr, ww, pp) for pp, rr, ww in zip(P.tolist(), R.tolist(),
                                                                   W.tolist())]
        return torch.tensor(rows, dtype=torch.long)

    Ova_w = op_ids_of(Pv, Tv, Wv)[:, :-1].contiguous()
    specials = sorted({vocab.PAD_ROOT, vocab.BOS_ROOT, vocab.EOS_ROOT, vocab.UNK_ROOT,
                       vocab.root2id['<PARTICLE>']} |
                      {i for i, r in enumerate(vocab.roots_list) if r.startswith('<P:')})
    keep_all = ~torch.isin(Tv[:, 1:].reshape(-1), torch.tensor(specials))
    return (Pv, Tv, Wv, Sv), Ova_w, keep_all, va4, va_st


@torch.no_grad()
def eval_head(model, head, h_batches, R, O, W, P, S, keep_all, tg, B=16):
    lg = []
    for i in range(0, len(R), B):
        sl = slice(i, min(i + B, len(R)))
        out = head(h_batches[sl], R[sl], O[sl], W[sl], P[sl], S[sl])
        lg.append(out['root_logits'].float())
    LG = torch.cat(lg).reshape(-1, model.vocab.num_roots)
    l, t = LG[keep_all], tg.reshape(-1)[keep_all]
    import torch.nn.functional as F
    ce = float(F.cross_entropy(l, t))
    sd = l.std(-1, keepdim=True).clamp_min(1e-6)
    cez = float(F.cross_entropy((l - l.mean(-1, keepdim=True)) / sd, t))
    return {'n': int(t.numel()), 'acc@1': float((l.argmax(-1) == t).float().mean()),
            'acc@5': float((l.topk(5, -1).indices == t.unsqueeze(-1)).any(-1).float().mean()),
            'ce_raw': ce, 'ce_z': cez, 'logit_scale': float(sd.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default='/workspace/head_fix/nrmp_cache_9490_aligned')
    ap.add_argument('--head', default='/workspace/head_fix/head_ALIGNED_FIX.pt')
    ap.add_argument('--trunk', default='', help='optional damaged trunk payload (state dict)')
    ap.add_argument('--ckpt', default=os.path.join(
        REL, 'checkpoints/rootformer_v19_2_synthesis_ar_backbone.awzan142.roots9490.tok10052.safetensors'))
    ap.add_argument('--val-windows', type=int, default=300)
    ap.add_argument('--cached', action='store_true')
    ap.add_argument('--live', action='store_true')
    ap.add_argument('--tag', default='probe')
    ap.add_argument('--out', default='')
    ap.add_argument('--threads', type=int, default=6)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    base, flash, model, vocab = build_model(args.ckpt, dtype=torch.float32)
    for p in model.parameters():
        p.requires_grad = False
    hsd = torch.load(args.head, map_location='cpu')
    r = model.nrmt_head.load_state_dict(hsd, strict=False)
    print(f'[*] FIX head loaded: missing={len(r.missing_keys)} unexpected={len(r.unexpected_keys)}',
          flush=True)
    if r.missing_keys or r.unexpected_keys:
        raise SystemExit(f'FIX head did not load cleanly: {r.missing_keys} {r.unexpected_keys}')

    if args.trunk:
        pay = torch.load(args.trunk, map_location='cpu')
        st = pay.get('state', pay)
        n = 0
        with torch.no_grad():
            for name, p in model.named_parameters():
                if name in st and name.startswith('backbone.layers.'):
                    p.data.copy_(st[name].float())
                    n += 1
        print(f'[*] damaged trunk applied: {n} tensors from {args.trunk}', flush=True)

    (Pv, Tv, Wv, Sv), Ova_w, keep_all, va4, va_st = build_meta(model, args.cache, args.val_windows)
    R, W, P, S = (Tv[:, :-1].contiguous(), Wv[:, :-1].contiguous(),
                  Pv[:, :-1].contiguous(), Sv[:, :-1].contiguous())
    tg = Tv[:, 1:].contiguous()
    print(f'[*] val windows {Tv.shape[0]} | radical positions {int(keep_all.sum())}', flush=True)

    res = {'tag': args.tag, 'val_windows': args.val_windows, 'n_positions': int(keep_all.sum()),
           'head': args.head, 'trunk': args.trunk or 'released'}
    if args.cached:
        import nrmt_train as T
        Hva = T.extract_backbone(model, va4, va_st, 'cpu', 'val')
        Hva_w = Hva[:, :-1, :].contiguous()
        res['cached'] = eval_head(model, model.nrmt_head, Hva_w, R, Ova_w, W, P, S, keep_all, tg)
        print(f"[{args.tag}] CACHED-H   acc@1 {100*res['cached']['acc@1']:.3f}% "
              f"acc@5 {100*res['cached']['acc@5']:.3f}% ce_z {res['cached']['ce_z']:.4f}", flush=True)
    if args.live:
        for _layer in model.backbone.layers:      # no stale root conditioning (see live path)
            _layer.self_attn.active_root_ids = None
            _layer.self_attn.active_wazn_ids = None
        hs = []
        with torch.no_grad():
            for i in range(0, Tv.shape[0], 8):
                sl = slice(i, min(i + 8, Tv.shape[0]))
                emb = model.morphemic_embed(Pv[sl], Tv[sl], Wv[sl], Sv[sl])
                out = model.backbone(inputs_embeds=emb)
                hs.append(model.final_norm(out.last_hidden_state).float()[:, :-1].contiguous())
        H = torch.cat(hs)
        res['live'] = eval_head(model, model.nrmt_head, H, R, Ova_w, W, P, S, keep_all, tg)
        print(f"[{args.tag}] LIVE fp32  acc@1 {100*res['live']['acc@1']:.3f}% "
              f"acc@5 {100*res['live']['acc@5']:.3f}% ce_z {res['live']['ce_z']:.4f}", flush=True)
    if args.out:
        json.dump(res, open(args.out, 'w'), indent=2)
    print('HEAD_PROBE_DONE')
    return res


if __name__ == '__main__':
    main()
