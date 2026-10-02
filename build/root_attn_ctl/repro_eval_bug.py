#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eval_saved_head.py -- READ-ONLY re-evaluation of a SAVED head (+ optional trunk/RCA payload)
with the trainer's CURRENT (val-stream, bug-fixed) live evaluator.

Why this exists
---------------
head_ROOTATTN.pt (step 5000 of the confounded run) and head_ROOTATTN.pt.trunk.pt were saved at
every eval, so the exact step-5000 model of Run A still exists on disk.  Its logged 0.23 % was
produced by the OLD live evaluator, which fed the TRAIN windows while pairing them with the VAL
targets.  Re-running the same weights through the fixed evaluator recovers the confounded arm's
TRUE step-5000 number WITHOUT retraining.

Three modes
-----------
  --trunk none                      released trunk, RCA gateway either absent or gate=0
  --trunk <head_*.pt.trunk.pt> --gates zero    damaged/unfrozen trunk, RCA ablated at inference
  --trunk <head_*.pt.trunk.pt> --gates trained damaged/unfrozen trunk, RCA live

The metric code is copied from nrmt_train.py::evaluate (scale-invariant acc@1/acc@5/ce_z on the
ALL_val and NOVEL_only masks) so the numbers are directly comparable to the run logs.

Usage:
  python eval_saved_head.py --head /workspace/root_attn/head_ROOTATTN.pt \
      --trunk /workspace/root_attn/head_ROOTATTN.pt.trunk.pt --gates trained --tag A_step5000
"""
import argparse
import json
import math
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, '/workspace/hf_v19_2_release')
sys.path.insert(0, '/workspace/hf_v19_2_release/models')
sys.path.insert(0, '/workspace/root_attn')

WIN, STRIDE, CTX = 128, 64, 12


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--head', required=True)
    ap.add_argument('--trunk', default='none',
                    help="'none' (released trunk) or a saved *.pt.trunk.pt payload")
    ap.add_argument('--gates', choices=['trained', 'zero'], default='trained')
    ap.add_argument('--checkpoint', default='/workspace/hf_v19_2_release/checkpoints/'
                    'rootformer_v19_2_synthesis_ar_backbone.awzan142.roots9490.tok10052.safetensors')
    ap.add_argument('--cache', default='/workspace/head_fix/nrmp_cache_9490_aligned')
    ap.add_argument('--val-windows', type=int, default=300)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--tag', default='eval')
    ap.add_argument('--buggy', action='store_true',
                    help='reproduce the ORIGINAL live-evaluator bug: feed the TRAIN windows to '
                         'live_h() while pairing with the VAL targets/head inputs')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    import nrmp_vocab as nv
    from models.unified_rootformer_v12 import UnifiedRootformerV12
    from deepseek_v4_1_flash_model import UnifiedRootformerV17_DeepSeekFlash
    from nrmt_arch import RootformerNRMT, build_operator_table
    from safetensors.torch import load_file

    device = torch.device('cuda')
    V = next(v for k, v in vars(nv).items() if isinstance(v, type) and 'MorphemicVocab' in k)
    vocab = V('/workspace/hf_v19_2_release/data/rootformer_v12_arabic_blueprint.json')
    bp = '/workspace/hf_v19_2_release/data/rootformer_v12_arabic_blueprint.json'
    base = UnifiedRootformerV12(bp, 'Qwen/Qwen2.5-0.5B', str(device), torch.bfloat16).to(device)
    flash = UnifiedRootformerV17_DeepSeekFlash(base, device, torch.bfloat16).to(device)
    # EXACTLY the trainer's construction (nrmt_train.py ~line 341): the head's parameter SET
    # depends on these kwargs -- with the defaults the `feat_gate` parameter does not exist and
    # the extra-conditioning branch is missing, which changes the metric.
    model = RootformerNRMT(flash, vocab, device, torch.bfloat16, hist=3, dropout=0.1,
                           use_features=True, feat_gate=True, feat_gate_proj_std=1e-3,
                           ngram_mode='none', ngram_orders=(1, 2, 3, 4), d_ng=32,
                           ngram_bucket_bits=20, ngram_vocabs=None, ngram_gate=True,
                           ngram_emb_init_std=0.02).to(device)
    miss, unexp = model.load_state_dict(load_file(args.checkpoint), strict=False)
    print(f'[{args.tag}] released checkpoint: missing={len(miss)} unexpected={len(unexp)}')
    # the trainer casts ONLY the head to fp32 (nrmt_train.py:378); the trunk stays bf16 and
    # live_h_windows returns `.float()`, so the head must be fp32 here too.
    model.nrmt_head.to(torch.float32)

    # ---- the ʿāmil (operator) stream, exactly as the trainer builds it ---------------------
    op_table = build_operator_table(vocab)
    try:
        from sibawayh_governor import SibawayhGovernor
        gov = SibawayhGovernor(vocab)
    except Exception as e:                                     # noqa: BLE001
        print(f'[{args.tag}] [warn] SibawayhGovernor unavailable: {e}')
        gov = None

    def op_ids_of(P, R, W):
        if gov is None:
            return op_table[R.clamp(min=0)]
        rows = [gov.op_ids_for_ids(rr, ww, pp) for pp, rr, ww in zip(P.tolist(), R.tolist(),
                                                                    W.tolist())]
        return torch.tensor(rows, dtype=torch.long)

    # ---- optional trunk/RCA payload ---------------------------------------------------------
    stack = None
    if args.trunk != 'none':
        from root_cross_attn import RootCrossAttentionStack
        blob = torch.load(args.trunk, map_location='cpu')
        meta = {k: v for k, v in blob.items() if k != 'state'}
        st = blob['state']
        print(f'[{args.tag}] payload metadata: {json.dumps(meta, default=str)}')
        rca_state = {k[len('root_cross.'):]: v for k, v in st.items()
                     if k.startswith('root_cross.')}
        trunk_state = {k: v for k, v in st.items() if not k.startswith('root_cross.')}
        layers = list(meta.get('rca_layers', [20, 21, 22, 23]))
        stack = RootCrossAttentionStack(
            model.d_model, model.morphemic_embed.root_embed, layers,
            num_heads=int(meta.get('rca_heads', 8)), dropout=0.0,
            out_norm=bool(meta.get('rca_out_norm', False)), dtype=torch.float32).to(device)
        stack.exclude_current = bool(meta.get('rca_exclude_current', False))
        m2, u2 = stack.load_state_dict(rca_state, strict=False)
        print(f'[{args.tag}] RCA load: missing={list(m2)} unexpected={list(u2)}')
        model.root_cross = stack
        stack.attach(model.backbone.layers)
        sd = model.state_dict()
        unknown = 0
        for k, v in trunk_state.items():
            if k in sd:
                sd[k] = v.to(sd[k].dtype)
            else:
                unknown += 1
        model.load_state_dict(sd, strict=False)
        print(f'[{args.tag}] loaded {len(trunk_state)} unfrozen-trunk tensors (unknown {unknown})')
        if args.gates == 'zero':
            stack.zero_gates()
        print(f'[{args.tag}] gates used: {[round(g, 6) for g in stack.gate_values()]}')

    # ---- head payload -----------------------------------------------------------------------
    hstate = torch.load(args.head, map_location='cpu')
    hmiss, hunexp = model.nrmt_head.load_state_dict(hstate, strict=False)
    print(f'[{args.tag}] head {args.head}: missing={list(hmiss)} unexpected={list(hunexp)}')

    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.backbone.eval()

    head = model.nrmt_head

    def live_h(Pr_w, Tr_w, Wr_w, Sr_w):
        emb = model.morphemic_embed(Pr_w, Tr_w, Wr_w, Sr_w)
        if stack is not None:
            stack.set_root_ids(Tr_w)
        out = model.backbone(inputs_embeds=emb)
        return model.final_norm(out.last_hidden_state).float()[:, :-1].contiguous()

    # ---- val windows, exactly as the trainer chooses them -----------------------------------
    va4 = [t.long() for t in torch.load(f'{args.cache}/val.pt', map_location='cpu')]
    s = list(range(0, va4[1].numel() - WIN - 1, STRIDE))
    np.random.default_rng(1).shuffle(s)
    va_st = sorted(s[:args.val_windows])
    Tv, Wv, Pv, Sv = (torch.stack([t[j:j + WIN] for j in va_st]) for t in (va4[1], va4[2],
                                                                           va4[0], va4[3]))
    Rt_va = Tv[:, 1:].contiguous()
    Ova_w = op_ids_of(Pv, Tv, Wv)[:, :-1].contiguous()

    if args.buggy:
        # the trainer's train-window selection
        tr4 = [t.long() for t in torch.load(f'{args.cache}/train.pt', map_location='cpu')]
        s2 = list(range(0, tr4[1].numel() - WIN - 1, STRIDE))
        np.random.default_rng(0).shuffle(s2)
        tr_st = sorted(s2[:3000])
        Ptr, Ttr, Wtr, Str = (torch.stack([t[j:j + WIN] for j in tr_st])
                              for t in (tr4[0], tr4[1], tr4[2], tr4[3]))
        print(f'[{args.tag}] BUGGY MODE: live_h() fed the TRAIN windows '
              f'{tuple(Ttr.shape)} while the head/targets come from the VAL windows '
              f'{tuple(Tv.shape)} -- the exact failure mode of the first patch version')

    specials = sorted({vocab.PAD_ROOT, vocab.BOS_ROOT, vocab.EOS_ROOT, vocab.UNK_ROOT,
                       vocab.root2id['<PARTICLE>']} |
                      {i for i, r in enumerate(vocab.roots_list) if r.startswith('<P:')})
    spec_t = torch.tensor(specials)
    keep_all = ~torch.isin(Rt_va.reshape(-1), spec_t)
    try:
        from nrmt_train import Novelty          # defined in nrmt_train.py (line 73), not nrmt_arch
        tr4 = [t.long() for t in torch.load(f'{args.cache}/train.pt', map_location='cpu')]
        nov = Novelty(tr4[1].tolist())
        per = WIN - 1
        nm = torch.zeros(Tv.shape[0] * per, dtype=torch.bool)
        for w in range(Tv.shape[0]):
            seg = Tv[w].tolist()
            for t in range(CTX, WIN):
                nm[w * per + (t - 1)] = nov.novel(seg[t - CTX:t])
        keep_novel = keep_all & nm
    except Exception as e:                                     # noqa: BLE001
        print(f'[{args.tag}] [warn] novelty mask unavailable ({e}); NOVEL_only will be skipped')
        keep_novel = torch.zeros_like(keep_all)

    print(f'[{args.tag}] val windows {len(va_st)} | radical positions all='
          f'{int(keep_all.sum())} novel={int(keep_novel.sum())}')

    lg = []
    with torch.no_grad():
        for i in range(0, len(va_st), args.batch):
            sl = slice(i, min(i + args.batch, len(va_st)))
            Pw, Tw, Ww, Sw = (x[sl].to(device) for x in (Pv, Tv, Wv, Sv))
            if args.buggy:
                Hsl = live_h(Ptr[sl].to(device), Ttr[sl].to(device),
                             Wtr[sl].to(device), Str[sl].to(device))
            else:
                Hsl = live_h(Pw, Tw, Ww, Sw)
            out = head(Hsl, Tw[:, :-1], Ova_w[sl].to(device), Ww[:, :-1], Pw[:, :-1], Sw[:, :-1])
            lg.append(out['root_logits'].float().cpu())
    LG = torch.cat(lg).reshape(-1, vocab.num_roots)
    tg = Rt_va.reshape(-1)
    assert LG.shape[0] == tg.shape[0], (LG.shape, tg.shape)

    res = {}
    for tag, m in (('ALL_val', keep_all), ('NOVEL_only', keep_novel)):
        l, t = LG[m], tg[m]
        if t.numel() == 0:
            continue
        ce_raw = float(F.cross_entropy(l, t))
        sd = l.std(-1, keepdim=True).clamp_min(1e-6)
        ce_z = float(F.cross_entropy((l - l.mean(-1, keepdim=True)) / sd, t))
        res[tag] = {'n': int(t.numel()),
                    'acc@1': float((l.argmax(-1) == t).float().mean()),
                    'acc@5': float((l.topk(5, -1).indices == t.unsqueeze(-1)).any(-1).float()
                                   .mean()),
                    'ce_z': ce_z, 'ppl_z': math.exp(min(ce_z, 25.0)),
                    'ce_raw_SCALE_DEPENDENT': ce_raw,
                    'logit_scale': float(sd.mean())}
    print(f'[{args.tag}] RESULT head={args.head.split("/")[-1]} trunk={args.trunk.split("/")[-1]} '
          f'gates={args.gates}')
    for k, v in res.items():
        print(f'[{args.tag}]   {k}: acc@1 {100*v["acc@1"]:.3f}%  acc@5 {100*v["acc@5"]:.3f}%  '
              f'CE_z {v["ce_z"]:.4f}  logit_scale {v["logit_scale"]:.3f}  (n={v["n"]})')
    out = {'tag': args.tag, 'head': args.head, 'trunk': args.trunk, 'gates': args.gates,
           'result': res}
    if args.out:
        json.dump(out, open(args.out, 'w'), indent=2)
        print(f'[{args.tag}] wrote {args.out}')


if __name__ == '__main__':
    main()
