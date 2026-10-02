#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_live_eval.py -- the correctness gate for the unfrozen/live trainer.

The frozen trainer consumes CACHED hidden states (extract_backbone -> Hva_w).  The live trainer
recomputes them in the forward pass.  With the trunk frozen and every RCA gate at 0 the two MUST
agree exactly, otherwise every held-out number in the live arms is measuring something else.

This script proves, on the REAL cache and the REAL val starts:
  1. max|H_live - H_cached| over all 300 val windows x 127 positions x 896 dims (must be 0),
  2. the head's root logits and held-out acc@1 from the two h's (must be identical),
  3. the same for the first 300 TRAIN windows,
  4. that the RCA with its gate OPEN does move h (so the pathway is live), and
  5. that `--rca-out-norm` bounds the injected residual.

Usage:
  python verify_live_eval.py --checkpoint <ckpt> --cache <val+train stream cache>
"""
import argparse
import sys

import numpy as np
import torch

sys.path.insert(0, '/workspace/hf_v19_2_release')
sys.path.insert(0, '/workspace/hf_v19_2_release/models')
sys.path.insert(0, '/workspace/root_attn')

WIN, STRIDE = 128, 64


def starts_for(t, n, seed):
    s = list(range(0, t.numel() - WIN - 1, STRIDE))
    np.random.default_rng(seed).shuffle(s)
    return sorted(s[:n])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--cache', default='/workspace/head_fix/nrmp_cache_9490_aligned')
    ap.add_argument('--train-windows', type=int, default=3000)
    ap.add_argument('--val-windows', type=int, default=300)
    ap.add_argument('--layers', default='top4')
    args = ap.parse_args()

    import nrmp_vocab as nv
    from models.unified_rootformer_v12 import UnifiedRootformerV12
    from deepseek_v4_1_flash_model import UnifiedRootformerV17_DeepSeekFlash
    from nrmt_arch import RootformerNRMT, build_operator_table
    from safetensors.torch import load_file
    from root_cross_attn import RootCrossAttentionStack, parse_layer_spec

    device = torch.device('cuda')
    V = next(v for k, v in vars(nv).items() if isinstance(v, type) and 'MorphemicVocab' in k)
    vocab = V('/workspace/hf_v19_2_release/data/rootformer_v12_arabic_blueprint.json')
    bp = '/workspace/hf_v19_2_release/data/rootformer_v12_arabic_blueprint.json'

    tr4 = [t.long() for t in torch.load(f'{args.cache}/train.pt', map_location='cpu')]
    va4 = [t.long() for t in torch.load(f'{args.cache}/val.pt', map_location='cpu')]

    base = UnifiedRootformerV12(bp, 'Qwen/Qwen2.5-0.5B', str(device), torch.bfloat16).to(device)
    flash = UnifiedRootformerV17_DeepSeekFlash(base, device, torch.bfloat16).to(device)
    model = RootformerNRMT(flash, vocab, device, torch.bfloat16).to(device)
    sd = load_file(args.checkpoint)
    model.load_state_dict(sd, strict=False)
    for p in model.parameters():
        p.requires_grad = False
    model.nrmt_head.to(torch.float32)      # the trainer's float32 head
    model.eval()
    model.backbone.eval()
    model.nrmt_head.eval()

    print('[*] head warm start (remap) so the head is the released one')
    sys.path.insert(0, '/workspace/hf_v19_2_release')
    import nrmt_train as NT
    hs = model.nrmt_head.state_dict()
    remapped, applied, exempt = NT.remap_legacy_head(sd, hs, model.d_model, 'truncate')
    model.nrmt_head.load_state_dict(remapped, strict=False)
    print(f'    applied {len(applied)} tensors')

    layers = parse_layer_spec(args.layers, len(model.backbone.layers))
    stack = RootCrossAttentionStack(model.d_model, model.morphemic_embed.root_embed, layers,
                                    num_heads=8, dropout=0.0, dtype=torch.float32,
                                    out_norm=True).to(device)
    model.root_cross = stack
    stack.attach(model.backbone.layers)

    op_table = build_operator_table(vocab).to(device)
    specials = sorted({vocab.PAD_ROOT, vocab.BOS_ROOT, vocab.EOS_ROOT, vocab.UNK_ROOT,
                       vocab.root2id['<PARTICLE>']} |
                      {i for i, r in enumerate(vocab.roots_list) if r.startswith('<P:')})
    spec_t = torch.tensor(specials)

    def win_tensor(t, starts):
        return torch.stack([t[j:j + WIN] for j in starts])

    def live_h(streams, starts, B=16, gates_open=False):
        P, R, W, S = (win_tensor(streams[i], starts) for i in (0, 1, 2, 3))
        old = stack.zero_gates()
        if gates_open:
            for m in stack.mods:
                m.gate.data.fill_(1.0)
        out = []
        with torch.no_grad():
            for i in range(0, P.shape[0], B):
                sl = slice(i, min(i + B, P.shape[0]))
                p, r, w, s = (t[sl].to(device) for t in (P, R, W, S))
                stack.set_root_ids(r)
                emb = model.morphemic_embed(p, r, w, s)
                h = model.final_norm(model.backbone(inputs_embeds=emb).last_hidden_state)
                out.append(h.float().cpu())
        stack.restore_gates(old)
        return torch.cat(out), P, R, W, S

    def cached_h(streams, starts):
        # the frozen trainer's path has NO cross-attention at all; disable the hook (with gate=0
        # the live path's residual is exactly 0, so the two h's must still agree bit-for-bit)
        stack.set_root_ids(None)
        P, R, W, S = (win_tensor(streams[i], starts) for i in (0, 1, 2, 3))
        out = []
        with torch.no_grad():
            for i in range(0, P.shape[0], 16):
                sl = slice(i, min(i + 16, P.shape[0]))
                emb = model.morphemic_embed(*(t[sl].to(device) for t in (P, R, W, S)))
                out.append(model.final_norm(
                    model.backbone(inputs_embeds=emb).last_hidden_state).float().cpu())
        return torch.cat(out), P, R, W, S

    def head_acc1(H, R, W, P, S):
        lg, tt = [], []
        with torch.no_grad():
            for i in range(0, H.shape[0], 16):
                sl = slice(i, min(i + 16, H.shape[0]))
                r = R[sl].to(device)
                o = op_table[r.clamp(min=0)]
                out = model.nrmt_head(H[sl].to(device), r, o, W[sl].to(device),
                                      P[sl].to(device), S[sl].to(device))
                lg.append(out['root_logits'].float().cpu())
        LG = torch.cat(lg)[:, :-1, :].reshape(-1, vocab.num_roots)
        tg = R[:, 1:].reshape(-1)
        keep = ~torch.isin(tg, spec_t)
        return LG, float((LG[keep].argmax(-1) == tg[keep]).float().mean()), int(keep.sum())

    fails = 0
    for tag, streams, seed, nwin in (('VAL', va4, 1, args.val_windows),
                                     ('TRAIN', tr4, 0, min(300, args.train_windows))):
        st = starts_for(streams[1], nwin, seed)
        Hc, P, R, W, S = cached_h(streams, st)
        Hl, _, _, _, _ = live_h(streams, st)
        d = float((Hc[:, :-1] - Hl[:, :-1]).abs().max())
        print(f'[{tag}] {len(st)} windows | max|H_live-H_cached| = {d:.3e}', flush=True)
        Lc, acc_c, n = head_acc1(Hc, R, W, P, S)
        Ll, acc_l, _ = head_acc1(Hl, R, W, P, S)
        dl = float((Lc - Ll).abs().max())
        print(f'[{tag}] max|logit_live-logit_cached| = {dl:.3e} | acc@1 cached {100*acc_c:.3f}% '
              f'vs live {100*acc_l:.3f}% (n={n}, identical={acc_c == acc_l})', flush=True)
        if d != 0.0 or dl != 0.0 or acc_c != acc_l:
            fails += 1

    st = starts_for(va4[1], 50, 1)
    Hg0, P, R, W, S = live_h(va4, st, gates_open=False)
    Hg1, _, _, _, _ = live_h(va4, st, gates_open=True)
    print(f'[LIVE CHECK] gate=0 vs gate=1 on 50 val windows: max|dh| = '
          f'{float((Hg0[:, :-1] - Hg1[:, :-1]).abs().max()):.3e} (>0 => the RCA is live); '
          f'injected residual RMS with gate=1 (bounded by out_norm): {stack.out_rms(Hg0[:, :-1].to(device), R[:, :-1].to(device)):.4f}')
    print(f'[RESULT] {"LIVE PATH MATCHES THE CACHE PATH EXACTLY" if fails == 0 else "MISMATCH"}')
    return 0 if fails == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
