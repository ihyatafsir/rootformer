#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""probe_rca_dynamics.py -- READ-ONLY diagnostics on a trained/saved root cross-attention stack.

Answers the brief's question (1): gradient norms (read separately from the run's trace.jsonl),
whether the gate opens, and whether the ATTENTION WEIGHTS are degenerate -- uniform, or collapsed
onto a single position.

Nothing is modified on disk: the RCA source is imported from /workspace/root_attn and the
attention recorder is a runtime monkeypatch inside THIS process only.

Usage:
  python probe_rca_dynamics.py --ckpt /workspace/root_attn/head_ROOTATTN.pt.trunk.pt \
      --tag ROOTATTN_step5000 --cache /workspace/head_fix/nrmp_cache_9490_aligned \
      --windows 8 --out /workspace/root_attn_ctl/probe_rca_ROOTATTN.json
"""
import argparse
import json
import math
import sys

import numpy as np
import torch

sys.path.insert(0, '/workspace/hf_v19_2_release')
sys.path.insert(0, '/workspace/hf_v19_2_release/models')
sys.path.insert(0, '/workspace/root_attn_width')

WIN, STRIDE = 128, 64


def install_attention_recorder():
    """Replace RootHistoryCrossAttention.forward with an identical body that also stashes the
    post-softmax attention and the pre-softmax logit scale.  Process-local only."""
    import root_cross_attn_width as rca
    orig = rca.RootHistoryCrossAttention.forward

    def forward(self, h, root_ids, root_weight, exclude_current=False):
        B, T, _ = h.shape
        H, D = self.num_heads, self.head_dim
        hf = h.float()
        q = self.q_norm(self.q_proj(hf).view(B, T, H, D)).transpose(1, 2)
        safe = root_ids.clamp_(0, root_weight.shape[0] - 1)
        r = root_weight.index_select(0, safe.reshape(-1)).view(B, T, -1).float()
        k = self.k_norm(self.k_proj(r).view(B, T, H, D)).transpose(1, 2)
        v = self.v_proj(r).view(B, T, H, D).transpose(1, 2)
        logits = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        pos = torch.arange(T, device=h.device)
        causal = pos.unsqueeze(0) >= pos.unsqueeze(1)
        if exclude_current:
            causal = pos.unsqueeze(0) > pos.unsqueeze(1)
        logits = logits.masked_fill(~causal, float('-inf'))
        att = torch.nan_to_num(logits.softmax(dim=-1), nan=0.0)
        ctx = torch.matmul(att, v).transpose(1, 2).reshape(B, T, H * D)
        out = self.o_proj(ctx)
        if self.out_norm is not None:
            out = self.out_norm(out)
        self._last_att = att.detach()
        self._last_logit_absmax = float(logits[torch.isfinite(logits)].abs().max()) \
            if bool(torch.isfinite(logits).any()) else float('nan')
        delta = self.gate.float() * self.drop(out)
        self._last_delta_rms = float(delta.detach().float().pow(2).mean().sqrt())
        return delta

    rca.RootHistoryCrossAttention.forward = forward
    return orig


def att_stats(att, exclude_current):
    """att: [B,H,T,T] post-softmax, already causal-masked."""
    B, H, T, _ = att.shape
    idx = torch.arange(T, device=att.device)
    tri = idx.unsqueeze(0) >= idx.unsqueeze(1)
    if exclude_current:
        tri = idx.unsqueeze(0) > idx.unsqueeze(1)
    nvis = tri.sum(-1).float()                                  # [T] visible keys per row
    p = att.float()
    ent = -(p.clamp_min(1e-12).log() * p).sum(-1)               # [B,H,T]
    logn = nvis.clamp_min(1.0).log().view(1, 1, T)
    ent_norm = torch.where(nvis.view(1, 1, T) >= 2, ent / logn.clamp_min(1e-9),
                           torch.full_like(ent, float('nan')))
    maxw = p.max(-1).values
    argmax = p.argmax(-1)                                       # [B,H,T]
    diag_mass = p.diagonal(dim1=-2, dim2=-1)                    # [B,H,T] mass on j == t
    # mean distribution over all (b,h,t) rows, and its spread -> uniformity detector
    mean_p = p.reshape(-1, T, T).mean(0).mean(0)                # [T] mean over queries of mean att
    # effective number of attended positions (exp of entropy), averaged over rows with nvis>=2
    eff = torch.exp(ent)
    valid = (nvis.view(1, 1, T) >= 2).expand_as(ent)
    effv = eff[valid]
    entv = ent_norm[valid]
    # how often is the argmax the CURRENT root (j==t) vs the FIRST root (j==0)?
    cur = (argmax == idx.view(1, 1, T)).float()
    first = (argmax == 0).float()
    cur = torch.where(valid, cur, torch.full_like(cur, float('nan')))
    first = torch.where(valid, first, torch.full_like(first, float('nan')))
    # distribution of attention mass over relative offsets, averaged (a shape fingerprint)
    off = (idx.view(1, 1, 1, T) - idx.view(1, 1, T, 1)).float()  # [1,1,T,T] j - t <= 0
    off_mean = []
    for d in range(0, min(T, 9)):
        m = (off == -float(d))
        off_mean.append(float((p * m).sum(-1).mean()))
    return {
        'rows': int(valid.sum()), 'n_heads': int(H),
        'entropy_norm_mean': float(entv.mean()), 'entropy_norm_p10': float(entv.quantile(0.10)),
        'entropy_norm_p90': float(entv.quantile(0.90)),
        'eff_positions_mean': float(effv.mean()), 'eff_positions_p10': float(effv.quantile(0.10)),
        'max_weight_mean': float(maxw[valid].mean()), 'max_weight_p90': float(maxw[valid].quantile(0.90)),
        'frac_rows_maxw_gt_0.9': float((maxw[valid] > 0.9).float().mean()),
        'diag_mass_mean': float(diag_mass[valid].mean()),
        'argmax_is_current_root_frac': float(cur[valid].mean()),
        'argmax_is_first_root_frac': float(first[valid].mean()),
        'mean_att_offset_mass_0..8': off_mean,
        'mean_att_over_queries_entropy_norm':
            float(-(mean_p.clamp_min(1e-12).log() * mean_p).sum()
                  / math.log(max(int(mean_p.numel()), 2))),
        'mean_att_over_queries_argmax_offset': int(mean_p.argmax()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default='/workspace/hf_v19_2_release/checkpoints/'
                    'rootformer_v19_2_synthesis_ar_backbone.awzan142.roots9490.tok10052.safetensors')
    ap.add_argument('--cache', default='/workspace/head_fix/nrmp_cache_9490_aligned')
    ap.add_argument('--ckpt', required=True, help='*.pt.trunk.pt (or plain root_cross state)')
    ap.add_argument('--tag', required=True)
    ap.add_argument('--layers', default=None, help='default: from the checkpoint metadata')
    ap.add_argument('--windows', type=int, default=8)
    ap.add_argument('--only-rca', action='store_true',
                    help='ignore an unfrozen trunk payload (measure the RCA on the RELEASED trunk)')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    import nrmp_vocab as nv
    from models.unified_rootformer_v12 import UnifiedRootformerV12
    from deepseek_v4_1_flash_model import UnifiedRootformerV17_DeepSeekFlash
    from nrmt_arch import RootformerNRMT
    from safetensors.torch import load_file
    from root_cross_attn_width import RootCrossAttentionStack

    install_attention_recorder()
    device = torch.device('cuda')
    V = next(v for k, v in vars(nv).items() if isinstance(v, type) and 'MorphemicVocab' in k)
    vocab = V('/workspace/hf_v19_2_release/data/rootformer_v12_arabic_blueprint.json')
    bp = '/workspace/hf_v19_2_release/data/rootformer_v12_arabic_blueprint.json'
    base = UnifiedRootformerV12(bp, 'Qwen/Qwen2.5-0.5B', str(device), torch.bfloat16).to(device)
    flash = UnifiedRootformerV17_DeepSeekFlash(base, device, torch.bfloat16).to(device)
    model = RootformerNRMT(flash, vocab, device, torch.bfloat16).to(device)
    model.load_state_dict(load_file(args.checkpoint), strict=False)

    blob = torch.load(args.ckpt, map_location='cpu')
    meta = {}
    if isinstance(blob, dict) and 'state' in blob:
        meta = {k: v for k, v in blob.items() if k != 'state'}
        st = blob['state']
    else:
        st = blob
    rca_state = {k[len('root_cross.'):]: v for k, v in st.items() if k.startswith('root_cross.')}
    trunk_state = {k: v for k, v in st.items() if not k.startswith('root_cross.')}
    print(f'[{args.tag}] metadata: {json.dumps(meta, default=str)}')
    print(f'[{args.tag}] rca tensors {len(rca_state)} '
          f'({sum(v.numel() for v in rca_state.values())/1e6:.3f}M), '
          f'trunk tensors {len(trunk_state)} '
          f'({sum(v.numel() for v in trunk_state.values())/1e6:.3f}M)')

    layers = ([int(x) for x in args.layers.split(',')] if args.layers
              else list(meta.get('rca_layers', [20, 21, 22, 23])))
    stack = RootCrossAttentionStack(
        model.d_model, model.morphemic_embed.root_embed, layers,
        num_heads=int(meta.get('rca_heads', 8)), dropout=0.0,
        out_norm=bool(meta.get('rca_out_norm', False)), dtype=torch.float32,
        d_attn=int(meta.get('rca_dim', model.d_model)) or None).to(device)
    stack.exclude_current = bool(meta.get('rca_exclude_current', False))
    miss, unexp = stack.load_state_dict(rca_state, strict=False)
    print(f'[{args.tag}] RCA load: missing={list(miss)} unexpected={list(unexp)}')
    model.root_cross = stack
    stack.attach(model.backbone.layers)
    if trunk_state and not args.only_rca:
        head_missing = [k for k in trunk_state if k not in model.state_dict()]
        sd = model.state_dict()
        for k, v in trunk_state.items():
            if k in sd:
                sd[k] = v.to(sd[k].dtype)
        model.load_state_dict(sd, strict=False)
        print(f'[{args.tag}] loaded {len(trunk_state)} unfrozen-trunk tensors '
              f'(unknown keys {len(head_missing)})')
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.backbone.eval()
    for m in stack.mods:
        m.eval()

    gates = stack.gate_values()
    print(f'[{args.tag}] gates: {[round(g, 6) for g in gates]}  '
          f'|gate| mean {sum(abs(g) for g in gates)/len(gates):.6f}')

    va4 = [t.long() for t in torch.load(f'{args.cache}/val.pt', map_location='cpu')]
    s = list(range(0, va4[1].numel() - WIN - 1, STRIDE))
    np.random.default_rng(1).shuffle(s)
    starts = sorted(s[:args.windows])
    P, R, W, S = (torch.stack([t[j:j + WIN] for j in starts]).to(device) for t in va4)

    emb = model.morphemic_embed(P, R, W, S)
    with torch.no_grad():
        # reference: released trunk, no RCA at all
        stack.zero_gates()
        h_ref = model.final_norm(model.backbone(inputs_embeds=emb).last_hidden_state).float()
        # trained gates, RCA live
        stack.restore_gates(gates)
        stack.set_root_ids(R)
        h_live = model.final_norm(model.backbone(inputs_embeds=emb).last_hidden_state).float()
    print(f'[{args.tag}] max|h(live) - h(no RCA)| on {args.windows} val windows = '
          f'{float((h_live - h_ref).abs().max()):.4e}   '
          f'rms(h)={float(h_ref.pow(2).mean().sqrt()):.4f}')

    rep = {'tag': args.tag, 'ckpt': args.ckpt, 'meta': {k: str(v) for k, v in meta.items()},
           'gates': gates, 'windows': args.windows,
           'max_abs_dh_live_vs_no_rca': float((h_live - h_ref).abs().max()),
           'h_rms_ref': float(h_ref.pow(2).mean().sqrt()), 'layers': {}}
    for pos, mod in enumerate(stack.mods):
        att = mod._last_att
        d = att_stats(att, stack.exclude_current)
        d['gate'] = float(mod.gate.detach().float())
        d['injected_residual_rms'] = float(mod._last_delta_rms)
        d['att_logit_absmax'] = mod._last_logit_absmax
        rep['layers'][f'layer{mod.layer_idx}'] = d
        print(f'  layer {mod.layer_idx}: gate {d["gate"]:+.5f} '
              f'injected_rms {d["injected_residual_rms"]:.4f} '
              f'ent_norm {d["entropy_norm_mean"]:.4f} (1.0 = uniform) '
              f'eff_pos {d["eff_positions_mean"]:.1f} '
              f'maxw {d["max_weight_mean"]:.3f} (frac>0.9 {d["frac_rows_maxw_gt_0.9"]:.3f}) '
              f'diag_mass {d["diag_mass_mean"]:.4f} '
              f'argmax=current {d["argmax_is_current_root_frac"]:.3f} '
              f'argmax=first {d["argmax_is_first_root_frac"]:.3f} '
              f'logit_absmax {d["att_logit_absmax"]:.2f}')

    if args.out:
        json.dump(rep, open(args.out, 'w'), indent=2)
        print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
