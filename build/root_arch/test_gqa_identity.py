#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_gqa_identity.py -- is `enable_gqa=True` bit-identical to explicit `repeat_interleave`?

WHY THIS MATTERS.  FLOOR_A started at 00:04:18Z, BEFORE `_sdpa` was switched from
`enable_gqa=True` to explicit expansion.  EARLYROOT_C later ran WITH the expansion.  If the two
formulations differ numerically, then A and C differ in the attention implementation as well as in
the root pathway -- a confound in the one comparison the whole session is about.  This measures it
instead of assuming it.

Method: build the v13 trunk, install the SDPA surface attention, then run the SAME full-trunk
forward twice -- once with `_sdpa` replaced by the old `enable_gqa` formulation and once with the
shipped explicit-expansion formulation -- and compare.
"""
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, '/workspace/root_arch')
import model_build as MB
import sdpa_attention as SA

DEV = 'cuda'
B, WIN = 8, 128


def old_sdpa(q_s, k_s, v, *, scale, causal, add_bias, attention_mask, dropout_p, num_kv_groups):
    """The PREVIOUS implementation: no manual expansion, enable_gqa=True."""
    T, Tk = q_s.shape[2], k_s.shape[3 - 2]
    Tk = k_s.shape[2]
    bias = None
    if attention_mask is not None:
        bias = attention_mask if attention_mask.dim() == 4 else \
            (1.0 - attention_mask[:, None, None, :].to(q_s.dtype)) * -10000.0
    if add_bias is not None:
        bias = add_bias.float() if bias is None else bias.float() + add_bias.float()
    if causal:
        band = SA._causal_band(T, Tk, torch.float32, q_s.device)
        bias = band if bias is None else bias + band
    q_s, k_s, v, actx = SA._unify_dtypes(q_s, k_s, v)
    if bias is not None:
        bias = bias.to(q_s.dtype)
    with actx:
        if bias is None:
            return F.scaled_dot_product_attention(q_s, k_s, v, is_causal=bool(causal),
                                                  scale=scale, dropout_p=dropout_p,
                                                  enable_gqa=bool(num_kv_groups > 1))
        return F.scaled_dot_product_attention(q_s, k_s, v, attn_mask=bias, is_causal=False,
                                              scale=scale, dropout_p=dropout_p,
                                              enable_gqa=bool(num_kv_groups > 1))


def trunk(m):
    P, R, W, S = MB.win_batch(B, WIN)
    m.eval()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        emb = m.morphemic_embed(P, R, W, S)
        out = m.backbone(inputs_embeds=emb)
        h = m.final_norm(out.last_hidden_state)
    return h.float().cpu()


def main():
    vocab, base, flash, model = MB.build(verbose=True)
    SA.install(verbose=False)
    layers = model.backbone.layers
    for l in layers:
        old = l.self_attn
        new = SA.SdpaIshtiqaqAttention(
            hidden_size=old.hidden_size, num_heads=old.num_heads,
            num_kv_heads=old.num_kv_heads, head_dim=old.head_dim,
            num_roots=old.root_embed.num_embeddings,
            num_awzan=old.wazn_embed.num_embeddings,
            ishtiqaq_init_strength=float(old.ishtiqaq_gamma.detach()),
            dropout=float(old.dropout.p), layer_idx=old.layer_idx,
            dtype=old.root_embed.weight.dtype).to(DEV)
        nsd = new.state_dict()
        new.load_state_dict({k: v for k, v in old.state_dict().items()
                             if k in nsd and tuple(v.shape) == tuple(nsd[k].shape)},
                            strict=False)
        l.self_attn = new

    shipped = SA._sdpa
    SA._sdpa = old_sdpa
    h_old = trunk(model)
    SA._sdpa = shipped
    h_new = trunk(model)

    d = (h_old - h_new).abs()
    eq = torch.equal(h_old, h_new)
    print()
    print('=' * 78)
    print('enable_gqa=True  vs  explicit repeat_interleave   (full 24-layer trunk, same weights)')
    print('=' * 78)
    print(f'  torch.equal          : {eq}')
    print(f'  max|d|               : {float(d.max()):.4e}')
    print(f'  rms  |d|             : {float(d.pow(2).mean().sqrt()):.4e}')
    print(f'  reference rms        : {float(h_old.pow(2).mean().sqrt()):.4e}')
    print(f'  max|d| / ref_rms     : {float(d.max() / h_old.pow(2).mean().sqrt()):.3e}')
    print()
    if eq:
        print('  VERDICT: BIT-IDENTICAL.  FLOOR_A is NOT confounded by the implementation change.')
    else:
        print('  VERDICT: NOT identical.  FLOOR_A must be re-run with the current implementation so')
        print('           that A and C differ ONLY in the root pathway.')
    raise SystemExit(0 if eq else 1)


if __name__ == '__main__':
    main()
