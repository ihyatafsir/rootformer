#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
probe_native_root_path.py (v2) -- WHAT does IshtiqaqAttentionV12's native root path compute,
and WHY is it inert?  CPU-only; no training.

Sections
  A  WIRING       -- trainer drives `model.backbone(inputs_embeds=emb)`, which never goes through
                     `UnifiedRootformerV12.forward()` (the only writer of `active_root_ids`).
  D  ID SPACES    -- native root_embed 9015 rows vs a 9490-id root stream; the model's own
                     token->root table is a *third*, near-empty mapping.
  B  DECOMPOSED FORCE-ENABLE -- root score bias / identical_root_bonus / Pillar I+II separately,
                     with RMS-relative effect sizes and the raw magnitude in NATS.
  C  EXACT RECONSTRUCTION -- the whole output delta equals o_proj((attn_on - attn_off) @ v),
                     i.e. the native root term is a pure reweighting of the surface value
                     stream and carries NO new information into the residual.
  E  TABLES       -- are the 24 per-layer (9015,64) tables trained, or at N(0,1) default init?
  F  SCALARS      -- stream_mix / ishtiqaq_gamma at init in every layer.
"""
import argparse
import functools
import json
import math
import sys

print = functools.partial(print, flush=True)  # never lose output to block buffering

import torch

sys.path.insert(0, '/workspace/hf_v19_2_release')
sys.path.insert(0, '/workspace/hf_v19_2_release/models')
sys.path.insert(0, '/workspace/root_attn')

WIN = 128


def rms(t):
    return float(t.float().pow(2).mean().sqrt())


def eff(a, b):
    """effect size of the delta: RMS(dh), max|dh|, RMS(dh)/RMS(a), max|dh|/max|a|"""
    d = (a.float() - b.float())
    return {'rms_dh': rms(d), 'max_dh': float(d.abs().max()),
            'rms_rel': rms(d) / max(1e-30, rms(a)),
            'max_rel': float(d.abs().max()) / max(1e-30, float(a.float().abs().max()))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--cache', required=True)
    ap.add_argument('--windows', type=int, default=8)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    OUT = {}

    import nrmp_vocab as nv
    from models.unified_rootformer_v12 import UnifiedRootformerV12
    from deepseek_v4_1_flash_model import UnifiedRootformerV17_DeepSeekFlash
    from nrmt_arch import RootformerNRMT
    from models.ishtiqaq_attention_v12 import IshtiqaqAttentionV12
    from safetensors.torch import load_file

    dev = torch.device(args.device)
    V = next(v for k, v in vars(nv).items() if isinstance(v, type) and 'MorphemicVocab' in k)
    BP = '/workspace/hf_v19_2_release/data/rootformer_v12_arabic_blueprint.json'
    vocab = V(BP)
    print(f'[*] vocab num_roots={vocab.num_roots} num_awzan={vocab.num_awzan}')

    counts = {'calls': 0, 'root_ids': 0, 'active': 0}
    _orig = IshtiqaqAttentionV12.forward

    def _wrapped(self, hidden_states, *a, **k):
        counts['calls'] += 1
        if k.get('root_ids') is not None:
            counts['root_ids'] += 1
        if getattr(self, 'active_root_ids', None) is not None:
            counts['active'] += 1
        return _orig(self, hidden_states, *a, **k)

    IshtiqaqAttentionV12.forward = _wrapped

    base = UnifiedRootformerV12(BP, 'Qwen/Qwen2.5-0.5B', str(dev), torch.bfloat16).to(dev)
    flash = UnifiedRootformerV17_DeepSeekFlash(base, str(dev), torch.bfloat16).to(dev)
    model = RootformerNRMT(flash, vocab, str(dev), torch.bfloat16).to(dev)
    missing, unexpected = model.load_state_dict(load_file(args.checkpoint), strict=False)
    print(f'[*] checkpoint missing={len(missing)} unexpected={len(unexpected)}')
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.backbone.eval()
    layers = model.backbone.layers
    nL = len(layers)

    tr4 = [t.long() for t in torch.load(f'{args.cache}/train.pt', map_location='cpu')]
    P_, R_, W_, S_ = tr4
    nw = min(args.windows, max(1, R_.shape[0] // WIN))
    P = torch.stack([P_[i * WIN:(i + 1) * WIN] for i in range(nw)]).to(dev)
    R = torch.stack([R_[i * WIN:(i + 1) * WIN] for i in range(nw)]).to(dev)
    W = torch.stack([W_[i * WIN:(i + 1) * WIN] for i in range(nw)]).to(dev)
    S = torch.stack([S_[i * WIN:(i + 1) * WIN] for i in range(nw)]).to(dev)

    def run(root_ids=None, tok_root=False):
        for L in layers:
            L.self_attn.active_root_ids = root_ids
        with torch.no_grad():
            emb = model.morphemic_embed(P, R, W, S)
            model._set_flash(R, W)
            h = model.final_norm(model.backbone(inputs_embeds=emb).last_hidden_state)
        for L in layers:
            L.self_attn.active_root_ids = None
        return h.float(), emb

    # ---------------- A ----------------
    counts.update(calls=0, root_ids=0, active=0)
    h_off, emb = run(None)
    OUT['A_wiring'] = dict(counts)
    print(f"\n=== A. WIRING ===")
    print(f"  attention calls={counts['calls']}  with root_ids={counts['root_ids']}  "
          f"active_root_ids set={counts['active']}")

    # ---------------- D ----------------
    rows = layers[0].self_attn.root_embed.num_embeddings
    ids = torch.unique(R)
    tbl = base.id_to_root_table
    mapped = tbl[P.clamp(0, tbl.numel() - 1)]
    same_pairs = ((R.unsqueeze(-1) == R.unsqueeze(-2)) & (R.unsqueeze(-1) != 0))
    off = torch.triu(torch.ones(R.shape[1], R.shape[1], dtype=torch.bool), 1)
    OUT['D_id_spaces'] = {
        'native_root_embed_rows': int(rows), 'vocab_num_roots': int(vocab.num_roots),
        'rca_table_shape': list(model.morphemic_embed.root_embed.weight.shape),
        'positions': int(R.numel()), 'nonzero_root_ids': int((R != 0).sum()),
        'positions_ge_9015': int((R >= rows).sum()),
        'distinct_ids': int(ids.numel()), 'distinct_ids_ge_9015': int((ids >= rows).sum()),
        'token_stream_max_id': int(P.max()),
        'token_stream_nonzero_roots': int((mapped != 0).sum()),
        'token_stream_frac_nonzero': float((mapped != 0).float().mean()),
        'same_root_pair_frac': float(same_pairs[:, ~off].float().mean()),
    }
    print(f"\n=== D. ROOT-ID SPACES ===")
    print(f"  native root_embed rows={rows} (hardcoded 9015)  vocab num_roots={vocab.num_roots}  "
          f"RCA K/V table={tuple(model.morphemic_embed.root_embed.weight.shape)}")
    print(f"  {int(R.numel())} positions, {int((R!=0).sum())} nonzero root ids")
    print(f"  ids >= {rows} alias to row {rows-1}: {int((R>=rows).sum())} positions "
          f"({100*float((R>=rows).float().mean()):.2f} %); {int((ids>=rows).sum())}/{int(ids.numel())} "
          f"distinct ids ALIASED")
    print(f"  fraction of causal pairs with r_i == r_j != 0: "
          f"{100*OUT['D_id_spaces']['same_root_pair_frac']:.4f} %")
    print(f"  extract_morphemic_ids(token stream): {int((mapped!=0).sum())}/{mapped.numel()} "
          f"nonzero ({100*OUT['D_id_spaces']['token_stream_frac_nonzero']:.4f} %)")

    # ---------------- B: full-model decomposition, RMS-relative ----------------
    print(f"\n=== B. FORCE-ENABLE, DECOMPOSED (RMS-relative effect sizes) ===")
    h_all, _ = run(R)
    OUT['B_all_on'] = eff(h_all, h_off)
    print(f"  all ON (bias+bonus+Pillars) : RMS(dh)/RMS(h) = "
          f"{100*OUT['B_all_on']['rms_rel']:.4f} %   max|dh| = {OUT['B_all_on']['max_dh']:.4e}")

    def bulk(**kw):
        for L in layers:
            for k, v in kw.items():
                getattr(L.self_attn, k).data.fill_(v)

    def sm1(v):
        for L in layers:
            L.self_attn.stream_mix.data[1] = v

    variants = {}
    bulk(ishtiqaq_gamma=0.0, coverage_weight=0.0, governance_strength=0.0); sm1(0.25)
    variants['score_bias_only'] = run(R)[0]
    bulk(ishtiqaq_gamma=0.25, coverage_weight=0.0, governance_strength=0.0); sm1(0.0)
    variants['identical_root_bonus_only'] = run(R)[0]
    bulk(ishtiqaq_gamma=0.0, coverage_weight=2.0, governance_strength=1.5); sm1(0.0)
    variants['pillar_I_II_only'] = run(R)[0]
    # EXACTLY what the combined arm switches on: both root SCORE terms, no Pillars
    bulk(ishtiqaq_gamma=0.25, coverage_weight=0.0, governance_strength=0.0); sm1(0.25)
    variants['root_terms_NO_PILLARS'] = run(R)[0]
    # and with the model's OWN token->root table (98.4 % zeros), Pillars still off
    tok0 = tbl[P.clamp(0, tbl.numel() - 1)]
    variants['token_table_root_terms_only'] = run(tok0)[0]
    bulk(ishtiqaq_gamma=0.25, coverage_weight=2.0, governance_strength=1.5); sm1(0.25)

    for k, h in variants.items():
        e = eff(h, h_off)
        OUT.setdefault('B_decomposed', {})[k] = e
        print(f"  {k:26s}: RMS(dh)/RMS(h) = {100*e['rms_rel']:.4f} %   "
              f"max|dh| = {e['max_dh']:.4e}   max_rel = {100*e['max_rel']:.3f} %")

    # what the model's OWN wiring would do (token -> root table, 98 % zeros)
    tok = tbl[P.clamp(0, tbl.numel() - 1)]
    h_tokroot, _ = run(tok)
    e = eff(h_tokroot, h_off)
    OUT['B_token_table_wiring'] = e
    print(f"  {'via extract_morphemic_ids':26s}: RMS(dh)/RMS(h) = {100*e['rms_rel']:.6f} %   "
          f"max|dh| = {e['max_dh']:.4e}")

    # ---------------- B2/C: layer-0 exact analysis on the REAL layer-0 input ----------------
    print(f"\n=== B2. RAW MAGNITUDES (nats) AND ATTENTION SHIFT, LAYER 0 ===")
    L0 = layers[0].self_attn
    h_in = emb
    B, T, _ = h_in.shape
    try:
        pos_ids = torch.arange(T, device=dev).unsqueeze(0)
        pos = model.backbone.rotary_emb(h_in, pos_ids)
    except Exception as e:
        print(f'    (rotary_emb unavailable: {e})')
        pos = None
    with torch.no_grad():
        o_off, w_off = _orig(L0, h_in, position_embeddings=pos, root_ids=None)
        o_on, w_on = _orig(L0, h_in, position_embeddings=pos, root_ids=R)
        safe = R.clamp(0, L0.root_embed.num_embeddings - 1)
        remb = L0.root_embed(safe)
        q_r = L0.root_q_proj(remb).view(B, T, L0.num_heads, L0.head_dim).transpose(1, 2).float()
        k_r = L0.root_k_proj(remb).view(B, T, L0.num_kv_heads, L0.head_dim).transpose(1, 2).float()
        k_r = k_r.repeat_interleave(L0.num_kv_groups, dim=1)
        s_root = torch.matmul(q_r, k_r.transpose(-2, -1)) * L0.scale
        has = (safe != 0).float()
        cond = (has.unsqueeze(-1) * has.unsqueeze(-2)).unsqueeze(1)
        bias = 0.25 * cond * s_root
        bonus = ((safe.unsqueeze(-1) == safe.unsqueeze(-2)) &
                 (safe.unsqueeze(-1) != 0)).float().unsqueeze(1) * 0.25
        q_s = L0.q_norm(L0.q_proj(h_in).view(B, T, L0.num_heads, L0.head_dim)).transpose(1, 2).float()
        k_s = L0.k_norm(L0.k_proj(h_in).view(B, T, L0.num_kv_heads, L0.head_dim)).transpose(1, 2).float()
        k_s = k_s.repeat_interleave(L0.num_kv_groups, dim=1)
        s_surf = torch.matmul(q_s, k_s.transpose(-2, -1)) * L0.scale
        m = ~off.bool()                      # [T,T] causal-future mask, broadcast on the last dim
        def sel(t):
            return t[..., m]
        cw = 0.75  # stream_mix[0]
        OUT['B2_nats'] = {
            'surface_score_rms': rms(sel(s_surf)),
            'surface_score_absmax': float(sel(s_surf).abs().max()),
            'root_score_raw_rms': rms(sel(s_root)),
            'root_bias_rms': rms(sel(bias)),
            'root_bias_absmax': float(sel(bias).abs().max()),
            'bonus_absmax': float(sel(bonus).abs().max()),
            'bias_over_surface_rms': rms(sel(bias)) / max(1e-30, rms(sel(s_surf))),
            'note_stream_mix0_scales_surface': cw,
        }
        for k, v in OUT['B2_nats'].items():
            print(f"    {k:28s} = {v:.6g}")
        dw = (w_on.float() - w_off.float())
        OUT['C_attn_shift'] = {
            'max_abs_delta_attn': float(dw.abs().max()),
            'rms_delta_attn': rms(dw),
            'mean_max_attn_off': float(w_off.float().max(-1).values.mean()),
            'mean_max_attn_on': float(w_on.float().max(-1).values.mean()),
        }
        for k, v in OUT['C_attn_shift'].items():
            print(f"    {k:28s} = {v:.6g}")
        # EXACT reconstruction: whole delta == o_proj((attn_on - attn_off) @ v)
        v = L0.v_proj(h_in).view(B, T, L0.num_kv_heads, L0.head_dim).transpose(1, 2)
        v = v.repeat_interleave(L0.num_kv_groups, dim=1).float()
        recon = L0.o_proj((dw @ v).transpose(1, 2).contiguous().view(
            B, T, L0.num_heads * L0.head_dim).to(L0.o_proj.weight.dtype)).float()
        raw = (o_on.float() - o_off.float())
        resid = (recon - raw)
        OUT['C_reconstruction'] = {
            'max_abs_raw_delta': float(raw.abs().max()),
            'max_abs_residual_after_recon': float(resid.abs().max()),
            'rel_residual': float(resid.abs().max() / max(1e-30, float(raw.abs().max()))),
            'v_rms': rms(v),
        }
        print(f"  === C. EXACT RECONSTRUCTION (layer 0) ===")
        for k, v_ in OUT['C_reconstruction'].items():
            print(f"    {k:34s} = {v_:.6g}")
        # root-sequence sensitivity of the native term
        Rr = torch.roll(R, 1, dims=1); Rr[:, 0] = R[:, 0]
        o_roll, _ = _orig(L0, h_in, position_embeddings=pos, root_ids=Rr)
        OUT['C_roll_sensitivity'] = {'max_abs_delta_vs_real_roots':
                                     float((o_roll.float() - o_on.float()).abs().max())}
        print(f"    max|o(roots) - o(rolled roots)|        = "
              f"{OUT['C_roll_sensitivity']['max_abs_delta_vs_real_roots']:.6g}")

    # ---------------- E/F ----------------
    print(f"\n=== E/F. CHECKPOINT WEIGHTS ===")
    tbls = [layers[i].self_attn.root_embed.weight.float() for i in range(nL)]
    ident = [bool(torch.equal(tbls[i], tbls[0])) for i in range(nL)]
    m_ = model.morphemic_embed.root_embed.weight.float()
    rn = tbls[0].norm(dim=-1)
    exp_mean = math.sqrt(2.0) * math.exp(math.lgamma(32.5) - math.lgamma(32.0))
    OUT['E_tables'] = {
        'n': nL, 'shape': list(tbls[0].shape), 'n_identical_to_layer0': int(sum(ident)),
        'l0_std': float(tbls[0].std()), 'l0_absmax': float(tbls[0].abs().max()),
        'l0_mean': float(tbls[0].mean()), 'l0_row0_norm': float(tbls[0][0].norm()),
        'row_norm_mean': float(rn.mean()), 'row_norm_std': float(rn.std()),
        'chi64_expected_mean': float(exp_mean),
        'at_default_N01_init': bool(sum(ident) == nL and abs(rn.mean() - exp_mean) < 0.05),
        'rca_table_shape': list(m_.shape), 'rca_table_std': float(m_.std()),
        'n_root_ids_never_reachable': int(max(0, vocab.num_roots - tbls[0].shape[0])),
        'root_q_proj_rms': rms(layers[0].self_attn.root_q_proj.weight),
        'root_k_proj_rms': rms(layers[0].self_attn.root_k_proj.weight),
    }
    print(f"  24 x root_embed {tuple(tbls[0].shape)}: identical to layer 0 = {sum(ident)}/{nL}")
    print(f"    L0 std={tbls[0].std():.4f} absmax={tbls[0].abs().max():.4f} "
          f"row-norm mean={rn.mean():.4f} vs chi_64={exp_mean:.4f} "
          f"-> {'AT DEFAULT N(0,1) INIT (never trained)' if abs(rn.mean()-exp_mean)<0.05 else 'DRIFTED'}")
    print(f"    RCA table {tuple(m_.shape)} std={m_.std():.4f}")
    print(f"    root ids with no native row: {OUT['E_tables']['n_root_ids_never_reachable']} "
          f"({100*OUT['E_tables']['n_root_ids_never_reachable']/vocab.num_roots:.2f} %)")
    sm = [tuple(round(float(x), 6) for x in layers[i].self_attn.stream_mix.float().tolist())
          for i in range(nL)]
    gm = [round(float(layers[i].self_attn.ishtiqaq_gamma.float()), 6) for i in range(nL)]
    cwv = [round(float(layers[i].self_attn.coverage_weight.float()), 6) for i in range(nL)]
    gsv = [round(float(layers[i].self_attn.governance_strength.float()), 6) for i in range(nL)]
    OUT['F_scalars'] = {'stream_mix_distinct': sorted(set(sm)), 'gamma_distinct': sorted(set(gm)),
                        'coverage_weight_distinct': sorted(set(cwv)),
                        'governance_strength_distinct': sorted(set(gsv))}
    print(f"  stream_mix distinct={sorted(set(sm))} (init [0.75,0.25])")
    print(f"  ishtiqaq_gamma distinct={sorted(set(gm))} (init 0.25)")
    print(f"  coverage_weight distinct={sorted(set(cwv))} (init 0.0)")
    print(f"  governance_strength distinct={sorted(set(gsv))} (init 0.0)")

    OUT['checkpoint'] = args.checkpoint
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(OUT, f, indent=2)
        print(f'\n[*] wrote {args.out}')


if __name__ == '__main__':
    main()
