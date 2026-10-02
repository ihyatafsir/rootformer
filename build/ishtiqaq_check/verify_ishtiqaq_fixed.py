#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_ishtiqaq_fixed.py -- PROVE the fixed native root path before spending a GPU on it.

Checks (all on the real checkpoint, CPU, no training):

  1  SHIPPED MODULE UNTOUCHED      md5 of models/ishtiqaq_attention_v12.py.
  2  GATE 0 IS A BITWISE NO-OP     fixed(root_gate=0, root_ids=None) == shipped(root_ids=None)
                                   torch.equal, all 24 layers, real weights, real inputs.
  3  GATE 1 REPRODUCES THE SHIPPED PATH EXACTLY
                                   fixed(root_gate=1, root_ids=R, source='native')
                                   == shipped(root_ids=R) bitwise.
  4  WHOLE-MODEL NO-OP             swap installed, gates 0, active_root_ids None
                                   => final_norm hidden states bitwise == un-swapped trunk.
  5  LIVE AT GATE 1                whole-model max|h(gate1) - h(gate0)| > 0, gate0 == gate-off.
  6  ROOT-SEQUENCE SENSITIVITY     rolled root ids change h at gate 1 and NOT at gate 0.
  7  SHARED SOURCE                 root_source='shared' (448-dim released table) is also a
                                   bitwise no-op at gate 0 and live at gate 1.
  8  STILL A PURE SCORE BIAS       reconstruction of the whole delta from
                                   o_proj((attn_on - attn_off) @ v) at gate 1.
"""
import argparse
import hashlib
import json
import sys

import torch

sys.path.insert(0, '/workspace/hf_v19_2_release')
sys.path.insert(0, '/workspace/hf_v19_2_release/models')
sys.path.insert(0, '/workspace/ishtiqaq_check')

WIN = 128


def md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def rms(t):
    return float(t.float().pow(2).mean().sqrt())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--cache', required=True)
    ap.add_argument('--windows', type=int, default=4)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--gates', type=float, nargs=2, default=[1.0, 0.0],
                    help='root_gate pillar_gate used for the LIVE checks')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    OUT = {'pass': True}

    import nrmp_vocab as nv
    from models.unified_rootformer_v12 import UnifiedRootformerV12
    from deepseek_v4_1_flash_model import UnifiedRootformerV17_DeepSeekFlash
    from nrmt_arch import RootformerNRMT
    from models.ishtiqaq_attention_v12 import IshtiqaqAttentionV12
    from ishtiqaq_root_bias import (IshtiqaqRootBiasAttention, swap_in_root_bias,
                                    zero_gates, restore_gates, gate_values)
    from safetensors.torch import load_file

    def check(name, ok, detail=''):
        OUT['pass'] = OUT['pass'] and bool(ok)
        print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
        OUT.setdefault('checks', []).append({'name': name, 'ok': bool(ok), 'detail': str(detail)})
        return ok

    SHIPPED = '/workspace/hf_v19_2_release/models/ishtiqaq_attention_v12.py'
    print('=== 1. SHIPPED MODULE UNTOUCHED ===')
    m = md5(SHIPPED)
    check('shipped ishtiqaq_attention_v12.py md5', m == 'd192ac9f3514f2a7f831a5ff03fe93b8', m)

    dev = torch.device(args.device)
    V = next(v for k, v in vars(nv).items() if isinstance(v, type) and 'MorphemicVocab' in k)
    BP = '/workspace/hf_v19_2_release/data/rootformer_v12_arabic_blueprint.json'
    vocab = V(BP)
    base = UnifiedRootformerV12(BP, 'Qwen/Qwen2.5-0.5B', str(dev), torch.bfloat16).to(dev)
    flash = UnifiedRootformerV17_DeepSeekFlash(base, str(dev), torch.bfloat16).to(dev)
    model = RootformerNRMT(flash, vocab, str(dev), torch.bfloat16).to(dev)
    missing, unexpected = model.load_state_dict(load_file(args.checkpoint), strict=False)
    print(f'  checkpoint missing={len(missing)} unexpected={len(unexpected)}')
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
    with torch.no_grad():
        emb = model.morphemic_embed(P, R, W, S)
    T = emb.shape[1]
    pos = None
    try:
        pos = model.backbone.rotary_emb(emb, torch.arange(T, device=dev).unsqueeze(0))
    except Exception as e:
        print(f'  (rotary_emb unavailable: {e})')

    # ---------------- 2/3. per-layer module equivalence ----------------
    print('=== 2/3. PER-LAYER: gate 0 == shipped(root_ids=None) ; gate 1 == shipped(root_ids=R) ===')
    n_eq0 = n_eq1 = 0
    worst0 = worst1 = 0.0
    native_mods, native_params = swap_in_root_bias(
        layers, list(range(nL)), shared_root_embed=model.morphemic_embed.root_embed,
        root_source='native', gate_init=0.0, pillar_gate_init=0.0)
    for i in range(nL):
        fixed = layers[i].self_attn
        # reconstruct the shipped module from the fixed one's surviving weights
        shipped = IshtiqaqAttentionV12(
            hidden_size=fixed.hidden_size, num_heads=fixed.num_heads,
            num_kv_heads=fixed.num_kv_heads, head_dim=fixed.head_dim,
            num_roots=fixed.root_embed.num_embeddings,
            num_awzan=fixed.wazn_embed.num_embeddings, layer_idx=i,
            dtype=fixed.root_embed.weight.dtype).to(dev)
        sd = {k: v for k, v in fixed.state_dict().items()
              if not k.startswith(('root_gate', 'pillar_gate'))}
        shipped.load_state_dict(sd)
        with torch.no_grad():
            fixed.root_gate.data.fill_(0.0)
            fixed.pillar_gate.data.fill_(0.0)
            o_f0, w_f0 = fixed(emb, position_embeddings=pos, root_ids=None)
            o_s0, w_s0 = shipped(emb, position_embeddings=pos, root_ids=None)
            # gate 1 + roots  ==  shipped + roots
            fixed.root_gate.data.fill_(1.0)
            fixed.pillar_gate.data.fill_(1.0)
            o_f1, w_f1 = fixed(emb, position_embeddings=pos, root_ids=R)
            o_s1, w_s1 = shipped(emb, position_embeddings=pos, root_ids=R)
        e0 = bool(torch.equal(o_f0, o_s0)) and bool(torch.equal(w_f0, w_s0))
        e1 = bool(torch.equal(o_f1, o_s1)) and bool(torch.equal(w_f1, w_s1))
        n_eq0 += e0
        n_eq1 += e1
        worst0 = max(worst0, float((o_f0.float() - o_s0.float()).abs().max()),
                     float((w_f0.float() - w_s0.float()).abs().max()))
        worst1 = max(worst1, float((o_f1.float() - o_s1.float()).abs().max()),
                     float((w_f1.float() - w_s1.float()).abs().max()))
        del shipped
    OUT['per_layer'] = {'n_equal_gate0': n_eq0, 'n_equal_gate1': n_eq1, 'n': nL,
                        'worst_abs_gate0': worst0, 'worst_abs_gate1': worst1}
    check('gate 0 == shipped(root_ids=None), bitwise, all layers', n_eq0 == nL,
          f'{n_eq0}/{nL}, worst|d|={worst0:.3e}')
    check("gate 1 + native source == shipped(root_ids=R), bitwise, all layers", n_eq1 == nL,
          f'{n_eq1}/{nL}, worst|d|={worst1:.3e}')

    # ---------------- 4/5/6. whole-model ----------------
    print('=== 4/5/6. WHOLE-MODEL ABLATION ===')

    def whole(gate, pillar, root_ids_for_layers, active=True):
        for i in range(nL):
            a = layers[i].self_attn
            a.root_gate.data.fill_(float(gate))
            a.pillar_gate.data.fill_(float(pillar))
            a.active_root_ids = root_ids_for_layers if active else None
        with torch.no_grad():
            model._set_flash(R, W)
            h = model.final_norm(model.backbone(inputs_embeds=emb).last_hidden_state)
        for i in range(nL):
            layers[i].self_attn.active_root_ids = None
        return h.float()

    # 4. gates 0, no roots
    for i in range(nL):
        layers[i].self_attn.root_gate.data.fill_(0.0)
        layers[i].self_attn.pillar_gate.data.fill_(0.0)
        layers[i].self_attn.active_root_ids = None
    # reference: rebuild the ORIGINAL trunk by swapping the shipped modules back
    saved_mods = [layers[i].self_attn for i in range(nL)]
    shipped_mods = []
    for i in range(nL):
        fx = saved_mods[i]
        sh = IshtiqaqAttentionV12(
            hidden_size=fx.hidden_size, num_heads=fx.num_heads, num_kv_heads=fx.num_kv_heads,
            head_dim=fx.head_dim, num_roots=fx.root_embed.num_embeddings,
            num_awzan=fx.wazn_embed.num_embeddings, layer_idx=i,
            dtype=fx.root_embed.weight.dtype).to(dev)
        sh.load_state_dict({k: v for k, v in fx.state_dict().items()
                            if not k.startswith(('root_gate', 'pillar_gate'))})
        sh.eval()
        layers[i].self_attn = sh
        shipped_mods.append(sh)
    with torch.no_grad():
        model._set_flash(R, W)
        h_ref = model.final_norm(model.backbone(inputs_embeds=emb).last_hidden_state).float()
    for i in range(nL):
        layers[i].self_attn = saved_mods[i]
    h_g0 = whole(0.0, 0.0, None, active=False)
    check('whole model: gates 0 + no root ids == un-swapped trunk, bitwise',
          bool(torch.equal(h_g0, h_ref)), f'max|d|={float((h_g0-h_ref).abs().max()):.3e}')

    g, pg = float(args.gates[0]), float(args.gates[1])
    h_g1 = whole(g, pg, R, active=True)
    d01 = float((h_g1 - h_g0).abs().max())
    OUT['whole_model'] = {'max_abs_dh_gate1_vs_gate0': d01,
                          'rms_rel': rms(h_g1 - h_g0) / max(1e-30, rms(h_g1)),
                          'gate': g, 'pillar_gate': pg,
                          'max_abs_h': float(h_g1.abs().max())}
    check('whole model: gate 1 is LIVE', d01 > 0, f'max|dh|={d01:.6e}, '
          f'RMS-rel={100*OUT["whole_model"]["rms_rel"]:.4f} %')

    # 6. rolled root sequence
    Rr = torch.roll(R, 1, dims=1)
    Rr[:, 0] = R[:, 0]
    h_g1_roll = whole(g, pg, Rr, active=True)
    d_roll = float((h_g1_roll - h_g1).abs().max())
    h_g0_roll = whole(0.0, 0.0, Rr, active=True)
    d_roll0 = float((h_g0_roll - h_g0).abs().max())
    OUT['roll'] = {'max_abs_dh_gate1_rolled': d_roll, 'max_abs_dh_gate0_rolled': d_roll0,
                   'n_positions_differing': int((Rr != R).sum())}
    check('whole model: gate 1 output DEPENDS on the root sequence', d_roll > 0,
          f'max|dh|={d_roll:.6e} over {int((Rr!=R).sum())} rolled positions')
    check('whole model: gate 0 output is INVARIANT to the root sequence', d_roll0 == 0.0,
          f'max|dh|={d_roll0:.3e}')

    # ---------------- 7. shared source ----------------
    print("=== 7. SHARED 448-dim root source ===")
    mods2, params2 = swap_in_root_bias(
        layers, list(range(nL)), shared_root_embed=model.morphemic_embed.root_embed,
        root_source='shared', gate_init=0.0, pillar_gate_init=0.0)
    h_s0 = whole(0.0, 0.0, R, active=True)
    h_s1 = whole(1.0, 0.0, R, active=True)
    d_s = float((h_s1 - h_s0).abs().max())
    OUT['shared_source'] = {
        'n_trainable_params_total': int(sum(p.numel() for p in params2)),
        'n_trainable_params_per_layer': int(sum(p.numel() for p in params2) // nL),
        'max_abs_dh_gate1_vs_gate0': d_s,
        'rms_rel': rms(h_s1 - h_s0) / max(1e-30, rms(h_s1)),
    }
    check('shared source: gate 0 == un-swapped trunk, bitwise', bool(torch.equal(h_s0, h_ref)),
          f'max|d|={float((h_s0-h_ref).abs().max()):.3e}')
    check('shared source: gate 1 is LIVE', d_s > 0, f'max|dh|={d_s:.6e}')

    # ---------------- 8. still a pure score bias ----------------
    # The trunk runs in bf16, so `o_on - o_off` is a difference of two O(1) bf16 tensors and
    # carries a ~2**-9 quantization floor.  To test the STRUCTURE (not the arithmetic) the same
    # module is rebuilt in float32 and the identity is checked there; the bf16 number is
    # reported alongside so the cancellation floor is visible rather than hidden.
    print('=== 8. THE FIXED PATH IS STILL A PURE SCORE BIAS (exact reconstruction) ===')
    L0 = layers[0].self_attn
    B, Tt, _ = emb.shape
    with torch.no_grad():
        L0.root_gate.data.fill_(1.0)
        L0.pillar_gate.data.fill_(0.0)
        o_off, w_off = L0(emb, position_embeddings=pos, root_ids=None)
        o_on, w_on = L0(emb, position_embeddings=pos, root_ids=R)
        v16 = L0.v_proj(emb).view(B, Tt, L0.num_kv_heads, L0.head_dim).transpose(1, 2)
        v16 = v16.repeat_interleave(L0.num_kv_groups, dim=1).float()
        dw16 = w_on.float() - w_off.float()
        recon16 = L0.o_proj((dw16 @ v16).transpose(1, 2).contiguous().view(
            B, Tt, L0.num_heads * L0.head_dim).to(L0.o_proj.weight.dtype)).float()
        raw16 = o_on.float() - o_off.float()
        rel16 = float((recon16 - raw16).abs().max() / max(1e-30, float(raw16.abs().max())))

        # float32 rebuild of the same layer, same weights
        f32 = IshtiqaqRootBiasAttention(
            hidden_size=L0.hidden_size, num_heads=L0.num_heads, num_kv_heads=L0.num_kv_heads,
            head_dim=L0.head_dim, num_roots=L0.root_embed.num_embeddings,
            num_awzan=L0.wazn_embed.num_embeddings, layer_idx=0, dtype=torch.float32,
            root_source=L0.root_source,
            shared_root_embed=getattr(L0, '_shared_root_embed', None)).to(dev)
        f32.load_state_dict({k: v.float() for k, v in L0.state_dict().items()}, strict=False)
        f32.root_gate.data.fill_(1.0)
        f32.pillar_gate.data.fill_(0.0)
        f32.eval()
        e32 = emb.float()
        pos32 = tuple(t.float() for t in pos) if pos is not None else None
        of_off, wf_off = f32(e32, position_embeddings=pos32, root_ids=None)
        of_on, wf_on = f32(e32, position_embeddings=pos32, root_ids=R)
        v32 = f32.v_proj(e32).view(B, Tt, f32.num_kv_heads, f32.head_dim).transpose(1, 2)
        v32 = v32.repeat_interleave(f32.num_kv_groups, dim=1)
        dw32 = wf_on - wf_off
        recon32 = f32.o_proj((dw32 @ v32).transpose(1, 2).contiguous().view(
            B, Tt, f32.num_heads * f32.head_dim))
        raw32 = of_on - of_off
        rel32 = float((recon32 - raw32).abs().max() / max(1e-30, float(raw32.abs().max())))
    OUT['reconstruction'] = {
        'root_source_of_layer0': L0.root_source,
        'max_abs_raw_delta_bf16': float(raw16.abs().max()),
        'rel_residual_bf16_cancellation_floor': rel16,
        'bf16_eps': 2.0 ** -9,
        'max_abs_raw_delta_fp32': float(raw32.abs().max()),
        'rel_residual_fp32': rel32,
        'max_abs_delta_attn': float(dw32.abs().max()),
    }
    check('delta == o_proj((attn_on - attn_off) @ v)  [float32]', rel32 < 1e-4,
          f'rel residual {rel32:.3e}, max|d attn| {float(dw32.abs().max()):.3e}; '
          f'bf16 cancellation floor {rel16:.3e} (raw bf16 delta {float(raw16.abs().max()):.3e})')

    n_par = sum(p.numel() for p in native_params)
    print(f'\n[*] native root path trainable params (all 24 layers, native source): '
          f'{n_par/1e6:.4f}M')
    OUT['native_params_all_layers'] = int(n_par)
    print(f"[*] OVERALL: {'ALL CHECKS PASS' if OUT['pass'] else 'FAILURES PRESENT'}")
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(OUT, f, indent=2)
        print(f'[*] wrote {args.out}')
    return 0 if OUT['pass'] else 1


if __name__ == '__main__':
    sys.exit(main())
