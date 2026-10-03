#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
proof_init_and_isolation.py -- the two proofs that must pass BEFORE anything is stacked.

PART 1  INIT EQUIVALENCE   the new pathway at gate 0 is BIT-IDENTICAL to no pathway.
                           Not "close": `torch.equal` on the full output, plus max|d| == 0.
PART 2  ISOLATION          the pathway ACTS, and it acts ON THE ROOTS.
                           (a) gate 0 vs gate 1 changes the output;
                           (b) with the TRUNK INPUT HELD FIXED, rolling only the root sequence
                               the pathway reads changes the output -- which is the direct
                               evidence that it is a root pathway and not a second MLP.
                           Part 2(b) is the one that matters: a pathway whose output does not
                           move when the roots move is not reading roots, and stacking it with
                           anything else loses the attribution.

Both mechanisms are proved separately and then together, because the brief asks for BOTH:
  residual   h <- h + gate*LN(o_proj(softmax(Wq h . Wk E(r)) @ Wv E(r)))
  score bias total_score += mix*cond*score_root + identical_root_bonus
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get('RF_ROOT', '/workspace/hf_v19_2_release')).resolve()
for p in (str(ROOT), str(ROOT / 'models'), '/workspace/transmute_v2', '/workspace/root_attn',
          '/workspace/ishtiqaq_check'):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch

from root_space import LIVE_ROOTS, LIVE_AWZAN, LIVE_VOCAB


def banner(s):
    print('\n' + '=' * 78 + f'\n{s}\n' + '=' * 78, flush=True)


def md5(p):
    import hashlib
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=str(ROOT / 'checkpoints' /
                    'rootformer_v19_2_synthesis_ar_backbone.awzan142.roots9490.tok10052.safetensors'))
    ap.add_argument('--schedule', default='all')
    ap.add_argument('--batch', type=int, default=2)
    ap.add_argument('--seq', type=int, default=16)
    ap.add_argument('--out', default='/workspace/transmute_v2/proof_pathway.json')
    args = ap.parse_args()

    import nrmp_vocab as nv
    from unified_rootformer_v13 import UnifiedRootformerV13
    from deepseek_v4_1_flash_model import UnifiedRootformerV17_DeepSeekFlash
    from nrmt_arch import RootformerNRMT
    from safetensors.torch import load_file
    from early_root_path import (parse_layer_spec_v13, attach_early_root_residual,
                                 attach_native_root_bias)

    report = {'ckpt': Path(args.ckpt).name, 'schedule': args.schedule, 'parts': {}}
    banner('0. BUILD (V13, corrected root-id space)')
    V = next(v for k, v in vars(nv).items() if isinstance(v, type) and 'MorphemicVocab' in k)
    bp = str(ROOT / 'data/rootformer_v12_arabic_blueprint.json')
    vocab = V(bp)
    dev, dt = 'cpu', torch.float32
    base = UnifiedRootformerV13(bp, 'Qwen/Qwen2.5-0.5B', dev, dt, vocab=vocab).to(dev)
    flash = UnifiedRootformerV17_DeepSeekFlash(base, dev, dt).to(dev)
    model = RootformerNRMT(flash, vocab, dev, dt).to(dev)
    sd = load_file(args.ckpt)
    stale = [k for k in sd if 'self_attn.root_embed.weight' in k
             or 'self_attn.wazn_embed.weight' in k]
    for k in stale:
        del sd[k]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f'  dropped {len(stale)} stale tables; missing={len(missing)} unexpected={len(unexpected)}')
    if len(stale) != 48:
        raise SystemExit(f'expected 48 stale tables, got {len(stale)}')
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    T = args.seq
    g = torch.Generator().manual_seed(7)
    P = torch.randint(1, vocab.num_prefixes, (args.batch, T), generator=g)
    R = torch.randint(5, LIVE_ROOTS, (args.batch, T), generator=g)
    W = torch.randint(3, LIVE_AWZAN, (args.batch, T), generator=g)
    S = torch.randint(3, vocab.num_suffixes, (args.batch, T), generator=g)
    # a rolled root sequence: SAME trunk input, DIFFERENT roots for the pathway to read
    R_roll = torch.roll(R, shifts=5, dims=1)
    if bool(torch.equal(R_roll, R)):
        raise SystemExit('rolled root sequence is identical -- the test would be vacuous')
    print(f'  root sequence roll: {int((R_roll != R).sum())}/{R.numel()} positions differ')

    def fwd(rca=None, ig_layers=(), root_ids=None):
        """The trainer's LIVE path: morphemic_embed -> backbone(inputs_embeds) -> final_norm."""
        with torch.no_grad():
            emb = model.morphemic_embed(P, R, W, S)
            if rca is not None:
                rca.set_root_ids(R if root_ids is None else root_ids)
            for i in ig_layers:
                model.backbone.layers[i].self_attn.active_root_ids = \
                    (R if root_ids is None else root_ids)
            out = model.backbone(inputs_embeds=emb)
            return model.final_norm(out.last_hidden_state).detach().clone()

    def cmp(a, b):
        return {'bit_equal': bool(torch.equal(a, b)),
                'max_abs_delta': float((a.float() - b.float()).abs().max())}

    layers = parse_layer_spec_v13(args.schedule, len(model.backbone.layers))
    if args.schedule == 'all':
        assert layers == list(range(24)), layers
    print(f'  schedule {args.schedule!r} -> {len(layers)} layers: {layers}')

    # ================================================================= PART 1
    banner('PART 1. INIT EQUIVALENCE -- gate 0 must be BIT-IDENTICAL to no pathway')
    ref_none = fwd()

    # determinism of the reference first
    d = cmp(ref_none, fwd())
    print(f'  reference determinism: bit_equal={d["bit_equal"]} max|d|={d["max_abs_delta"]:.3e}')
    if not d['bit_equal']:
        raise SystemExit('the trunk forward is not bit-deterministic; every claim below is void')

    out = {}

    # ---- 1a. residual injector at gate 0 -----------------------------------------------
    rca, rca_info = attach_early_root_residual(model, model.backbone.layers, layers,
                                              model.morphemic_embed.root_embed,
                                              num_heads=8, out_norm=True)
    print(f'  attached RESIDUAL injector: {rca_info}')
    d = cmp(ref_none, fwd(rca=rca))
    out['residual_gate0'] = d
    print(f'  [1a] residual gate=0 vs NO pathway : bit_equal={d["bit_equal"]} '
          f'max|d|={d["max_abs_delta"]:.3e}')
    if not d['bit_equal']:
        raise SystemExit('RESIDUAL PATHWAY AT GATE 0 IS NOT BIT-IDENTICAL')

    # ---- 1b. native score bias at gate 0 ----------------------------------------------
    ig_mods, ig_params, ig_info = attach_native_root_bias(
        model, model.backbone.layers, layers, model.morphemic_embed.root_embed,
        gate_init=0.0, pillar_gate_init=0.0)
    print(f'  attached SCORE BIAS: {ig_info}')
    d = cmp(ref_none, fwd(rca=rca, ig_layers=layers))
    out['score_bias_gate0'] = d
    print(f'  [1b] score bias gate=0 vs NO pathway: bit_equal={d["bit_equal"]} '
          f'max|d|={d["max_abs_delta"]:.3e}')
    if not d['bit_equal']:
        raise SystemExit('SCORE BIAS AT GATE 0 IS NOT BIT-IDENTICAL')

    # ---- 1c. both at gate 0 -----------------------------------------------------------
    d = cmp(ref_none, fwd(rca=rca, ig_layers=layers))
    out['both_gate0'] = d
    print(f'  [1c] BOTH at gate 0 vs NO pathway  : bit_equal={d["bit_equal"]} '
          f'max|d|={d["max_abs_delta"]:.3e}')
    if not d['bit_equal']:
        raise SystemExit('BOTH PATHWAYS AT GATE 0 ARE NOT BIT-IDENTICAL')
    report['parts']['init_equivalence'] = out

    # ================================================================= PART 2
    banner('PART 2. ISOLATION -- does the pathway ACT, and does it act ON THE ROOTS?')

    # ---- 2a. residual: gate 0 vs gate 1, then roll the roots it reads ------------------
    out2 = {}
    with torch.no_grad():
        [m.gate.fill_(1.0) for m in rca.mods]
    o_on = fwd(rca=rca)
    d = cmp(ref_none, o_on)
    out2['residual_gate1_vs_none'] = d
    print(f'  [2a] residual gate=1 vs gate=0    : bit_equal={d["bit_equal"]} '
          f'max|d|={d["max_abs_delta"]:.3e}  <- MUST differ')
    if d['bit_equal']:
        raise SystemExit('RESIDUAL GATE 1 IS INERT: the pathway does not act')
    d = cmp(o_on, fwd(rca=rca, root_ids=R_roll))
    out2['residual_gate1_root_roll'] = d
    print(f'  [2b] residual gate=1, ROOTS ROLLED, trunk input fixed: bit_equal='
          f'{d["bit_equal"]} max|d|={d["max_abs_delta"]:.3e}  <- MUST differ')
    if d['bit_equal']:
        raise SystemExit('RESIDUAL PATHWAY DOES NOT READ THE ROOT SEQUENCE')
    # and at gate 0 the roll must be inert -- the pathway is the ONLY thing that moved
    with torch.no_grad():
        [m.gate.fill_(0.0) for m in rca.mods]
    d = cmp(fwd(rca=rca), fwd(rca=rca, root_ids=R_roll))
    out2['residual_gate0_root_roll'] = d
    print(f'  [2c] residual gate=0, ROOTS ROLLED: bit_equal={d["bit_equal"]} '
          f'max|d|={d["max_abs_delta"]:.3e}  <- MUST be identical')
    if not d['bit_equal']:
        raise SystemExit('at gate 0 the rolled roots changed the output -- the gate is leaking')

    # ---- 2d. score bias: gate 0 vs gate 1, then roll --------------------------------
    with torch.no_grad():
        [m.root_gate.fill_(1.0) for m in ig_mods]
    ig_on = fwd(rca=rca, ig_layers=layers)
    d = cmp(ref_none, ig_on)
    out2['score_bias_gate1_vs_none'] = d
    print(f'  [2d] score bias gate=1 vs gate=0  : bit_equal={d["bit_equal"]} '
          f'max|d|={d["max_abs_delta"]:.3e}  <- MUST differ')
    if d['bit_equal']:
        raise SystemExit('SCORE BIAS GATE 1 IS INERT: the native path does not act')
    d = cmp(ig_on, fwd(rca=rca, ig_layers=layers, root_ids=R_roll))
    out2['score_bias_gate1_root_roll'] = d
    print(f'  [2e] score bias gate=1, ROOTS ROLLED, trunk input fixed: bit_equal='
          f'{d["bit_equal"]} max|d|={d["max_abs_delta"]:.3e}  <- MUST differ')
    if d['bit_equal']:
        raise SystemExit('SCORE BIAS DOES NOT READ THE ROOT SEQUENCE')
    with torch.no_grad():
        [m.root_gate.fill_(0.0) for m in ig_mods]
        [m.pillar_gate.fill_(0.0) for m in ig_mods]
    d = cmp(fwd(rca=rca, ig_layers=layers), fwd(rca=rca, ig_layers=layers, root_ids=R_roll))
    out2['score_bias_gate0_root_roll'] = d
    print(f'  [2f] score bias gate=0, ROOTS ROLLED: bit_equal={d["bit_equal"]} '
          f'max|d|={d["max_abs_delta"]:.3e}  <- MUST be identical')
    if not d['bit_equal']:
        raise SystemExit('at gate 0 the rolled roots changed the output -- the gate is leaking')

    # ---- 2g. the two mechanisms are COMPLEMENTARY, not the same function ------------
    # residual ON, bias OFF  vs  residual OFF, bias ON  vs both OFF
    with torch.no_grad():
        [m.gate.fill_(1.0) for m in rca.mods]
        [m.root_gate.fill_(0.0) for m in ig_mods]
    only_res = fwd(rca=rca, ig_layers=layers)
    with torch.no_grad():
        [m.gate.fill_(0.0) for m in rca.mods]
        [m.root_gate.fill_(1.0) for m in ig_mods]
    only_bias = fwd(rca=rca, ig_layers=layers)
    with torch.no_grad():
        [m.gate.fill_(1.0) for m in rca.mods]
    both = fwd(rca=rca, ig_layers=layers)
    d_ab = cmp(only_res, only_bias)
    d_sum = cmp(both, (only_res + only_bias - ref_none.float().to(only_res.dtype)))
    out2['complementarity'] = {
        'residual_only_vs_bias_only': d_ab,
        'both_vs_sum_of_parts': d_sum,
        'residual_effect_rms': float((only_res - ref_none).pow(2).mean().sqrt()),
        'bias_effect_rms': float((only_bias - ref_none).pow(2).mean().sqrt()),
    }
    print(f'  [2g] residual-only vs bias-only: bit_equal={d_ab["bit_equal"]} '
          f'max|d|={d_ab["max_abs_delta"]:.3e} (different mechanisms -> MUST differ)')
    print(f'       residual effect RMS={out2["complementarity"]["residual_effect_rms"]:.4e}, '
          f'bias effect RMS={out2["complementarity"]["bias_effect_rms"]:.4e}')
    print(f'       both vs (residual+bias-none): max|d|={d_sum["max_abs_delta"]:.3e} '
          f'(not additive -> they interact, which is the point of building both)')
    if d_ab['bit_equal']:
        raise SystemExit('the two mechanisms compute the SAME function -- not complementary')
    report['parts']['isolation'] = out2

    report['new_params'] = {'residual_M': rca_info['new_params_M'],
                            'score_bias_M': ig_info['new_params_M'],
                            'schedule_layers': layers}
    Path(args.out).write_text(json.dumps(report, indent=2))
    banner('VERDICT')
    print('  PART 1 PASS -- both pathways are BIT-IDENTICAL to no pathway at gate 0.')
    print('  PART 2 PASS -- both pathways ACT (gate 1 changes the output), both READ THE ROOT')
    print('               SEQUENCE (rolling the roots with the trunk input fixed changes the')
    print('               output), and at gate 0 the roll is inert.')
    print(f'  wrote {args.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
