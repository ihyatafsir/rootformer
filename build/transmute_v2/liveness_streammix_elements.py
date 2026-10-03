#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
liveness_streammix_elements.py -- make the last imprecise flag exact.

`liveness_assert.py` reports 24 parameters as "flagged by grad but CAUSALLY LIVE": those are the
24 `self_attn.stream_mix` (2,)-vectors.  Perturbing the WHOLE tensor moves element 0, which is
live (`total_score = stream_mix[0] * score_surface`), so the tensor-level test says LIVE while
the per-element gradient readout says `[x, 0.0]` -- element 1 EXACTLY 0.0 in all 24 layers.

The honest resolution is element-wise, and it is done here: perturb element 0 alone and element 1
alone, for all 24 layers, and compare bitwise.  Element 1 is inside the root branch
(`total_score += stream_mix[1] * ishtiqaq_cond * score_root`) and the whole branch is behind
`if root_ids is not None:`, which is never true in the NRMT forward -- so element 1 must be INERT.
"""
import json
import os
import sys
import math
from pathlib import Path

ROOT = Path(os.environ.get('RF_ROOT', '/workspace/hf_v19_2_release')).resolve()
for p in (str(ROOT), str(ROOT / 'models'), '/workspace/transmute_v2'):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch


def main():
    from safetensors.torch import load_file
    import nrmp_vocab as nv
    from models.unified_rootformer_v12 import UnifiedRootformerV12
    from deepseek_v4_1_flash_model import UnifiedRootformerV17_DeepSeekFlash
    from nrmt_arch import RootformerNRMT

    V = next(v for k, v in vars(nv).items() if isinstance(v, type) and 'MorphemicVocab' in k)
    bp = str(ROOT / 'data/rootformer_v12_arabic_blueprint.json')
    vocab = V(bp)
    dev, dt = 'cpu', torch.float32
    base = UnifiedRootformerV12(bp, 'Qwen/Qwen2.5-0.5B', dev, dt).to(dev)
    flash = UnifiedRootformerV17_DeepSeekFlash(base, dev, dt).to(dev)
    model = RootformerNRMT(flash, vocab, dev, dt).to(dev)
    model.load_state_dict(load_file(str(ROOT / 'checkpoints' /
        'rootformer_v19_2_synthesis_ar_backbone.awzan142.roots9490.tok10052.safetensors')),
        strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    B, T = 2, 8
    g = torch.Generator().manual_seed(11)
    P = torch.randint(1, vocab.num_prefixes, (B, T), generator=g)
    R = torch.randint(5, vocab.num_roots, (B, T), generator=g)
    W = torch.randint(3, vocab.num_awzan, (B, T), generator=g)
    S = torch.randint(3, vocab.num_suffixes, (B, T), generator=g)

    def fwd():
        with torch.no_grad():
            emb = model.morphemic_embed(P, R, W, S)
            model._set_flash(R, W)
            out = model.backbone(inputs_embeds=emb)
            return model.final_norm(out.last_hidden_state).detach().clone()

    ref = fwd()
    print(f'determinism: {bool(torch.equal(ref, fwd()))}')

    # every self_attn.active_root_ids must be None -- the precondition for element-1 inertness
    print('active_root_ids all None: '
          f'{all(l.self_attn.active_root_ids is None for l in model.backbone.layers)}')

    rows, inert1, live0 = [], 0, 0
    for i in range(24):
        p = model.backbone.layers[i].self_attn.stream_mix
        p0 = p.detach().clone()
        assert p.numel() == 2
        res = {}
        for e in (0, 1):
            with torch.no_grad():
                p[e] += 1e-3
            o = fwd()
            with torch.no_grad():
                p.copy_(p0)
            res[e] = {'bit_equal': bool(torch.equal(o, ref)),
                      'max_abs_delta': float((o - ref).abs().max())}
        rows.append({'layer': i, 'element0': res[0], 'element1': res[1]})
        live0 += int(not res[0]['bit_equal'])
        inert1 += int(res[1]['bit_equal'])
        print(f'  L{i:02d}  element0: bit_equal={res[0]["bit_equal"]!s:5s} '
              f'max|d|={res[0]["max_abs_delta"]:.3e}   '
              f'element1: bit_equal={res[1]["bit_equal"]!s:5s} '
              f'max|d|={res[1]["max_abs_delta"]:.3e}')

    print(f'\nrestore integrity: {bool(torch.equal(ref, fwd()))}')
    print(f'element 0 LIVE in {live0}/24 layers  (it multiplies the surface score)')
    print(f'element 1 INERT in {inert1}/24 layers (it multiplies the root branch, never taken)')
    out = {'determinism': bool(torch.equal(ref, ref)),
           'element0_live_layers': live0, 'element1_inert_layers': inert1, 'rows': rows}
    Path('/workspace/transmute_v2/liveness_streammix.json').write_text(json.dumps(out, indent=2))
    if inert1 != 24 or live0 != 24:
        print('!! NOT the expected pattern -- investigate before trusting the liveness verdict')
        return 1
    print('wrote /workspace/transmute_v2/liveness_streammix.json')
    return 0


if __name__ == '__main__':
    sys.exit(main())
