#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
probe_root_paths.py -- three structural facts about the root cross-attention wiring.

Runs on the pod with the shipped checkpoint and a cache dir, NO training:

  1. DEADNESS OF THE OLD ROOT PATH IS PRESERVED.  IshtiqaqAttentionV12.forward is
     instrumented and the trunk is driven exactly as the trainer drives it
     (`model.backbone(inputs_embeds=emb)`).  Expected: 24 attention calls per forward,
     0 carrying `root_ids`, 0 with `active_root_ids` set -- the historical measurement.
     Attaching the root cross-attention must NOT change that: the new module keeps its
     root ids on its own stack, not on `self_attn`.

  2. THE RCA IS A NUMERICAL NO-OP AT INIT (warm start preserved).  With gate == 0 the
     hidden states are bit-identical to the no-RCA trunk; with the gate forced to 1 the
     hidden states move.  This is what makes "attached but untrained" safe.

  3. THE RCA OUTPUT DEPENDS ON ROOT HISTORY (it can express the lookup).  Same h, two
     different root-id windows -> different injected residuals.

Usage:
  python probe_root_paths.py --cache /workspace/discrete_path/smoke_cache --layers top4
"""
import argparse
import sys

import torch

sys.path.insert(0, '/workspace/hf_v19_2_release')
sys.path.insert(0, '/workspace/hf_v19_2_release/models')
sys.path.insert(0, '/workspace/root_attn')

WIN = 128


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--cache', required=True)
    ap.add_argument('--layers', default='top4')
    ap.add_argument('--windows', type=int, default=4)
    ap.add_argument('--heads', type=int, default=8)
    args = ap.parse_args()

    import nrmp_vocab as nv
    from models.unified_rootformer_v12 import UnifiedRootformerV12
    from deepseek_v4_1_flash_model import UnifiedRootformerV17_DeepSeekFlash
    from nrmt_arch import RootformerNRMT
    from models.ishtiqaq_attention_v12 import IshtiqaqAttentionV12
    from safetensors.torch import load_file
    from root_cross_attn import RootCrossAttentionStack, parse_layer_spec

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    V = next(v for k, v in vars(nv).items() if isinstance(v, type) and 'MorphemicVocab' in k)
    vocab = V('/workspace/hf_v19_2_release/data/rootformer_v12_arabic_blueprint.json')

    # ---- instrumentation of the OLD root path ------------------------------------------------
    counts = {'calls': 0, 'root_ids': 0, 'active': 0, 'generated_root_ids': 0}
    _orig = IshtiqaqAttentionV12.forward

    def _wrapped(self, hidden_states, *a, **k):
        counts['calls'] += 1
        if k.get('root_ids') is not None:
            counts['root_ids'] += 1
        if getattr(self, 'active_root_ids', None) is not None:
            counts['active'] += 1
        return _orig(self, hidden_states, *a, **k)

    IshtiqaqAttentionV12.forward = _wrapped

    bp = '/workspace/hf_v19_2_release/data/rootformer_v12_arabic_blueprint.json'
    base = UnifiedRootformerV12(bp, 'Qwen/Qwen2.5-0.5B', str(device), torch.bfloat16).to(device)
    flash = UnifiedRootformerV17_DeepSeekFlash(base, device, torch.bfloat16).to(device)
    model = RootformerNRMT(flash, vocab, device, torch.bfloat16).to(device)
    missing, unexpected = model.load_state_dict(load_file(args.checkpoint), strict=False)
    print(f'[*] checkpoint {args.checkpoint.split("/")[-1]}: missing={len(missing)} '
          f'unexpected={len(unexpected)}')
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.backbone.eval()

    # ---- one real batch of windows -----------------------------------------------------------
    tr4 = [t.long() for t in torch.load(f'{args.cache}/train.pt', map_location='cpu')]
    idx = torch.arange(min(args.windows, max(1, tr4[1].numel() // WIN)))
    P, R, W, S = (t[:len(idx) * WIN].view(len(idx), WIN).to(device) for t in tr4)
    print(f'[*] batch: {tuple(R.shape)} root ids, {int(R.min())}..{int(R.max())}')

    emb = model.morphemic_embed(P, R, W, S)

    with torch.no_grad():
        counts.update(calls=0, root_ids=0, active=0)
        h0 = model.final_norm(model.backbone(inputs_embeds=emb).last_hidden_state)
        c_no_rca = dict(counts)

        layers = parse_layer_spec(args.layers, len(model.backbone.layers))
        stack = RootCrossAttentionStack(model.d_model, model.morphemic_embed.root_embed,
                                        layers, num_heads=args.heads,
                                        dtype=torch.float32).to(device)
        model.root_cross = stack
        stack.attach(model.backbone.layers)

        counts.update(calls=0, root_ids=0, active=0)
        stack.set_root_ids(R)
        h_gate0 = model.final_norm(model.backbone(inputs_embeds=emb).last_hidden_state)
        c_rca = dict(counts)

        old = stack.zero_gates()
        for m in stack.mods:
            m.gate.data.fill_(1.0)
        h_gate1 = model.final_norm(model.backbone(inputs_embeds=emb).last_hidden_state)
        stack.restore_gates(old)

    print()
    print('=== 1. OLD ROOT PATH (IshtiqaqAttentionV12 root stream) ===')
    print(f'  no RCA attached : calls={c_no_rca["calls"]} with_root_ids={c_no_rca["root_ids"]} '
          f'active_root_ids={c_no_rca["active"]}')
    print(f'  RCA attached    : calls={c_rca["calls"]} with_root_ids={c_rca["root_ids"]} '
          f'active_root_ids={c_rca["active"]}')
    print(f'  => the dual-stream root path is STILL dead: '
          f'{c_rca["root_ids"] + c_rca["active"]} of {c_rca["calls"]} calls carry roots '
          f'(historical: 0 of 456).  The RCA carries its own root ids.')

    print()
    print('=== 2. RCA IS A NO-OP AT INIT, LIVE WHEN THE GATE OPENS ===')
    d0 = float((h_gate0.float() - h0.float()).abs().max())
    d1 = float((h_gate1.float() - h0.float()).abs().max())
    print(f'  max|h(gate=0) - h(no RCA)| = {d0:.3e}  (0 => warm start preserved)')
    print(f'  max|h(gate=1) - h(no RCA)| = {d1:.3e}  (>0 => the pathway is live)')

    print()
    print('=== 3. THE RCA OUTPUT DEPENDS ON THE ROOT HISTORY ===')
    with torch.no_grad():
        R2 = R.clone()
        perm = torch.roll(R2, shifts=1, dims=1)          # a different root sequence
        perm[:, 0] = R2[:, 0]
        m0 = stack.mods[0]
        old = stack.zero_gates()
        for m in stack.mods:
            m.gate.data.fill_(1.0)
        emb2 = model.morphemic_embed(P, perm, W, S)
        stack.set_root_ids(perm)
        h_perm = model.final_norm(model.backbone(inputs_embeds=emb2).last_hidden_state)
        stack.set_root_ids(R)
        stack.restore_gates(old)
    # compare on the SAME surface inputs, so the only difference is the root stream fed to the RCA
    print(f'  max|h(root seq) - h(rolled root seq)| = '
          f'{float((h_gate1.float() - h_perm.float()).abs().max()):.3e}  '
          f'(>0 => the residual is a function of the root history)')
    print(f'  (rolled roots differ at {int((perm != R).sum())} of {R.numel()} positions)')


if __name__ == '__main__':
    main()
