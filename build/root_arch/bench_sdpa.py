#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bench_sdpa.py -- Phase 0a measurement: what does the SDPA rewrite actually buy?

THE POINT OF THE T SWEEP
------------------------
The NRMT trainer windows at WIN=128, and at T=128 the eager `[B,14,T,T]` matrix is only 14.7 MB at
B=32 -- small next to the MLP activations, so the SDPA win there is real but modest.  The two OOMs
on record were at **T ~ 251-592** (the brief's window-token p50/p99/max), where the SAME matrix is
(512/128)^2 = 16x larger and becomes the dominant term.  So the bench measures BOTH:

    T=128   the trainer's actual shape
    T=512   the shape that OOM'd

and for each, walks the batch up until the card says no.  Same v13 trunk, same weights, same loss;
the ONLY variable is the attention implementation.  `max_memory_allocated` is the allocator peak
over the whole fwd+bwd step, which is the quantity that OOM'd.

GUARD: before each measurement the free VRAM is checked against a projection from the previous
measurement at the same (mode, T).  An OOM in THIS process is harmless to a resident arm (CUDA OOM
is process-local), but the guard means we should essentially never cause one.
"""
import json
import sys
import time

import torch

sys.path.insert(0, '/workspace/root_arch')
import model_build as MB
import sdpa_attention as SA

DEV = 'cuda'
OUT = {}
MODES = ('eager', 'sdpa', 'sdpa+root')
CONFIGS = [(128, (4, 8, 16, 32, 64)), (512, (2, 4, 8, 16))]
CKPTS = (False, True)


def log(*a):
    print(*a, flush=True)


def free_gb():
    f, t = torch.cuda.mem_get_info()
    return f / 2 ** 30, t / 2 ** 30


vocab, base, flash, model = MB.build()
layers = model.backbone.layers
share = model.morphemic_embed.root_embed
fg, tg = free_gb()
log('[*] VRAM free %.2f / %.2f GB  (a resident arm holds the rest; budget for it)' % (fg, tg))
_tr = [t.long() for t in torch.load(MB.CACHE + '/train.pt', map_location='cpu')]


def make_batch(B, T):
    """A real (P,R,W,S) batch of shape [B,T] sliced from the corpus streams."""
    if T == 128:
        return tuple(t[:B * T].view(B, T).to(DEV) for t in _tr[:4])
    reps = (T + 127) // 128
    return tuple(t[:B * 128].view(B, 128).repeat(1, reps)[:, :T].to(DEV) for t in _tr[:4])


def set_mode(mode, gate=1.0):
    SA.set_dtype_mode('bf16')
    for l in layers:
        old = l.self_attn
        cls = {'eager': SA._ORIG_BASE, 'sdpa': SA.SdpaIshtiqaqAttention,
               'sdpa+root': SA.SdpaIshtiqaqRootBiasAttention}[mode]
        kw = (dict(root_source='shared', shared_root_embed=share, gate_init=gate,
                   pillar_gate_init=0.0) if mode == 'sdpa+root' else {})
        new = cls(hidden_size=old.hidden_size, num_heads=old.num_heads,
                  num_kv_heads=old.num_kv_heads, head_dim=old.head_dim,
                  num_roots=old.root_embed.num_embeddings,
                  num_awzan=old.wazn_embed.num_embeddings,
                  ishtiqaq_init_strength=float(old.ishtiqaq_gamma.detach()),
                  dropout=float(old.dropout.p), layer_idx=old.layer_idx,
                  dtype=old.root_embed.weight.dtype, **kw).to(DEV)
        _nsd = new.state_dict()
        keep = {k: v for k, v in old.state_dict().items()
                if k in _nsd and tuple(v.shape) == tuple(_nsd[k].shape)}
        new.load_state_dict(keep, strict=False)
        if mode == 'sdpa+root':
            new.root_embed.weight.requires_grad_(False)
        l.self_attn = new


def run_step(B, T, mode, ckpt, iters=3):
    set_mode(mode)
    model.train()
    if ckpt:
        model.backbone.gradient_checkpointing_enable()
    else:
        model.backbone.gradient_checkpointing_disable()
    P, R, W, S = make_batch(B, T)
    if mode == 'sdpa+root':
        MB.set_active_root_ids(model, R)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    try:
        for _ in range(iters):
            model.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                emb = model.morphemic_embed(P, R, W, S)
                out = model.backbone(inputs_embeds=emb)
                loss = model.final_norm(out.last_hidden_state).float().pow(2).mean()
            loss.backward()
    except (torch.OutOfMemoryError, RuntimeError) as e:
        if 'out of memory' not in str(e).lower():
            raise
        torch.cuda.empty_cache()
        MB.set_active_root_ids(model, None)
        return {'oom': True, 'err': str(e)[:150]}
    torch.cuda.synchronize()
    dt = (time.time() - t0) / iters
    peak = torch.cuda.max_memory_allocated() / 2 ** 30
    MB.set_active_root_ids(model, None)
    torch.cuda.empty_cache()
    return {'oom': False, 'sec_per_step': dt, 'it_per_s': 1.0 / dt, 'peak_GB': peak}


log('\n' + '=' * 92)
log('BENCH: eager vs sdpa vs sdpa+root -- fwd+bwd, identical weights, T=128 (trainer) and T=512')
log('=' * 92)
for ckpt in CKPTS:
    for T, batches in CONFIGS:
        for mode in MODES:
            prev = None
            for B in batches:
                fg, _ = free_gb()
                need = 3.5 if prev is None else max(3.5, 2.2 * prev)
                key = '%s|B%d|T%d|ckpt%d' % (mode, B, T, int(ckpt))
                if fg < need:
                    log('  ckpt=%d T=%3d %-10s B=%2d  SKIP (need ~%.1f GB, free %.2f GB)'
                        % (int(ckpt), T, mode, B, need, fg))
                    OUT[key] = {'skipped': True, 'free_GB': fg}
                    continue
                r = run_step(B, T, mode, ckpt)
                OUT[key] = dict(r, free_GB_before=fg)
                if r['oom']:
                    log('  ckpt=%d T=%3d %-10s B=%2d  OOM (free was %.2f GB)'
                        % (int(ckpt), T, mode, B, fg))
                    break
                prev = r['peak_GB']
                log('  ckpt=%d T=%3d %-10s B=%2d  %6.2f it/s  %8.1f ms  peak %6.2f GB '
                    '(free before %.2f GB)'
                    % (int(ckpt), T, mode, B, r['it_per_s'], r['sec_per_step'] * 1000,
                       r['peak_GB'], fg))

json.dump(OUT, open('/workspace/root_arch/bench_sdpa.json', 'w'), indent=2)
log('\nwrote /workspace/root_arch/bench_sdpa.json')

log('\n' + '=' * 92)
log('SUMMARY  (it/s @ peak GB, or OOM)')
log('=' * 92)
for ckpt in CKPTS:
    for T, batches in CONFIGS:
        log('  gradient checkpointing %s   T=%d' % ('OFF' if not ckpt else 'ON ', T))
        for B in batches:
            cells = []
            for mode in MODES:
                r = OUT.get('%s|B%d|T%d|ckpt%d' % (mode, B, T, int(ckpt)), {})
                if r.get('oom'):
                    cells.append('%s=OOM' % mode)
                elif r.get('skipped') or 'it_per_s' not in r:
                    cells.append('%s=skip' % mode)
                else:
                    cells.append('%s=%.2fit/s@%.1fG' % (mode, r['it_per_s'], r['peak_GB']))
            log('    B=%2d  %s' % (B, '  '.join(cells)))
        for mode in MODES:
            ok = [(B, OUT.get('%s|B%d|T%d|ckpt%d' % (mode, B, T, int(ckpt)), {}))
                  for B in batches]
            ok = [(B, r) for B, r in ok if 'it_per_s' in r]
            if ok:
                B, r = max(ok, key=lambda x: x[0])
                log('    max batch %-10s = %2d  (%.2f it/s, peak %.2f GB)'
                    % (mode, B, r['it_per_s'], r['peak_GB']))
            else:
                log('    max batch %-10s = none completed' % mode)
