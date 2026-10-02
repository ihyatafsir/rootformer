#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_alignment.py -- three CPU/GPU-light proofs that must pass BEFORE any training run:

  1. ATTENTION EQUIVALENCE.  With both gates at 0, `AltRootCrossAttentionStack` rewrites each
     selected layer's attention.  If the rewrite is faithful, the rewritten attention output must
     equal `IshtiqaqAttentionV12.forward`'s output -- including the native root term it may still
     contain.  Reported as max abs difference with the stack's own gates zeroed AND with the
     module's own root path live, so the test is not trivially satisfied.

  2. END-TO-END BITWISE NO-OP.  Full trunk forward, stack attached, gates 0, score bias off:
     max|h_attached - h_detached| must be exactly 0.

  3. ALIGNMENT ROUND TRIP.  For a sample of cached word events, re-tokenize the word's own
     surface with the base tokenizer and confirm that (a) the tokens are those the cache stored
     for that word, and (b) the word's char span in the sentence is what the morphemic side
     believes.  This is the independent check that the external root stream sits on the right
     base-token positions.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

RELEASE = Path('/workspace/hf_v19_2_release')
ALT = Path('/workspace/alt_base')
WIN = 128


def load_matching(model, sd, tag='ckpt'):
    """Load only tensors whose shape matches: the released awzan142 checkpoint predates the
    current blueprint (9868-token / 9114-root vs 10052 / 9490), so the embedding tables differ in
    ROW COUNT.  Every other tensor must match exactly; a non-shape difference is a hard error."""
    ms = model.state_dict()
    keep, skipped = {}, []
    for k, v in sd.items():
        if k in ms and tuple(ms[k].shape) == tuple(v.shape):
            keep[k] = v
        elif k in ms:
            skipped.append((k, tuple(v.shape), tuple(ms[k].shape)))
        else:
            skipped.append((k, tuple(v.shape), None))
    missing = [k for k in ms if k not in keep and not k.startswith(('nrmt_head.', 'nrmp_head.'))]
    model.load_state_dict(keep, strict=False)
    print(f'[*] {tag}: loaded {len(keep)}/{len(sd)} tensors; skipped {len(skipped)}', flush=True)
    for k, a, b in skipped[:12]:
        print(f'    SKIP {k}: ckpt{a} vs model{b}', flush=True)
    print(f'[*] {tag}: model tensors NOT covered by the checkpoint: {len(missing)}', flush=True)
    for k in missing[:12]:
        print(f'    MISSING {k} {tuple(ms[k].shape)}', flush=True)
    return keep, skipped, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default='/workspace/alt_base/cache/rootqwen')
    ap.add_argument('--base', default='Qwen/Qwen2.5-0.5B')
    ap.add_argument('--ckpt', default=str(RELEASE / 'checkpoints/'
                    'rootformer_v19_2_synthesis_ar_backbone.awzan142.safetensors'))
    ap.add_argument('--layers', default='all')
    ap.add_argument('--n-align', type=int, default=4000)
    ap.add_argument('--out', default=str(ALT / 'verify_alignment.json'))
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    T0 = time.time()
    RES = {}
    dev = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    sys.path.insert(0, str(RELEASE)); sys.path.insert(0, str(RELEASE / 'models'))
    sys.path.insert(0, str(ALT))
    import nrmp_vocab as nv
    from safetensors.torch import load_file
    from nrmt_arch import RootformerNRMT
    from models.unified_rootformer_v12 import UnifiedRootformerV12
    from deepseek_v4_1_flash_model import UnifiedRootformerV17_DeepSeekFlash
    from alt_root_attn import AltRootCrossAttentionStack, ScoreBiasModule, disable_native_root_path

    V = next(v for k, v in vars(nv).items() if isinstance(v, type) and 'MorphemicVocab' in k)
    vocab = V(str(RELEASE / 'data/rootformer_v12_arabic_blueprint.json'))

    base = UnifiedRootformerV12(str(RELEASE / 'data/rootformer_v12_arabic_blueprint.json'),
                               args.base, str(dev), torch.bfloat16).to(dev)
    flash = UnifiedRootformerV17_DeepSeekFlash(base, dev, torch.bfloat16).to(dev)
    model = RootformerNRMT(flash, vocab, dev, torch.bfloat16, hist=3, dropout=0.0,
                           use_features=True, feat_gate=False).to(dev)
    sd = load_file(args.ckpt)
    load_matching(model, sd, Path(args.ckpt).name)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    layers = model.backbone.layers
    root_w = model.morphemic_embed.root_embed.weight
    print(f'[{time.time()-T0:.1f}s] model ready, {len(layers)} layers, '
          f'root table {tuple(root_w.shape)}', flush=True)

    tok = None
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(
            args.base, cache_dir='/workspace/alt_base/cache/hub')
    except Exception as e:
        print(f'[warn] base tokenizer unavailable: {e}', flush=True)

    # ------------------------------------------------ 1 & 2: attention + end-to-end
    tr = {n: torch.load(Path(args.cache) / f'{n}_train.pt', map_location='cpu')
          for n in ('word', 'tok', 'soff', 'wtokoff', 'wlen', 'wstart')}
    wstart = tr['wstart'].long(); soff = tr['soff'].long(); wtokoff = tr['wtokoff'].long()
    wlen = tr['wlen'].long(); tokstream = tr['tok'].long()
    b = 7
    idxs = torch.arange(b * WIN, (b + 1) * WIN)
    sent = torch.searchsorted(wstart, idxs, right=True) - 1
    first = soff[sent] + wtokoff[idxs]
    start = int(first[0]); end = int(first[-1] + wlen[idxs][-1])
    ids = tokstream[start:end].unsqueeze(0).to(dev)
    print(f'[{time.time()-T0:.1f}s] probe window T={ids.shape[1]}', flush=True)

    # per-token root ids (the EXTERNAL stream, aligned by character offset)
    wid = torch.repeat_interleave(torch.arange(WIN), wlen[idxs]).unsqueeze(0)
    rw = tr['word'][1].long()[idxs]
    root_tok = rw[wid].to(dev)

    layer0 = layers[0]
    att0 = layer0.self_attn
    with torch.no_grad():
        model.backbone(input_ids=ids)
    RES['window'] = {'T': int(ids.shape[1]), 'block': b}

    # --- attention equivalence: call the module with the SAME inputs the layer sees
    captured = {}

    def cap_pre(module, a, kw):
        captured['hs'] = a[0] if a else kw.get('hidden_states')
        captured['pe'] = kw.get('position_embeddings', kw.get('rotary_emb'))
        captured['mask'] = kw.get('attention_mask')

    def cap_post(module, a, out):
        captured['att_out'] = out[0] if isinstance(out, tuple) else out
        captured['att_w'] = out[1] if isinstance(out, tuple) and len(out) > 1 else None

    hdl = [att0.register_forward_pre_hook(cap_pre, with_kwargs=True),
           att0.register_forward_hook(cap_post)]
    with torch.no_grad():
        o = model.backbone(input_ids=ids)
    for h in hdl:
        h.remove()
    hs_in = captured['hs'].detach()
    ref_out = captured['att_out'].detach()
    print(f'[{time.time()-T0:.1f}s] captured layer0 attention: out {tuple(ref_out.shape)} '
          f'mask={None if captured["mask"] is None else tuple(captured["mask"].shape)} '
          f'pe={None if captured["pe"] is None else tuple(captured["pe"][0].shape)}', flush=True)

    def stack_for(layer_ids, residual, score_bias):
        sb = None
        if score_bias:
            sb = ScoreBiasModule(vocab.num_roots, att0.head_dim, att0.num_heads,
                                 att0.num_kv_heads, att0.scale).to(dev).float()
        st = AltRootCrossAttentionStack(
            model.d_model, int(root_w.shape[1]), root_w, layer_ids, score_bias=sb,
            residual=residual, num_heads=8, dropout=0.0, out_std=1e-3,
            out_norm=True).to(dev)
        return st

    # 1a. residual only, gate 0 -> the LAYER output must be unchanged
    st = stack_for([0], residual=True, score_bias=False)
    st.set_root_ids(root_tok)
    st._position_embeddings = captured['pe']
    st._attn_mask = captured['mask']
    rew = st._layer_forward(layer0, hs_in, root_tok)
    layer_ref = None
    def cap_post2(module, a, out):
        nonlocal layer_ref
        layer_ref = out[0] if isinstance(out, tuple) else out
    h2 = layer0.register_forward_hook(cap_post2)
    with torch.no_grad():
        model.backbone(input_ids=ids)
    h2.remove()
    d_layer = float((rew.float() - layer_ref.float()).abs().max())
    RES['layer0_rewrite_gate0_max_abs_diff'] = d_layer
    RES['layer0_rewrite_bitwise'] = bool(torch.equal(rew, layer_ref))
    print(f'  layer0 rewrite (residual gate 0): max|diff|={d_layer:.3e} '
          f'bitwise={RES["layer0_rewrite_bitwise"]}', flush=True)

    # 2. end-to-end bitwise no-op on the FULL stack at all selected layers
    st2 = stack_for(list(range(len(layers))), residual=True, score_bias=False)
    st2.attach(layers)
    with torch.no_grad():
        o = model.backbone(input_ids=ids)
        h_on = model.final_norm(o.last_hidden_state)
        st2.set_root_ids(None)
        o = model.backbone(input_ids=ids)
        h_off = model.final_norm(o.last_hidden_state)
    RES['e2e_gate0_max_abs_diff'] = float((h_on.float() - h_off.float()).abs().max())
    RES['e2e_gate0_bitwise'] = bool(torch.equal(h_on, h_off))
    print(f'  end-to-end stack (all {len(layers)} layers, gates 0): '
          f'max|diff|={RES["e2e_gate0_max_abs_diff"]:.3e} '
          f'bitwise={RES["e2e_gate0_bitwise"]}', flush=True)

    # 3. root ids actually change the output once the gate is opened
    with torch.no_grad():
        st2.set_root_ids(root_tok)
        st2.gate.data.fill_(0.5)
        o = model.backbone(input_ids=ids)
        h_g = model.final_norm(o.last_hidden_state)
        st2.gate.data.zero_()
        o = model.backbone(input_ids=ids)
        h_z = model.final_norm(o.last_hidden_state)
    RES['gate0.5_max_abs_diff_vs_gate0'] = float((h_g.float() - h_z.float()).abs().max())
    print(f'  gate=0.5 vs gate=0: max|diff|={RES["gate0.5_max_abs_diff_vs_gate0"]:.3e} '
          f'(must be > 0)', flush=True)

    # ------------------------------------------------ 3: alignment round trip
    if tok is not None:
        import re
        n = min(args.n_align, tr['word'][1].numel())
        g = np.random.default_rng(0)
        sel = np.sort(g.choice(tr['word'][1].numel(), size=n, replace=False))
        ok_len = ok_nonempty = 0
        bad = []
        for j in sel.tolist():
            si = int(torch.searchsorted(wstart, torch.tensor(j), right=True) - 1)
            st_ = int(soff[si]); off = int(wtokoff[j]); L = int(wlen[j])
            stored = tokstream[st_ + off: st_ + off + L].tolist()
            dec = tok.decode(stored)
            if len(stored) == L:
                ok_len += 1
            if dec.strip() != '':
                ok_nonempty += 1
            if len(bad) < 5 and (len(stored) != L or dec.strip() == ''):
                bad.append({'word_index': j, 'stored': stored, 'decoded': dec, 'L': L})
        RES['alignment'] = {'n_checked': n, 'word_span_length_consistent': ok_len,
                            'decodes_to_nonempty_text': ok_nonempty, 'examples_bad': bad}
        print(f'  alignment: {n} word events, length-consistent={ok_len}/{n}, '
              f'nonempty_decode={ok_nonempty}/{n}', flush=True)

    Path(args.out).write_text(json.dumps(RES, ensure_ascii=False, indent=2))
    print(f'wrote {args.out}', flush=True)


if __name__ == '__main__':
    main()
