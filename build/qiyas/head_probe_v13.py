#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
head_probe_v13.py -- the head-independent trunk-damage control, fixed for the LIVE id space.

WHY THIS EXISTS
---------------
`/workspace/ghazali_forget/ewc_head_probe.py` is the project's head-independent control (the one
that caught arm P: self-report 16.34 % vs head-independent 2.676 %).  It CANNOT be run on the
current arms.  I ran it and it dies at `ewc_head_probe.py:123`:

    RuntimeError: The size of tensor a (9015) must match the size of tensor b (9490)

Isolated cause: `ewc_fisher.build_model` reads the CURRENT blueprint, so `vocab.num_roots` prints
9490 -- but it constructs `UnifiedRootformerV12`, whose attention root tables are built at the OLD
9015, while the arm's `.trunk.pt` payload carries (9490, 64).  Passing the correct `--ckpt` does not
help: the mismatch is in the architecture, not the weights.  This is the same stale-id-space class
as `...roots9313` and the 48 dropped-SPACE tables, in a fourth place.

THE FIX
-------
Build with `UnifiedRootformerV13` and hand it the SAME vocab the trainer uses, exactly as
`nrmt_train_v13_sdpa.py:414-421` does:

    base = UnifiedRootformerV13(bp, 'Qwen/Qwen2.5-0.5B', device, dtype, vocab=vocab)

V13 asserts the live space (9490/142).  Everything else -- `build_meta`, `eval_head`, the trunk
overwrite, the RCA-absent/native-ids-nulled condition -- is IMPORTED UNCHANGED from the original
probe, so the numbers are directly comparable to the arm-P precedent.

Two hardening changes over the original, both prompted by its own failure modes:
  * the vocabulary class is named explicitly (`FarāhīdianMorphemicVocab`), not discovered by the
    fragile `next(... 'MorphemicVocab' in k)` idiom that works only by dict-iteration order;
  * the trunk copy COUNTS and ASSERTS how many tensors it applied, because the original only
    prints `n` and a silent name-match miss would leave a released trunk in place while reporting
    a "damaged trunk applied" line.

CPU only.  Reads the pod's files; writes one small JSON.  No GPU, no process disturbed.

usage:
  python head_probe_v13.py --trunk /tmp/root_arch_arms/head_FLOOR_A.pt.trunk.pt \
      --tag FLOOR_A --out /tmp/probe_head_FLOOR_A.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, '/workspace/ghazali_forget')
sys.path.insert(0, '/workspace/hf_v19_2_release')
sys.path.insert(0, '/workspace/hf_v19_2_release/models')
sys.path.insert(0, '/workspace/transmute_v2')
sys.path.insert(0, '/workspace/head_fix')

REL = '/workspace/hf_v19_2_release'
BP = os.path.join(REL, 'data/rootformer_v12_arabic_blueprint.json')

# imported UNCHANGED from the original probe so the numbers are comparable to its precedent
from ewc_head_probe import build_meta, eval_head          # noqa: E402


def build_model_v13(ckpt, dtype=torch.float32):
    from unified_rootformer_v13 import UnifiedRootformerV13
    from deepseek_v4_1_flash_model import UnifiedRootformerV17_DeepSeekFlash
    from safetensors.torch import load_file
    import nrmp_vocab as nv
    from nrmt_arch import RootformerNRMT

    vocab = nv.FarāhīdianMorphemicVocab(BP)          # explicit, not the next() idiom
    print('[*] vocab roots=%s awzan=%s' % (vocab.num_roots, vocab.num_awzan), flush=True)
    assert (int(vocab.num_roots), int(vocab.num_awzan)) == (9490, 142), 'live space changed'

    base = UnifiedRootformerV13(BP, 'Qwen/Qwen2.5-0.5B', 'cpu', dtype, vocab=vocab)
    flash = UnifiedRootformerV17_DeepSeekFlash(base, 'cpu', dtype)
    model = RootformerNRMT(flash, vocab, 'cpu', dtype, hist=3, dropout=0.0,
                           use_features=True, feat_gate=True)
    sa = model.backbone.layers[0].self_attn.root_embed.weight
    print('[*] model self_attn.root_embed.weight = %s' % (tuple(sa.shape),), flush=True)
    assert sa.shape[0] == 9490, ('STALE ID SPACE: attention root table is %d, expected 9490 -- '
                                 'the fix did not take' % sa.shape[0])

    sd = load_file(ckpt)
    # Replicate the trainer's 48-key drop (nrmt_train_v13_sdpa.py:476-489).  The released /
    # pre-alignment checkpoint still carries self_attn root/wazn tables at the OLD (9015,64) and
    # (128,64); `load_state_dict` raises on shape mismatch even with strict=False, so they must be
    # removed first.  V13's own release drops exactly these 48.
    _want = dict(model.state_dict())
    _drop = {}
    for _k in list(sd):
        if ('self_attn.root_embed.weight' in _k or 'self_attn.wazn_embed.weight' in _k):
            if _k in _want and tuple(sd[_k].shape) != tuple(_want[_k].shape):
                _drop[_k] = (tuple(sd[_k].shape), tuple(_want[_k].shape))
                del sd[_k]
            elif _k not in _want:
                _drop[_k] = (tuple(sd[_k].shape), None)
                del sd[_k]
    _rows = sorted({v[0][0] for v in _drop.values()})
    print('[*] dropped %d stale-SPACE self_attn tables (rows %s -> live %d/%d)'
          % (len(_drop), _rows, vocab.num_roots, vocab.num_awzan), flush=True)
    if len(_drop) != 48:
        raise SystemExit('[V13] expected exactly 48 stale tables dropped, got %d' % len(_drop))

    missing, unexpected = model.load_state_dict(sd, strict=False)
    # The 48 dropped tables are EXPECTED to be missing: V13 already built them at the live
    # 9490/142 and the checkpoint has no correct values for them.  The trainer
    # (nrmt_train_v13_sdpa.py:491-494) likewise tolerates them and only inspects nrmt_head keys.
    # So subtract the 48 we deliberately dropped from the missing set before judging.
    dropped_names = {k for k in _drop}
    bb_missing = [k for k in missing if k.startswith('backbone.') and k not in dropped_names]
    real_unexp = [k for k in unexpected if not k.startswith('nrmp_head.')]
    print('[*] load: missing=%d (backbone missing=%d, of which dropped-stale=%d) unexpected=%d '
          '(non-legacy-head=%d)'
          % (len(missing), len(bb_missing), len(missing) - len(bb_missing) - 0,
             len(unexpected), len(real_unexp)), flush=True)
    if bb_missing or real_unexp:
        raise SystemExit('CHECKPOINT DID NOT LOAD CLEANLY: bb_missing=%s unexpected=%s'
                         % (bb_missing[:4], real_unexp[:4]))
    return base, flash, model, vocab


def linear_readout(model, Pv, Tv, Wv, Sv, n_layers, threads=6,
                   train_frac=0.5, alpha=1.0, seed=0, max_samples=40000, split='root'):
    """HEAD-INDEPENDENT, FOREIGN-HEAD-FREE control: is the CURRENT position's root linearly
    decodable from each trunk layer's hidden state?

    This is the project's own decodability protocol (`probe_orig_base.py:121-129`: closed-form
    ridge to one-hot, no fitted network, no transfer from another trunk) applied to a live arm.
    It answers the question the fixed-head probe cannot answer without a head/trunk confound:
    does this trunk still CARRY root information?

    Published reference on the same protocol: raw base 29.30 % (unseen 23.34 %), transmuted trunk
    92.34 %, and a local isolated probe decaying 21.83 % @ layer 4 -> 13.00 % @ layer 23.

    Target convention: hidden state at position i vs the root OF THAT position (Tv[i]), matching
    `probe_orig_base.py`.  Note `eval_head` instead predicts Tv[:, 1:] (next-token) -- a DIFFERENT
    target, deliberately.
    """
    import numpy as np
    torch.set_num_threads(threads)

    # capture every backbone layer's output for the whole val set
    buf = {}

    def mk_hook(li):
        def hook(module, inp, out):
            h = out[0] if isinstance(out, (tuple, list)) else out
            buf[li] = h.detach()
        return hook

    handles = [model.backbone.layers[i].register_forward_hook(mk_hook(i))
               for i in range(n_layers)]
    hs = []
    with torch.no_grad():
        for i in range(0, Tv.shape[0], 8):
            sl = slice(i, min(i + 8, Tv.shape[0]))
            emb = model.morphemic_embed(Pv[sl], Tv[sl], Wv[sl], Sv[sl])
            model.backbone(inputs_embeds=emb)
            hs.append({li: buf[li].float().clone() for li in range(n_layers)})
    for h in handles:
        h.remove()

    y_all = Tv.reshape(-1)
    valid = (y_all >= 0)
    out = {}
    for li in range(n_layers):
        H = torch.cat([d[li] for d in hs])            # [W, T, d]
        X_all = H.reshape(-1, H.shape[-1])
        keep = valid & (y_all < model.vocab.num_roots)
        if int(keep.sum()) < 100:
            continue
        X = X_all[keep].numpy().astype(np.float32)
        y = y_all[keep].numpy().astype(np.int64)
        # SPLIT BY WORD IDENTITY, not by position.  A random split over positions LEAKS: the same
        # word type occurs in many windows, so a near-duplicate of a test instance sits in train
        # and the probe scores high by memorisation.  `probe_orig_base.py` reports BOTH an
        # overall and an `unseen` figure precisely for this reason, and the two differ materially
        # (92.34 % overall vs 83.92 % unseen).  Splitting on the word id removes the leak.
        wid = X_all[keep, :0]  # placeholder, replaced below
        rng = np.random.default_rng(seed)
        # word id = the root target is NOT the word; use the token id stream instead
        # (Tv is the root/token stream aligned to positions, so use it as the identity key)
        if split == 'root':
            # hold out ROOTS -> the UNSEEN figure.  Compare against probe_orig_base's 83.92 %.
            wkey = y.copy()
            uw = np.unique(wkey)
            rng.shuffle(uw)
            n_wtr = int(train_frac * len(uw))
            tr = np.isin(wkey, uw[:n_wtr])
            te = ~tr
        else:
            # random over positions -> the OVERALL figure.  Compare against 92.34 %.
            ii = rng.permutation(len(y))
            k = int(train_frac * len(ii))
            tr = np.zeros(len(y), dtype=bool)
            te = np.zeros(len(y), dtype=bool)
            tr[ii[:k]] = True
            te[ii[k:]] = True
        # cap the ridge cost: O(n*d^2) Gram + a d x C multiply.  A decodability probe does not
        # need every position; `probe_orig_base.py` is order-of-magnitude comparable at this scale.
        cap = max_samples
        if int(tr.sum()) > cap:
            tri = np.flatnonzero(tr)[:cap]
        else:
            tri = np.flatnonzero(tr)
        tei = np.flatnonzero(te)
        if len(tei) > cap:
            tei = tei[:cap]
        tr, te = tri, tei
        C = int(model.vocab.num_roots)
        Y = np.zeros((len(tr), C), dtype=np.float32)
        Y[np.arange(len(tr)), y[tr]] = 1.0
        Xtr = X[tr]
        mu = Xtr.mean(0, keepdims=True)
        Xtr = Xtr - mu
        Xte = X[te] - mu
        A = Xtr.T @ Xtr + alpha * np.eye(Xtr.shape[1], dtype=np.float32)
        W = np.linalg.solve(A, Xtr.T @ Y)
        S = Xte @ W
        a1 = float((S.argmax(1) == y[te]).mean())
        k = min(5, C)
        top5 = np.argpartition(-S, k - 1, axis=1)[:, :k]
        a5 = float((top5 == y[te, None]).any(1).mean())
        out[li] = {'acc@1_pct': 100 * a1, 'acc@5_pct': 100 * a5, 'n': int(len(y))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default='/workspace/head_fix/nrmp_cache_9490_aligned')
    ap.add_argument('--head', default='/workspace/head_fix/head_ALIGNED_FIX.pt')
    ap.add_argument('--trunk', default='')
    ap.add_argument('--ckpt', default=os.path.join(
        REL, 'checkpoints/'
             'rootformer_v19_2_synthesis_ar_backbone.awzan142.roots9490.tok10052.safetensors'))
    ap.add_argument('--val-windows', type=int, default=300)
    ap.add_argument('--live', action='store_true')
    ap.add_argument('--tag', default='probe')
    ap.add_argument('--out', default='')
    ap.add_argument('--threads', type=int, default=6)
    ap.add_argument('--split', choices=('root', 'random'), default='root',
                    help='root=hold out ROOTS (unseen figure); random=over positions (overall)')
    ap.add_argument('--max-samples', type=int, default=40000,
                    help='cap positions fed to the ridge readout')
    ap.add_argument('--linear-readout', action='store_true',
                    help='head-independent ridge decodability per layer (no fitted head)')
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    base, flash, model, vocab = build_model_v13(args.ckpt, dtype=torch.float32)
    for p in model.parameters():
        p.requires_grad = False

    hsd = torch.load(args.head, map_location='cpu')
    r = model.nrmt_head.load_state_dict(hsd, strict=False)
    print('[*] FIX head loaded: missing=%d unexpected=%d'
          % (len(r.missing_keys), len(r.unexpected_keys)), flush=True)
    if r.missing_keys or r.unexpected_keys:
        raise SystemExit('FIX head did not load cleanly: %s %s' % (r.missing_keys,
                                                                  r.unexpected_keys))

    applied = 0
    expected = 0
    if args.trunk:
        pay = torch.load(args.trunk, map_location='cpu')
        st = pay.get('state', pay)
        with torch.no_grad():
            for name, p in model.named_parameters():
                if name.startswith('backbone.layers.'):
                    expected += 1
                    if name in st:
                        if tuple(st[name].shape) != tuple(p.shape):
                            raise SystemExit('SHAPE MISMATCH on %s: payload %s vs model %s'
                                             % (name, tuple(st[name].shape), tuple(p.shape)))
                        p.data.copy_(st[name].float())
                        applied += 1
        print('[*] trunk applied: %d/%d backbone.layers tensors from %s'
              % (applied, expected, args.trunk), flush=True)
        # the original probe only PRINTED this; a silent miss would mean we measured a
        # released trunk while claiming a damaged one -- assert instead
        if applied != expected:
            raise SystemExit('TRUNK OVERWRITE INCOMPLETE: %d/%d applied -- refusing to report'
                             % (applied, expected))

    (Pv, Tv, Wv, Sv), Ova_w, keep_all, va4, va_st = build_meta(model, args.cache,
                                                              args.val_windows)
    R, W, P, S = (Tv[:, :-1].contiguous(), Wv[:, :-1].contiguous(),
                  Pv[:, :-1].contiguous(), Sv[:, :-1].contiguous())
    tg = Tv[:, 1:].contiguous()
    print('[*] val windows %d | radical positions %d' % (Tv.shape[0], int(keep_all.sum())),
          flush=True)

    res = {'tag': args.tag, 'val_windows': args.val_windows,
           'n_positions': int(keep_all.sum()), 'head': args.head,
           'trunk': args.trunk or 'released',
           'trunk_tensors_applied': applied, 'trunk_tensors_expected': expected,
           'model': 'UnifiedRootformerV13', 'vocab_roots': int(vocab.num_roots)}

    if args.live:
        for _layer in model.backbone.layers:      # RCA absent; native ids nulled
            _layer.self_attn.active_root_ids = None
            _layer.self_attn.active_wazn_ids = None
        hs = []
        with torch.no_grad():
            for i in range(0, Tv.shape[0], 8):
                sl = slice(i, min(i + 8, Tv.shape[0]))
                emb = model.morphemic_embed(Pv[sl], Tv[sl], Wv[sl], Sv[sl])
                out = model.backbone(inputs_embeds=emb)
                hs.append(model.final_norm(out.last_hidden_state).float()[:, :-1].contiguous())
        H = torch.cat(hs)
        res['live'] = eval_head(model, model.nrmt_head, H, R, Ova_w, W, P, S, keep_all, tg)
        print('[%s] LIVE fp32 (V13) acc@1 %.3f%% acc@5 %.3f%% ce_z %.4f'
              % (args.tag, 100 * res['live']['acc@1'], 100 * res['live']['acc@5'],
                 res['live']['ce_z']), flush=True)

    if args.linear_readout:
        print('[*] linear decodability readout (closed-form ridge, no fitted head) ...', flush=True)
        lr_res = linear_readout(model, Pv, Tv, Wv, Sv, len(model.backbone.layers),
                               threads=args.threads, max_samples=args.max_samples,
                               split=args.split)
        res['linear_readout'] = {str(k): v for k, v in lr_res.items()}
        print('    layer   acc@1     acc@5', flush=True)
        for li in sorted(lr_res):
            r = lr_res[li]
            print('    %5d  %7.3f%%  %7.3f%%' % (li, r['acc@1_pct'], r['acc@5_pct']), flush=True)
        if lr_res:
            best = max(lr_res, key=lambda k: lr_res[k]['acc@1_pct'])
            print('    BEST layer %d : acc@1 %.3f%%' % (best, lr_res[best]['acc@1_pct']), flush=True)

    if args.out:
        json.dump(res, open(args.out, 'w'), indent=2)
    print('HEAD_PROBE_V13_DONE')
    return res


if __name__ == '__main__':
    main()
