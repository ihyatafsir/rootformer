#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nrmt_train.py -- train the NRMT architecture and watch root PPL on ARABIC ONLY, leak-free.

Protocol
--------
* backbone frozen; hidden states cached once (valid because they do not depend on the head)
* the NRMT head is trained on the cached states with
      L = L_root + 0.5 L_wazn + 0.25 L_prefix + 0.25 L_suffix
          + 0.1 * impossibility hinge   (forbidden roots pushed below a margin)
          + lambda_orbit * Al-Taqalib orbit consistency
* WARM START: the previous generation's `nrmp_head.*` tensors are remapped onto `nrmt_head`.
  MEASURED: with the shipped morphemic embedding (d_root=448) ALL 8 legacy tensors match the new
  head exactly, including cond_proj.0.weight (896,1344).  If a variant with a different d_root is
  used, cond_proj.0.weight becomes (896, 896+d_root); the remap then keeps the shared h block
  [0:896] and zero-inits the new root block (see `remap_legacy_head`).
  `--head-init scratch` restores the old random-init behaviour.
* SCHEDULED SAMPLING on the morphological conditioning: `--ss-prob` is p(use the model's own
  greedy next root); the complement uses the GOLD next root (teacher forcing).  `--ss-start`
  ramps that probability linearly from `--ss-start` to `--ss-prob`.  0.0 = full teacher
  forcing, >= 1.0 = free running (the historical behaviour).
* evaluation reports TWO root numbers on the file-level held-out stream:
      ALL val     -- the usual PPL
      NOVEL only  -- positions whose 12-root context never occurs in train
  plus CE/accuracy for the factorised wazn/prefix/suffix heads.
  Root PPL on NOVEL positions is the only number that proves real progress.

Usage:
  python nrmt_train.py --checkpoint checkpoints/rootformer_v20_nrmp_master.safetensors \
      --train-windows 3000 --val-windows 300 --steps 20000
"""
import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / 'models'))

WIN, STRIDE, CTX = 128, 64, 12
HASH_B = np.uint64(1000003)


def rolling_hashes(stream, n, B=HASH_B):
    a = np.asarray(stream, dtype=np.uint64)
    L = len(a) - n + 1
    if L <= 0:
        return np.zeros(0, dtype=np.uint64)
    h = np.zeros(L, dtype=np.uint64)
    for k in range(n):
        h = h * B + a[k:k + L]
    return h


def hash_ctx(ctx, B=1000003):
    # Python ints with explicit 64-bit masking: identical arithmetic to rolling_hashes
    # (uint64 wraparound) but without numpy overflow warnings.
    h = 0
    for x in ctx:
        h = (h * B + int(x)) & ((1 << 64) - 1)
    return np.uint64(h)


class Novelty:
    def __init__(self, train_stream, ctx=CTX):
        t0 = time.time()
        self.h = np.unique(rolling_hashes(train_stream, ctx))
        print(f'    novelty: {len(self.h)} distinct {ctx}-gram hashes ({time.time()-t0:.0f}s)',
              flush=True)

    def novel(self, ctx):
        h = hash_ctx(ctx)
        i = np.searchsorted(self.h, h)
        return not (i < len(self.h) and self.h[i] == h)


@torch.no_grad()
def extract_backbone(model, streams, starts, device, tag):
    P, R, W, S = streams
    hs = []
    bs = 16
    t0 = time.time()
    for i in range(0, len(starts), bs):
        idx = starts[i:i + bs]
        p = torch.stack([P[j:j + WIN] for j in idx]).to(device)
        r = torch.stack([R[j:j + WIN] for j in idx]).to(device)
        w = torch.stack([W[j:j + WIN] for j in idx]).to(device)
        s = torch.stack([S[j:j + WIN] for j in idx]).to(device)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            emb = model.morphemic_embed(p, r, w, s)
            model._set_flash(r, w)
            o = model.backbone(inputs_embeds=emb)
            h = model.final_norm(o.last_hidden_state)
        hs.append(h.float().cpu())
    print(f'    {tag}: {len(starts)} windows, hidden {torch.cat(hs).shape} in {time.time()-t0:.0f}s',
          flush=True)
    return torch.cat(hs)


def remap_legacy_head(sd, head_state, d_model, cond_init='truncate', allow_local=False):
    """Map legacy `nrmp_head.*` (or already-`nrmt_head.*`) tensors onto the new head.

    `head_state` is the NEW head's own state_dict, i.e. keys WITHOUT the `nrmt_head.` prefix.
    Returns (local_state_dict, applied, exempt) where applied/exempt are human-readable lists.

    `allow_local=True` additionally accepts keys already in head-local form -- that is what the
    trainer's own saved head `.pt` looks like (and what nrmt_governed_eval.py expects), so a
    previous NRMT run can warm start the next one.  It MUST stay False for the backbone
    safetensors: those 568 keys include ~560 non-head tensors which would otherwise all be
    reported as "no counterpart".

    The ONLY possible shape mismatch between the generations is `cond_proj.0.weight`:
        legacy Linear(d_model + 448, d_model)         ->  (896, 1344)
        new    Linear(d_model + d_root, d_model)      ->  (896, 896+d_root)
    With the SHIPPED morphemic embedding d_root == 448, so even this one matches exactly (the
    trainer logs the target shape at startup).  When d_root != 448 the h block is columns
    0:d_model in BOTH (verified against rootformer_v18_nrmp_model.py, where cond_proj is fed
    torch.cat([h, e_chosen_root], -1) with h first), so we copy the trained h block verbatim and
    ZERO-INIT the new d_root-wide root-conditioning block -- consistent with the head's own
    zero-init convention (feat_proj), and unavoidable because the old 448-d root embedding space
    does not map onto a different d_root-d one.
    `cond_init='scratch'` discards the whole tensor instead (pure re-init).
    """
    applied, exempt, out = [], [], {}
    for k, v in sd.items():
        if k.startswith('nrmp_head.'):
            local = k[len('nrmp_head.'):]
        elif k.startswith('nrmt_head.'):
            local = k[len('nrmt_head.'):]
        elif allow_local and k in head_state:
            local = k
        else:
            continue
        if local not in head_state:
            exempt.append(f'{local} {tuple(v.shape)} -> no counterpart (skipped)')
            continue
        tgt = head_state[local]
        if tuple(v.shape) == tuple(tgt.shape):
            out[local] = v.to(tgt.dtype)
            applied.append(f'{local} {tuple(v.shape)} (exact)')
        elif local == 'cond_proj.0.weight':
            if cond_init == 'scratch':
                exempt.append(f'{local} {tuple(v.shape)} -> {tuple(tgt.shape)} (re-initialised)')
                continue
            w = torch.zeros_like(tgt)
            hcols = min(d_model, v.shape[1], tgt.shape[1])
            w[:, :hcols] = v[:, :hcols].to(tgt.dtype)
            out[local] = w
            applied.append(f'{local} {tuple(v.shape)} -> {tuple(tgt.shape)} '
                           f'(h-block[0:{hcols}] copied, root block zero-init)')
        else:
            exempt.append(f'{local} {tuple(v.shape)} -> {tuple(tgt.shape)} (shape mismatch)')
    return out, applied, exempt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default=None)
    ap.add_argument('--cache', default='/workspace/nrmp_cache')
    ap.add_argument('--train-windows', type=int, default=3000)
    ap.add_argument('--val-windows', type=int, default=300)
    ap.add_argument('--steps', type=int, default=20000)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--ss-prob', type=float, default=1.0,
                    help='p(condition the morph heads on the MODEL greedy root); the rest uses '
                         'the GOLD root.  0 = full teacher forcing, 1 = free running (default: '
                         'free running, because gold conditioning measured WORSE held-out morph '
                         'accuracy: wazn acc 50%% -> 20%% at 0.0, 36%% at a 0->0.3 ramp)')
    ap.add_argument('--ss-start', type=float, default=None,
                    help='linear schedule start for --ss-prob (None = constant --ss-prob); '
                         'e.g. --ss-start 0.0 --ss-prob 0.3 anneals teacher forcing -> free')
    ap.add_argument('--head-init', choices=('remap', 'scratch'), default='remap',
                    help='remap the legacy nrmp_head weights (warm start) or random-init')
    ap.add_argument('--cond-init', choices=('truncate', 'scratch'), default='truncate',
                    help='what to do with the shape-mismatched cond_proj.0.weight')
    ap.add_argument('--head-checkpoint', default=None,
                    help='optional path to a previously saved head .pt/.safetensors (local keys)')
    ap.add_argument('--seed', type=int, default=-1,
                    help='seed torch/numpy/cuda (-1 = unseeded, the historical behaviour)')
    ap.add_argument('--lambda-orbit', type=float, default=0.0)
    ap.add_argument('--orbit-rng-isolate', action='store_true',
                    help='do not let the orbit sampler perturb the training RNG stream')
    ap.add_argument('--orbit-grad-probe', action='store_true',
                    help='measure whether the orbit term changes any parameter gradient, then exit')
    ap.add_argument('--margin', type=float, default=-2.0)
    # ---- v18fix flags: ALL default to the historical behaviour -------------------------------
    ap.add_argument('--margin-ramp', type=int, default=0,
                    help='v18fix: linearly anneal the impossibility-hinge margin from '
                         '--margin-start down to --margin over the first N steps '
                         '(0 = off, historical)')
    ap.add_argument('--margin-start', type=float, default=50.0,
                    help='v18fix: margin at step 1 when --margin-ramp is on.  The hinge is '
                         'relu(z - margin)^2, so the margin is an UPPER BOUND on the forbidden '
                         'logit and a MORE NEGATIVE margin is a TIGHTER constraint.  The ramp '
                         'must therefore start HIGH/loose and anneal DOWN to --margin; starting '
                         'at a very negative value (measured) makes the step-1 penalty 2345 and '
                         'the grad-norm 43,351 -- far worse than no ramp at all')
    ap.add_argument('--grad-warmup', type=int, default=0,
                    help='v18fix: during the first N steps, SKIP the optimizer step whenever the '
                         'pre-clip grad-norm exceeds --grad-warmup-thresh (0 = off, historical)')
    ap.add_argument('--grad-warmup-thresh', type=float, default=50.0,
                    help='v18fix: pre-clip grad-norm above which a warmup step is skipped.  NOTE: '
                         'tightening --grad-clip itself is a NO-OP for Adam (its update is '
                         'invariant to a global gradient rescale), which is why this skips instead')
    ap.add_argument('--grad-clip', type=float, default=1.0,
                    help='v18fix: grad-norm clip value (historical hard-coded value: 1.0)')
    ap.add_argument('--extra-norm-l2', type=float, default=0.0,
                    help='v18fix: coefficient of an L2 penalty on ||feat_proj.weight||^2, i.e. on '
                         'the reported extra_norm (0.0 = off, historical).  MEASURED to be a '
                         'mathematical no-op: the LayerNorm downstream makes the loss exactly '
                         'invariant to that norm, so it constrains a pure gauge.')
    ap.add_argument('--feat-gate', action='store_true',
                    help='v18fix: multiply the root-history/operator conditioning branch by a '
                         'zero-initialised scalar so it FADES IN.  Without it the LayerNorm after '
                         'the zero-init feat_proj makes the branch snap from 0 to full unit-RMS '
                         'magnitude in one Adam step (step-2 loss 90.53, grad_norm 12,410)')
    ap.add_argument('--feat-gate-proj-std', type=float, default=1e-3,
                    help='v18fix: std of the small-random init for feat_proj when --feat-gate is '
                         'on.  The branch OUTPUT is still exactly 0 (the gate starts at 0, so no '
                         'slam), but this gives the gate a non-zero gradient to bootstrap from. '
                         '0.0 reproduces the DEAD-SADDLE arm exactly (every branch gradient is 0 '
                         'with feat_proj == 0, measured; the branch never turns on in 20k steps)')
    ap.add_argument('--logit-scale', choices=('raw', 'ln', 'l2'), default='raw',
                    help='v18fix: normalise the ROOT logits before the root cross-entropy. '
                         'raw = historical; ln = per-position LayerNorm over the class axis '
                         '(parameter-free, removes the unbounded logit-scale direction from the '
                         'objective); l2 = per-position L2 projection to RMS sqrt(C)')
    ap.add_argument('--hist', type=int, default=3)
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--no-features', action='store_true',
                    help='CONTROL: disable root-history/operator/morph conditioning (h-only head)')
    # ---- DISCRETE root-history pathway (default 'none' = the historical path, reproducible) ----
    ap.add_argument('--ngram-features', choices=('none', 'vocab', 'hash'), default='none',
                    help='DISCRETE root-history pathway: every distinct order-k root n-gram owns '
                         'a learnable embedding row; the projected sum is ADDED to h_t.  `vocab` '
                         'is collision-free (distinct contexts of the TRAIN stream get compact '
                         'ids, everything else = one OOV row); `hash` buckets into 2**bits.  '
                         'none = historical behaviour, byte-for-byte')
    ap.add_argument('--ngram-orders', default='1,2,3,4',
                    help='comma-separated orders k (context = the k roots ENDING at the position)')
    ap.add_argument('--ngram-dim', type=int, default=32, help='embedding width d_ng per order')
    ap.add_argument('--ngram-bucket-bits', type=int, default=20,
                    help='hash mode: 2**bits buckets per order')
    ap.add_argument('--ngram-emb-std', type=float, default=0.02,
                    help='std of the n-gram embedding init (must be > 0 with the zero gate, else '
                         'the arm is a dead saddle)')
    ap.add_argument('--no-ngram-gate', action='store_true',
                    help='disable the zero-init fade-in gate on the discrete arm (NOT recommended: '
                         'the arm slams from 0 to full RMS in one Adam step)')
    ap.add_argument('--tag', default='full')
    ap.add_argument('--eval-every', type=int, default=2000)
    ap.add_argument('--probe-out', default='',
                    help='optional jsonl path: per-step loss components + grad norm')
    ap.add_argument('--out', default='/workspace/nrmt_results.json')
    ap.add_argument('--save', default='/workspace/nrmt_head.pt')
    args = ap.parse_args()

    if args.seed >= 0:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        print(f'[*] seeded run: torch/numpy/cuda seed = {args.seed}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    import nrmp_vocab as nv
    from models.unified_rootformer_v12 import UnifiedRootformerV12
    from deepseek_v4_1_flash_model import UnifiedRootformerV17_DeepSeekFlash
    from safetensors.torch import load_file
    from nrmt_arch import RootformerNRMT, build_operator_table, ngram_order_vocabs

    V = next(v for k, v in vars(nv).items() if isinstance(v, type) and 'MorphemicVocab' in k)
    bp = str(ROOT_DIR / 'data/rootformer_v12_arabic_blueprint.json')
    vocab = V(bp)

    # (0) the cache is loaded BEFORE the head is built, because the discrete pathway's `vocab`
    # mode needs the train root stream to size its tables (one row per distinct context).
    tr4 = [t.long() for t in torch.load(Path(args.cache) / 'train.pt', map_location='cpu')]
    va4 = [t.long() for t in torch.load(Path(args.cache) / 'val.pt', map_location='cpu')]
    ng_orders = tuple(int(x) for x in str(args.ngram_orders).split(',') if x.strip())
    ng_vocabs = None
    if args.ngram_features == 'vocab':
        _t = time.time()
        ng_vocabs = ngram_order_vocabs(tr4[1].numpy().astype('int64'), vocab.num_roots, ng_orders)
        print('[*] ngram vocab (train stream): ' +
              ', '.join(f'k{k}={len(v)}' for k, v in zip(ng_orders, ng_vocabs)) +
              f' rows ({time.time()-_t:.0f}s)', flush=True)

    base = UnifiedRootformerV12(bp, 'Qwen/Qwen2.5-0.5B', str(device), torch.bfloat16).to(device)
    flash = UnifiedRootformerV17_DeepSeekFlash(base, device, torch.bfloat16).to(device)
    model = RootformerNRMT(flash, vocab, device, torch.bfloat16, hist=args.hist,
                           dropout=args.dropout, use_features=not args.no_features,
                           feat_gate=args.feat_gate,
                           feat_gate_proj_std=args.feat_gate_proj_std,
                           ngram_mode=args.ngram_features, ngram_orders=ng_orders,
                           d_ng=args.ngram_dim, ngram_bucket_bits=args.ngram_bucket_bits,
                           ngram_vocabs=ng_vocabs, ngram_gate=not args.no_ngram_gate,
                           ngram_emb_init_std=args.ngram_emb_std).to(device)
    if args.ngram_features != 'none':
        print(f'[*] DISCRETE arm: mode={args.ngram_features} orders={ng_orders} '
              f'd_ng={args.ngram_dim} gate={not args.no_ngram_gate} '
              f'tables={model.nrmt_head.ngram_table_stats()} '
              f'params={model.nrmt_head.ngram_params()/1e6:.2f}M')
    print(f'[*] ARM [{args.tag}] use_features={not args.no_features} dropout={args.dropout}')
    print(f'[*] v18fix: feat_gate={args.feat_gate} (proj_std {args.feat_gate_proj_std}) '
          f'logit_scale={args.logit_scale} '
          f'margin_ramp={args.margin_ramp} (start {args.margin_start} -> {args.margin}) '
          f'grad_warmup={args.grad_warmup}@thresh{args.grad_warmup_thresh} '
          f'extra_norm_l2={args.extra_norm_l2} grad_clip={args.grad_clip}')

    # DEFAULT: the awzan142-extended checkpoint.  The awzan ids 0-129 were restored to their
    # historical order (append-only), so the original 130-wide checkpoint is 12 wazn rows and
    # 12 embed rows short of the current blueprint (9868 tokens / 142 awzan) and RAISES on load.
    # The extension appends 12 zero rows to wazn_embed/wazn_head and INSERTS 12 zero rows into
    # embed_tokens at index 353, preserving the trained token->row association.
    _DEFAULT_CKPT = ROOT_DIR / 'checkpoints/rootformer_v19_2_synthesis_ar_backbone.awzan142.safetensors'
    if not _DEFAULT_CKPT.exists():
        _DEFAULT_CKPT = ROOT_DIR / 'checkpoints/rootformer_v19_2_synthesis_ar_backbone.safetensors'
    ckpt = args.checkpoint or str(_DEFAULT_CKPT)
    sd = load_file(ckpt)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    new_keys = [k for k in missing if k.startswith('nrmt_head.')]
    print(f'[*] checkpoint {Path(ckpt).name}: missing={len(missing)} '
          f'(of which nrmt_head new={len(new_keys)}) unexpected={len(unexpected)}')

    for p in model.parameters():
        p.requires_grad = False
    # the frozen backbone's cached states are float32, so train this small head in float32 too
    model.nrmt_head.to(torch.float32)

    # ---- (a) HEAD WARM START -------------------------------------------------------------
    # The released checkpoints carry `nrmp_head.*` (previous generation), not `nrmt_head.*`.
    # Without this remap every one of the 18 nrmt_head tensors is randomly initialised and the
    # step-1 root CE sits at ~ln(9114)=9.1176.  Remap BEFORE the head's requires_grad is enabled
    # (cheap either way) but AFTER the float32 cast so the copy is exact rather than bf16-rounded.
    if args.head_init == 'remap':
        head_state = model.nrmt_head.state_dict()
        print(f'[*] new head cond_proj.0.weight target {tuple(head_state["cond_proj.0.weight"].shape)} '
              f'(d_root={model.morphemic_embed.d_root}, d_model={model.d_model})')
        remapped, applied, exempt = remap_legacy_head(sd, head_state, model.d_model, args.cond_init)
        res = model.nrmt_head.load_state_dict(remapped, strict=False)
        n_legacy = len(applied) + len(exempt)
        print(f'[*] head warm start [{args.head_init}/{args.cond_init}]: '
              f'applied {len(applied)}/{n_legacy} legacy head tensors')
        for a in applied:
            print(f'      + {a}')
        for e in exempt:
            print(f'      ! {e}')
        if res.missing_keys:
            print(f'      [info] {len(res.missing_keys)} new head tensors stay at their init '
                  f'(history/operator/morph embeddings, feat_norm)')
    else:
        print('[*] head warm start DISABLED (--head-init scratch): nrmt_head random-init')

    if args.head_checkpoint:
        hsd = (load_file(args.head_checkpoint) if args.head_checkpoint.endswith('.safetensors')
               else torch.load(args.head_checkpoint, map_location='cpu'))
        head_state = model.nrmt_head.state_dict()
        remapped, applied, exempt = remap_legacy_head(hsd, head_state, model.d_model,
                                                      'truncate', allow_local=True)
        res = model.nrmt_head.load_state_dict(remapped, strict=False)
        print(f'[*] head checkpoint {Path(args.head_checkpoint).name}: '
              f'applied {len(applied)} tensors, missing={len(res.missing_keys)}')

    print(f'[*] extra-conditioning norm at init: {model.nrmt_head.extra_norm():.6f} '
          f'(GAUGE -- no logit effect; the real branch magnitude is feat_scale '
          f'{model.nrmt_head.feat_scale():.6f})')
    print(f'[*] Al-Taqalib orbit pairs: {len(model.orbit_pairs)}')
    _ss_start = args.ss_start if args.ss_start is not None else args.ss_prob
    print(f'[*] morph conditioning: gold->model root schedule '
          f'ss_prob {_ss_start:.3f} -> {args.ss_prob:.3f} '
          f'(p = P(use MODEL greedy root); 0 = full teacher forcing)')

    for p in model.nrmt_head.parameters():
        p.requires_grad = True
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'[*] trainable parameters: {n_tr/1e6:.2f}M (backbone frozen)')

    if args.lambda_orbit != 0.0:
        # A structural probe: if the orbit term has no grad_fn it contributes exactly zero
        # gradient and lambda cannot matter.  RNG-isolated: this diagnostic must not shift the
        # dropout stream that the hidden-state extraction and training then consume.
        _rc = torch.get_rng_state()
        _rd = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
        _l = model.orbit_consistency_loss()
        torch.set_rng_state(_rc)
        if _rd is not None:
            torch.cuda.set_rng_state(_rd)
        _re = model.morphemic_embed.root_embed.weight
        print(f'[*] orbit loss probe @init: value={float(_l):.6f} requires_grad={_l.requires_grad} '
              f'grad_fn={type(_l.grad_fn).__name__ if _l.grad_fn is not None else None} | '
              f'root_embed.weight.requires_grad={_re.requires_grad}')

    # (the cache was loaded at (0) above, before the head was built)
    print(f'[*] train {tr4[1].numel()} | val {va4[1].numel()} root tokens')
    nov = Novelty(tr4[1].tolist())

    def starts_for(t, n, seed):
        s = list(range(0, t.numel() - WIN - 1, STRIDE))
        np.random.default_rng(seed).shuffle(s)
        return sorted(s[:n])

    tr_st = starts_for(tr4[1], args.train_windows, 0)
    va_st = starts_for(va4[1], args.val_windows, 1)

    print('[*] caching frozen backbone hidden states', flush=True)
    Htr = extract_backbone(model, tr4, tr_st, device, 'train')
    Hva = extract_backbone(model, va4, va_st, device, 'val')

    def win_tensor(t, starts):
        return torch.stack([t[j:j + WIN] for j in starts])

    Tr, Wr, Pr, Sr = (win_tensor(tr4[i], tr_st) for i in (1, 2, 0, 3))
    Tv, Wv, Pv, Sv = (win_tensor(va4[i], va_st) for i in (1, 2, 0, 3))

    op_table = build_operator_table(vocab)

    # Sibawayh's ʿāmil, PERSISTENT across the window.  The table above recomputes the state
    # from t-1 alone, so government dies after one word; the governor keeps it in force until
    # its act is cut.  al-Kitab 1/421: «فأما النعت الذى جرى على المنعوت فقولك: مررت برجل
    # ظريف قبل، فصار النعت مجرورا مثل المنعوت لأنهما كالاسم الواحد».
    _sib_gov = None
    try:
        from sibawayh_governor import SibawayhGovernor
        _sib_gov = SibawayhGovernor(vocab)
        print('[*] ʿāmil stream: SibawayhGovernor (persistent, al-Kitab 1/421)')
    except Exception as _e:                                    # pragma: no cover
        print(f'[warn] SibawayhGovernor unavailable; falling back to the t-1 operator table: {_e}')

    def op_ids_of(P, R, W):
        # the operator state IN FORCE at each position (persistence), not the t-1 state
        if _sib_gov is None:
            return op_table[R.clamp(min=0)]
        rows = [_sib_gov.op_ids_for_ids(rr, ww, pp)
                for pp, rr, ww in zip(P.tolist(), R.tolist(), W.tolist())]
        return torch.tensor(rows, dtype=torch.long)

    # forbidden static set: Al-Khalil phonotactics (corrected) + control strings
    try:
        from classical_governance_v2 import AlKhalilV2
        forbidden_ids = sorted(AlKhalilV2(vocab).mask())
    except Exception as e:
        print(f'[warn] {e}')
        forbidden_ids = [vocab.PAD_ROOT, vocab.BOS_ROOT, vocab.UNK_ROOT,
                         vocab.root2id['<PARTICLE>']]
    fmask_root = torch.zeros(vocab.num_roots, dtype=torch.bool)
    fmask_root[torch.tensor(forbidden_ids)] = True
    print(f'[*] static forbidden roots (impossibility hinge): {int(fmask_root.sum())}')

    specials = sorted({vocab.PAD_ROOT, vocab.BOS_ROOT, vocab.EOS_ROOT, vocab.UNK_ROOT,
                       vocab.root2id['<PARTICLE>']} |
                      {i for i, r in enumerate(vocab.roots_list) if r.startswith('<P:')})
    spec_t = torch.tensor(specials)

    # per-WINDOW tensors: inputs are positions 0..T-2, targets are positions 1..T-1.
    # (An earlier version flattened [N,T,d] -> [N*T,d], destroying the time axis.)
    Htr_w = Htr[:, :-1, :].contiguous()          # [N, T-1, d]
    Hva_w = Hva[:, :-1, :].contiguous()
    Rt_tr, Rt_va = Tr[:, 1:].contiguous(), Tv[:, 1:].contiguous()
    # operator state established by the word AT each position (Sibawayh's ʿāmil)
    Otr_w = op_ids_of(Pr, Tr, Wr)[:, :-1].contiguous()
    Ova_w = op_ids_of(Pv, Tv, Wv)[:, :-1].contiguous()
    n_win, T1 = Htr_w.shape[0], Htr_w.shape[1]
    print(f'[*] training windows: {n_win} x {T1} positions = {n_win*T1}', flush=True)

    # context-novelty mask over val INPUT positions 0..T-2 (window-major order)
    per = WIN - 1
    novel_mask = torch.zeros(Tv.shape[0] * per, dtype=torch.bool)
    for w in range(Tv.shape[0]):
        seg = Tv[w].tolist()
        for t in range(CTX, WIN):
            novel_mask[w * per + (t - 1)] = nov.novel(seg[t - CTX:t])
    keep_all = ~torch.isin(Rt_va.reshape(-1), spec_t)
    keep_novel = keep_all & novel_mask
    print(f'[*] val radical positions: all={int(keep_all.sum())} novel={int(keep_novel.sum())} '
          f'({100*int(keep_novel.sum())/max(int(keep_all.sum()),1):.1f}%)')

    head = model.nrmt_head
    opt = torch.optim.AdamW([p for p in head.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.steps,
        pct_start=max(0.03, min(0.3, 500.0 / max(args.steps, 1))))

    def ss_prob_at(step):
        """p(use the model's own greedy root) at this step."""
        if args.ss_start is None:
            return args.ss_prob
        frac = (step - 1) / max(args.steps - 1, 1)
        return args.ss_start + (args.ss_prob - args.ss_start) * frac

    def step_loss(h, R, W, P, S, O, tgt_r, tgt_w, tgt_p, tgt_s, p_ss, eff_margin=None,
                  logit_scale='raw'):
        """h:[B,T1,d]; R/W/P/S/O:[B,T1] aligned inputs; tgt_*:[B,T1] next-position targets.

        `tgt_r` is the GOLD next root, i.e. exactly the root whose wazn/prefix/suffix the
        factorised heads are asked to predict, so it is the correct teacher-forcing target.
        p_ss <= 0 => full teacher forcing; 0 < p_ss < 1 => scheduled sampling; p_ss >= 1 =>
        cond_roots=None, i.e. the head's own argmax (the pre-wiring behaviour).

        v18fix `logit_scale` normalises the ROOT logits before the root CE only:
          'raw' -> historical;  'ln' -> per-position LayerNorm over the class axis;
          'l2'  -> per-position L2 projection to RMS sqrt(num_roots).
        The impossibility hinge keeps using the RAW logits, because its margin is defined in
        raw-logit units.
        """
        cond = None
        if p_ss <= 0.0:
            cond = tgt_r
        elif p_ss < 1.0:
            with torch.no_grad():
                h_aug = h + head.build_features(R, O, W, P, S).to(h.dtype)
                pred = head.root_head(h_aug).argmax(-1)
            use_pred = torch.rand(pred.shape, device=pred.device) < p_ss
            cond = torch.where(use_pred, pred, tgt_r)
        out = head(h, R, O, W, P, S, cond_roots=cond)
        keep = ~torch.isin(tgt_r, spec_t.to(tgt_r.device))
        rl_raw = out['root_logits']
        if logit_scale == 'ln':
            rl = (rl_raw - rl_raw.mean(-1, keepdim=True)) / \
                rl_raw.std(-1, keepdim=True).clamp_min(1e-6)
        elif logit_scale == 'l2':
            rl = rl_raw / rl_raw.norm(dim=-1, keepdim=True).clamp_min(1e-6) \
                * math.sqrt(vocab.num_roots)
        else:
            rl = rl_raw
        l_root = (F.cross_entropy(rl[keep], tgt_r[keep]) if keep.any()
                  else rl.sum() * 0.0)
        l_wazn = F.cross_entropy(out['wazn_logits'].reshape(-1, vocab.num_awzan),
                                 tgt_w.reshape(-1), ignore_index=vocab.PAD_WAZN)
        l_pref = F.cross_entropy(out['prefix_logits'].reshape(-1, vocab.num_prefixes),
                                 tgt_p.reshape(-1), ignore_index=vocab.PAD_PREFIX)
        l_suff = F.cross_entropy(out['suffix_logits'].reshape(-1, vocab.num_suffixes),
                                 tgt_s.reshape(-1), ignore_index=vocab.PAD_SUFFIX)
        loss = l_root + 0.5 * l_wazn + 0.25 * l_pref + 0.25 * l_suff
        info = {'root': float(l_root.detach()), 'wazn': float(l_wazn.detach()),
                'prefix': float(l_pref.detach()), 'suffix': float(l_suff.detach())}
        if keep.any():
            # ALWAYS log both root CEs so arms are comparable whichever objective they train under:
            #   root_raw = historical scale-dependent raw-logit CE
            #   root_cez = scale-invariant per-position-standardised CE (the capability number)
            rr = rl_raw[keep]
            info['root_raw'] = float(F.cross_entropy(rr, tgt_r[keep]).detach())
            rs = rr.std(-1, keepdim=True).clamp_min(1e-6)
            info['root_cez'] = float(
                F.cross_entropy((rr - rr.mean(-1, keepdim=True)) / rs, tgt_r[keep]).detach())
        if args.extra_norm_l2 != 0.0:
            pen_norm = head.feat_proj.weight.pow(2).sum()
            loss = loss + args.extra_norm_l2 * pen_norm
            info['extra_l2'] = float(pen_norm.detach())
        fm = fmask_root.to(tgt_r.device)[tgt_r]
        m = args.margin if eff_margin is None else eff_margin
        if fm.any() and m != 0.0:
            pen = F.relu(rl_raw[fm] - m) ** 2
            loss = loss + 0.1 * pen.mean()
            info['impossible'] = float(pen.mean().detach())
            info['margin'] = float(m)
        if args.lambda_orbit > 0:
            if args.orbit_rng_isolate:
                # keep the sampler from shifting the dropout / batch-order stream
                rc = torch.get_rng_state()
                rd = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
                l_orb = model.orbit_consistency_loss()
                torch.set_rng_state(rc)
                if rd is not None:
                    torch.cuda.set_rng_state(rd)
            else:
                l_orb = model.orbit_consistency_loss()
            loss = loss + args.lambda_orbit * l_orb
            info['orbit'] = float(l_orb.detach())
        return loss, info

    @torch.no_grad()
    def evaluate(H, R, W, P, S, O, B=16):
        """H:[N,T1,d]; R/W/P/S/O:[N,T1] inputs. Root metrics on ALL/NOVEL positions, plus the
        factorised morph heads (always free-running: cond_roots=None, as at inference).

        v18fix METRIC CONTRACT -- read this before quoting any number:
          acc@1, acc@5        scale-invariant by construction (argmax/topk are invariant to any
                              positive per-position rescaling of the logits).
          ce_z / ppl_z        PRIMARY.  Cross-entropy after per-position standardisation
                              z = (logit - mean_c logit) / std_c logit.  Parameter-free, fit on
                              nothing, invariant to any per-position affine gauge.  This is the
                              only CE here that reflects CAPABILITY.
          ppl_raw / ce_raw    SCALE-DEPENDENT AND NOT A CAPABILITY MEASURE.  Retained only for
                              continuity with the reference run.  It moved 1,197 -> 65,394 while
                              acc@1 was flat and ce_z *improved*.
          logit_scale         mean per-position std of the root logits: the "temperature" that
                              was inflating.  Reported so the raw PPL is always interpretable.
        """
        head.eval()
        lg, wlg, plg, slg = [], [], [], []
        nb = H.shape[0]
        for i in range(0, nb, B):
            sl = slice(i, min(i + B, nb))
            out = head(H[sl].to(device), R[sl].to(device), O[sl].to(device),
                       W[sl].to(device), P[sl].to(device), S[sl].to(device))
            lg.append(out['root_logits'].float().cpu())
            wlg.append(out['wazn_logits'].float().cpu())
            plg.append(out['prefix_logits'].float().cpu())
            slg.append(out['suffix_logits'].float().cpu())
        LG = torch.cat(lg).reshape(-1, vocab.num_roots)
        tg = Rt_va.reshape(-1)
        assert LG.shape[0] == tg.shape[0], (LG.shape, tg.shape)
        res = {}
        for tag, m in (('ALL_val', keep_all), ('NOVEL_only', keep_novel)):
            l, t = LG[m], tg[m]
            if t.numel() == 0:
                continue
            ce_raw = float(F.cross_entropy(l, t))
            sd = l.std(-1, keepdim=True).clamp_min(1e-6)
            ce_z = float(F.cross_entropy((l - l.mean(-1, keepdim=True)) / sd, t))
            res[tag] = {'n': int(t.numel()),
                        'acc@1': float((l.argmax(-1) == t).float().mean()),
                        'acc@5': float((l.topk(5, -1).indices == t.unsqueeze(-1)).any(-1)
                                       .float().mean()),
                        'ce_z': ce_z, 'ppl_z': math.exp(min(ce_z, 25.0)),
                        'ce_raw_SCALE_DEPENDENT': ce_raw,
                        'logit_scale': float(sd.mean()),
                        # legacy key, kept so older readers of this JSON do not crash
                        'ppl': math.exp(min(ce_raw, 25.0))}
        WG = torch.cat(wlg).reshape(-1, vocab.num_awzan)
        PG = torch.cat(plg).reshape(-1, vocab.num_prefixes)
        SG = torch.cat(slg).reshape(-1, vocab.num_suffixes)
        for tag, logits, tgt, pad in (
                ('wazn', WG, Wv[:, 1:].reshape(-1), vocab.PAD_WAZN),
                ('prefix', PG, Pv[:, 1:].reshape(-1), vocab.PAD_PREFIX),
                ('suffix', SG, Sv[:, 1:].reshape(-1), vocab.PAD_SUFFIX)):
            k = tgt != pad
            if not bool(k.any()):
                continue
            ll, tt = logits[k], tgt[k]
            ce_raw = float(F.cross_entropy(ll, tt))
            sdm = ll.std(-1, keepdim=True).clamp_min(1e-6)
            res[tag] = {'n': int(tt.numel()), 'ce': ce_raw,
                        'ce_z': float(F.cross_entropy((ll - ll.mean(-1, keepdim=True)) / sdm, tt)),
                        'logit_scale': float(sdm.mean()),
                        'acc@1': float((ll.argmax(-1) == tt).float().mean())}
        head.train()
        return res

    if args.orbit_grad_probe:
        # Decisive test for item 3: does adding lambda*orbit_consistency_loss() change ANY
        # parameter gradient?  We seed the RNG identically before each of two identical
        # forwards (so dropout is identical), then backward once without and once with the
        # orbit term, and compare the gradients bitwise.
        assert args.lambda_orbit == 0.0, 'pass --lambda-orbit 0 with --orbit-grad-probe'
        idx = torch.randint(0, n_win, (args.batch_size,))
        probe_batch = (Htr_w[idx].to(device),
                       Tr[idx][:, :-1].to(device), Wr[idx][:, :-1].to(device),
                       Pr[idx][:, :-1].to(device), Sr[idx][:, :-1].to(device),
                       Otr_w[idx].to(device), Rt_tr[idx].to(device),
                       Wr[idx][:, 1:].to(device), Pr[idx][:, 1:].to(device),
                       Sr[idx][:, 1:].to(device))

        def _grads(with_orbit):
            torch.manual_seed(20261001)
            torch.cuda.manual_seed_all(20261001)
            head.zero_grad(set_to_none=True)
            l, _ = step_loss(*probe_batch, 1.0)
            if with_orbit:
                l = l + 1.0 * model.orbit_consistency_loss()
            l.backward()
            return {n: p.grad.detach().clone() for n, p in head.named_parameters()
                    if p.grad is not None}

        g0 = _grads(False)
        g1 = _grads(True)
        diffs = {n: float((g0[n] - g1[n]).abs().max()) for n in g0}
        worst = sorted(diffs.items(), key=lambda kv: -kv[1])[:4]
        print(f'[*] orbit grad probe: params with grad={len(g0)}; '
              f'max |dgrad| over ALL = {max(diffs.values()):.3e}')
        print(f'[*] orbit grad probe: top diffs {worst}')
        print(f'[*] orbit grad probe: gradients bitwise identical = '
              f'{all(torch.equal(g0[n], g1[n]) for n in g0)}  '
              f'-> orbit term contributes {0.0 if all(torch.equal(g0[n], g1[n]) for n in g0) else 1.0}'
              f' gradient')
        return

    _pf = open(args.probe_out, 'w') if args.probe_out else None
    hist = []
    t0 = time.time()
    head.train()

    def margin_at(step):
        """v18fix hinge-margin ramp: from --margin-start at step 1 to --margin at --margin-ramp."""
        if args.margin_ramp <= 0:
            return args.margin
        f = min(1.0, step / float(args.margin_ramp))
        return args.margin_start + (args.margin - args.margin_start) * f

    for step in range(1, args.steps + 1):
        p_ss = ss_prob_at(step)
        idx = torch.randint(0, n_win, (args.batch_size,))
        loss, info = step_loss(
            Htr_w[idx].to(device),
            Tr[idx][:, :-1].to(device), Wr[idx][:, :-1].to(device),
            Pr[idx][:, :-1].to(device), Sr[idx][:, :-1].to(device),
            Otr_w[idx].to(device),
            Rt_tr[idx].to(device), Wr[idx][:, 1:].to(device),
            Pr[idx][:, 1:].to(device), Sr[idx][:, 1:].to(device), p_ss,
            eff_margin=margin_at(step), logit_scale=args.logit_scale)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(
            [p for p in head.parameters() if p.requires_grad], args.grad_clip)
        # v18fix: a tighter --grad-clip is a NO-OP for Adam (m_hat/sqrt(v_hat) is invariant to a
        # global gradient rescale), so the pathological warm-up steps are SKIPPED instead.
        skipped = False
        if args.grad_warmup > 0 and step <= args.grad_warmup \
                and float(gnorm) > args.grad_warmup_thresh:
            skipped = True
            opt.zero_grad(set_to_none=True)
        else:
            opt.step()
        sched.step()
        if _pf is not None:
            _pf.write(json.dumps({'step': step, 'loss': float(loss),
                                  'grad_norm': float(gnorm), 'p_ss': p_ss,
                                  'skipped': skipped,
                                  'lr': opt.param_groups[0]['lr'], **info}) + '\n')
            if step % 25 == 0:
                _pf.flush()
        if step % 2000 == 0:
            print(f'  step {step}/{args.steps} loss {float(loss):.4f} '
                  f'(root {info["root"]:.3f} wazn {info["wazn"]:.3f} '
                  f'pref {info["prefix"]:.3f} suff {info["suffix"]:.3f}) '
                  f'{step/(time.time()-t0):.0f} it/s', flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            m = evaluate(Hva_w, Tv[:, :-1], Wv[:, :-1], Pv[:, :-1], Sv[:, :-1], Ova_w)
            parts = []
            for k, v in m.items():
                if 'ppl_z' in v:
                    parts.append(f'{k}: acc@1 {100*v["acc@1"]:.2f}% acc@5 {100*v["acc@5"]:.2f}% '
                                 f'CE_z {v["ce_z"]:.4f} [scale-invariant] | '
                                 f'raw PPL {v["ppl"]:.1f} (SCALE-DEPENDENT, logit_scale '
                                 f'{v["logit_scale"]:.2f}) (n={v["n"]})')
                else:
                    parts.append(f'{k}: ce {v["ce"]:.4f} ce_z {v["ce_z"]:.4f} '
                                 f'acc@1 {100*v["acc@1"]:.2f}% (n={v["n"]})')
            extra = (f' | extra_norm {model.nrmt_head.extra_norm():.4f} (GAUGE)'
                     f' | feat_scale {model.nrmt_head.feat_scale():.4f} (real branch RMS x '
                     f'{model.nrmt_head.feat_scale()/math.sqrt(model.d_model):.4f})')
            if args.ngram_features != 'none':
                extra += (f' | ngram gate {model.nrmt_head.ngram_gate():.5f} '
                          f'scale {model.nrmt_head.ngram_scale():.4f} (RMS x '
                          f'{model.nrmt_head.ngram_scale():.4f})')
            print(f'  [eval @{step}] ' + ' | '.join(parts) + extra, flush=True)
            hist.append({'step': step, **m, 'extra_norm': model.nrmt_head.extra_norm(),
                         'feat_scale': model.nrmt_head.feat_scale(),
                         'ngram_gate': model.nrmt_head.ngram_gate(),
                         'ngram_scale': model.nrmt_head.ngram_scale()})
            torch.save({k: v.cpu() for k, v in head.state_dict().items()}, args.save)

    if _pf is not None:
        _pf.flush()
        _pf.close()
    json.dump({'history': hist, 'args': vars(args)}, open(args.out, 'w'), indent=2)
    print(f'\nwrote {args.out} and {args.save}')


if __name__ == '__main__':
    main()
