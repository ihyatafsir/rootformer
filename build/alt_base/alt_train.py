#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
alt_train.py -- train the NRMT head (+ optional root cross-attention) on an ARBITRARY base
model whose root annotation arrives as an EXTERNAL stream aligned by character offset.

Design in one paragraph
-----------------------
The cache (`build_root_cache.py`) is at WORD granularity: `word_<split>[1]` is one root id per
word event, which is the unit every published number uses.  Each word also knows which base
tokens its surface occupies (`wtokoff` + `wlen` inside its sentence's `tok` slice).  A window is
128 consecutive word events; the trunk is fed the CONCATENATION of those words' base tokens, and
the head reads the hidden state of each word's LAST token.  Padding only ever appends, and the
causal mask means a real token can never attend forward to it, so real positions are unaffected.

Arms (the A/C contrast the brief asks for)
-----------------------------------------
  A  head only, `--no-features`          -> h-only floor on a fresh base (no root pathway)
  C  head + `--root-cross-attn`          -> the mechanism known to give 19.53 % on the old trunk

Both share the SAME cached hidden states, so A-vs-C isolates the RCA exactly.

Init equivalence
----------------
`--rca-init-check` runs the trunk with the stack ATTACHED and every gate at 0, and compares the
hidden states bitwise against the identical forward without the stack.  With `--rca-out-norm`
the comparison is done there too.  Zero tolerance: any difference is a failure.
"""
import argparse
import json
import math
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

RELEASE = Path('/workspace/hf_v19_2_release')
ALT = Path('/workspace/alt_base')
WIN = 128
ARABIC = None


# ----------------------------------------------------------------------------- head
class NRMTHead(nn.Module):
    """Byte-faithful re-implementation of `nrmt_arch.NRMTHead` (linear-features path only)."""

    def __init__(self, d_model, num_roots, num_awzan, num_prefixes, num_suffixes, d_root,
                 hist=3, d_hist=64, n_ops=1, d_op=32, d_morph=32, dropout=0.1,
                 use_features=True, feat_gate=False, feat_gate_proj_std=1e-3):
        super().__init__()
        self.d_model, self.d_root, self.hist = d_model, d_root, hist
        self.num_roots, self.use_features = num_roots, use_features
        self.feat_gate_enabled = feat_gate
        self.feat_gate_proj_std = feat_gate_proj_std
        if feat_gate:
            self.feat_gate = nn.Parameter(torch.zeros(()))
        self.hist_emb = nn.ModuleList([nn.Embedding(num_roots, d_hist) for _ in range(hist)])
        self.op_emb = nn.Embedding(n_ops, d_op)
        self.prev_wazn_emb = nn.Embedding(num_awzan, d_morph)
        self.prev_pref_emb = nn.Embedding(num_prefixes, d_morph)
        self.prev_suff_emb = nn.Embedding(num_suffixes, d_morph)
        d_extra = hist * d_hist + d_op + 3 * d_morph
        self.feat_proj = nn.Linear(d_extra, d_model, bias=False)
        self.feat_norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.root_head = nn.Linear(d_model, num_roots, bias=False)
        self.cond_proj = nn.Sequential(
            nn.Linear(d_model + d_root, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.wazn_head = nn.Linear(d_model, num_awzan, bias=False)
        self.prefix_head = nn.Linear(d_model, num_prefixes, bias=False)
        self.suffix_head = nn.Linear(d_model, num_suffixes, bias=False)
        if feat_gate and feat_gate_proj_std > 0.0:
            nn.init.normal_(self.feat_proj.weight, std=feat_gate_proj_std)
        else:
            nn.init.zeros_(self.feat_proj.weight)

    def build_features(self, root_ids, op_ids, w_ids, p_ids, s_ids):
        if not self.use_features:
            return torch.zeros(*root_ids.shape, self.d_model,
                               device=root_ids.device, dtype=self.feat_proj.weight.dtype)
        parts = []
        for k in range(1, self.hist + 1):
            sh = torch.roll(root_ids, shifts=k, dims=1)
            sh[:, :k] = 0
            parts.append(self.hist_emb[k - 1](sh))
        pw = torch.roll(w_ids, 1, dims=1); pw[:, 0] = 0
        pp = torch.roll(p_ids, 1, dims=1); pp[:, 0] = 0
        ps = torch.roll(s_ids, 1, dims=1); ps[:, 0] = 0
        parts += [self.op_emb(op_ids), self.prev_wazn_emb(pw),
                  self.prev_pref_emb(pp), self.prev_suff_emb(ps)]
        out = self.feat_norm(self.feat_proj(torch.cat(parts, dim=-1)))
        if self.feat_gate_enabled:
            out = out * self.feat_gate
        return self.drop(out)

    def forward(self, h, root_ids, op_ids, w_ids, p_ids, s_ids, cond_roots=None,
                root_embed=None):
        h_aug = h + self.build_features(root_ids, op_ids, w_ids, p_ids, s_ids).to(h.dtype)
        root_logits = self.root_head(h_aug)
        if root_embed is not None:
            chosen = cond_roots if cond_roots is not None else root_logits.argmax(-1)
            e_root = root_embed(chosen)
            h_cond = self.cond_proj(torch.cat([h_aug, e_root.to(h_aug.dtype)], dim=-1))
        else:
            h_cond = h_aug
        return {'root_logits': root_logits, 'wazn_logits': self.wazn_head(h_cond),
                'prefix_logits': self.prefix_head(h_cond), 'suffix_logits': self.suffix_head(h_cond)}


def load_matching(model, sd, tag='ckpt'):
    """Load only shape-matching tensors; report every skip.  The released awzan142 checkpoint
    predates the current blueprint (9868-token/9114-root vs 10052/9490), so the two embedding
    tables differ in ROW COUNT.  Everything else must match."""
    ms = model.state_dict()
    keep, skipped = {}, []
    for k, v in sd.items():
        if k in ms and tuple(ms[k].shape) == tuple(v.shape):
            keep[k] = v
        elif k in ms:
            skipped.append((k, tuple(v.shape), tuple(ms[k].shape)))
        else:
            skipped.append((k, tuple(v.shape), None))
    model.load_state_dict(keep, strict=False)
    print(f'[*] {tag}: loaded {len(keep)}/{len(sd)} tensors; skipped {len(skipped)}', flush=True)
    for k, a, b in skipped[:10]:
        print(f'    SKIP {k}: ckpt{a} vs model{b}', flush=True)
    return keep, skipped


def remap_legacy_head(sd, head_state, d_model, cond_init='truncate'):
    """Map legacy `nrmp_head.*` tensors onto this head (see nrmt_train_width.remap_legacy_head)."""
    applied, exempt = [], []
    out = {}
    for k, v in head_state.items():
        src = 'nrmp_head.' + k
        if src not in sd:
            exempt.append(k)
            continue
        t = sd[src]
        if tuple(t.shape) == tuple(v.shape):
            out[k] = t
            applied.append(k)
        elif k == 'cond_proj.0.weight' and t.dim() == 2 and t.shape[0] == v.shape[0]:
            new = torch.zeros_like(v)
            d_old = t.shape[1] - d_model
            keep = min(d_model, d_old)
            new[:, :keep] = t[:, :keep]
            out[k] = new
            applied.append(k + f'[truncate {tuple(t.shape)}->{tuple(v.shape)}]')
        else:
            exempt.append(k + f'[shape {tuple(t.shape)}]')
    return out, applied, exempt


# ----------------------------------------------------------------------------- data
def load_cache(d):
    tr = {n: torch.load(Path(d) / f'{n}_train.pt', map_location='cpu')
          for n in ('word', 'tok', 'soff', 'wtokoff', 'wstart', 'wlen')}
    va = {n: torch.load(Path(d) / f'{n}_val.pt', map_location='cpu')
          for n in ('word', 'tok', 'soff', 'wtokoff', 'wstart', 'wlen')}
    return tr, va


def blocks_for(wlen, win=WIN):
    nw = int(wlen.numel())
    nb = nw // win
    return torch.arange(nb, dtype=torch.long) * win, nb


def sent_of_word(wstart, idx):
    """sentence index for each word index (wstart is per-sentence word start)."""
    return torch.searchsorted(wstart, idx, right=True) - 1


def gather_window(tr, b, tok_wstart=None):
    """Contiguous token span of 128 consecutive word events.

    `tok_wstart[i]` = index of word i's FIRST base token in the global token stream.  Words are
    stored sentence by sentence, and within a sentence consecutively, so 128 consecutive words
    occupy a CONTIGUOUS slice of the token stream.  Taking the union of the first and last word's
    sentence spans (as an earlier version did) picks up other sentences' tokens in between, which
    both wastes compute and pushes `last_local` past the end of the gathered tensor.
    """
    ws, we = b * WIN, (b + 1) * WIN
    wl = tr['wlen'][ws:we].long()
    if tok_wstart is None:
        tok_wstart = token_cumstart(tr)
    tok = tr['tok'].long()
    start = int(tok_wstart[ws])
    end = int(tok_wstart[we - 1] + wl[-1])
    ids = tok[start:end]
    wid = torch.repeat_interleave(torch.arange(WIN, dtype=torch.long), wl)
    first = tok_wstart[ws:we]
    last_local = (first - start) + wl - 1
    assert int(last_local.max()) < ids.numel(), (int(last_local.max()), ids.numel())
    words = torch.arange(ws, we)
    return ids, wid, last_local, wl, words


def build_novel_mask(tr_roots, va_roots, va_blocks, log=None, ctx=12):
    """Novelty of each val position's `ctx`-root context w.r.t. the TRAIN root stream."""
    a = np.asarray(tr_roots.tolist(), dtype=np.uint64)
    B_ = np.uint64(1000003)
    L = max(len(a) - ctx + 1, 0)
    hh = np.zeros(L, dtype=np.uint64)
    for k in range(ctx):
        hh = hh * B_ + a[k:k + L]
    seen = set(np.unique(hh).tolist()) if L else set()
    n = len(va_blocks)
    mask = torch.zeros(n, WIN - 1, dtype=torch.bool)
    for wi, b in enumerate(va_blocks):
        seg = va_roots[int(b) * WIN:(int(b) + 1) * WIN].tolist()
        for t in range(ctx, WIN):
            h_ = 0
            for x in seg[t - ctx:t]:
                h_ = (h_ * 1000003 + int(x)) & ((1 << 64) - 1)
            mask[wi, t - 1] = h_ not in seen
    if log:
        log(f'novelty: {len(seen)} distinct {ctx}-gram hashes; val novel fraction='
            f'{100.0*float(mask.sum())/max(mask.numel(),1):.1f}%')
    return mask


def token_cumstart(d):
    """Index of each word's first base token in the global token stream (cumulative wlen)."""
    wl = d['wlen'].long()
    cs = torch.zeros(wl.numel() + 1, dtype=torch.long)
    torch.cumsum(wl, 0, out=cs[1:])
    return cs


class LRows:
    """Lazy block -> (ids, wid, last_local).  Only the sampled blocks are ever materialised."""

    def __init__(self, d, blocks):
        self.d = d
        self.cs = token_cumstart(d)
        self.blocks = [int(b) for b in blocks]

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, b):
        ids, wid, last_local, wl, _ = gather_window(self.d, int(b), self.cs)
        return ids.to(torch.int32), wid.to(torch.int32), last_local


def build_word_rows(tr, perm, device, fp16=True):
    """Precompute per-window token ids and the per-word hidden-state index map.

    Returns list of tuples (ids[T] int32, wid[T] int32, last_local[128] int64).
    """
    rows = []
    cs = token_cumstart(tr)
    for b in perm:
        ids, wid, last_local, wl, _ = gather_window(tr, int(b), cs)
        rows.append((ids.to(torch.int32), wid.to(torch.int32), last_local))
    return rows


def _word_ids_of(bidx, win=WIN):
    return torch.stack([torch.arange(int(b) * win, (int(b) + 1) * win) for b in bidx])


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default='/workspace/alt_base/cache/rootqwen')
    ap.add_argument('--base', default='Qwen/Qwen2.5-0.5B')
    ap.add_argument('--ckpt', default=str(RELEASE / 'checkpoints/'
                    'rootformer_v19_2_synthesis_ar_backbone.awzan142.safetensors'))
    ap.add_argument('--tag', default='A')
    ap.add_argument('--steps', type=int, default=6000)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--eval-every', type=int, default=1000)
    ap.add_argument('--train-windows', type=int, default=0, help='0 = all 128-word blocks')
    ap.add_argument('--val-windows', type=int, default=400)
    ap.add_argument('--no-features', action='store_true')
    ap.add_argument('--root-cross-attn', default='none', help='none | all | topN | everyK | a,b,c')
    ap.add_argument('--score-bias', action='store_true',
                    help='ALSO add the native Farahidian root term to the attention SCORE '
                         '(reweights, does not inject).  Only meaningful with --root-cross-attn.')
    ap.add_argument('--rca-heads', type=int, default=8)
    ap.add_argument('--rca-dim', type=int, default=0, help='0 = d_model')
    ap.add_argument('--rca-dropout', type=float, default=0.0)
    ap.add_argument('--rca-out-std', type=float, default=1e-3)
    ap.add_argument('--rca-out-norm', action='store_true')
    ap.add_argument('--rca-lr-scale', type=float, default=1.0)
    ap.add_argument('--rca-ablate-eval', action='store_true')
    ap.add_argument('--feat-gate', action='store_true')
    ap.add_argument('--feat-gate-proj-std', type=float, default=1e-3)
    ap.add_argument('--margin-ramp', type=int, default=0)
    ap.add_argument('--margin', type=float, default=-2.0)
    ap.add_argument('--margin-start', type=float, default=50.0)
    ap.add_argument('--grad-warmup', type=int, default=0)
    ap.add_argument('--grad-warmup-thresh', type=float, default=50.0)
    ap.add_argument('--grad-clip', type=float, default=1.0)
    ap.add_argument('--logit-scale', choices=('raw', 'ln', 'l2'), default='raw')
    ap.add_argument('--hist', type=int, default=3)
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--head-init', choices=('remap', 'scratch'), default='remap')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--max-tokens', type=int, default=0, help='0 = derive from cache p99.9')
    ap.add_argument('--rca-init-check', action='store_true')
    ap.add_argument('--precompute-only', action='store_true')
    ap.add_argument('--live', action='store_true',
                    help='END-TO-END: run the trunk in the forward pass every step (trunk '
                         'gradients enabled).  Required for the root pathway to be trainable '
                         'from the INPUT stage upward.  Bypasses the frozen hidden-state cache.')
    ap.add_argument('--trunk-lr-scale', type=float, default=1.0,
                    help='lr of the pretrained trunk weights as a fraction of --lr (1.0 = full LR; '
                         'English preservation is explicitly NOT a requirement)')
    ap.add_argument('--bench', type=int, default=0,
                    help='>0: time that many steps, print s/step and VRAM, then exit')
    ap.add_argument('--out', default='')
    ap.add_argument('--save', default='')
    ap.add_argument('--probe-out', default='')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    T0 = time.time()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats()
    dev = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    RD = Path(args.out).parent if args.out else ALT
    RD.mkdir(parents=True, exist_ok=True)
    args.out = args.out or str(RD / f'results_{args.tag}.json')
    args.save = args.save or str(RD / f'head_{args.tag}.pt')
    probe = open(args.probe_out, 'w') if args.probe_out else None

    def log(*a):
        m = f'[{time.time()-T0:7.1f}s] ' + ' '.join(str(x) for x in a)
        print(m, flush=True)
        if probe:
            probe.write(json.dumps({'t': round(time.time() - T0, 1), 'msg': m}) + '\n')
            probe.flush()

    # ---- vocab + operator table -------------------------------------------------------
    sys.path.insert(0, str(RELEASE)); sys.path.insert(0, str(RELEASE / 'models'))
    import nrmp_vocab as nv
    V = next(v for k, v in vars(nv).items() if isinstance(v, type) and 'MorphemicVocab' in k)
    vocab = V(str(RELEASE / 'data/rootformer_v12_arabic_blueprint.json'))
    log(f'vocab roots={vocab.num_roots} awzan={vocab.num_awzan} prefixes={vocab.num_prefixes} '
        f'suffixes={vocab.num_suffixes}')
    specials = sorted({vocab.PAD_ROOT, vocab.BOS_ROOT, vocab.EOS_ROOT, vocab.UNK_ROOT,
                       vocab.root2id['<PARTICLE>']} |
                      {i for i, r in enumerate(vocab.roots_list) if r.startswith('<P:')})
    spec_t = torch.tensor(specials)
    try:
        from classical_governance_v2 import AlKhalilV2
        forbidden = sorted(AlKhalilV2(vocab).mask())
        log(f'forbidden roots from AlKhalilV2: {len(forbidden)}')
    except Exception as e:
        log(f'[warn] AlKhalilV2 unavailable ({e}); using specials only for the margin hinge')
        forbidden = specials
    fmask = torch.zeros(vocab.num_roots, dtype=torch.bool)
    fmask[torch.tensor(forbidden)] = True

    # ---- cache -------------------------------------------------------------------------
    tr, va = load_cache(args.cache)
    log(f'cache {args.cache}: train words={tr["word"][1].numel()} tokens={tr["tok"].numel()} | '
        f'val words={va["word"][1].numel()} tokens={va["tok"].numel()}')
    ntr = int(tr['word'][1].numel()) // WIN
    nva = int(va['word'][1].numel()) // WIN
    tr_blocks = np.arange(ntr)
    if args.train_windows:
        rng = np.random.default_rng(args.seed)
        tr_blocks = np.sort(rng.choice(ntr, size=min(args.train_windows, ntr), replace=False))
    va_blocks = np.linspace(0, max(nva - 1, 0), min(args.val_windows, nva)).astype(int)
    va_blocks = np.unique(va_blocks)
    log(f'blocks: train {len(tr_blocks)}/{ntr}  val {len(va_blocks)}/{nva}')

    # ---- base model + morphemic root table ---------------------------------------------
    from models.unified_rootformer_v12 import UnifiedRootformerV12
    from deepseek_v4_1_flash_model import UnifiedRootformerV17_DeepSeekFlash
    from safetensors.torch import load_file
    from nrmt_arch import RootformerNRMT
    base = UnifiedRootformerV12(str(RELEASE / 'data/rootformer_v12_arabic_blueprint.json'),
                                args.base, str(dev), torch.bfloat16).to(dev)
    flash = UnifiedRootformerV17_DeepSeekFlash(base, dev, torch.bfloat16).to(dev)
    model = RootformerNRMT(flash, vocab, dev, torch.bfloat16, hist=args.hist,
                           dropout=args.dropout, use_features=not args.no_features,
                           feat_gate=args.feat_gate,
                           feat_gate_proj_std=args.feat_gate_proj_std).to(dev)
    sd = load_file(args.ckpt)
    load_matching(model, sd, Path(args.ckpt).name)
    # PURE BASE: the native root term reads SUBWORD token ids clamped to [0,9014], which are
    # meaningless for a 151,643-entry base vocabulary.  Disable it so the trunk is the untouched
    # base model and OUR root path is the only one carrying root information.
    for _l in model.backbone.layers:
        _l.self_attn.active_root_ids = None
        _l.self_attn.active_wazn_ids = None
    log('native root path DISABLED on all layers (trunk = pure base)')
    for p in model.parameters():
        p.requires_grad = False
    model.nrmt_head.to(torch.float32)
    root_embed = model.morphemic_embed.root_embed          # (9490, 448), the LIVE root table
    log(f'root_embed table {tuple(root_embed.weight.shape)} '
        f'requires_grad={root_embed.weight.requires_grad}')

    # ---- our own head (replaces the checkpoint's, which is a different d_root space) ----
    from nrmt_arch import OP_STATES
    head = NRMTHead(model.d_model, vocab.num_roots, vocab.num_awzan, vocab.num_prefixes,
                    vocab.num_suffixes, int(root_embed.weight.shape[1]), hist=args.hist,
                    n_ops=len(OP_STATES),
                    dropout=args.dropout, use_features=not args.no_features,
                    feat_gate=args.feat_gate,
                    feat_gate_proj_std=args.feat_gate_proj_std).to(dev).float()
    hs = head.state_dict()
    if args.head_init == 'remap':
        rm, applied, exempt = remap_legacy_head(sd, hs, model.d_model)
        res = head.load_state_dict(rm, strict=False)
        log(f'head warm start: applied={len(applied)} exempt={len(exempt)} '
            f'missing_after={len(res.missing_keys)}')
        for k in applied:
            log(f'    applied {k}')
        for k in exempt:
            log(f'    EXEMPT  {k}')
    else:
        log('head scratch init')

    # ---- IN-ARCHITECTURE root attention -------------------------------------------------
    rca_stack = None
    rca_params = []
    if str(args.root_cross_attn).strip().lower() not in ('', 'none', 'off', '0'):
        import importlib.util as _iu
        _spec = _iu.spec_from_file_location('alt_root_attn', '/workspace/alt_base/alt_root_attn.py')
        _mod = _iu.module_from_spec(_spec); _spec.loader.exec_module(_mod)
        ScoreBiasModule = _mod.ScoreBiasModule
        AltRootCrossAttentionStack = _mod.AltRootCrossAttentionStack
        nl = len(model.backbone.layers)
        if str(args.root_cross_attn).strip().lower() == 'all':
            layers_idx = list(range(nl))
        else:
            from root_cross_attn_width import parse_layer_spec
            layers_idx = parse_layer_spec(args.root_cross_attn, nl)
        sb = None
        if args.score_bias:
            a0 = model.backbone.layers[0].self_attn
            sb = ScoreBiasModule(vocab.num_roots, a0.head_dim, a0.num_heads, a0.num_kv_heads,
                                 a0.scale, dtype=torch.float32).to(dev)
        rca_stack = AltRootCrossAttentionStack(
            model.d_model, int(root_embed.weight.shape[1]), root_embed.weight, layers_idx,
            score_bias=sb, residual=True, num_heads=args.rca_heads, dropout=args.rca_dropout,
            out_std=args.rca_out_std, out_norm=bool(args.rca_out_norm),
            dtype=torch.float32).to(dev)
        rca_stack.attach(model.backbone.layers)
        rca_params = list(rca_stack.parameters())
        log(f'RCA layers={layers_idx[:6]}{"..." if len(layers_idx)>6 else ""} '
            f'params={sum(p.numel() for p in rca_params)/1e6:.2f}M score_bias={sb is not None} '
            f'gates0={rca_stack.gate_values()}')

    head_params = [p for p in head.parameters() if p.requires_grad]
    all_params = head_params + rca_params
    RES = {'meta': {'tag': args.tag, 'base': args.base, 'steps': args.steps,
                    'batch_size': args.batch_size, 'cache': args.cache,
                    'no_features': args.no_features, 'root_cross_attn': args.root_cross_attn,
                    'live': bool(args.live),
                    'trunk_lr_scale': args.trunk_lr_scale,
                    'rca_layers': rca_stack.layer_indices if rca_stack else [],
                    'rca_params_m': sum(p.numel() for p in rca_params) / 1e6,
                    'head_params_m': sum(p.numel() for p in head_params) / 1e6,
                    'logit_scale': args.logit_scale, 'feat_gate': args.feat_gate,
                    'rca_out_norm': args.rca_out_norm, 'seed': args.seed},
           'arms': {}, 'history': []}
    log(f'head params={sum(p.numel() for p in head_params)/1e6:.3f}M (+rca '
        f'{sum(p.numel() for p in rca_params)/1e6:.3f}M)')


    # ---- token budget ------------------------------------------------------------------
    rows_tr = build_word_rows(tr, tr_blocks, dev)
    rows_tr_by_b = {int(b): rows_tr[i] for i, b in enumerate(tr_blocks)}
    lens = np.array([int(r[0].numel()) for r in rows_tr])
    if args.max_tokens:
        TMAX = args.max_tokens
    else:
        TMAX = int(np.percentile(lens, 99.9)) + 2
    log(f'token lengths: p50={int(np.median(lens))} p99={int(np.percentile(lens,99))} '
        f'max={int(lens.max())} -> TMAX={TMAX} (windows clipped: '
        f'{int((lens>=TMAX).sum())})')

    @torch.no_grad()
    def run_trunk(ids, wid, last_local, root_ids=None, mask=None, out_all=False):
        """ids [B,T] long -> per-word hidden states [B,128,d] (float32)."""
        B, T = ids.shape
        if rca_stack is not None:
            rca_stack._attn_mask = mask.to(dev) if mask is not None else None
        o = model.backbone(input_ids=ids.to(dev), attention_mask=mask.to(dev) if mask is not None else None)
        h = model.final_norm(o.last_hidden_state)
        if out_all:
            return h.float()
        ll = last_local.to(dev)                              # [B,128]
        idx = ll.unsqueeze(-1).expand(-1, -1, h.shape[-1])
        return h.gather(1, idx).float()

    def make_batch(bidx):
        bs = [rows_tr[b] for b in bidx]
        T = max(int(x[0].numel()) for x in bs)
        B = len(bs)
        ids = torch.zeros(B, T, dtype=torch.long)
        wid = torch.zeros(B, T, dtype=torch.long)
        mask = torch.zeros(B, T, dtype=torch.long)
        ll = torch.zeros(B, WIN, dtype=torch.long)
        for i, (a, b, c) in enumerate(bs):
            n = min(int(a.numel()), T)
            ids[i, :n] = a[:n].long(); wid[i, :n] = b[:n].long()
            mask[i, :n] = 1
            ll[i] = c.clamp(max=T - 1)
        return ids, wid, mask, ll

    def root_tok_of_window(wid, words):
        """per-token root id [B,T] from the per-token word index, using the word root stream."""
        R = tr['word'][1].long()
        rw = R[words].to(wid.device)                         # [B,128]
        return rw.gather(1, wid)

    # ---- init-equivalence: gate 0 must be a bitwise no-op ------------------------------
    if args.rca_init_check:
        log('=== INIT EQUIVALENCE CHECK (gate forced to 0) ===')
        _ib = [int(tr_blocks[0]), int(tr_blocks[1])]
        ids, wid, mask, ll = make_batch(_ib)
        rca_stack.set_root_ids(None)
        h_off = run_trunk(ids, wid, ll, mask=mask, out_all=True)
        rca_stack.set_root_ids(root_tok_of_window(wid, _word_ids_of(_ib)))
        rca_stack.counts['calls'] = 0; rca_stack.counts['with_root_ids'] = 0
        h_on = run_trunk(ids, wid, ll, mask=mask, out_all=True)
        d = (h_on.float() - h_off.float()).abs().max().item()
        bits = bool(torch.equal(h_on, h_off))
        log(f'  rca calls={rca_stack.counts["calls"]} with_root_ids='
            f'{rca_stack.counts["with_root_ids"]} residual={rca_stack.counts["residual"]}')
        log(f'  T={ids.shape[1]} tokens, {int(mask.sum())} real; '
            f'max|h_with_stack - h_without| = {d:.3e}   bitwise_identical={bits}')
        log(f'  gates at init = {rca_stack.gate_values()}')
        if not bits:
            log('  *** FAIL: the attached stack is NOT a numerical no-op at init ***')
            raise SystemExit(2)
        log('  PASS: attached-but-untrained is a bitwise no-op')

    # ---- precompute hidden states ------------------------------------------------------
    def precompute(split_rows, blocks, tag):
        out = []
        order = np.argsort(np.array([int(split_rows[b][0].numel()) for b in blocks]))
        log(f'precompute {tag}: {len(blocks)} windows (length-sorted)')
        for s in range(0, len(blocks), args.batch_size):
            bidx = [int(blocks[i]) for i in order[s:s + args.batch_size]]
            ids, wid, mask, ll = make_batch(bidx)
            h = run_trunk(ids, wid, ll, mask=mask)
            out.append((h.half().cpu(), torch.tensor(bidx)))
        H = torch.cat([x[0] for x in out])
        I = torch.cat([x[1] for x in out])
        log(f'  {tag} hidden {tuple(H.shape)} (dtype {H.dtype})')
        return H, I

    rows_va = build_word_rows(va, va_blocks, dev)
    rows_va_by_b = {int(b): rows_va[i] for i, b in enumerate(va_blocks)}

    def make_batch_from(rows_map, bidx):
        bs = [rows_map[int(b)] for b in bidx]
        T = max(int(x[0].numel()) for x in bs)
        B = len(bs)
        ids = torch.zeros(B, T, dtype=torch.long); wid = torch.zeros(B, T, dtype=torch.long)
        mask = torch.zeros(B, T, dtype=torch.long); ll = torch.zeros(B, WIN, dtype=torch.long)
        for i, (a, b, c) in enumerate(bs):
            n = min(int(a.numel()), T)
            ids[i, :n] = a[:n].long(); wid[i, :n] = b[:n].long(); mask[i, :n] = 1
            ll[i] = c.clamp(max=T - 1)
        return ids, wid, mask, ll, torch.tensor([int(b) for b in bidx])

    # ---- aligned per-word tensors ------------------------------------------------------
    def aligned(split, blocks, H, I):
        """Reorder hidden states to the `blocks` order and build per-word input tensors."""
        rows = build_word_rows(split, blocks)
        assert len(rows) == H.shape[0], (len(rows), H.shape)
        # I[] holds the block ids in the order they were computed; H rows follow that order.
        pos = {int(b): i for i, b in enumerate(I.tolist())}
        idx = torch.tensor([pos[int(b)] for b in blocks])
        Hr = H[idx].to(dev).float()
        R = split['word'][1].long(); Wz = split['word'][2].long()
        P = split['word'][0].long(); S = split['word'][3].long()
        nb = len(blocks)
        Rw = torch.stack([R[int(b) * WIN:(int(b) + 1) * WIN] for b in blocks])
        Ww = torch.stack([Wz[int(b) * WIN:(int(b) + 1) * WIN] for b in blocks])
        Pw = torch.stack([P[int(b) * WIN:(int(b) + 1) * WIN] for b in blocks])
        Sw = torch.stack([S[int(b) * WIN:(int(b) + 1) * WIN] for b in blocks])
        return Hr, Rw.to(dev), Ww.to(dev), Pw.to(dev), Sw.to(dev), rows, nb

    Htr_a, Rtr, Wtr, Ptr, Str, rows_trA, ntrA = aligned(tr, tr_blocks, Htr, Itr)
    Hva_a, Rva, Wva, Pva, Sva, rows_vaA, nvaA = aligned(va, va_blocks, Hva, Iva)
    log(f'word tensors: train {tuple(Htr_a.shape)} {tuple(Rtr.shape)} | '
        f'val {tuple(Hva_a.shape)} peak_vram={torch.cuda.max_memory_allocated()/2**20:.0f}MiB')

    # targets are the NEXT word's tuple (positions 0..T-2 -> targets 1..T-1)
    def slices(H, R, W, P, S, i):
        h = H[i, :-1].contiguous()
        r = R[i, :-1].contiguous()
        tgt_r = R[i, 1:].contiguous()
        tgt_w = W[i, 1:].contiguous()
        tgt_p = P[i, 1:].contiguous()
        tgt_s = S[i, 1:].contiguous()
        return h, r, W[i, :-1].contiguous(), P[i, :-1].contiguous(), S[i, :-1].contiguous(), \
            tgt_r, tgt_w, tgt_p, tgt_s

    # ---- operator ids (Sibawayh ʿāmil when available, else the t-1 operator table) -----
    op_table = None
    try:
        from nrmt_arch import build_operator_table
        op_table = build_operator_table(vocab).to(dev)
        log(f'operator table {tuple(op_table.shape)}')
    except Exception as e:
        log(f'[warn] build_operator_table failed: {e}')

    def op_of(Rw, Ww, Pw):
        if op_table is None:
            return torch.zeros_like(Rw)
        return op_table[Rw.clamp(min=0)]

    # ---- loss --------------------------------------------------------------------------
    def compute_loss(H, R, W, P, S, O, tgt_r, tgt_w, tgt_p, tgt_s, margin=None):
        O = op_of(R, W, P) if O is None else O
        out = head(H, R, O, W, P, S, cond_roots=tgt_r, root_embed=root_embed)
        keep = ~torch.isin(tgt_r, spec_t.to(tgt_r.device))
        rl_raw = out['root_logits']
        if args.logit_scale == 'ln':
            rl = (rl_raw - rl_raw.mean(-1, keepdim=True)) / \
                rl_raw.std(-1, keepdim=True).clamp_min(1e-6)
        elif args.logit_scale == 'l2':
            rl = rl_raw / rl_raw.norm(dim=-1, keepdim=True).clamp_min(1e-6) \
                * math.sqrt(vocab.num_roots)
        else:
            rl = rl_raw
        info = {}
        l_root = F.cross_entropy(rl[keep], tgt_r[keep]) if keep.any() else rl.sum() * 0.0
        l_wazn = F.cross_entropy(out['wazn_logits'].reshape(-1, vocab.num_awzan),
                                 tgt_w.reshape(-1), ignore_index=vocab.PAD_WAZN)
        l_pref = F.cross_entropy(out['prefix_logits'].reshape(-1, vocab.num_prefixes),
                                 tgt_p.reshape(-1), ignore_index=vocab.PAD_PREFIX)
        l_suff = F.cross_entropy(out['suffix_logits'].reshape(-1, vocab.num_suffixes),
                                 tgt_s.reshape(-1), ignore_index=vocab.PAD_SUFFIX)
        loss = l_root + 0.5 * l_wazn + 0.25 * l_pref + 0.25 * l_suff
        info.update({'root': float(l_root.detach()), 'wazn': float(l_wazn.detach()),
                     'prefix': float(l_pref.detach()), 'suffix': float(l_suff.detach())})
        if keep.any():
            rr = rl_raw[keep]
            rs = rr.std(-1, keepdim=True).clamp_min(1e-6)
            info['root_cez'] = float(F.cross_entropy(
                (rr - rr.mean(-1, keepdim=True)) / rs, tgt_r[keep]).detach())
        fm = fmask.to(tgt_r.device)[tgt_r]
        m = args.margin if margin is None else margin
        if fm.any() and m != 0.0:
            pen = F.relu(rl_raw[fm] - m) ** 2
            loss = loss + 0.1 * pen.mean()
            info['impossible'] = float(pen.mean().detach()); info['margin'] = m
        # accuracy on this batch (all / novel not available live, so report all)
        with torch.no_grad():
            if keep.any():
                info['acc1'] = float((rl_raw[keep].argmax(-1) == tgt_r[keep]).float().mean())
                info['n'] = int(keep.sum())
        return loss, info

    def margin_at(step):
        if args.margin_ramp <= 0:
            return args.margin
        f = min(1.0, (step - 1) / max(args.margin_ramp - 1, 1))
        return args.margin_start + (args.margin - args.margin_start) * f

    # ---- evaluation (frozen protocol: free-running condition, gold targets) ------------
    @torch.no_grad()
    def evaluate(H, R, W, P, S, ablate=False, bs=32):
        head.eval()
        if ablate and rca_stack is not None:
            old = rca_stack.zero_gates()
        n_all = n_nov = c_all = c_nov = 0
        morph = Counter(); morph_n = 0
        for s in range(0, H.shape[0], bs):
            sl = slice(s, min(s + bs, H.shape[0]))
            h = H[sl, :-1]
            r = R[sl, :-1]; tgt = R[sl, 1:]
            O = op_of(r, W[sl, :-1], P[sl, :-1])
            out = head(h, r, O, W[sl, :-1], P[sl, :-1], S[sl, :-1], cond_roots=None,
                       root_embed=root_embed)
            rl = out['root_logits']
            if args.logit_scale == 'ln':
                rl = (rl - rl.mean(-1, keepdim=True)) / rl.std(-1, keepdim=True).clamp_min(1e-6)
            keep = ~torch.isin(tgt, spec_t.to(tgt.device))
            pred = out['root_logits'].argmax(-1)
            nov = novel_mask[sl]
            c_all += int(((pred == tgt) & keep).sum()); n_all += int(keep.sum())
            c_nov += int(((pred == tgt) & keep & nov).sum()); n_nov += int((keep & nov).sum())
            for nm, lg, tg, ig in (('wazn', out['wazn_logits'], W[sl, 1:], vocab.PAD_WAZN),
                                   ('prefix', out['prefix_logits'], P[sl, 1:], vocab.PAD_PREFIX),
                                   ('suffix', out['suffix_logits'], S[sl, 1:], vocab.PAD_SUFFIX)):
                p = lg.argmax(-1); m = tg != ig
                morph[nm] += int(((p == tg) & m).sum()); 
            morph_n += int((W[sl, 1:] != vocab.PAD_WAZN).sum())
        if ablate and rca_stack is not None:
            rca_stack.restore_gates(old)
        head.train()
        return {'root_acc_all_pct': 100.0 * c_all / max(n_all, 1),
                'root_acc_novel_pct': 100.0 * c_nov / max(n_nov, 1),
                'n_all': n_all, 'n_novel': n_nov,
                'wazn_acc_pct': 100.0 * morph['wazn'] / max(morph_n, 1),
                'prefix_acc_pct': 100.0 * morph['prefix'] / max(morph_n, 1),
                'suffix_acc_pct': 100.0 * morph['suffix'] / max(morph_n, 1)}

    # novel mask over val word positions (order = va_blocks order, 127 positions each)
    trR = tr['word'][1].tolist()
    B_ = np.uint64(1000003)
    a = np.asarray(trR, dtype=np.uint64)
    L = len(a) - 12 + 1
    hh = np.zeros(L, dtype=np.uint64)
    for k in range(12):
        hh = hh * B_ + a[k:k + L]
    nov_hash = set(np.unique(hh).tolist())
    novel_mask = torch.zeros(len(va_blocks), WIN - 1, dtype=torch.bool)
    for wi, b in enumerate(va_blocks):
        seg = va['word'][1][int(b) * WIN:(int(b) + 1) * WIN].tolist()
        for t in range(12, WIN):
            ctx = seg[t - 12:t]
            h_ = 0
            for x in ctx:
                h_ = (h_ * 1000003 + int(x)) & ((1 << 64) - 1)
            novel_mask[wi, t - 1] = h_ not in nov_hash
    log(f'val windows={len(va_blocks)} radical(val) novel fraction='
        f'{100.0*float(novel_mask.sum())/novel_mask.numel():.1f}%')

    if args.live:
        # END-TO-END: the trunk is TRAINED at full LR, so the frozen hidden-state cache is not
        # used (and must not be, or the trunk would never receive a gradient).  Rows are built
        # lazily per sampled block -- materialising all 55k windows would be pointless work.
        live_tr_by_b = LRows(tr, tr_blocks)
        live_va_by_b = LRows(va, va_blocks)
        log('live rows: lazy per-block construction')
        return run_live_suite(args, dev, T0, log, model, head, rca_stack, rca_params,
                              root_embed, spec_t, fmask, vocab, live_tr_by_b, live_va_by_b,
                              tr_blocks, va_blocks,
                              lambda rm, bi: make_batch_from(rm, bi),
                              novel_mask, RES, args.out, args.save, probe)

    # ---- optimizer ---------------------------------------------------------------------
    groups = [{'params': head_params, 'lr': args.lr, 'grp': 'head'}]
    if rca_params:
        groups.append({'params': rca_params, 'lr': args.lr * args.rca_lr_scale, 'grp': 'rca'})
    opt = torch.optim.AdamW(groups, lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[g['lr'] for g in groups], total_steps=args.steps,
        pct_start=max(0.03, min(0.3, 500.0 / max(args.steps, 1))))
    log('optimizer: ' + ' | '.join(f"{g['grp']} n={len(g['params'])} max_lr={g['lr']:g}"
                                   for g in opt.param_groups))

    # ---- train -------------------------------------------------------------------------
    head.train()
    rng = np.random.default_rng(args.seed)
    nb_tr = Htr_a.shape[0]
    RES['meta'].update({'n_train_windows': int(nb_tr),
                        'n_val_windows': int(len(va_blocks))})
    Hva_ab = None
    for step in range(1, args.steps + 1):
        bidx = rng.integers(0, nb_tr, size=min(args.batch_size, nb_tr))
        h, r, w, p, s, tgt_r, tgt_w, tgt_p, tgt_s = slices(Htr_a, Rtr, Wtr, Ptr, Str, bidx)
        O = op_of(r, w, p)
        if rca_stack is not None:
            ids, wid, mask, ll, _ = make_batch_from(
                rows_tr_by_b, [int(tr_blocks[i]) for i in bidx])
            Rtok = root_tok_of_window(wid, _word_ids_of(bidx))
            rca_stack.set_root_ids(Rtok)
        loss, info = compute_loss(h, r, w, p, s, O, tgt_r, tgt_w, tgt_p, tgt_s,
                                  margin=margin_at(step))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(all_params, args.grad_clip))
        skip = args.grad_warmup > 0 and step <= args.grad_warmup and gn > args.grad_warmup_thresh
        if not skip:
            opt.step()
        sched.step()
        if rca_stack is not None:
            rca_stack.set_root_ids(None)
        if step % 100 == 0 or step == 1:
            g = rca_stack.gate_values() if rca_stack else []
            log(f'step {step:>6} loss {float(loss):.4f} root {info["root"]:.4f} '
                f'cez {info.get("root_cez", float("nan")):.4f} wazn {info["wazn"]:.4f} '
                f'pref {info["prefix"]:.4f} suff {info["suffix"]:.4f} '
                f'acc1 {info.get("acc1", float("nan")):.4f} gnorm {gn:.3f}'
                + (f' gates [{", ".join(f"{x:.3e}" for x in g)}]' if g else ''))
            RES['history'].append({'step': step, 'loss': float(loss), **info, 'gnorm': gn,
                                   'gates': g})
        if args.eval_every and (step % args.eval_every == 0 or step == args.steps):
            ev = evaluate(Hva_a, Rva, Wva, Pva, Sva)
            log(f'  EVAL step {step}: root acc@1 ALL {ev["root_acc_all_pct"]:.2f}% '
                f'NOVEL {ev["root_acc_novel_pct"]:.2f}% (n={ev["n_all"]}/{ev["n_novel"]}) '
                f'wazn {ev["wazn_acc_pct"]:.2f}% pref {ev["prefix_acc_pct"]:.2f}% '
                f'suff {ev["suffix_acc_pct"]:.2f}%')
            RES['arms'][f'step{step}'] = ev
            if rca_stack is not None and args.rca_ablate_eval:
                ev2 = evaluate(Hva_a, Rva, Wva, Pva, Sva, ablate=True)
                log(f'  EVAL step {step} RCA-GATE-OFF: root acc@1 ALL '
                    f'{ev2["root_acc_all_pct"]:.2f}% NOVEL {ev2["root_acc_novel_pct"]:.2f}%')
                RES['arms'][f'step{step}_gateoff'] = ev2

    RES['meta'].update({
        'peak_vram_allocated_mib': round(torch.cuda.max_memory_allocated() / 2**20, 1),
        'peak_vram_reserved_mib': round(torch.cuda.max_memory_reserved() / 2**20, 1),
        'wall_seconds': round(time.time() - T0, 1),
        'final_gates': rca_stack.gate_values() if rca_stack else []})
    torch.save({'head': head.state_dict(),
                'rca': rca_stack.state_dict() if rca_stack else None,
                'args': vars(args)}, args.save)
    Path(args.out).write_text(json.dumps(RES, ensure_ascii=False, indent=2))
    log(f'wrote {args.out} and {args.save}  wall={RES["meta"]["wall_seconds"]}s '
        f'peak_vram={RES["meta"]["peak_vram_allocated_mib"]}MiB')
    if probe:
        probe.close()


def run_live_suite(args, dev, T0, log, model, head, rca_stack, rca_params, root_embed, spec_t,
                   fmask, vocab, rows_tr_by_b, rows_va_by_b, tr_blocks, va_blocks,
                   make_batch_from, novel_mask, RES, Res_path, save_path, probe):
    """END-TO-END branch: unfreeze the trunk, run it live every step, train at full LR."""
    import json as _json
    log('=== LIVE END-TO-END MODE: trunk is TRAINED (full LR); no frozen-cache bypass ===')
    for p in model.parameters():
        p.requires_grad = True
    head_params = [p for p in head.parameters() if p.requires_grad]
    trunk_params = [p for n, p in model.named_parameters() if p.requires_grad]
    groups = [{'params': head_params, 'lr': args.lr, 'grp': 'head'},
              {'params': trunk_params, 'lr': args.lr * args.trunk_lr_scale, 'grp': 'trunk'}]
    if rca_params:
        groups.insert(1, {'params': rca_params, 'lr': args.lr * args.rca_lr_scale, 'grp': 'rca'})
    opt = torch.optim.AdamW(groups, lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[g['lr'] for g in groups], total_steps=max(args.steps, 1),
        pct_start=max(0.03, min(0.3, 500.0 / max(args.steps, 1))))
    log('optimizer: ' + ' | '.join(f"{g['grp']} n={len(g['params'])} max_lr={g['lr']:g}"
                                   for g in opt.param_groups))
    all_train = head_params + trunk_params + rca_params

    def op_of(Rw, Ww, Pw):
        return torch.zeros_like(Rw)

    def margin_at(step):
        if args.margin_ramp <= 0:
            return args.margin
        f = min(1.0, (step - 1) / max(args.margin_ramp - 1, 1))
        return args.margin_start + (args.margin - args.margin_start) * f

    def compute(h, r, w, p, s, tgt_r, tgt_w, tgt_p, tgt_s, margin=None):
        out = head(h, r, op_of(r, w, p), w, p, s, cond_roots=tgt_r, root_embed=root_embed)
        keep = ~torch.isin(tgt_r, spec_t.to(tgt_r.device))
        rl_raw = out['root_logits']
        if args.logit_scale == 'ln':
            rl = (rl_raw - rl_raw.mean(-1, keepdim=True)) / \
                rl_raw.std(-1, keepdim=True).clamp_min(1e-6)
        elif args.logit_scale == 'l2':
            rl = rl_raw / rl_raw.norm(dim=-1, keepdim=True).clamp_min(1e-6) * math.sqrt(
                vocab.num_roots)
        else:
            rl = rl_raw
        info = {}
        l_root = F.cross_entropy(rl[keep], tgt_r[keep]) if keep.any() else rl.sum() * 0.0
        l_wazn = F.cross_entropy(out['wazn_logits'].reshape(-1, vocab.num_awzan),
                                 tgt_w.reshape(-1), ignore_index=vocab.PAD_WAZN)
        l_pref = F.cross_entropy(out['prefix_logits'].reshape(-1, vocab.num_prefixes),
                                 tgt_p.reshape(-1), ignore_index=vocab.PAD_PREFIX)
        l_suff = F.cross_entropy(out['suffix_logits'].reshape(-1, vocab.num_suffixes),
                                 tgt_s.reshape(-1), ignore_index=vocab.PAD_SUFFIX)
        loss = l_root + 0.5 * l_wazn + 0.25 * l_pref + 0.25 * l_suff
        info.update({'root': float(l_root.detach()), 'wazn': float(l_wazn.detach()),
                     'prefix': float(l_pref.detach()), 'suffix': float(l_suff.detach())})
        if keep.any():
            rr = rl_raw[keep]; rs = rr.std(-1, keepdim=True).clamp_min(1e-6)
            info['root_cez'] = float(F.cross_entropy(
                (rr - rr.mean(-1, keepdim=True)) / rs, tgt_r[keep]).detach())
            info['acc1'] = float((rl_raw[keep].argmax(-1) == tgt_r[keep]).float().mean())
            info['n'] = int(keep.sum())
        fm = fmask.to(tgt_r.device)[tgt_r]
        m = args.margin if margin is None else margin
        if fm.any() and m != 0.0:
            pen = F.relu(rl_raw[fm] - m) ** 2
            loss = loss + 0.1 * pen.mean()
            info['impossible'] = float(pen.mean().detach()); info['margin'] = m
        return loss, info

    def batch_for(rows_map, bidx):
        ids, wid, mask, ll, wids = make_batch_from(rows_map, bidx)
        return ids, wid, mask, ll, wids

    def forward_words(ids, wid, mask, ll, wids, Rw, global_R):
        T = ids.shape[1]
        if rca_stack is not None:
            rca_stack._attn_mask = mask.to(dev)
            Rtok = global_R[wids].to(dev).gather(1, wid.to(dev))
            rca_stack.set_root_ids(Rtok)
        o = model.backbone(input_ids=ids.to(dev),
                           attention_mask=mask.to(dev))
        h = model.final_norm(o.last_hidden_state)
        ll2 = ll.to(dev).clamp(max=T - 1)
        hw = h.gather(1, ll2.unsqueeze(-1).expand(-1, -1, h.shape[-1]))
        W_ = (tr if global_R is tr['word'][1] else va)['word']
        wl = W_[2][wids].long().to(dev); pl = W_[0][wids].long().to(dev)
        sl = W_[3][wids].long().to(dev)
        n = wl.shape[1]
        return hw[:, :n - 1].contiguous(), wl[:, :n - 1], pl[:, :n - 1], sl[:, :n - 1], \
            wl[:, 1:].contiguous(), pl[:, 1:].contiguous(), sl[:, 1:].contiguous(), \
            Rw.to(dev)[:, :n - 1].contiguous(), Rw.to(dev)[:, 1:].contiguous()

    @torch.no_grad()
    def evaluate_live(blocks, rows_map, global_R, ablate=False, maxb=None):
        head.eval(); model.eval()
        if ablate and rca_stack is not None:
            old = rca_stack.zero_gates()
        n_all = n_nov = c_all = c_nov = 0
        mg = Counter(); mn = 0
        bl = list(blocks[:maxb] if maxb else blocks)
        for s in range(0, len(bl), max(1, args.batch_size)):
            bidx = [int(b) for b in bl[s:s + args.batch_size]]
            ids, wid, mask, ll, wids = batch_for(rows_map, bidx)
            Rw = global_R[wids]
            hw, wi, pi, si, tw, tp, ts, ri, tr_ = forward_words(ids, wid, mask, ll, wids, Rw,
                                                                global_R)
            out = head(hw, ri, torch.zeros_like(ri), wi, pi, si, cond_roots=None,
                       root_embed=root_embed)
            keep = ~torch.isin(tr_, spec_t.to(tr_.device))
            pred = out['root_logits'].argmax(-1)
            nov = novel_mask[torch.tensor([list(blocks).index(b) for b in bidx])] \
                if False else None
            c_all += int(((pred == tr_) & keep).sum()); n_all += int(keep.sum())
            if nov is not None:
                c_nov += int(((pred == tr_) & keep & nov.to(dev)).sum())
                n_nov += int((keep & nov.to(dev)).sum())
            for lg, tg, ig, nm in ((out['wazn_logits'], tw, vocab.PAD_WAZN, 'wazn'),
                                   (out['prefix_logits'], tp, vocab.PAD_PREFIX, 'prefix'),
                                   (out['suffix_logits'], ts, vocab.PAD_SUFFIX, 'suffix')):
                m_ = tg != ig
                mg[nm] += int(((lg.argmax(-1) == tg) & m_).sum())
            mn += int((tw != vocab.PAD_WAZN).sum())
        if ablate and rca_stack is not None:
            rca_stack.restore_gates(old)
        head.train(); model.train()
        return {'root_acc_all_pct': 100.0 * c_all / max(n_all, 1),
                'root_acc_novel_pct': (100.0 * c_nov / max(n_nov, 1)) if n_nov else float('nan'),
                'n_all': n_all, 'n_novel': n_nov,
                'wazn_acc_pct': 100.0 * mg['wazn'] / max(mn, 1),
                'prefix_acc_pct': 100.0 * mg['prefix'] / max(mn, 1),
                'suffix_acc_pct': 100.0 * mg['suffix'] / max(mn, 1)}

    model.train()
    rng = np.random.default_rng(args.seed)
    nb_tr = len(tr_blocks)
    if args.bench:
        t0 = time.time()
        for step in range(1, args.bench + 1):
            bidx = rng.integers(0, nb_tr, size=min(args.batch_size, nb_tr))
            ids, wid, mask, ll, wids = batch_for(rows_tr_by_b, [int(tr_blocks[i]) for i in bidx])
            Rw = tr['word'][1][wids]
            if rca_stack is not None:
                Rtok = tr['word'][1][torch.tensor(wids)].to(dev).gather(1, wid.to(dev))
                rca_stack.set_root_ids(Rtok)
            hw, wi, pi, si, tw, tp, ts, ri, tr_ = forward_words(ids, wid, mask, ll, wids, Rw,
                                                                tr['word'][1])
            loss, info = compute(hw, ri, wi, pi, si, tr_, tw, tp, ts, margin=margin_at(step))
            opt.zero_grad(set_to_none=True); loss.backward()
            gn = float(torch.nn.utils.clip_grad_norm_(all_train, args.grad_clip))
            opt.step(); sched.step()
            if rca_stack is not None:
                rca_stack.set_root_ids(None)
            if step % 5 == 0 or step == 1:
                log(f'bench {step}/{args.bench}: loss {float(loss):.4f} gnorm {gn:.3f} '
                    f'{time.time()-t0:.1f}s elapsed ({(time.time()-t0)/step:.2f} s/step) '
                    f'peak_vram={torch.cuda.max_memory_allocated()/2**20:.0f}MiB')
        log(f'BENCH DONE {args.bench} steps in {time.time()-t0:.1f}s = '
            f'{(time.time()-t0)/args.bench:.3f} s/step')
        return

    for step in range(1, args.steps + 1):
        bidx = rng.integers(0, nb_tr, size=min(args.batch_size, nb_tr))
        bs = [int(tr_blocks[i]) for i in bidx]
        ids, wid, mask, ll, wids = batch_for(rows_tr_by_b, bs)
        Rw = tr['word'][1][wids]
        if rca_stack is not None:
            Rtok = tr['word'][1][torch.tensor(wids)].to(dev).gather(1, wid.to(dev))
            rca_stack.set_root_ids(Rtok)
        hw, wi, pi, si, tw, tp, ts, ri, tr_ = forward_words(ids, wid, mask, ll, wids, Rw,
                                                            tr['word'][1])
        loss, info = compute(hw, ri, wi, pi, si, tr_, tw, tp, ts, margin=margin_at(step))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(all_train, args.grad_clip))
        skip = args.grad_warmup > 0 and step <= args.grad_warmup and gn > args.grad_warmup_thresh
        if not skip:
            opt.step()
        sched.step()
        if rca_stack is not None:
            rca_stack.set_root_ids(None)
        if step % 100 == 0 or step == 1:
            g = rca_stack.gate_values() if rca_stack else []
            log(f'step {step:>6} loss {float(loss):.4f} root {info["root"]:.4f} '
                f'cez {info.get("root_cez", float("nan")):.4f} wazn {info["wazn"]:.4f} '
                f'pref {info["prefix"]:.4f} suff {info["suffix"]:.4f} '
                f'acc1 {info.get("acc1", float("nan")):.4f} gnorm {gn:.3f}'
                + (f' gates [{", ".join(f"{x:.3e}" for x in g)}]' if g else ''))
            RES['history'].append({'step': step, 'loss': float(loss), **info, 'gnorm': gn,
                                   'gates': g})
        if args.eval_every and (step % args.eval_every == 0 or step == args.steps):
            ev = evaluate_live(va_blocks, rows_va_by_b, va['word'][1], maxb=args.val_windows)
            log(f'  EVAL step {step}: root acc@1 ALL {ev["root_acc_all_pct"]:.2f}% '
                f'NOVEL {ev["root_acc_novel_pct"]:.2f}% (n={ev["n_all"]}/{ev["n_novel"]}) '
                f'wazn {ev["wazn_acc_pct"]:.2f}% pref {ev["prefix_acc_pct"]:.2f}% '
                f'suff {ev["suffix_acc_pct"]:.2f}%')
            RES['arms'][f'step{step}'] = ev
            if rca_stack is not None and args.rca_ablate_eval:
                ev2 = evaluate_live(va_blocks, rows_va_by_b, va['word'][1], ablate=True,
                                    maxb=args.val_windows)
                log(f'  EVAL step {step} GATE-OFF: root acc@1 ALL {ev2["root_acc_all_pct"]:.2f}% '
                    f'NOVEL {ev2["root_acc_novel_pct"]:.2f}%')
                RES['arms'][f'step{step}_gateoff'] = ev2

    RES['meta'].update({'peak_vram_allocated_mib': round(torch.cuda.max_memory_allocated() / 2**20, 1),
                        'peak_vram_reserved_mib': round(torch.cuda.max_memory_reserved() / 2**20, 1),
                        'wall_seconds': round(time.time() - T0, 1), 'live': True,
                        'trunk_lr_scale': args.trunk_lr_scale,
                        'final_gates': rca_stack.gate_values() if rca_stack else []})
    torch.save({'head': head.state_dict(),
                'rca': rca_stack.state_dict() if rca_stack else None,
                'trunk': model.state_dict() if args.save else None, 'args': vars(args)},
               save_path)
    Path(Res_path).write_text(_json.dumps(RES, ensure_ascii=False, indent=2))
    log(f'wrote {Res_path} and {save_path} wall={RES["meta"]["wall_seconds"]}s '
        f'peak_vram={RES["meta"]["peak_vram_allocated_mib"]}MiB')
    if probe:
        probe.close()


if __name__ == "__main__":
    main()
