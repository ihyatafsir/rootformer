#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
echo_identity.py -- ONE training-free experiment separating

  (A) the frozen backbone's clamped root stream destroys the root signal, from
  (B) next-root prediction is simply a hard task for this ~32M head.

The manipulation is: change only WHAT IS ASKED of the SAME trained head on the SAME
frozen hidden states -- predict the CURRENT word's root (identity / echo) instead of the
NEXT word's root (next-root) -- then split every metric by the clamp boundary (9014/9015).

Arms
  1  reproduction        next-root acc@1/@5 on the exact 300 val windows of the CONTROL 20k run
  2  IDENTITY (window)   identical hidden states + identical head outputs, target = own root
  3  IDENTITY (len-1)    each held-out word fed as its own length-1 context (the echo test)
  4  knockout            len-1 with the root input id replaced -> isolates the root route
  5  clamp activation    is the clamped attention-root stream even ON in the NRMT path?
  6  aliasing microscale h / argmax for roots that collide on attention row 9014
  7  readout-fair probe  ridge  h -> root_t  and  h -> root_{t+1}  (same readout, two questions)
  8  train-set echo      memorisation check (trained head + probe)
  9  root-stream ON      h with the clamped stream force-enabled (backbone's native config)

READ-ONLY w.r.t. every existing module (only in-process monkeypatching for instrumentation).
Writes only under /workspace/echo_test/.
"""
import json
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

RELEASE = Path('/workspace/hf_v19_2_release')
OUTDIR = Path('/workspace/echo_test')
sys.path.insert(0, str(RELEASE))
sys.path.insert(0, str(RELEASE / 'models'))

WIN, STRIDE, CTX = 128, 64, 12
CUT = 9015                      # attention root_embed rows = 9015 -> ids >= 9015 clamp onto row 9014
CKPT = RELEASE / 'checkpoints/rootformer_v19_2_synthesis_ar_backbone.awzan142.roots9490.tok10052.safetensors'
CACHE = Path('/workspace/nrmp_cache_9490')
HEAD_FILES = {
    'CONTROL_20k': Path('/workspace/v18fix/head_CONTROL.pt'),   # the 5.71 -> 5.72 flat run
    'FIX_20k': Path('/workspace/v18fix/head_FIX.pt'),           # currently training; may be mid-write
}
T0 = time.time()
RES = {'meta': {}, 'arms': {}}


def log(*a):
    print(f'[{time.time()-T0:7.1f}s]', *a, flush=True)


# ----------------------------------------------------------------------------- metrics
class Sub:
    __slots__ = ('n', 'a1', 'a5', 'pred_alias', 'pred_9014', 'pred_gt', 'pred_le')

    def __init__(self):
        self.n = self.a1 = self.a5 = 0
        self.pred_alias = self.pred_9014 = self.pred_gt = self.pred_le = 0

    def out(self):
        if self.n == 0:
            return {'n': 0}
        d = {'n': self.n, 'acc@1': self.a1 / self.n, 'acc@5': self.a5 / self.n,
             'acc@1_pct': 100.0 * self.a1 / self.n, 'acc@5_pct': 100.0 * self.a5 / self.n}
        if self.pred_alias or self.pred_9014 or self.pred_gt or self.pred_le:
            d.update({'pred_in_alias_class_pct': 100.0 * self.pred_alias / self.n,
                      'pred_eq_9014_pct': 100.0 * self.pred_9014 / self.n,
                      'pred_gt_9014_pct': 100.0 * self.pred_gt / self.n,
                      'pred_le_9014_pct': 100.0 * self.pred_le / self.n})
        return d


def topk_acc(lg, tgt):
    """lg [N,C] float32, tgt [N] long -> (top5 idx [N,5], a1 bool [N], a5 bool [N])"""
    if not bool(torch.isfinite(lg).all()):
        raise FloatingPointError(f'non-finite logits: {int((~torch.isfinite(lg)).sum())} entries')
    top = torch.topk(lg, 5, dim=-1).indices
    a1 = top[:, 0] == tgt
    a5 = (top == tgt.unsqueeze(-1)).any(-1)
    return top, a1, a5


def split_update(subs, top, a1, a5, tgt, m_all):
    """subs: {'all':Sub,'le':Sub,'gt':Sub}; records predictions for the >9014 group."""
    for name, m in m_all.items():
        if m is None:
            continue
        s = subs[name]
        k = int(m.sum())
        if k == 0:
            continue
        s.n += k
        s.a1 += int(a1[m].sum())
        s.a5 += int(a5[m].sum())
    mg = m_all.get('gt')
    if mg is not None and bool(mg.any()):
        s = subs['gt']
        top1 = top[:, 0]
        tg = tgt[mg]
        p1 = top1[mg]
        s.pred_alias += int(((p1 >= CUT - 1) & (p1 < 9490)).sum())   # any id in the collapsed class
        s.pred_9014 += int((p1 == CUT - 1).sum())
        s.pred_gt += int((p1 >= CUT).sum())
        s.pred_le += int((p1 < CUT - 1).sum())


def majority_rate(ids, C):
    """acc@1 of the best constant predictor, and its acc@5 (top-5 most frequent ids)."""
    if ids.numel() == 0:
        return None
    cnt = torch.bincount(ids, minlength=C)
    top = torch.topk(cnt, 5).indices
    return {'n': int(ids.numel()),
            'majority_acc@1': float(cnt[top[0]].item() / ids.numel()),
            'majority_acc@5': float(cnt[top].sum().item() / ids.numel()),
            'majority_root_id': int(top[0].item())}


# ----------------------------------------------------------------------------- backbone
@torch.no_grad()
def extract_h(model, streams, starts, device, tag, activate_root_stream=False,
              p_transform=None, bs=16):
    P, R, W, S = streams
    hs, t0 = [], time.time()
    for i in range(0, len(starts), bs):
        idx = starts[i:i + bs]
        p = torch.stack([P[j:j + WIN] for j in idx]).to(device)
        r = torch.stack([R[j:j + WIN] for j in idx]).to(device)
        w = torch.stack([W[j:j + WIN] for j in idx]).to(device)
        s = torch.stack([S[j:j + WIN] for j in idx]).to(device)
        if p_transform is not None:
            p = p_transform(p)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            emb = model.morphemic_embed(p, r, w, s)
            model._set_flash(r, w)
            if activate_root_stream:
                for layer in model.backbone.layers:
                    layer.self_attn.active_root_ids = r
            o = model.backbone(inputs_embeds=emb)
            h = model.final_norm(o.last_hidden_state)
        hs.append(h.float().cpu())
    if activate_root_stream:
        for layer in model.backbone.layers:
            layer.self_attn.active_root_ids = None
    H = torch.cat(hs)
    log(f'    extract[{tag}] root_stream={"ON" if activate_root_stream else "OFF"}'
        f'{", prefix-ablated" if p_transform is not None else ""}: '
        f'{tuple(H.shape)} in {time.time()-t0:.1f}s')
    return H


@torch.no_grad()
def head_logits_stream(head, H, R, O, W, P, S, device, bs=16):
    """Yields (logits [b,T,C] float32) for H [N,T,d] and per-position inputs."""
    for i in range(0, H.shape[0], bs):
        sl = slice(i, min(i + bs, H.shape[0]))
        out = head(H[sl].to(device), R[sl].to(device), O[sl].to(device),
                   W[sl].to(device), P[sl].to(device), S[sl].to(device))
        yield out['root_logits'].float()


def window_metrics(head, H, R, O, W, P, S, tgt_id, tgt_nw, keep_id, keep_nw,
                   device, num_roots, bs=16):
    """Both targets evaluated on the SAME head outputs, over the SAME positions."""
    m = {'identity': {'all': Sub(), 'le': Sub(), 'gt': Sub()},
         'next_word': {'all': Sub(), 'le': Sub(), 'gt': Sub()}}
    T1 = H.shape[1]
    tgt_id_f, tgt_nw_f = tgt_id.reshape(-1), tgt_nw.reshape(-1)
    keep_id_f, keep_nw_f = keep_id.reshape(-1), keep_nw.reshape(-1)
    # window-major flatten: flat index = window * T1 + position
    for b, lg in enumerate(head_logits_stream(head, H, R, O, W, P, S, device, bs)):
        start = b * bs * T1
        nb = lg.shape[0] * T1
        sl = slice(start, start + nb)
        C = lg.shape[-1]
        flat = lg.reshape(-1, C)
        for name, tgtf, keepf in (('identity', tgt_id_f, keep_id_f),
                                  ('next_word', tgt_nw_f, keep_nw_f)):
            t = tgtf[sl].to(flat.device)
            k = keepf[sl].to(flat.device)
            top, a1, a5 = topk_acc(flat, t)
            grp = t >= CUT
            split_update(m[name],
                         top, a1, a5, t,
                         {'all': k, 'le': k & ~grp, 'gt': k & grp})
    return m


# ----------------------------------------------------------------------------- len-1 echo
@torch.no_grad()
def identify_word_level(model, head, P, R, W, S, device, num_roots,
                        bs=512, mask_root=None, tag=''):
    """Each unique word as its own length-1 context; read the root head at that position."""
    subs = {'all': Sub(), 'le': Sub(), 'gt': Sub()}
    for i in range(0, P.numel(), bs):
        sl = slice(i, min(i + bs, P.numel()))
        p = P[sl].view(-1, 1).to(device)
        r_true = R[sl].view(-1, 1).to(device)
        w = W[sl].view(-1, 1).to(device)
        s = S[sl].view(-1, 1).to(device)
        r_in = mask_root(r_true) if mask_root is not None else r_true
        with torch.autocast('cuda', dtype=torch.bfloat16):
            emb = model.morphemic_embed(p, r_in, w, s)
            model._set_flash(r_in, w)
            h = model.final_norm(model.backbone(inputs_embeds=emb).last_hidden_state).float()
        op = torch.zeros_like(r_in)
        lg = head(h, r_in, op, w, p, s)['root_logits'][:, 0, :].float()
        t = r_true[:, 0]
        top, a1, a5 = topk_acc(lg, t)
        grp = t >= CUT
        split_update(subs, top, a1, a5, t, {'all': None, 'le': ~grp, 'gt': grp})
        subs['all'].n += t.numel()
        subs['all'].a1 += int(a1.sum())
        subs['all'].a5 += int(a5.sum())
    log(f'    echo[{tag}] n={subs["all"].n}')
    return {k: v.out() for k, v in subs.items()}


# ----------------------------------------------------------------------------- ridge probe
def fit_ridge(X, y, num_roots, device, lam_frac=3e-3, bs=32768):
    """Closed-form linear readout (with intercept) from frozen features to root ids."""
    d = X.shape[1]
    XTX = torch.zeros(d + 1, d + 1, dtype=torch.float32, device=device)
    XTY = torch.zeros(d + 1, num_roots, dtype=torch.float32, device=device)
    for i in range(0, X.shape[0], bs):
        xb = X[i:i + bs].to(device)
        yb = y[i:i + bs].to(device)
        xb1 = torch.cat([xb, torch.ones(xb.shape[0], 1, device=device)], dim=1)
        XTX += xb1.t() @ xb1
        XTY.index_add_(1, yb, xb1.t())
    lam = lam_frac * float(torch.diagonal(XTX[:d, :d]).mean())
    A = XTX.double() + lam * torch.eye(d + 1, dtype=torch.float64, device=device)
    W = torch.linalg.solve(A, XTY.double()).float()
    return W, lam


@torch.no_grad()
def eval_ridge(W, X, y, keep, num_roots, device, C, bs=32768):
    subs = {'all': Sub(), 'le': Sub(), 'gt': Sub()}
    d = X.shape[1]
    for i in range(0, X.shape[0], bs):
        sl = slice(i, min(i + bs, X.shape[0]))
        xb = X[sl].to(device)
        xb1 = torch.cat([xb, torch.ones(xb.shape[0], 1, device=device)], dim=1)
        lg = xb1 @ W
        t = y[sl].to(device)
        k = keep[sl].to(device)
        top, a1, a5 = topk_acc(lg, t)
        grp = t >= CUT
        split_update(subs, top, a1, a5, t, {'all': k, 'le': k & ~grp, 'gt': k & grp})
    return {kk: v.out() for kk, v in subs.items()}


# ----------------------------------------------------------------------------- main
def unique_words(streams, num_roots, num_awzan, num_suffixes):
    """Dedupe the (prefix, root, wazn, suffix) tuples exactly as the consumer pairs them
    (index-wise).  The cache's stream 0 carries 2 extra boundary markers per sentence, so all
    streams are truncated to the common length first (the consumer can only pair indices that
    exist in every stream)."""
    L = min(int(t.numel()) for t in streams)
    P, R, W, S = [t.long()[:L] for t in streams]
    key = (((P * num_roots + R) * num_awzan + W) * num_suffixes + S)
    u = torch.unique(key)
    s = u % num_suffixes
    u = u // num_suffixes
    w = u % num_awzan
    u = u // num_awzan
    r = u % num_roots
    p = u // num_roots
    return p.contiguous(), r.contiguous(), w.contiguous(), s.contiguous()


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device('cuda')
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    import nrmp_vocab as nv
    from models.unified_rootformer_v12 import UnifiedRootformerV12
    from deepseek_v4_1_flash_model import UnifiedRootformerV17_DeepSeekFlash
    from safetensors.torch import load_file
    from nrmt_arch import RootformerNRMT, build_operator_table
    import models.ishtiqaq_attention_v12 as ia

    V = next(v for k, v in vars(nv).items() if isinstance(v, type) and 'MorphemicVocab' in k)
    bp = str(RELEASE / 'data/rootformer_v12_arabic_blueprint.json')
    vocab = V(bp)
    log(f'vocab: num_roots={vocab.num_roots} awzan={vocab.num_awzan} '
        f'prefixes={vocab.num_prefixes} suffixes={vocab.num_suffixes}')
    RES['meta'].update({'vocab_num_roots': int(vocab.num_roots),
                        'vocab_num_awzan': int(vocab.num_awzan),
                        'vocab_num_prefixes': int(vocab.num_prefixes),
                        'vocab_num_suffixes': int(vocab.num_suffixes),
                        'checkpoint': str(CKPT), 'cache': str(CACHE), 'clamp_boundary': CUT})

    base = UnifiedRootformerV12(bp, 'Qwen/Qwen2.5-0.5B', str(device), torch.bfloat16).to(device)
    flash = UnifiedRootformerV17_DeepSeekFlash(base, device, torch.bfloat16).to(device)
    model = RootformerNRMT(flash, vocab, device, torch.bfloat16, hist=3, dropout=0.1,
                           use_features=True, feat_gate=False).to(device)
    sd = load_file(str(CKPT))
    miss, unexp = model.load_state_dict(sd, strict=False)
    n_new = len([k for k in miss if k.startswith('nrmt_head.')])
    log(f'backbone load: missing={len(miss)} (new head={n_new}) unexpected={len(unexp)}')
    for p in model.parameters():
        p.requires_grad = False
    model.nrmt_head.to(torch.float32)

    # clamp census on the two different root embeddings
    att_rows = int(model.backbone.layers[0].self_attn.root_embed.num_embeddings)
    morph_rows = int(model.morphemic_embed.root_embed.num_embeddings)
    RES['meta'].update({'attention_root_embed_rows': att_rows,
                        'morphemic_root_embed_rows': morph_rows,
                        'ids_aliased_onto_row_9014': max(0, morph_rows - att_rows + 1)})
    log(f'attention root_embed rows = {att_rows}; morphemic root_embed rows = {morph_rows}; '
        f'ids >= {att_rows} alias onto row {att_rows-1}')

    head = model.nrmt_head
    head.eval()
    model.eval()

    # ---------------- arm 5a: is the clamped attention-root stream even ON in this path?
    counts = {'attn_calls': 0, 'root_ids_kwarg_not_none': 0, 'active_root_ids_not_none': 0}
    _orig = ia.IshtiqaqAttentionV12.forward

    def _wrap(self, hidden_states, *a, **kw):
        counts['attn_calls'] += 1
        if kw.get('root_ids', None) is not None:
            counts['root_ids_kwarg_not_none'] += 1
        if getattr(self, 'active_root_ids', None) is not None:
            counts['active_root_ids_not_none'] += 1
        return _orig(self, hidden_states, *a, **kw)

    ia.IshtiqaqAttentionV12.forward = _wrap

    # ---------------- data
    tr4 = [t.long() for t in torch.load(CACHE / 'train.pt', map_location='cpu')]
    va4 = [t.long() for t in torch.load(CACHE / 'val.pt', map_location='cpu')]
    log(f'train {tr4[1].numel()} root tokens | val {va4[1].numel()} root tokens')

    # ---------------- cache integrity census (stream 0 vs the word streams)
    # nrmp_train.cmd_prepare appends BOS_ROOT before and EOS_ROOT after each sentence to STREAM 0
    # only, so stream 0 has exactly 2 extra entries per sentence and can never be index-aligned
    # with the (root, wazn, suffix) streams that the consumer pairs it with.
    for tag, ss in (('train', tr4), ('val', va4)):
        L0, L1 = int(ss[0].numel()), int(ss[1].numel())
        RES['meta'].setdefault('cache_stream_census', {})[tag] = {
            'len_stream0_prefix': L0, 'len_stream1_root': L1, 'len_diff': L0 - L1,
            'implied_sentence_count': (L0 - L1) / 2,
            'count_p_eq_BOS_ROOT': int((ss[0] == vocab.BOS_ROOT).sum()),
            'count_p_eq_EOS_ROOT': int((ss[0] == vocab.EOS_ROOT).sum()),
            'stream0_len_equals_stream1_len': L0 == L1,
        }
    c = RES['meta']['cache_stream_census']
    log(f'  cache census: val stream0={c["val"]["len_stream0_prefix"]} '
        f'stream1={c["val"]["len_stream1_root"]} diff={c["val"]["len_diff"]} '
        f'== 2*{c["val"]["implied_sentence_count"]:.0f} sentences; '
        f'prefix==EOS marker count {c["val"]["count_p_eq_EOS_ROOT"]}')

    def starts_for(t, n, seed):
        s = list(range(0, t.numel() - WIN - 1, STRIDE))
        np.random.default_rng(seed).shuffle(s)
        return sorted(s[:n])

    va_st = starts_for(va4[1], 300, 1)          # exactly the 20k runs' val windows
    tr_st = starts_for(tr4[1], 3000, 0)         # exactly the 20k runs' train windows

    def win_tensor(t, st):
        return torch.stack([t[j:j + WIN] for j in st])

    Tv, Wv, Pv, Sv = (win_tensor(va4[i], va_st) for i in (1, 2, 0, 3))
    Tr, Wr, Pr, Sr = (win_tensor(tr4[i], tr_st) for i in (1, 2, 0, 3))

    op_table = build_operator_table(vocab)
    try:
        from sibawayh_governor import SibawayhGovernor
        gov = SibawayhGovernor(vocab)
        op_ids_of = lambda P, R, W: torch.tensor(
            [gov.op_ids_for_ids(rr, ww, pp) for pp, rr, ww in zip(P.tolist(), R.tolist(), W.tolist())],
            dtype=torch.long)
        RES['meta']['operator_stream'] = 'SibawayhGovernor'
    except Exception as e:                                     # pragma: no cover
        log(f'[warn] SibawayhGovernor unavailable: {e!r}')
        op_ids_of = lambda P, R, W: op_table[R.clamp(min=0)]
        RES['meta']['operator_stream'] = f'fallback: {e!r}'

    Ova = op_ids_of(Pv, Tv, Wv)
    Otr = op_ids_of(Pr, Tr, Wr)

    specials = sorted({vocab.PAD_ROOT, vocab.BOS_ROOT, vocab.EOS_ROOT, vocab.UNK_ROOT,
                       vocab.root2id['<PARTICLE>']} |
                      {i for i, r in enumerate(vocab.roots_list) if r.startswith('<P:')})
    spec_t = torch.tensor(specials)

    Hva = extract_h(model, va4, va_st, device, 'val')
    ia.IshtiqaqAttentionV12.forward = _orig
    RES['arms']['clamp_stream_activity'] = counts
    log(f'attention calls={counts["attn_calls"]} '
        f'root_ids_kwarg={counts["root_ids_kwarg_not_none"]} '
        f'active_root_ids={counts["active_root_ids_not_none"]}')

    Hva_w = Hva[:, :-1, :].contiguous()
    Rt_id, Rt_nw = Tv[:, :-1].contiguous(), Tv[:, 1:].contiguous()
    keep_id = ~torch.isin(Rt_id.reshape(-1), spec_t)
    keep_nw = ~torch.isin(Rt_nw.reshape(-1), spec_t)
    log(f'val radical positions: identity n={int(keep_id.sum())} next-word n={int(keep_nw.sum())}')

    # ---------------- the trained head under test (the 5.71 -> 5.72 flat run)
    hp = HEAD_FILES['CONTROL_20k']
    hsd = torch.load(str(hp), map_location='cpu')
    r = model.nrmt_head.load_state_dict(hsd, strict=True)
    log(f'head {hp.name}: {len(hsd)} tensors, load={r}')
    RES['meta']['head_file'] = str(hp)
    RES['meta']['head_file_mtime'] = time.strftime('%Y-%m-%dT%H:%M:%S',
                                                   time.localtime(hp.stat().st_mtime))
    try:
        ch = json.load(open('/workspace/v18fix/results_CONTROL.json'))['history']
        RES['meta']['control_run_reference'] = {'step': ch[-1]['step'],
                                                'ALL_val': ch[-1]['ALL_val']['acc@1'],
                                                'NOVEL_only': ch[-1]['NOVEL_only']['acc@1']}
        log(f'CONTROL run final eval @{ch[-1]["step"]}: ALL {100*ch[-1]["ALL_val"]["acc@1"]:.2f}% '
            f'NOVEL {100*ch[-1]["NOVEL_only"]["acc@1"]:.2f}%')
    except Exception as e:                                     # pragma: no cover
        log(f'[warn] could not read results_CONTROL.json: {e!r}')

    # ---------------- arm 1+2: same hidden states, same head outputs, two targets
    t0 = time.time()
    wm = window_metrics(head, Hva_w, Tv[:, :-1], Ova[:, :-1], Wv[:, :-1], Pv[:, :-1], Sv[:, :-1],
                        Rt_id, Rt_nw, keep_id, keep_nw, device, vocab.num_roots)
    log(f'window metrics in {time.time()-t0:.1f}s')
    RES['arms']['window_same_hidden_states'] = {
        'note': 'identical h, identical head outputs; only the label position changes '
                '(t vs t+1). identity = predict the CURRENT word root; next_word = the run\'s task.',
        'identity': {k: v.out() for k, v in wm['identity'].items()},
        'next_word': {k: v.out() for k, v in wm['next_word'].items()},
    }
    marg_nw = majority_rate(Rt_nw.reshape(-1)[keep_nw], vocab.num_roots)
    marg_id = majority_rate(Rt_id.reshape(-1)[keep_id], vocab.num_roots)
    grp_nw = Rt_nw.reshape(-1)[keep_nw]
    grp_id = Rt_id.reshape(-1)[keep_id]
    RES['arms']['marginals'] = {
        'next_word_all': marg_nw,
        'identity_all': marg_id,
        'identity_le_9014': majority_rate(grp_id[grp_id < CUT], vocab.num_roots),
        'identity_gt_9014': majority_rate(grp_id[grp_id >= CUT], vocab.num_roots),
        'next_word_le_9014': majority_rate(grp_nw[grp_nw < CUT], vocab.num_roots),
        'next_word_gt_9014': majority_rate(grp_nw[grp_nw >= CUT], vocab.num_roots),
    }
    log(f'  identity ALL : {RES["arms"]["window_same_hidden_states"]["identity"]["all"]}')
    log(f'  next-word ALL: {RES["arms"]["window_same_hidden_states"]["next_word"]["all"]}')

    for tag, (st, streams, tgt) in (('train_windows', (tr_st, tr4, Tr)),
                                    ('val_windows', (va_st, va4, Tv))):
        y = tgt.reshape(-1)
        n_spec = int(torch.isin(y, spec_t).sum())
        RES['arms'].setdefault('group_counts', {})[tag] = {
            'positions_total': int(y.numel()),
            'positions_special': n_spec,
            'identity_target_positions': (int((y < CUT).sum()), int((y >= CUT).sum())),
            'distinct_ids_le_9014': int(torch.unique(y[y < CUT]).numel()),
            'distinct_ids_gt_9014': int(torch.unique(y[y >= CUT]).numel()),
        }
    for tag, stream in (('train_full_stream', tr4[1]), ('val_full_stream', va4[1])):
        y = stream
        gt = int((y >= CUT).sum())
        RES['arms'].setdefault('group_counts', {})[tag] = {
            'root_tokens_total': int(y.numel()),
            'root_tokens_gt_9014': gt,
            'pct_gt_9014': 100.0 * gt / max(1, int(y.numel())),
            'root_tokens_eq_9014': int((y == CUT - 1).sum()),
            'distinct_ids_gt_9014': int(torch.unique(y[y >= CUT]).numel()),
            'distinct_ids_total': int(torch.unique(y).numel()),
        }
    log(f'  group counts: {json.dumps(RES["arms"]["group_counts"], indent=1)}')

    # ---------------- arm 3+4: length-1 echo on unique held-out words
    vp, vr, vw, vs = unique_words(va4, vocab.num_roots, vocab.num_awzan, vocab.num_suffixes)
    keep_w = ~torch.isin(vr, spec_t)
    vp, vr, vw, vs = vp[keep_w], vr[keep_w], vw[keep_w], vs[keep_w]
    log(f'unique val words: {vp.numel()} (of which >=9015 roots: {int((vr >= CUT).sum())}, '
        f'distinct {int(torch.unique(vr[vr >= CUT]).numel())})')
    RES['arms']['echo_length1'] = {
        'n_unique_val_words': int(vp.numel()),
        'n_unique_val_words_gt_9014': int((vr >= CUT).sum()),
        'true_root': identify_word_level(model, head, vp, vr, vw, vs, device,
                                         vocab.num_roots, tag='val/true'),
        'root_id_knockout': identify_word_level(
            model, head, vp, vr, vw, vs, device, vocab.num_roots,
            mask_root=lambda r: torch.full_like(r, int(vocab.UNK_ROOT)), tag='val/knockout'),
    }
    log(f'  echo true   : {RES["arms"]["echo_length1"]["true_root"]["all"]}')
    log(f'  echo knockout: {RES["arms"]["echo_length1"]["root_id_knockout"]["all"]}')

    # ---------------- arm 8: train-set echo (memorisation)
    tp, tr_, tw, ts = unique_words(tr4, vocab.num_roots, vocab.num_awzan, vocab.num_suffixes)
    keep_tw = ~torch.isin(tr_, spec_t)
    tp, tr_, tw, ts = tp[keep_tw], tr_[keep_tw], tw[keep_tw], ts[keep_tw]
    if tp.numel() > 400000:
        sel = torch.randperm(tp.numel())[:400000]
        tp, tr_, tw, ts = tp[sel], tr_[sel], tw[sel], ts[sel]
    log(f'unique train words (capped): {tp.numel()}')
    RES['arms']['train_echo_length1'] = {
        'n_unique_train_words': int(tp.numel()),
        'true_root': identify_word_level(model, head, tp, tr_, tw, ts, device,
                                         vocab.num_roots, tag='train/true'),
        'root_id_knockout': identify_word_level(
            model, head, tp, tr_, tw, ts, device, vocab.num_roots,
            mask_root=lambda r: torch.full_like(r, int(vocab.UNK_ROOT)), tag='train/knockout'),
    }
    log(f'  train echo true: {RES["arms"]["train_echo_length1"]["true_root"]["all"]}')

    # ---------------- arm 6: aliasing microscale (training-free)
    n_micro = 512
    sel = torch.randperm(vp.numel())[:n_micro]
    mp, mw, ms = vp[sel], vw[sel], vs[sel]
    variants = [9014] + [x for x in (9015, 9200, 9489) if x < vocab.num_roots]
    hs = {}
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for rv in variants:
            rr = torch.full_like(mp, rv).view(-1, 1).to(device)
            p = mp.view(-1, 1).to(device); w = mw.view(-1, 1).to(device); s = ms.view(-1, 1).to(device)
            emb = model.morphemic_embed(p, rr, w, s)
            model._set_flash(rr, w)
            h = model.final_norm(model.backbone(inputs_embeds=emb).last_hidden_state)[:, 0, :].float()
            lg = head(h.unsqueeze(1), rr, torch.zeros_like(rr), w, p, s)['root_logits'][:, 0, :].float()
            hs[rv] = (h.cpu(), lg.argmax(-1).cpu(), lg.cpu())
    base_h = hs[9014][0]
    micro = {'n_contexts': int(n_micro), 'variants': variants, 'cnorm': float(base_h.norm(dim=-1).mean())}
    for rv in variants[1:]:
        dh = (hs[rv][0] - base_h).norm(dim=-1)
        micro[f'root_{rv}'] = {
            'mean_rel_dh_vs_9014': float((dh / base_h.norm(dim=-1)).mean()),
            'max_abs_dlogit_vs_9014': float((hs[rv][2] - hs[9014][2]).abs().max()),
            'argmax_changes_pct': float(100.0 * (hs[rv][1] != hs[9014][1]).float().mean()),
        }
    att = model.backbone.layers[0].self_attn
    with torch.no_grad():
        e = att.root_embed.weight.float()
        micro['attention_rows_identical_9014_vs_gt'] = {
            int(rv): bool(torch.equal(e[9014], e[min(rv, e.shape[0] - 1)])) for rv in variants[1:]}
        em = model.morphemic_embed.root_embed.weight.float()
        micro['morphemic_rows_identical_9014_vs_gt'] = {
            int(rv): bool(torch.equal(em[9014], em[rv])) for rv in variants[1:]}
    RES['arms']['aliasing_microscale'] = micro
    log(f'  microscale: {json.dumps({k: v for k, v in micro.items() if k.startswith("root_")})}')

    # ---------------- arm 7: readout-fair ridge probes on the same windows
    log('fitting ridge probes (h -> root_t and h -> root_{t+1}) ...')
    Htr = extract_h(model, tr4, tr_st, device, 'train')
    Htr_w = Htr[:, :-1, :].contiguous()
    Xtr = Htr_w.reshape(-1, Htr_w.shape[-1])
    ytr_id = Tr[:, :-1].reshape(-1)
    ytr_nw = Tr[:, 1:].reshape(-1)
    ktr_id = ~torch.isin(ytr_id, spec_t)
    ktr_nw = ~torch.isin(ytr_nw, spec_t)
    Xv = Hva_w.reshape(-1, Hva_w.shape[-1])
    yv_id, yv_nw = Rt_id.reshape(-1), Rt_nw.reshape(-1)
    del Htr_w, Htr

    p0 = time.time()
    W_id, lam_id = fit_ridge(Xtr[ktr_id], ytr_id[ktr_id], vocab.num_roots, device)
    W_nw, lam_nw = fit_ridge(Xtr[ktr_nw], ytr_nw[ktr_nw], vocab.num_roots, device)
    log(f'  ridge fit in {time.time()-p0:.1f}s (lam_id={lam_id:.3g} lam_nw={lam_nw:.3g})')
    probe = {
        'identity': eval_ridge(W_id, Xv, yv_id, keep_id, vocab.num_roots, device, vocab.num_roots),
        'next_word': eval_ridge(W_nw, Xv, yv_nw, keep_nw, vocab.num_roots, device, vocab.num_roots),
    }
    # identity probe restricted to roots <= 9014 only (train + eval) -- isolates the clamp question
    m_le = ktr_id & (ytr_id < CUT)
    W_id_le, _ = fit_ridge(Xtr[m_le], ytr_id[m_le], vocab.num_roots, device)
    probe['identity_train_le_9014_only'] = eval_ridge(
        W_id_le, Xv, yv_id, keep_id & (yv_id < CUT), vocab.num_roots, device, vocab.num_roots)
    # train-set accuracy of the identity probe (memorisation reference)
    Xt = Xtr[ktr_id]
    probe['identity_train'] = eval_ridge(W_id, Xt, ytr_id[ktr_id], ktr_id[ktr_id],
                                         vocab.num_roots, device, vocab.num_roots)
    RES['arms']['ridge_probe_readout_fair'] = probe
    log(f'  probe identity : {probe["identity"]["all"]}')
    log(f'  probe next-word: {probe["next_word"]["all"]}')

    # ---------------- arm 9: clamped root stream force-ON (backbone's native config)
    Hva_on = extract_h(model, va4, va_st, device, 'val-rootstreamON', activate_root_stream=True)
    d = (Hva_on - Hva)
    rel = float((d.norm(dim=-1) / Hva.norm(dim=-1).clamp_min(1e-6)).mean())
    wm_on = window_metrics(head, Hva_on[:, :-1, :].contiguous(), Tv[:, :-1], Ova[:, :-1],
                           Wv[:, :-1], Pv[:, :-1], Sv[:, :-1], Rt_id, Rt_nw,
                           keep_id, keep_nw, device, vocab.num_roots)
    RES['arms']['root_stream_forced_on'] = {
        'note': 'the head was trained on OFF features, so ON numbers are off-distribution; '
                'included only to size the backbone-config mismatch.',
        'mean_relative_dh_vs_off': rel,
        'identity': {k: v.out() for k, v in wm_on['identity'].items()},
        'next_word': {k: v.out() for k, v in wm_on['next_word'].items()},
    }
    log(f'  root-stream ON: mean rel dh={rel:.4f} '
        f'next-word {RES["arms"]["root_stream_forced_on"]["next_word"]["all"]}')

    # ---------------- arm 10: how much does the (misaligned) prefix stream matter at all?
    PADP = int(vocab.PAD_PREFIX)
    Hva_np = extract_h(model, va4, va_st, device, 'val-prefix-ablated',
                       p_transform=lambda x: torch.full_like(x, PADP))
    rel_p = float(((Hva_np - Hva).norm(dim=-1) / Hva.norm(dim=-1).clamp_min(1e-6)).mean())
    Pv_abl = torch.full_like(Pv, PADP)
    wm_np = window_metrics(head, Hva_np[:, :-1, :].contiguous(), Tv[:, :-1], Ova[:, :-1],
                           Wv[:, :-1], Pv_abl[:, :-1], Sv[:, :-1], Rt_id, Rt_nw,
                           keep_id, keep_nw, device, vocab.num_roots)
    RES['arms']['prefix_stream_ablated'] = {
        'note': 'prefix id replaced by PAD_PREFIX everywhere (embedding + head features). Bounds '
                'how much the prefix slot can contribute, aligned or not. Off-distribution for '
                'the trained head, exactly like arm 9.',
        'mean_relative_dh_vs_unaltered': rel_p,
        'identity': {k: v.out() for k, v in wm_np['identity'].items()},
        'next_word': {k: v.out() for k, v in wm_np['next_word'].items()},
    }
    log(f'  prefix ablated: mean rel dh={rel_p:.4f} '
        f'next-word {RES["arms"]["prefix_stream_ablated"]["next_word"]["all"]}')
    del Hva_np, wm_on, wm_np
    torch.cuda.empty_cache()

    # ---------------- finish
    RES['meta']['peak_vram_allocated_mib'] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    RES['meta']['peak_vram_reserved_mib'] = round(torch.cuda.max_memory_reserved() / 2**20, 1)
    RES['meta']['wall_seconds'] = round(time.time() - T0, 1)
    RES["meta"]["finite_logits"] = True   # guaranteed: topk_acc raises on any non-finite logit
    out = OUTDIR / 'echo_results.json'
    json.dump(RES, open(out, 'w'), indent=2)
    log(f'wrote {out}  peak_vram={RES["meta"]["peak_vram_allocated_mib"]}MiB '
        f'wall={RES["meta"]["wall_seconds"]}s')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        RES['meta']['error'] = traceback.format_exc()
        json.dump(RES, open(OUTDIR / 'echo_results.json', 'w'), indent=2)
        sys.exit(1)
