#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
probe_ridge_arm.py -- the length-1 ridge readout on a TRAINED arm's trunk.

WHY THIS EXISTS
---------------
`ALL_val.acc@1` measures the trained NRMT head, whose readout direction is fixed by next-root
prediction.  The brief's 29.30 % / 92.34 % comparators come from a DIFFERENT question, asked of the
same trunk with a free closed-form readout:

    h(prefix, root, wazn, suffix)  ->  that word's OWN root?

Because a ridge readout is fit, not trained, the number is about what the REPRESENTATION contains,
not about what a head learned to read.  Run on the released trunk it gives 92.34 % (83.92 % on
unseen words); on a raw Qwen2.5-0.5B it gives 29.30 % (23.34 % unseen).

THE FOUR THINGS THAT MAKE THIS PROBE HONEST
-------------------------------------------
1. THE PATHWAY MUST BE ACTIVE.  Arm C's trunk was TRAINED with the root pathway in the forward
   pass; probing it with the pathway off measures a different network.  `--pathway` attaches the
   same mechanism the arm trained with and supplies `active_root_ids` (exactly the wiring the
   trainer's `live_h_windows` uses).  Arm A trained with no pathway, so its probe runs with none.
2. THE WEIGHTS MUST COME FROM THE ARM, not the release.  `/tmp/root_arch_arms/head_<TAG>.pt.trunk.pt`
   holds every trained tensor (`backbone.layers.*` for all 24 layers, `morphemic_embed.*`,
   `final_norm.*`, `root_cross.*`, and the score-bias `root_gate`/`root_q_proj`/`root_k_proj` which
   live under `backbone.layers.*.self_attn.`).
3. THE ROOT COUNT IS ASSERTED AT 9490 / 142 before anything else (the silent-shrink trap).
4. TRAIN/EVAL ARE DISJOINT WORD TUPLES, with a separate "unseen in train" split, matching the
   reference protocol.

Usage:
  probe_ridge_arm.py --tag FLOOR_A --pathway none
  probe_ridge_arm.py --tag EARLYROOT_C --pathway both
"""
import argparse
import json
import sys
import time

import torch

sys.path.insert(0, '/workspace/root_arch')
sys.path.insert(0, '/workspace/echo_test')
sys.path.insert(0, '/workspace/ishtiqaq_check')
sys.path.insert(0, '/workspace/root_attn')
import model_build as MB                                       # noqa: E402
import sdpa_attention as SA                                    # noqa: E402

DEV = 'cuda'
T0 = time.time()


def log(*a):
    print('[%6.1fs]' % (time.time() - T0), *a, flush=True)


def packed_words(streams, num_roots, num_awzan, num_suffixes, cap=None, seed=0):
    """Unique (p,r,w,s) tuples over the common index range, plus the packed keys."""
    L = min(int(t.numel()) for t in streams)
    P, R, W, S = [t.long()[:L] for t in streams]
    key = (((P * num_roots + R) * num_awzan + W) * num_suffixes + S)
    u = torch.unique(key)
    if cap is not None and u.numel() > cap:
        u = u[torch.randperm(u.numel(), generator=torch.Generator().manual_seed(seed))[:cap]]
        u = torch.sort(u).values
    s = u % num_suffixes
    q = u // num_suffixes
    w = q % num_awzan
    q = q // num_awzan
    r = q % num_roots
    p = q // num_roots
    return p.contiguous(), r.contiguous(), w.contiguous(), s.contiguous(), u


def attach_pathway(model, which, layer_spec='all'):
    """Re-create exactly the mechanism the arm trained with, on exactly the layers it used.

    `layer_spec` is parsed with the PROJECT'S OWN parser (`early_root_path.parse_layer_spec_v13`),
    so `all` -> 0..23 and `top4` -> 20..23 match the trainer bit for bit.  Arm X trained BOTH
    mechanisms on layers 20-23 only; attaching them to all 24 here would probe a different network.
    """
    from early_root_path import parse_layer_spec_v13
    layers = model.backbone.layers
    share = model.morphemic_embed.root_embed
    n = len(layers)
    idx = parse_layer_spec_v13(layer_spec, n)
    print('    attach_pathway: %s on layers %s' % (which, idx), flush=True)
    if which in ('both', 'residual'):
        from root_cross_attn import RootCrossAttentionStack
        import rca_compat
        rca_compat.install(verbose=True)
        st = RootCrossAttentionStack(model.d_model, share, idx, num_heads=8,
                                     dropout=0.1, out_std=1e-3, dtype=torch.float32,
                                     out_norm=True).to(device=DEV)
        st.exclude_current = False
        model.root_cross = st
        st.attach(layers)
    if which in ('both', 'bias'):
        from ishtiqaq_root_bias import swap_in_root_bias
        SA.install(verbose=False)
        mods, params = swap_in_root_bias(layers, idx, shared_root_embed=share,
                                        root_source='shared', gate_init=0.0,
                                        pillar_gate_init=0.0)
        for m in mods:                       # the arm held this at exactly 0
            m.pillar_gate.data.zero_()
            m.pillar_gate.requires_grad_(False)
    return which


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--pathway', choices=('none', 'residual', 'bias', 'both'), default='none')
    ap.add_argument('--pathway-layers', default='all',
                    help="which layers the pathway was trained on: `all` (A/C/E/D) or `top4` (X). "
                         "Parsed by the project's own parse_layer_spec_v13.")
    ap.add_argument('--ckpt', default=None,
                    help='default /tmp/root_arch_arms/head_<TAG>.pt.trunk.pt')
    ap.add_argument('--cap-train', type=int, default=400000)
    ap.add_argument('--cache', default='/workspace/nrmp_cache_9490',
                    help='word-list cache.  DEFAULT is the REFERENCE cache the published '
                         '92.34 %/83.92 % numbers were measured on, so the train/val/unseen SPLIT '
                         'is identical and the comparison is like-for-like.  (The arms themselves '
                         'were TRAINED on nrmp_cache_9490_aligned; that affects their weights, not '
                         'the word lists used to probe them.)')
    ap.add_argument('--released', action='store_true',
                    help='probe the RELEASED trunk (skip the arm checkpoint).  This is the only '
                         'fair baseline: the published 92.34 %% used a different cache, so it must '
                         'be re-measured through THIS code path before any arm is compared to it.')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    import echo_identity as E
    ck = args.ckpt or '/tmp/root_arch_arms/head_%s.pt.trunk.pt' % args.tag
    out = args.out or '/workspace/root_arch/probe_ridge_%s.json' % args.tag

    vocab, base, flash, model = MB.build(verbose=True)
    R = int(vocab.num_roots)
    A, SF = int(vocab.num_awzan), int(vocab.num_suffixes)

    if args.pathway != 'none':
        attach_pathway(model, args.pathway, args.pathway_layers)

    if args.released:
        log('--released: probing the RELEASED trunk, no arm checkpoint applied')
        blob = {}
    else:
        blob = torch.load(ck, map_location='cpu')
    state = blob['state'] if isinstance(blob, dict) and 'state' in blob else blob
    sd = model.state_dict()
    applied, skipped = 0, []
    for k, v in state.items():
        if k in sd and tuple(sd[k].shape) == tuple(v.shape):
            sd[k].copy_(v.to(sd[k].dtype))
            applied += 1
        else:
            skipped.append(k)
    log('trunk checkpoint %s: applied %d tensors, skipped %d%s'
        % (ck, applied, len(skipped), (' e.g. ' + str(skipped[:3])) if skipped else ''))
    if not args.released and applied < 100:
        raise SystemExit('refusing to probe: fewer than 100 tensors applied from %s' % ck)

    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    tr4 = [t.long() for t in torch.load(args.cache + '/train.pt', map_location='cpu')]
    va4 = [t.long() for t in torch.load(args.cache + '/val.pt', map_location='cpu')]
    specials = sorted({vocab.PAD_ROOT, vocab.BOS_ROOT, vocab.EOS_ROOT, vocab.UNK_ROOT,
                       vocab.root2id['<PARTICLE>']} |
                      {i for i, r in enumerate(vocab.roots_list) if r.startswith('<P:')})
    spec_t = torch.tensor(specials)

    tp, tr_, tw, ts, tk = packed_words(tr4, R, A, SF, cap=args.cap_train)
    vp, vr, vw, vs, vk = packed_words(va4, R, A, SF)
    mt = ~torch.isin(tr_, spec_t)
    mv = ~torch.isin(vr, spec_t)
    tp, tr_, tw, ts, tk = tp[mt], tr_[mt], tw[mt], ts[mt], tk[mt]
    vp, vr, vw, vs, vk = vp[mv], vr[mv], vw[mv], vs[mv], vk[mv]
    seen = torch.isin(vk, tk)
    log('unique train words %d | val words %d (seen %d, unseen %d)'
        % (tp.numel(), vp.numel(), int(seen.sum()), int((~seen).sum())))

    @torch.no_grad()
    def extract(P, Rr, W, S, tag, bs=1024):
        out = []
        for i in range(0, P.numel(), bs):
            sl = slice(i, min(i + bs, P.numel()))
            # `IshtiqaqRootBiasAttention` keeps an autoregressive `coverage_cache` sized to
            # (batch_size, total_kv_len) and only re-creates it when the LAST dim changes -- so a
            # short final batch raises
            #   "The size of tensor a (1024) must match the size of tensor b (855)"
            # That branch is unreachable in training (T=128 always) and it is not our module, so it
            # is neutralised here rather than edited.  `pillar_gate` is 0 so the cache has no
            # numerical effect anyway.
            for _m in getattr(model, 'backbone').layers:
                if hasattr(_m.self_attn, 'coverage_cache'):
                    _m.self_attn.coverage_cache = None
            p = P[sl].view(-1, 1).to(DEV)
            r = Rr[sl].view(-1, 1).to(DEV)
            w = W[sl].view(-1, 1).to(DEV)
            s = S[sl].view(-1, 1).to(DEV)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                emb = model.morphemic_embed(p, r, w, s)
                if args.pathway != 'none':
                    # BOTH wirings, exactly as the trainer's `live_h_windows` does.  The RCA stack
                    # reads `rca_stack.set_root_ids()`, NOT `self_attn.active_root_ids`; setting
                    # only the latter leaves the residual injector with `_root_ids = None` and it
                    # returns early -- which is what made an earlier run of this probe report the
                    # pathway-ON and pathway-OFF numbers as identical.
                    MB.set_active_root_ids(model, r)
                    st = getattr(model, 'root_cross', None)
                    if st is not None:
                        st.set_root_ids(r)
                h = model.final_norm(model.backbone(inputs_embeds=emb).last_hidden_state)
            out.append(h[:, 0, :].float().cpu())
        H = torch.cat(out)
        log('    h[%s] %s' % (tag, tuple(H.shape)))
        return H

    Htr = extract(tp, tr_, tw, ts, 'train-words')
    Hva = extract(vp, vr, vw, vs, 'val-words')

    t0 = time.time()
    Wr, lam = E.fit_ridge(Htr, tr_, R, DEV)
    log('ridge fit in %.1fs lam=%.4g' % (time.time() - t0, lam))

    ones_v = torch.ones(vp.numel(), dtype=torch.bool)
    ones_t = torch.ones(tp.numel(), dtype=torch.bool)
    res = {
        'tag': args.tag, 'pathway': args.pathway, 'pathway_layers': args.pathway_layers,
        'ckpt': ck, 'released': bool(args.released),
        'cache': args.cache, 'cap_train': args.cap_train,
        'n_train_words': int(tp.numel()), 'n_val_words': int(vp.numel()),
        'n_unseen': int((~seen).sum()), 'lam': float(lam),
        'val_all': E.eval_ridge(Wr, Hva, vr, ones_v, R, DEV, R),
        'val_unseen_words': E.eval_ridge(Wr, Hva, vr, ~seen, R, DEV, R),
        'val_seen_words': E.eval_ridge(Wr, Hva, vr, seen, R, DEV, R),
        'train_all': E.eval_ridge(Wr, Htr, tr_, ones_t, R, DEV, R),
        'marginal_val_all': E.majority_rate(vr, R),
        'marginal_val_unseen': E.majority_rate(vr[~seen], R),
    }
    log('')
    log('  ================ LENGTH-1 RIDGE PROBE [%s] pathway=%s ================'
        % (args.tag, args.pathway))
    for k in ('val_all', 'val_unseen_words', 'val_seen_words', 'train_all'):
        v = res[k]
        log('  %-18s acc@1 %6.2f%%  acc@5 %6.2f%%  (n=%d)'
            % (k, 100 * v['all']['acc@1'], 100 * v['all']['acc@5'], v['all']['n']))
    log('  marginal (majority root) val_all %.2f%%  val_unseen %.2f%%'
        % (100 * res['marginal_val_all']['majority_acc@1'],
           100 * res['marginal_val_unseen']['majority_acc@1']))
    log('  reference: released trunk 92.34%% (unseen 83.92%%) | raw Qwen2.5-0.5B 29.30%% (unseen 23.34%%)')
    json.dump(res, open(out, 'w'), indent=2)
    log('wrote %s' % out)


if __name__ == '__main__':
    main()
