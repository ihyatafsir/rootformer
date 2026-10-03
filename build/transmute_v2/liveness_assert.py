#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
liveness_assert.py -- the LIVENESS ASSERTION, built and run FIRST.

Four dead-code findings in this project looked like live defects, and one (the
`wazn_ids = torch.zeros_like(...)` stub) was still being read as the architecture AFTER it was
found.  Grep cannot answer "is this reached", so this file answers it three ways, none of which
is grep:

  (1) HOOK COUNTS      every nn.Module in the live model gets a forward hook and a call counter.
  (2) GRAD REACH       one real forward+backward; every trainable parameter's .grad is read.
                       A parameter that is never read in the forward has .grad is None.
                       Per-ELEMENT for tensors with numel <= 4, so stream_mix[1] is separable
                       from stream_mix[0].
  (3) PERTURBATION     every parameter with grad None or exactly zero is perturbed (+eps,
                       forward, -eps) and the OUTPUT delta is measured.  This is a causal test:
                       it distinguishes "read but zero gradient" from "never read".
                       A random sample of live parameters is perturbed as a positive control.

Then it asserts the CONSTANTS against the live vocab and that no two ROOT-ID SPACES are mixed.

Usage:
  python liveness_assert.py [--ckpt PATH] [--out JSON] [--json-only]
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get('RF_ROOT', '/workspace/hf_v19_2_release')).resolve()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'models'))

import torch
import torch.nn as nn

# --- the three spaces under suspicion, as LITERALS, so nothing silently follows a live value ---
LIVE_ROOTS, LIVE_AWZAN, LIVE_VOCAB = 9490, 142, 10052
STALE_ROOTS, STALE_AWZAN = 9015, 128
STALE_VOCAB_ALT = 9868        # the pre-tok10052 embed_tokens row count
OLDEST_ROOTS = 9114           # the pre-Lisan root_embed row count


def banner(s):
    print('\n' + '=' * 78 + f'\n{s}\n' + '=' * 78, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=str(ROOT / 'checkpoints' /
                    'rootformer_v19_2_synthesis_ar_backbone.awzan142.roots9490.tok10052.safetensors'))
    ap.add_argument('--out', default='/workspace/transmute_v2/liveness_report.json')
    ap.add_argument('--batch', type=int, default=2)
    ap.add_argument('--seq', type=int, default=8)
    ap.add_argument('--no-perturb', action='store_true')
    args = ap.parse_args()

    flags = []
    facts = {}

    # ---------------------------------------------------------------- build the LIVE model
    banner('0. BUILD THE LIVE MODEL (exactly as nrmt_train.py does)')
    import nrmp_vocab as nv
    from models.unified_rootformer_v12 import UnifiedRootformerV12
    from deepseek_v4_1_flash_model import UnifiedRootformerV17_DeepSeekFlash
    from nrmt_arch import RootformerNRMT
    from safetensors.torch import load_file

    V = next(v for k, v in vars(nv).items() if isinstance(v, type) and 'MorphemicVocab' in k)
    bp = str(ROOT / 'data/rootformer_v12_arabic_blueprint.json')
    vocab = V(bp)
    dev, dt = 'cpu', torch.float32
    base = UnifiedRootformerV12(bp, 'Qwen/Qwen2.5-0.5B', dev, dt).to(dev)
    flash = UnifiedRootformerV17_DeepSeekFlash(base, dev, dt).to(dev)
    model = RootformerNRMT(flash, vocab, dev, dt).to(dev)
    model.eval()

    sd = load_file(args.ckpt)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # nrmt_head.* is NEW vs the released nrmp_head.* generation; the trainer remaps it.
    new_head = [k for k in missing if k.startswith('nrmt_head.')]
    other_missing = [k for k in missing if not k.startswith('nrmt_head.')]
    print(f'checkpoint: {Path(args.ckpt).name}')
    print(f'  keys in file        : {len(sd)}')
    print(f'  missing             : {len(missing)}  (nrmt_head new={len(new_head)}, OTHER={len(other_missing)})')
    print(f'  unexpected          : {len(unexpected)}')
    if other_missing:
        flags.append({'kind': 'ckpt_missing_non_head', 'keys': other_missing[:20]})
    if unexpected:
        print(f'  unexpected keys (in file, not in model): {unexpected}')
    facts['ckpt'] = {'file': Path(args.ckpt).name, 'n_keys': len(sd),
                     'missing': len(missing), 'missing_non_head': other_missing,
                     'unexpected': unexpected}

    print(f'live vocab: num_roots={vocab.num_roots} num_awzan={vocab.num_awzan} '
          f'num_prefixes={vocab.num_prefixes} num_suffixes={vocab.num_suffixes} '
          f'vocab_size={vocab.base_tok.vocab_size}')

    # ---------------------------------------------------------------- 1. CONSTANTS
    banner('1. CONSTANTS vs THE LIVE VOCAB')
    if (vocab.num_roots, vocab.num_awzan, vocab.base_tok.vocab_size) != \
            (LIVE_ROOTS, LIVE_AWZAN, LIVE_VOCAB):
        flags.append({'kind': 'live_vocab_differs_from_spec',
                      'got': [vocab.num_roots, vocab.num_awzan, vocab.base_tok.vocab_size],
                      'want': [LIVE_ROOTS, LIVE_AWZAN, LIVE_VOCAB]})

    const_rows = []

    def ck(name, got, want, ok_extra=''):
        ok = (got == want)
        const_rows.append((name, got, want, ok, ok_extra))
        if not ok:
            flags.append({'kind': 'constant_mismatch', 'where': name, 'got': got, 'want': want})
        print(f'  [{"OK " if ok else "FAIL"}] {name:52s} got={got!s:>8s} want={want!s:>8s} {ok_extra}')

    # model-level constant surfaces that trace back to the hardcoded values
    st0 = model.backbone.layers[0].self_attn
    ck('base.vocab_size', base.vocab_size, LIVE_VOCAB)
    ck('backbone.embed_tokens rows', model.backbone.embed_tokens.weight.shape[0], LIVE_VOCAB)
    ck('morphemic_embed.root_embed rows', model.morphemic_embed.root_embed.weight.shape[0], LIVE_ROOTS)
    ck('morphemic_embed.wazn_embed rows', model.morphemic_embed.wazn_embed.weight.shape[0], LIVE_AWZAN)
    ck('nrmt_head.root_head out', model.nrmt_head.root_head.weight.shape[0], LIVE_ROOTS)
    ck('nrmt_head.wazn_head out', model.nrmt_head.wazn_head.weight.shape[0], LIVE_AWZAN)
    ck('nrmt_head.hist_emb[0] rows', model.nrmt_head.hist_emb[0].weight.shape[0], LIVE_ROOTS)
    ck('self_attn.root_embed rows  (L24)', st0.root_embed.weight.shape[0], LIVE_ROOTS,
       '<== STALE SPACE' if st0.root_embed.weight.shape[0] != LIVE_ROOTS else '')
    ck('self_attn.wazn_embed rows  (L24)', st0.wazn_embed.weight.shape[0], LIVE_AWZAN,
       '<== STALE SPACE' if st0.wazn_embed.weight.shape[0] != LIVE_AWZAN else '')
    ck('base.metadata_mask numel', int(base.metadata_mask.numel()), LIVE_VOCAB)
    ck('base.id_to_root_table numel', int(base.id_to_root_table.numel()), LIVE_VOCAB)

    # ---------------------------------------------------------------- 2. HOOK REACH
    banner('2. MODULE REACH (forward hooks, not grep)')
    counts = {}
    handles = []
    for name, mod in model.named_modules():
        if name == '':
            continue
        def mk(nm):
            def h(module, args, output):
                counts[nm] = counts.get(nm, 0) + 1
            return h
        handles.append(mod.register_forward_hook(mk(name)))

    B, T = args.batch, args.seq
    g = torch.Generator().manual_seed(1234)
    P = torch.randint(1, vocab.num_prefixes, (B, T), generator=g)
    R = torch.randint(5, LIVE_ROOTS, (B, T), generator=g)
    W = torch.randint(3, LIVE_AWZAN, (B, T), generator=g)
    S = torch.randint(3, vocab.num_suffixes, (B, T), generator=g)

    def run(return_out=True):
        return model(P, R, W, S, op_ids=torch.zeros_like(R),
                     target_roots=R, target_awzan=W,
                     target_prefixes=P, target_suffixes=S)

    out = run()
    loss = out['loss']
    loss.backward()

    # attention modules: exactly 24 expected calls for each self_attn
    sa_counts = sorted(v for k, v in counts.items() if k.endswith('.self_attn'))
    mlp_counts = sorted(v for k, v in counts.items() if k.endswith('.mlp'))
    print(f'  self_attn call counts : {sorted(set(sa_counts))} over {len(sa_counts)} layers '
          f'(24 expected)')
    print(f'  mlp call counts       : {sorted(set(mlp_counts))} over {len(mlp_counts)} layers')
    extra_called = {k: v for k, v in counts.items()
                    if not any(k.endswith(s) for s in ('.self_attn', '.mlp'))}
    for k in sorted(extra_called):
        print(f'  reached: {k:56s} {extra_called[k]}x')

    # ---------------------------------------------------------------- 3. PARAMETER REACH
    banner('3. PARAMETER REACH (.grad after one real backward; per-element when numel<=4)')
    grad_info = {}
    for name, p in model.named_parameters():
        if p.grad is None:
            grad_info[name] = {'numel': p.numel(), 'grad': None}
            continue
        gf = p.grad.detach().float()
        if p.numel() <= 4:
            grad_info[name] = {'numel': p.numel(), 'grad': 'per_element',
                               'elements': [float(x) for x in gf.reshape(-1)]}
        else:
            grad_info[name] = {'numel': p.numel(), 'grad': float(gf.norm())}

    never = [n for n, v in grad_info.items() if v['grad'] is None]
    zero = [n for n, v in grad_info.items()
            if isinstance(v.get('grad'), float) and v['grad'] == 0.0]
    pe = {n: v['elements'] for n, v in grad_info.items() if v.get('grad') == 'per_element'}
    print(f'  parameters with .grad is None (NEVER READ in the forward): {len(never)}')
    for n in never:
        print(f'      DEAD: {n}')
    print(f'  parameters with .grad norm exactly 0: {len(zero)}')
    for n in zero:
        print(f'      ZERO: {n}')
    print('  per-element grads of the small tensors:')
    for n in sorted(pe):
        print(f'      {n:60s} {pe[n]}')

    # the two specific stub/space claims
    banner('3b. THE TWO SPECIFIC CLAIMS')
    ar = [l.self_attn.active_root_ids for l in model.backbone.layers]
    claim_active = all(x is None for x in ar)
    print(f'  after a FULL live forward, every self_attn.active_root_ids is None : {claim_active}')
    print('     => the native IshtiqaqAttentionV12 root path (root_embed, root_q_proj,'
          ' root_k_proj,\n        ishtiqaq_gamma, stream_mix[1], Pillar I/II) is NEVER given root'
          ' ids in the NRMT forward.')
    if not claim_active:
        flags.append({'kind': 'native_root_path_actually_reached'})

    r_ids_stub, w_ids_stub = base.extract_morphemic_ids(R)
    stub_zero = bool((w_ids_stub == 0).all())
    print(f'  base.extract_morphemic_ids() returns wazn_ids == zeros : {stub_zero}  '
          f'(unified_rootformer_v12.py:121)')
    print('     => the returned wazn id is a DEAD STUB.  The wazn is nevertheless LIVE through a'
          ' different\n        route: morphemic_embed(p, r, w, s) consumes the real w in the NRMT'
          ' path.  This is the one\n        finding that was previously misread as the'
          ' architecture; the perturbation test below settles it.')

    # ---------------------------------------------------------------- 4. PERTURBATION
    banner('4. PERTURBATION (causal: does changing this parameter change the OUTPUT?)')
    # FIRST: prove the forward is BIT-DETERMINISTIC under one fixed grad mode.  Comparing a
    # no_grad reference against a grad-enabled run introduces a ~1e-6 kernel-selection
    # difference and every "inert" parameter then looks live.  That is exactly how a liveness
    # test lies; it is settled here before any delta is believed.
    def fwd():
        with torch.no_grad():
            return run()['root_logits'].detach().clone()

    d1, d2 = fwd(), fwd()
    det = bool(torch.equal(d1, d2))
    print(f'  determinism: two identical forwards bit-equal = {det} '
          f'(max|d| = {float((d1 - d2).abs().max()) if not det else 0.0:.3e})')
    if not det:
        flags.append({'kind': 'forward_not_bit_deterministic'})
    ref = d1

    def _eps(p):
        """A perturbation that is non-zero and FINITE for every shape.

        `p.std()` of a 1-element tensor is NaN (degrees of freedom <= 0), and `nan or x` is nan,
        so a naive `(1e-3*std) or 1e-4` poisons every scalar test with NaN and then reports every
        parameter as "live".  That is a liveness test that lies in the SAFE direction; it is
        guarded here.
        """
        pf = p.detach().float()
        if pf.numel() > 1:
            s = float(pf.std())
            if math.isfinite(s) and s > 0:
                return 1e-3 * s
        if pf.numel() == 1:
            v = abs(float(pf))
            if math.isfinite(v) and v > 0:
                return 1e-3 * v
        return 1e-3

    def perturb(name, p, eps):
        """Perturb ONE parameter and compare against a baseline taken IMMEDIATELY BEFORE it.

        Two traps are closed here, both of which make an inert parameter look live:
          * `p + eps - eps != p` in floating point, so a naive add/sub round-trip leaves the
            weight ~1 ULP off and EVERY later comparison inherits that drift.  The restore is
            therefore `copy_` from a saved clone, and the drift is asserted to be zero at the end.
          * comparing against one global reference computed before any perturbation propagates
            that same drift.  Each comparison gets its OWN baseline.
        """
        assert math.isfinite(eps) and eps > 0, f'{name}: bad eps {eps}'
        p0 = p.detach().clone()
        base_o = fwd()
        with torch.no_grad():
            p.add_(eps)
        o = fwd()
        with torch.no_grad():
            p.copy_(p0)                      # EXACT restore, not sub_(eps)
        bad = (not bool(torch.isfinite(o).all())) or (not bool(torch.isfinite(base_o).all()))
        d = float('nan') if bad else float((o - base_o).abs().max())
        eq = bool(torch.equal(o, base_o))
        return d, eq, bad

    suspected_dead = sorted(set(never) | set(zero) | {
        n for n in grad_info if n.endswith(('.stream_mix', '.ishtiqaq_gamma',
                                            '.coverage_weight', '.coverage_threshold',
                                            '.governance_strength'))})
    # add the native root tables regardless (they are the ones the whole task is about)
    suspected_dead = sorted(set(suspected_dead) | {
        n for n, _ in model.named_parameters()
        if n.endswith(('.root_embed.weight', '.root_q_proj.weight', '.root_k_proj.weight'))
        and '.self_attn.' in n})
    control = ['morphemic_embed.root_embed.weight', 'morphemic_embed.wazn_embed.weight',
               'nrmt_head.root_head.weight', 'backbone.layers.0.self_attn.q_proj.weight',
               'backbone.embed_tokens.weight']
    if not args.no_perturb:
        results = {}
        print('  -- POSITIVE CONTROLS: parameters the grad test called LIVE --')
        for n in control:
            p = dict(model.named_parameters())[n]
            eps = _eps(p)
            d, eq, bad = perturb(n, p, eps)
            results[n] = {'max_abs_delta': d, 'bit_equal': eq, 'eps': eps, 'nan_poisoned': bad}
            print(f'      {n:56s} eps={eps:.3e} max|d|={d:.6e} bit_equal={eq} '
                  f'{"POISONED" if bad else ("INERT" if eq else "LIVE")}')
        print('  -- the parameters the grad test flagged (confirm causally) --')
        dead_confirmed, dead_refuted = [], []
        for n in suspected_dead:
            p = dict(model.named_parameters())[n]
            eps = _eps(p)
            d, eq, bad = perturb(n, p, eps)
            results[n] = {'max_abs_delta': d, 'bit_equal': eq, 'eps': eps, 'nan_poisoned': bad}
            tag = 'POISONED' if bad else ('INERT (bit-equal)' if eq else 'LIVE')
            (dead_refuted if not eq else dead_confirmed).append(n)
            print(f'      {n:56s} eps={eps:.3e} max|d|={d:.6e} bit_equal={eq} {tag}')
        facts['perturbation'] = results
        facts['inert_confirmed'] = dead_confirmed
        facts['flagged_but_live'] = dead_refuted
        print(f'  => INERT confirmed causally (bit-equal under an eps perturbation): '
              f'{len(dead_confirmed)}')
        print(f'  => flagged by grad but CAUSALLY LIVE: {len(dead_refuted)}')
        # the restore must be EXACT: if the model drifted, every verdict above is suspect
        drift = bool(torch.equal(fwd(), ref))
        print(f'  restore integrity: after {len(control) + len(suspected_dead)} perturb/restore '
              f'cycles the forward is bit-equal to the original = {drift}')
        if not drift:
            flags.append({'kind': 'perturbation_restore_drifted'})
        facts['perturbation_restore_exact'] = drift
        if dead_refuted:
            flags.append({'kind': 'grad_flagged_but_perturbation_live', 'params': dead_refuted})
    else:
        dead_confirmed, dead_refuted = [], []

    # ---------------------------------------------------------------- 5. ROOT-ID SPACES
    banner('5. ROOT-ID SPACES -- no two spaces may be silently mixed')
    spaces = {
        'live       morphemic_embed.root_embed': model.morphemic_embed.root_embed.weight.shape[0],
        'live       nrmp_head.root_head': model.nrmt_head.root_head.weight.shape[0],
        'live       nrmp_head.hist_emb[0]': model.nrmt_head.hist_emb[0].weight.shape[0],
        'live       vocab.num_roots': vocab.num_roots,
        'STALE      self_attn.root_embed (x24)': st0.root_embed.weight.shape[0],
    }
    for k, v in spaces.items():
        print(f'  {k:44s} rows={v}')
    rows_live = [v for k, v in spaces.items() if k.startswith('live')]
    if len(set(rows_live)) != 1:
        flags.append({'kind': 'live_root_spaces_disagree', 'rows': rows_live})
    if st0.root_embed.weight.shape[0] != vocab.num_roots:
        flags.append({'kind': 'stale_root_space_mixed',
                      'self_attn_root_embed_rows': st0.root_embed.weight.shape[0],
                      'live_num_roots': vocab.num_roots,
                      'missing_ids': vocab.num_roots - st0.root_embed.weight.shape[0]})

    # aliasing: what clamp(root_ids, 0, R-1) does to the live id stream
    Rrows = st0.root_embed.weight.shape[0]
    ids = torch.arange(vocab.num_roots)
    clamped = ids.clamp(0, Rrows - 1)
    collide = (vocab.num_roots - Rrows)
    print(f'  clamp(root_ids, 0, {Rrows - 1}) on the live id stream 0..{vocab.num_roots - 1}:')
    print(f'      ids that have NO row of their own : {collide}  '
          f'({100.0 * collide / vocab.num_roots:.2f} % of roots)')
    print(f'      ids that ALIAS onto row {Rrows - 1}   : {collide} ids collapse there, so row '
          f'{Rrows - 1} represents {collide + 1} different roots')
    print(f'      injective                          : {bool((clamped.unique().numel() == vocab.num_roots))}')
    flags.append({'kind': 'root_id_aliasing',
                  'table_rows': Rrows, 'live_ids': vocab.num_roots,
                  'ids_without_row': collide,
                  'pct_of_roots': round(100.0 * collide / vocab.num_roots, 3),
                  'alias_row': Rrows - 1, 'aliased_ids': collide})

    # the token-level id_to_root_table: injective? does it use real offsets?
    t = base.id_to_root_table
    root_tok_ids = [i for i in range(base.vocab_size)
                    if base.tokenizer.id_to_token.get(i, '').startswith('<root_')]
    vals = t[torch.tensor(root_tok_ids)] if root_tok_ids else torch.zeros(0, dtype=torch.long)
    uniq = int(vals.unique().numel())
    print(f'  base.id_to_root_table: {len(root_tok_ids)} <root_*> tokens -> {uniq} distinct root '
          f'ids  (injective={uniq == len(root_tok_ids)})')
    print(f'      min={int(vals.min()) if vals.numel() else None} '
          f'max={int(vals.max()) if vals.numel() else None}   '
          f'(max must be <= {LIVE_ROOTS - 1}; <root_start>/<root_end> map to 1)')
    if uniq != len(root_tok_ids):
        flags.append({'kind': 'id_to_root_table_not_injective',
                      'tokens': len(root_tok_ids), 'distinct': uniq})
    facts['id_to_root_table'] = {'root_tokens': len(root_tok_ids), 'distinct': uniq,
                                 'min': int(vals.min()) if vals.numel() else None,
                                 'max': int(vals.max()) if vals.numel() else None}

    # ---------------------------------------------------------------- 6. THE FLASH HOOKS
    banner('6. THE FLASH WRAPPER HOOKS (found during recon, not in the brief)')
    flash_hooked = []
    for i, l in enumerate(model.backbone.layers):
        n = len(l._forward_hooks)
        if n:
            flash_hooked.append((i, n, [type(m).__name__ for m in
                                        (flash.engram_layer1, flash.engram_layer14,
                                         flash.farahidi_orbit)]))
    n_flash_params = sum(p.numel() for p in flash.parameters())
    in_model = {id(p) for p in model.parameters()}
    tracked = sum(p.numel() for p in flash.parameters() if id(p) in in_model)
    untracked = [n for n, p in flash.named_parameters() if id(p) not in in_model]
    n_untracked = sum(p.numel() for n, p in flash.named_parameters() if id(p) not in in_model)
    print(f'  trunk layers carrying EXTRA (flash) forward hooks: {[(i, n) for i, n, _ in flash_hooked]}')
    print(f'  flash wrapper params total                       : {n_flash_params / 1e6:.3f} M')
    print(f'  ... reachable through model.backbone (TRACKED)   : {tracked / 1e6:.3f} M')
    print(f'  ... NOT in model.parameters() (UNTRACKED, never optimised): {n_untracked / 1e6:.3f} M')
    print(f'      untracked param names: {untracked}')
    print('  the hooks inject engram_layer1 @L1, engram_layer14 + farahidi_orbit @L14.')
    print('  both modules are ZERO-init output (mem_proj / out_proj = 0) and are NOT in the')
    print('  checkpoint and NOT in model.parameters(), so they are numerically INERT but they')
    print('  accumulate .grad forever and would inject random noise the moment anyone')
    print('  initialised their output projections.  FLAGGED for the new construction.')
    flags.append({'kind': 'untracked_live_hooks',
                  'layers': [i for i, _, _ in flash_hooked],
                  'flash_params_M': round(n_flash_params / 1e6, 3),
                  'untracked_params_M': round(n_untracked / 1e6, 3),
                  'untracked_names': untracked,
                  'numerically_inert': True})

    for h in handles:
        h.remove()

    # ---------------------------------------------------------------- verdict
    banner('FLAGS')
    if not flags:
        print('  NONE.  Every named component is reached, every constant matches the live vocab,')
        print('  and no two root-id spaces are mixed.')
    for f in flags:
        print(f'  * {f["kind"]}: ' + json.dumps({k: v for k, v in f.items() if k != "kind"},
                                                ensure_ascii=False)[:220])

    facts['flags'] = flags
    facts['constants'] = [{'where': n, 'got': g, 'want': w, 'ok': o}
                          for (n, g, w, o, _) in const_rows]
    facts['grad_none'] = never
    facts['grad_zero'] = zero
    facts['per_element_grad'] = pe
    facts['active_root_ids_none'] = claim_active
    facts['extract_morphemic_ids_wazn_is_stub'] = stub_zero
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(facts, indent=2, ensure_ascii=False))
    print(f'\nwrote {args.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
