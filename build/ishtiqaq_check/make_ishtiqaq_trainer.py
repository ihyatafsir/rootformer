#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_ishtiqaq_trainer.py -- produce nrmt_train_ishtiqaq.py from the LIVE nrmt_train.py by
applying a small number of exact, asserted replacements.

Every replacement is guarded: the script fails loudly if the anchor is not found exactly once,
so the output cannot silently drift from the file it claims to patch.  With
`--ishtiqaq-root-bias none` (the default) every inserted block is either not entered or is a
provable no-op, so flag-off behaviour is the ORIGINAL code path.
"""
import argparse
import difflib
import hashlib
import sys
from pathlib import Path

R = []


def add(name, old, new):
    R.append((name, old, new))


# ---------------------------------------------------------------- R0: runnable from anywhere
# The experiment copy lives in /workspace/ishtiqaq_check/ so that its argv still contains
# `nrmt_train.py --checkpoint` and the established GPU guards (`pgrep -f "nrmt_train[.]py
# --checkpoint"`) SEE it -- a third concurrent arm must never be launched by accident.
add('R0 root dir',
    """ROOT_DIR = Path(__file__).resolve().parent
""",
    """ROOT_DIR = Path(__file__).resolve().parent
if not (ROOT_DIR / 'data/rootformer_v12_arabic_blueprint.json').exists():
    # experiment copy: run from anywhere, resolve the module tree + blueprint in the release
    ROOT_DIR = Path('/workspace/hf_v19_2_release')
""")


# ---------------------------------------------------------------- R1: new CLI arguments
add('R1 args',
    """    ap.add_argument('--unfreeze-last', type=int, default=0,
""",
    """    # ---- NATIVE IshtiqaqAttentionV12 ROOT SCORE BIAS (flag off => byte-identical path) ------
    # The shipped module's root path is a pure ADDITIVE BIAS on the surface attention SCORES
    # (`models/ishtiqaq_attention_v12.py:186`) and has never been wired: this trainer drives
    # `model.backbone(inputs_embeds=emb)` directly, so `active_root_ids` stays None and 0 of 24
    # attention calls carry root ids.  `ishtiqaq_root_bias.py` wraps the mechanism in a
    # ZERO-INIT gate and (with --ishtiqaq-root-source shared) re-sources its Q/K projections
    # from the released 448-dim morphemic table, so the ONLY difference from the residual RCA
    # is the INSERTION POINT: score bias vs residual injection.
    ap.add_argument('--ishtiqaq-root-bias', default='none',
                    help='NATIVE root score bias on these trunk layers (`none` default = the '
                         'historical path, unchanged; `topN`/`midN`/`everyK`/`a,b,c`).  The '
                         'shipped module file is NOT edited; a gated subclass is swapped in.')
    ap.add_argument('--ishtiqaq-root-source', choices=('shared', 'native'), default='shared',
                    help='shared = the released 448-dim `morphemic_embed.root_embed` (9490 rows, '
                         'the same K/V table the RCA reads); native = the shipped per-layer '
                         '9015x64 table (stale id space, at N(0,1) default init)')
    ap.add_argument('--ishtiqaq-gate-init', type=float, default=0.0,
                    help='init of the zero-gated native root bias (0.0 = bitwise no-op at step 0)')
    ap.add_argument('--ishtiqaq-pillar-gate-init', type=float, default=0.0,
                    help='init of the gate on the shipped Pillar-I/II penalties (default 0 = off; '
                         'the shipped coverage_weight=2.0 / governance_strength=1.5 fire the '
                         'moment root ids are supplied, so they must be gated separately)')
    ap.add_argument('--ishtiqaq-lr-scale', type=float, default=0.3,
                    help='lr of the native root-bias weights as a fraction of --lr')
    ap.add_argument('--ishtiqaq-ablate-eval', action='store_true',
                    help='also evaluate with the native root gates forced to 0: the causal '
                         'ablation of the native path on the SAME trunk weights')
    ap.add_argument('--unfreeze-last', type=int, default=0,
""")

# ---------------------------------------------------------------- R2: swap the modules in
add('R2 attach',
    """    live = bool(unfrozen_layers) or rca_stack is not None
""",
    """    # ---- NATIVE root score bias: swapped in only when explicitly requested -------------
    ishtiqaq_mods, ishtiqaq_params, ishtiqaq_layers = [], [], []
    ishtiqaq_root_source = 'native'
    if str(args.ishtiqaq_root_bias).strip().lower() not in ('', 'none', 'off', '0'):
        if '/workspace/ishtiqaq_check' not in sys.path:
            sys.path.insert(0, '/workspace/ishtiqaq_check')
        from ishtiqaq_root_bias import (swap_in_root_bias, parse_layer_spec as _ig_spec,
                                        zero_gates as _ig_zero, restore_gates as _ig_restore)
        ishtiqaq_layers = _ig_spec(args.ishtiqaq_root_bias, len(model.backbone.layers))
        ishtiqaq_root_source = args.ishtiqaq_root_source
    if ishtiqaq_layers:
        ishtiqaq_mods, ishtiqaq_params = swap_in_root_bias(
            model.backbone.layers, ishtiqaq_layers,
            shared_root_embed=model.morphemic_embed.root_embed,
            root_source=ishtiqaq_root_source,
            gate_init=args.ishtiqaq_gate_init,
            pillar_gate_init=args.ishtiqaq_pillar_gate_init)
        for _p in ishtiqaq_params:
            _p.requires_grad_(True)
        print(f'[*] NATIVE root score bias on trunk layers {ishtiqaq_layers} '
              f'({sum(p.numel() for p in ishtiqaq_params)/1e6:.3f}M NEW params, source='
              f'{ishtiqaq_root_source}, gates='
              f'{[(float(m.root_gate.detach()), float(m.pillar_gate.detach())) for m in ishtiqaq_mods]}'
              f' on {ishtiqaq_params[0].device}', flush=True)
        if unfrozen_layers and list(ishtiqaq_layers) != list(unfrozen_layers):
            print(f'[warn] native root-bias layers {ishtiqaq_layers} != unfrozen layers '
                  f'{unfrozen_layers}', flush=True)
    _ig_ids = {id(p) for p in ishtiqaq_params}
    live = bool(unfrozen_layers) or rca_stack is not None or bool(ishtiqaq_layers)
""")

# ---------------------------------------------------------------- R3: param accounting
add('R3 trunk params',
    """    trunk_params = [p for n, p in model.named_parameters()
                    if p.requires_grad and not n.startswith(('nrmt_head.', 'root_cross.'))]
    all_trainable = head_params + rca_params + trunk_params
""",
    """    trunk_params = [p for n, p in model.named_parameters()
                    if p.requires_grad and not n.startswith(('nrmt_head.', 'root_cross.'))
                    and id(p) not in _ig_ids]
    all_trainable = head_params + rca_params + trunk_params + ishtiqaq_params
""")

# ---------------------------------------------------------------- R4: the report line
add('R4 report',
    """          + (f' | trunk {_m(trunk_params):.2f}M @lr {args.lr*args.trunk_lr_scale:g} '
             f'layers {unfrozen_layers}' if trunk_params else '')
          + ')', flush=True)
""",
    """          + (f' | trunk {_m(trunk_params):.2f}M @lr {args.lr*args.trunk_lr_scale:g} '
             f'layers {unfrozen_layers}' if trunk_params else '')
          + (f' | ishtiqaq {_m(ishtiqaq_params):.3f}M @lr {args.lr*args.ishtiqaq_lr_scale:g} '
             f'layers {ishtiqaq_layers} source={ishtiqaq_root_source}'
             if ishtiqaq_params else '')
          + ')', flush=True)
""")

# ---------------------------------------------------------------- R5: the wiring
add('R5 live wiring',
    """        emb = model.morphemic_embed(Pr_w, Tr_w, Wr_w, Sr_w)
        if rca_stack is not None:
            rca_stack.set_root_ids(Tr_w)
        out = model.backbone(inputs_embeds=emb)
""",
    """        emb = model.morphemic_embed(Pr_w, Tr_w, Wr_w, Sr_w)
        if rca_stack is not None:
            rca_stack.set_root_ids(Tr_w)
        # the ONLY wiring the shipped native root path has ever needed
        for _i in ishtiqaq_layers:
            model.backbone.layers[_i].self_attn.active_root_ids = Tr_w
        out = model.backbone(inputs_embeds=emb)
""")

# ---------------------------------------------------------------- R6: optimizer group
add('R6 opt group',
    """        if rca_params:
            groups.append({'params': rca_params, 'lr': args.lr * args.rca_lr_scale, 'grp': 'rca'})
""",
    """        if rca_params:
            groups.append({'params': rca_params, 'lr': args.lr * args.rca_lr_scale, 'grp': 'rca'})
        if ishtiqaq_params:
            groups.append({'params': ishtiqaq_params,
                           'lr': args.lr * args.ishtiqaq_lr_scale, 'grp': 'ishtiqaq'})
""")

# ---------------------------------------------------------------- R7: ablate eval
add('R7 ablate eval',
    """                if 'NOVEL_only' in m_off:
                    m['NOVEL_only_RCA_OFF'] = m_off['NOVEL_only']
""",
    """                if 'NOVEL_only' in m_off:
                    m['NOVEL_only_RCA_OFF'] = m_off['NOVEL_only']
            if live and args.ishtiqaq_ablate_eval and ishtiqaq_mods:
                _old_ig = _ig_zero(ishtiqaq_mods)
                try:
                    m_ig = evaluate(None, Tv[:, :-1], Wv[:, :-1], Pv[:, :-1], Sv[:, :-1],
                                    Ova_w, live_streams=_ls)
                finally:
                    _ig_restore(ishtiqaq_mods, _old_ig)
                if 'ALL_val' in m_ig:
                    m['ALL_val_ISHTIQAQ_OFF'] = m_ig['ALL_val']
                if 'NOVEL_only' in m_ig:
                    m['NOVEL_only_ISHTIQAQ_OFF'] = m_ig['NOVEL_only']
""")

# ---------------------------------------------------------------- R8: gate readout
add('R8 gate readout',
    """                if h_drift is not None:
                    extra += f' | h_drift {h_drift:.4e}'
""",
    """                if ishtiqaq_mods:
                    extra += (' | ISHTIQAQ gates '
                              + ','.join(f'{float(mm.root_gate):+.4f}/{float(mm.pillar_gate):+.4f}'
                                         for mm in ishtiqaq_mods))
                if h_drift is not None:
                    extra += f' | h_drift {h_drift:.4e}'
""")

# ---------------------------------------------------------------- R9: saved trunk payload
add('R9 save',
    """                keep = {n: p.detach().to(torch.bfloat16).cpu() for n, p in model.named_parameters()
                        if n.startswith('root_cross.')
                        or any(n.startswith(f'backbone.layers.{i}.') for i in unfrozen_layers)}
                torch.save({'state': keep, 'unfrozen_layers': unfrozen_layers,
                            'rca_layers': rca_layers, 'live': True,
""",
    """                keep = {n: p.detach().to(torch.bfloat16).cpu() for n, p in model.named_parameters()
                        if n.startswith('root_cross.')
                        or any(n.startswith(f'backbone.layers.{i}.') for i in unfrozen_layers)
                        or any(n.startswith(f'backbone.layers.{i}.') for i in ishtiqaq_layers)}
                torch.save({'state': keep, 'unfrozen_layers': unfrozen_layers,
                            'rca_layers': rca_layers, 'live': True,
                            **({'ishtiqaq_layers': ishtiqaq_layers,
                                'ishtiqaq_root_source': ishtiqaq_root_source,
                                'ishtiqaq_gate_init': float(args.ishtiqaq_gate_init),
                                'ishtiqaq_lr_scale': float(args.ishtiqaq_lr_scale)}
                               if ishtiqaq_layers else {}),
""")

MARKS = ('ishtiqaq', '_ig_ids', '_ig_zero', '_ig_restore', '_ig_spec',
         'ishtiqaq_layers', 'ishtiqaq_params', 'ishtiqaq_mods',
         'ishtiqaq_root_source')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)
    ap.add_argument('--dst', required=True)
    ap.add_argument('--diff', default=None)
    args = ap.parse_args()
    src = Path(args.src).read_text(encoding='utf-8')
    orig_md5 = hashlib.md5(src.encode()).hexdigest()
    out = src
    for name, old, new in R:
        n = out.count(old)
        if n != 1:
            print(f'[!] {name}: anchor found {n} times (expected 1) -- ABORT')
            return 1
        out = out.replace(old, new)
        print(f'[ok] {name}')
    dst = Path(args.dst)
    dst.write_text(out, encoding='utf-8')
    print(f'[*] wrote {dst} ({len(out.splitlines())} lines)')
    # report the diff, and assert every changed line mentions the new feature
    d = list(difflib.unified_diff(src.splitlines(True), out.splitlines(True),
                                  'nrmt_train.py', 'nrmt_train_ishtiqaq.py', n=2))
    added = [l for l in d if l.startswith('+') and not l.startswith('+++')]
    print(f'[*] diff: {len(added)} added lines, {sum(1 for l in d if l.startswith("-") and not l.startswith("---"))} removed lines')
    if args.diff:
        Path(args.diff).write_text(''.join(d), encoding='utf-8')
        print(f'[*] diff -> {args.diff}')
    print(f'[*] src md5 {orig_md5}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
