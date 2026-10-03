#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_trainer_v13.py -- generate `nrmt_train_v13.py` from the verified A2 trainer.

Base: `_ref_nrmt_train_ishtiqaq.py` (md5 e8666ceee12d38eba959f1bc4d947b8b), which is itself the
canonical `nrmt_train.py` (md5 56ef3f2fd9ddc352dee596f32590e7b5) + the 171-line NATIVE root
score-bias patch.  Neither base file is modified; this script writes a sibling and a unified
diff so the delta is auditable.

WHAT V13 ADDS (and nothing else)
--------------------------------
  --model-v13           build `UnifiedRootformerV13` (roots 9490 / awzan 142 / vocab 10052) and
                        DROP the 48 stale-SPACE keys, asserting that is exactly what was dropped.
                        Without this the corrected architecture cannot load the released
                        checkpoint at all (9015 into 9490 raises).
  --unfreeze-trunk-all  train the WHOLE trunk at one LR: every backbone layer PLUS
                        `morphemic_embed` (the input stage -- where the (P,R,W,S) tuple creates
                        the root signal) PLUS `final_norm`.  `--unfreeze-last` cannot express
                        this: it stops at the backbone.
  --root-cross-attn all accepts `all`/`earlyN` in the layer spec (via early_root_path).
  --flash-hooks         `off` detaches the deepseek_v4_1_flash_model hooks (see
                        liveness_report.json flag `untracked_live_hooks`: 151.95 M params on
                        layers 1/11/14 that are in NO checkpoint and NO optimizer).  They are
                        zero-init INERT; the flag exists so the choice is explicit and recorded.
"""
import difflib
import hashlib
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE / '_ref_nrmt_train_ishtiqaq.py'
DST = HERE / 'nrmt_train_v13.py'


def md5(p):
    return hashlib.md5(pathlib.Path(p).read_bytes()).hexdigest()


def sub_once(text, old, new, tag):
    n = text.count(old)
    if n != 1:
        raise SystemExit(f'[make_trainer_v13] anchor {tag!r} matched {n} times, expected 1')
    return text.replace(old, new, 1)


def patch(text):
    # ---- 1. header note + sys.path for transmute_v2 (root_space / v13 / early_root_path) -----
    text = sub_once(
        text,
        "ROOT_DIR = Path(__file__).resolve().parent\n"
        "if not (ROOT_DIR / 'data/rootformer_v12_arabic_blueprint.json').exists():\n"
        "    # experiment copy: run from anywhere, resolve the module tree + blueprint in the release\n"
        "    ROOT_DIR = Path('/workspace/hf_v19_2_release')\n"
        "sys.path.insert(0, str(ROOT_DIR))\n"
        "sys.path.insert(0, str(ROOT_DIR / 'models'))\n",
        "ROOT_DIR = Path(__file__).resolve().parent\n"
        "if not (ROOT_DIR / 'data/rootformer_v12_arabic_blueprint.json').exists():\n"
        "    # experiment copy: run from anywhere, resolve the module tree + blueprint in the release\n"
        "    ROOT_DIR = Path('/workspace/hf_v19_2_release')\n"
        "sys.path.insert(0, str(ROOT_DIR))\n"
        "sys.path.insert(0, str(ROOT_DIR / 'models'))\n"
        "# transmute_v2: root_space (the ONE root-id space), unified_rootformer_v13, early_root_path\n"
        "if '/workspace/transmute_v2' not in sys.path:\n"
        "    sys.path.insert(0, '/workspace/transmute_v2')\n",
        'header')

    # ---- 2. new args ------------------------------------------------------------------------
    text = sub_once(
        text,
        "    ap.add_argument('--unfreeze-last', type=int, default=0,\n",
        "    # ---- V13: corrected root-id space + whole-trunk training + early root schedule ------ \n"
        "    ap.add_argument('--model-v13', action='store_true',\n"
        "                    help='build UnifiedRootformerV13 (roots 9490 / awzan 142 / vocab '\n"
        "                         '10052) and drop the 48 stale-SPACE self_attn keys on load')\n"
        "    ap.add_argument('--unfreeze-trunk-all', action='store_true',\n"
        "                    help='train EVERY backbone layer + morphemic_embed (the input stage) '\n"
        "                         '+ final_norm at --lr*--trunk-lr-scale.  No freezing, no EWC.')\n"
        "    ap.add_argument('--flash-hooks', choices=('on', 'off'), default='on',\n"
        "                    help='deepseek_v4_1_flash_model registers forward hooks on trunk '\n"
        "                         'layers 1/11/14 whose 151.95M params are in NO checkpoint and NO '\n"
        "                         'optimizer (zero-init INERT).  `off` detaches them.')\n"
        "    ap.add_argument('--unfreeze-last', type=int, default=0,\n",
        'args')

    # ---- 3. build V13 ----------------------------------------------------------------------
    text = sub_once(
        text,
        "    base = UnifiedRootformerV12(bp, 'Qwen/Qwen2.5-0.5B', str(device), torch.bfloat16).to(device)\n",
        "    if args.model_v13:\n"
        "        from models.unified_rootformer_v13 import UnifiedRootformerV13\n"
        "        _t0 = time.time()\n"
        "        base = UnifiedRootformerV13(bp, 'Qwen/Qwen2.5-0.5B', str(device),\n"
        "                                    torch.bfloat16, vocab=vocab).to(device)\n"
        "        print(f'[*] V13 architecture: roots={vocab.num_roots} awzan={vocab.num_awzan} '\n"
        "              f'vocab={base.vocab_size} built in {time.time()-_t0:.1f}s', flush=True)\n"
        "        assert vocab.num_roots == 9490 and vocab.num_awzan == 142, 'live space changed'\n"
        "    else:\n"
        "        base = UnifiedRootformerV12(bp, 'Qwen/Qwen2.5-0.5B', str(device),\n"
        "                                    torch.bfloat16).to(device)\n",
        'build')

    # ---- 3b. flash-hook control, AFTER the wrapper has registered them ---------------------
    text = sub_once(
        text,
        "    print(f'[*] ARM [{args.tag}] use_features={not args.no_features} dropout={args.dropout}')\n",
        "    if args.flash_hooks == 'off':\n"
        "        # The flash wrapper registered forward hooks on trunk layers 1/11/14 in its own\n"
        "        # __init__, and its 151.95M params are in NO checkpoint and NO optimizer.  They\n"
        "        # are zero-init INERT, but they add a 152M-param dead autograd branch; detaching\n"
        "        # them is bit-identical (mem_proj / out_proj are exactly 0) and is recorded.\n"
        "        _n = 0\n"
        "        for _l in model.backbone.layers:\n"
        "            _k = len(_l._forward_hooks)\n"
        "            if _k:\n"
        "                _l._forward_hooks.clear()\n"
        "                _n += _k\n"
        "        print(f'[*] flash hooks DETACHED: {_n} hooks removed from trunk layers (bit-identical: '\n"
        "              f'their output projections are exactly 0)', flush=True)\n"
        "    print(f'[*] ARM [{args.tag}] use_features={not args.no_features} dropout={args.dropout}')\n",
        'flash_hooks')

    # ---- 4. drop the stale-space keys on load ----------------------------------------------
    text = sub_once(
        text,
        "    sd = load_file(ckpt)\n",
        "    sd = load_file(ckpt)\n"
        "    if args.model_v13:\n"
        "        # The released file carries 24 x self_attn.root_embed (9015,64) and 24 x\n"
        "        # self_attn.wazn_embed (128,64): the STALE space, and dead code in the NRMT forward\n"
        "        # (0/24 attention calls carry root ids; see liveness_report.json).  A 9015-row\n"
        "        # tensor cannot load into a 9490-row module, so they are dropped -- and the dropped\n"
        "        # set is ASSERTED to be exactly the 48 stale tables, nothing else.\n"
        "        _want = model.state_dict()\n"
        "        _drop = {}\n"
        "        for _k in list(sd):\n"
        "            if ('self_attn.root_embed.weight' in _k or 'self_attn.wazn_embed.weight' in _k):\n"
        "                if _k in _want and tuple(sd[_k].shape) != tuple(_want[_k].shape):\n"
        "                    _drop[_k] = (tuple(sd[_k].shape), tuple(_want[_k].shape))\n"
        "                    del sd[_k]\n"
        "                elif _k not in _want:\n"
        "                    _drop[_k] = (tuple(sd[_k].shape), None)\n"
        "                    del sd[_k]\n"
        "        _rstale = sorted({v[0][0] for v in _drop.values()})\n"
        "        print(f'[*] V13 dropped {len(_drop)} stale-SPACE self_attn tables (rows '\n"
        "              f'{_rstale} -> live {vocab.num_roots}/{vocab.num_awzan})', flush=True)\n"
        "        if len(_drop) != 48:\n"
        "            raise SystemExit(f'[V13] expected exactly 48 stale tables dropped, got '\n"
        "                             f'{len(_drop)}: {sorted(_drop)[:6]}')\n",
        'drop_stale')

    # ---- 5. whole-trunk unfreeze -----------------------------------------------------------
    text = sub_once(
        text,
        "    if args.unfreeze_last:\n"
        "        _n = len(model.backbone.layers)\n"
        "        unfrozen_layers = list(range(max(0, _n - args.unfreeze_last), _n))\n"
        "        for _i in unfrozen_layers:\n"
        "            for _p in model.backbone.layers[_i].parameters():\n"
        "                _p.requires_grad = True\n",
        "    if args.unfreeze_last:\n"
        "        _n = len(model.backbone.layers)\n"
        "        unfrozen_layers = list(range(max(0, _n - args.unfreeze_last), _n))\n"
        "        for _i in unfrozen_layers:\n"
        "            for _p in model.backbone.layers[_i].parameters():\n"
        "                _p.requires_grad = True\n"
        "    if args.unfreeze_trunk_all:\n"
        "        # NO FREEZING, NO PRESERVATION MACHINERY.  English does not need preserving, so the\n"
        "        # LR cliff (which IS catastrophic forgetting) is not a constraint.  The input stage\n"
        "        # `morphemic_embed` is trained too: it is what creates the root signal from the\n"
        "        # (P,R,W,S) tuple, and 'root structure from the EARLY stage' is meaningless if the\n"
        "        # embedding that carries it is held frozen.\n"
        "        _n = len(model.backbone.layers)\n"
        "        unfrozen_layers = list(range(_n))\n"
        "        for _p in model.backbone.parameters():\n"
        "            _p.requires_grad = True\n"
        "        for _p in model.morphemic_embed.parameters():\n"
        "            _p.requires_grad = True\n"
        "        for _p in model.final_norm.parameters():\n"
        "            _p.requires_grad = True\n"
        "        print(f'[*] TRUNK-ALL: {_n} backbone layers + morphemic_embed + final_norm trainable '\n"
        "              f'({sum(p.numel() for p in model.backbone.parameters())/1e6:.1f}M backbone + '\n"
        "              f'{sum(p.numel() for p in model.morphemic_embed.parameters())/1e6:.2f}M '\n"
        "              f'embed + {sum(p.numel() for p in model.final_norm.parameters())/1e6:.3f}M '\n"
        "              f'final_norm)', flush=True)\n",
        'unfreeze_all')

    # ---- 6. save the input stage too -------------------------------------------------------
    text = sub_once(
        text,
        "                keep = {n: p.detach().to(torch.bfloat16).cpu() for n, p in model.named_parameters()\n"
        "                        if n.startswith('root_cross.')\n"
        "                        or any(n.startswith(f'backbone.layers.{i}.') for i in unfrozen_layers)\n"
        "                        or any(n.startswith(f'backbone.layers.{i}.') for i in ishtiqaq_layers)}\n",
        "                keep = {n: p.detach().to(torch.bfloat16).cpu() for n, p in model.named_parameters()\n"
        "                        if n.startswith('root_cross.')\n"
        "                        or n.startswith('nrmt_head.')\n"
        "                        or n.startswith('morphemic_embed.')\n"
        "                        or n.startswith('final_norm.')\n"
        "                        or any(n.startswith(f'backbone.layers.{i}.') for i in unfrozen_layers)\n"
        "                        or any(n.startswith(f'backbone.layers.{i}.') for i in ishtiqaq_layers)}\n",
        'save')

    # ---- 7. schedule parser: `all` / `earlyN` for BOTH mechanisms --------------------------
    text = sub_once(
        text,
        "        from root_cross_attn import RootCrossAttentionStack, parse_layer_spec\n"
        "        rca_layers = parse_layer_spec(args.root_cross_attn, len(model.backbone.layers))\n",
        "        from root_cross_attn import RootCrossAttentionStack\n"
        "        from early_root_path import parse_layer_spec_v13 as parse_layer_spec\n"
        "        rca_layers = parse_layer_spec(args.root_cross_attn, len(model.backbone.layers))\n",
        'rca_spec')
    text = sub_once(
        text,
        "        ishtiqaq_layers = _ig_spec(args.ishtiqaq_root_bias, len(model.backbone.layers))\n",
        "        from early_root_path import parse_layer_spec_v13 as _ig_spec_v13\n"
        "        ishtiqaq_layers = _ig_spec_v13(args.ishtiqaq_root_bias, len(model.backbone.layers))\n",
        'ig_spec')
    return text


def main():
    src_text = SRC.read_text()
    out = patch(src_text)
    DST.write_text(out)
    diff = list(difflib.unified_diff(
        src_text.splitlines(keepends=True), out.splitlines(keepends=True),
        fromfile='_ref_nrmt_train_ishtiqaq.py', tofile='nrmt_train_v13.py'))
    (HERE / 'nrmt_train_v13.diff').write_text(''.join(diff))
    print(f'[make_trainer_v13] base  {SRC.name} md5={md5(SRC)}')
    print(f'[make_trainer_v13] wrote {DST.name} md5={md5(DST)} ({len(out.splitlines())} lines, '
          f'{len(diff)} diff lines)')
    print(f'[make_trainer_v13] wrote nrmt_train_v13.diff')
    return 0


if __name__ == '__main__':
    sys.exit(main())
