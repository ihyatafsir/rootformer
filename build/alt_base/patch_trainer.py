#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""patch_trainer.py -- generate nrmt_train_width.py from the SHIPPED, UNTOUCHED
/workspace/hf_v19_2_release/nrmt_train.py (md5 56ef3f2fd9ddc352dee596f32590e7b5, the parent's
recorded trainer that carries the live-evaluator bugfix).  The shipped file is never modified,
so the parent's chain and the two live arms keep reading exactly what they already loaded.

FIVE edits, all additive and all defaulted to the shipped behaviour:

  1. ROOT_DIR is pinned to /workspace/hf_v19_2_release (the copy lives elsewhere, but `nrmt_arch`,
     `nrmp_vocab`, `models/` must still resolve, and the checkpoint/cache args are relative to cwd).
  2. new flag `--rca-dim W` (0 = shipped = d_model).
  3. import the WIDTH module from /workspace/root_attn_width, not the shared /workspace/root_attn.
  4. thread `d_attn=args.rca_dim or None` into RootCrossAttentionStack.
  5. record `rca_dim` in the *.pt.trunk.pt metadata so the read-only dynamics probe can rebuild
     the module at the right width.

With `--rca-dim 0` (the default) the generated trainer is behaviourally identical to the shipped
one; equiv_check.py proves the MODULE is bit-identical and a 1-step smoke run proves the wiring.
"""
import difflib
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE / 'nrmt_train.py.orig'
DST = HERE / 'nrmt_train_width.py'

EDITS = [
    # ---- 1. pin ROOT_DIR -------------------------------------------------------------
    (
        "ROOT_DIR = Path(__file__).resolve().parent\n",
        "# WIDTH FORK: pinned (this copy does not live next to models/ and nrmt_arch.py)\n"
        "ROOT_DIR = Path('/workspace/hf_v19_2_release')\n",
    ),
    # ---- 2. new flag -----------------------------------------------------------------
    (
        "    ap.add_argument('--rca-heads', type=int, default=8, help='heads in the cross-attention')\n",
        "    ap.add_argument('--rca-heads', type=int, default=8, help='heads in the cross-attention')\n"
        "    ap.add_argument('--rca-dim', type=int, default=0,\n"
        "                    help='WIDTH LEVER: internal attention width d_attn of the cross-attention'\n"
        "                         ' (the space q/k/v and the attended context live in; per-head dim ='\n"
        "                         ' d_attn/heads).  0 = shipped = d_model.  --rca-dim 1792 doubles it'\n"
        "                         ' (heads 8x224).  NOTE: k_proj/v_proj read the FROZEN 448-dim'\n"
        "                         ' morphemic_embed.root_embed, so this raises CAPACITY, not the'\n"
        "                         ' information rate of the root channel (which is already 448/896)')\n",
    ),
    # ---- 3. import the width module --------------------------------------------------
    (
        "        if '/workspace/root_attn' not in sys.path:\n"
        "            sys.path.insert(0, '/workspace/root_attn')\n"
        "        from root_cross_attn import RootCrossAttentionStack, parse_layer_spec\n",
        "        if '/workspace/root_attn_width' not in sys.path:\n"
        "            sys.path.insert(0, '/workspace/root_attn_width')\n"
        "        from root_cross_attn_width import RootCrossAttentionStack, parse_layer_spec\n",
    ),
    # ---- 4. thread d_attn ------------------------------------------------------------
    (
        "            num_heads=args.rca_heads, dropout=args.rca_dropout, out_std=args.rca_out_std,\n"
        "            dtype=torch.float32, out_norm=bool(args.rca_out_norm)).to(device=device)\n",
        "            num_heads=args.rca_heads, dropout=args.rca_dropout, out_std=args.rca_out_std,\n"
        "            dtype=torch.float32, out_norm=bool(args.rca_out_norm),\n"
        "            d_attn=(int(args.rca_dim) or None)).to(device=device)\n",
    ),
    # ---- 5. record the width in the trunk payload metadata ---------------------------
    (
        "                            'rca_heads': int(args.rca_heads),\n",
        "                            'rca_heads': int(args.rca_heads),\n"
        "                            'rca_dim': int(args.rca_dim or model.d_model),\n",
    ),
]


def main():
    src = SRC.read_text()
    out = src
    for i, (old, new) in enumerate(EDITS, 1):
        n = out.count(old)
        if n != 1:
            print(f'FATAL: edit {i} matched {n} times, expected 1', file=sys.stderr)
            return 2
        out = out.replace(old, new)
    DST.write_text(out)
    diff = difflib.unified_diff(src.splitlines(True), out.splitlines(True),
                                'nrmt_train.py', 'nrmt_train_width.py')
    (HERE / 'nrmt_train_width.diff').write_text(''.join(diff))
    print(f'wrote {DST.name}: {len(src.splitlines())} -> {len(out.splitlines())} lines, '
          f'{len(EDITS)} edits applied')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
