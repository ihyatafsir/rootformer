#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""patch_probe.py -- generate probe_width.py from the parent's read-only
/workspace/root_attn_ctl/probe_rca_dynamics.py (attention entropy / gate / residual diagnostics).

Three edits: import the WIDTH module, and rebuild the stack at the width recorded in the
checkpoint metadata (`rca_dim`).  Everything else -- the attention recorder, att_stats, the
fixed val windows, the zero-gate / live-gate h comparison -- is the parent's code verbatim.
"""
import difflib
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE / 'probe_rca_dynamics.py.orig'
DST = HERE / 'probe_width.py'

EDITS = [
    ("sys.path.insert(0, '/workspace/root_attn')\n",
     "sys.path.insert(0, '/workspace/root_attn_width')\n"),
    ("    import root_cross_attn as rca\n",
     "    import root_cross_attn_width as rca\n"),
    ("    from root_cross_attn import RootCrossAttentionStack\n",
     "    from root_cross_attn_width import RootCrossAttentionStack\n"),
    ("        num_heads=int(meta.get('rca_heads', 8)), dropout=0.0,\n"
     "        out_norm=bool(meta.get('rca_out_norm', False)), dtype=torch.float32).to(device)\n",
     "        num_heads=int(meta.get('rca_heads', 8)), dropout=0.0,\n"
     "        out_norm=bool(meta.get('rca_out_norm', False)), dtype=torch.float32,\n"
     "        d_attn=int(meta.get('rca_dim', model.d_model)) or None).to(device)\n"),
]


def main():
    src = SRC.read_text()
    out = src
    for i, (old, new) in enumerate(EDITS, 1):
        n = out.count(old)
        if n != 1:
            print(f'FATAL: edit {i} matched {n} times, expected 1')
            return 2
        out = out.replace(old, new)
    DST.write_text(out)
    (HERE / 'probe_width.diff').write_text(''.join(difflib.unified_diff(
        src.splitlines(True), out.splitlines(True), 'probe_rca_dynamics.py', 'probe_width.py')))
    print(f'wrote {DST.name}: {len(EDITS)} edits applied')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
