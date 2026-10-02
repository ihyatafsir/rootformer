#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extend_root_checkpoint.py -- widen an NRMT/backbone checkpoint's ROOT-ID-KEYED tables to a
target root count, APPEND-ONLY and RE-RUNNABLE.

Precedent: `extend_awzan_checkpoint.py` (wazn 130 -> 142, 2026-10-01).

WHY
---
`nrmp_vocab.py` builds its root id space by concatenation, never by sorting:

    roots_list = special_roots + particle_roots + raw_roots + appended_particle_roots

so appending surfaces cannot move an existing root id.  The 2026-10-01 round-trip fix appended
199 `<P:..>` closed-class surfaces: `num_roots` went 9114 -> 9313.  PROOF that this was strictly
append-only: the pre-fix backup's 9114-entry roots_list equals the new `roots_list[:9114]`
exactly (checked in this session against
`nrmp_vocab.py.bak.rtfix-20261001-191450`).

A checkpoint whose root tables are still 9114 rows RAISES inside `load_state_dict`;
`strict=False` does not help because a size mismatch is not a missing/unexpected key:

    RuntimeError: Error(s) in loading state_dict for RootformerNRMT:
        size mismatch for morphemic_embed.root_embed.weight:
          copying a param with shape torch.Size([9114, 448]) from checkpoint,
          the shape in current model is torch.Size([9313, 448]).

The root inventory was ALSO incomplete, and that fix landed while this script was being written:
177 genuine roots (mostly triliterals beginning with ya') were appended on 2026-10-01 18:07 UTC,
taking `num_roots` 9114 -> 9313 -> 9490.  This script is therefore PARAMETERISED by
`--target-roots`: re-run it for any count.  Verified append-only across both moves:
`roots_list[:9114]` is unchanged and `roots_list[:9313]` is unchanged.
Every pass keeps rows `0..old-1` BIT-IDENTICAL to the checkpoint it was given, so the composition
of passes keeps rows 0..9113 bit-identical to the ORIGINAL checkpoint.  `--against REF` asserts
that directly against an arbitrary older checkpoint.

WHAT IT CHANGES (and nothing else)
----------------------------------
Every ROOT-ID-KEYED tensor -- a tensor whose name mentions a root AND whose leading dimension
equals the source root count (see `ROOT_NAME_HINT`; the dimension test is what makes the
detection safe) -- is grown by APPENDING `target - old` rows:

    morphemic_embed.root_embed.weight   [9114, 448] -> [9313, 448]
    nrmp_head.root_head.weight          [9114, 896] -> [9313, 896]

Rows `0..old-1` are copied VERBATIM; the appended rows get the documented init below.  Every
other tensor in the checkpoint is copied verbatim (bit-identical).

The `nrmp_head.root_head.weight` table matters as much as the embedding: `nrmt_train.py`
warm-starts the new head with `remap_legacy_head()`, which copies `nrmp_head.root_head.weight`
onto `nrmt_head.root_head.weight` only when the shapes match.  If it is left at 9114 the remap
reports it as a shape mismatch and the NRMT root readout silently stays at its random init.
This is exactly the shape of the awzan precedent, which extended BOTH `wazn_embed` and
`wazn_head`.

THE SAME ROOT EXPANSION ALSO BREAKS THE TRANSMUTER CHECKPOINT
-------------------------------------------------------------
`rootformer_v19_2_synthesis_transmuter_master.safetensors` carries another root-keyed table in
the same 9114-row root-id space:

    root_gate_embed.weight  [9114, 512]     (NeuralFarahidianTransmuterHead, `neural_transmuter_head.py:101`)

Both in-repo constructors pass `num_roots=vocab.num_roots` (9490 as of 2026-10-01 18:07 UTC), so
loading that checkpoint with `strict=False` RAISES:

    size mismatch for root_gate_embed.weight: copying a param with shape torch.Size([9114, 512])
    from checkpoint, the shape in current model is torch.Size([9490, 512])

This script handles it too (it is a root-named table with dim0 == the source root count), e.g.

    ... --src checkpoints/rootformer_v19_2_synthesis_transmuter_master.safetensors \
        --target-roots 9313
    # -> ...roots9313.safetensors, then point transmute_quickstart.py at it

NOT CHANGED (reported, deliberately)
------------------------------------
The 24 `backbone.layers.*.self_attn.root_embed.weight` tensors are `[9015, 64]` -- a THIRD,
older root-id space, hardcoded as `num_roots=9015` at
`models/unified_rootformer_v12.py:80` and `:91`.  `IshtiqaqAttentionV12` reads it under
`torch.clamp(root_ids, 0, num_embeddings - 1)` (`models/ishtiqaq_attention_v12.py:169` and
`:208`), so every root id >= 9015 ALREADY collapses onto row 9014 -- true before this change
(ids 9015..9113) and more so after (ids 9015..9312).  These are frozen backbone parameters in a
different id space; resizing them is a design decision, not a shape fix, so the script reports
them and leaves them alone.

INITIALISATION OF THE APPENDED ROWS
-----------------------------------
`--init mean+noise` (DEFAULT):  row ~ N(mu_d, (noise_frac * sigma_d)^2) per dimension d, where
mu_d / sigma_d are the per-dimension mean / std of the EXISTING rows -- i.e. new rows are drawn
from a diagonal-Gaussian fit to the trained table.

MEASURED on the 9114-row source (why not zeros, and why `noise_frac` defaults to 1.0):

    tensor                            ||mu||   typical ||row||   ||mu||/typical
    morphemic_embed.root_embed.weight  0.216        21.147          0.010
    nrmp_head.root_head.weight         1.143         1.262          0.906

  * `--init zeros` (the awzan precedent) puts every new row at the ORIGIN of the trained space.
    For `root_embed` that is ~21 units away from every trained row -- a degenerate input the
    backbone never saw -- and all 199 new roots start indistinguishable from one another.
  * "mean plus a LITTLE noise" taken literally is equally degenerate for `root_embed`, because
    its per-dimension mean is ~0 (`||mu||` is 1% of a typical row norm): `noise_frac=0.05` would
    give rows of norm ~1.06, ~20x too small.  A `mean+small-noise` row is a zero row for all
    practical purposes.
  * The diagonal-Gaussian fit (noise_frac=1.0) lands each table at its own trained scale:
    E||row|| = 21.159 vs 21.147 measured for `root_embed`, and 1.277 vs 1.262 for `root_head`
    (both ~100% of typical).  For `root_embed` the fit is essentially the table's own init
    distribution (per-dim sigma = 0.9996, mu = 0.008), so new rows are statistically
    indistinguishable from trained rows rather than special.  Independent draws per row make the
    199 rows mutually distinguishable, so each receives its own gradient.

`--init small-random` uses N(0, 0.02) (the IshtiqaqAttention init convention) and is provided
for control runs only; it is at the wrong scale for `root_embed`.

THE APPENDED ROWS ARE UNTRAINED BY CONSTRUCTION.  Extending the checkpoint only makes them
LOADABLE; it does not make them predictive.  Until they are trained the model cannot predict the
appended surfaces, and it cannot even train them from the shipped token cache, because
`/workspace/nrmp_cache/{train,val}.pt` was built on 2026-09-30 with the 9114-root vocab (stream
`r` max id = 9113) and therefore contains NO occurrence of any newer root id.  The cache must be
rebuilt from the current `nrmp_vocab.py` before the appended rows receive any gradient at all.

TOKEN TABLES
------------
`--append NAME=NEWWIDTH` extends any other append-only table by the same rule.  This is needed
when the BLUEPRINT grows at the tail (a new partition after `simd_padding`), which makes the
model's `backbone.embed_tokens.weight` wider than the checkpoint's:

    backbone.embed_tokens.weight  [9868, 896] -> [10052, 896]    (blueprint 9868 -> 10052)

NOTE the difference from `extend_awzan_checkpoint.py`: there the blueprint partition grew in the
MIDDLE and rows had to be INSERTED (shifting the tail).  Here the new
`classical_roots_lisan_extension` partition is at the END, so a pure append is correct.  Always
check WHERE the new partition sits before choosing.

USAGE
-----
  # PASS 1 -- 2026-10-01, round-trip fix appended 199 <P:..> surfaces (9114 -> 9313)
  /workspace/venvs/rootformer/bin/python extend_root_checkpoint.py \
      --src checkpoints/rootformer_v19_2_synthesis_ar_backbone.awzan142.safetensors \
      --dst checkpoints/rootformer_v19_2_synthesis_ar_backbone.awzan142.roots9313.safetensors \
      --target-roots 9313 \
      --against checkpoints/rootformer_v19_2_synthesis_ar_backbone.awzan142.safetensors

  # PASS 2 -- the lisan root fix appended 177 genuine roots (9313 -> 9490).  A one-liner.
  ... --src ...awzan142.roots9313.safetensors \
      --dst ...awzan142.roots9490.safetensors --target-roots 9490 --against <the ORIGINAL>

  # PASS 3 -- the same fix appended a blueprint tail partition (9868 -> 10052 tokens)
  ... --src ...roots9490.safetensors --dst ...roots9490.tok10052.safetensors \
      --target-roots 9490 --append backbone.embed_tokens.weight=10052 --against <the ORIGINAL>

`--against <ORIGINAL>` on every pass directly asserts that rows 0..9113 are still bit-identical
to the original 9114-row checkpoint.
"""
import argparse
import hashlib
import json
import sys
import time
import zlib
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

ROOT_DIR = Path(__file__).resolve().parent

#: A tensor is a CANDIDATE root-id-keyed table iff its name mentions a root and its leading
#: dimension equals the source root count.  The dim0 test is what keeps this safe: tables in a
#: DIFFERENT root-id space (the frozen backbone's hardcoded 9015-row ones) or in a different
#: space entirely (root_k_proj 128, root_q_proj 896) can never be resized by accident.
#:
#: This catches, without hardcoding names:
#:   morphemic_embed.root_embed.weight   [9114, 448]  NRMT input embedding
#:   nrmp_head.root_head.weight          [9114, 896]  NRMT/legacy root readout
#:   root_gate_embed.weight              [9114, 512]  transmuter root gate (see --help)
#: Pass --tensor NAME to force an extra table that does not mention "root".
ROOT_NAME_HINT = 'root'
DEFAULT_SEED = 20261001
INITS = ('mean+noise', 'zeros', 'small-random')


def md5_file(path, chunk=1 << 22):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def detect_old_roots(sd, override=None):
    if override is not None:
        return int(override)
    t = sd.get('morphemic_embed.root_embed.weight')
    if t is not None:
        return int(t.shape[0])
    widths = sorted({int(v.shape[0]) for k, v in sd.items()
                     if ROOT_NAME_HINT in k.lower() and v.dim() >= 1})
    if len(widths) != 1:
        raise SystemExit(f'cannot infer the source root count (candidates {widths}); '
                         f'pass --old-roots')
    return widths[0]


def all_root_named(sd):
    """Every root-id-keyed CANDIDATE table in the checkpoint (any leading dimension)."""
    return sorted(k for k, v in sd.items()
                  if ROOT_NAME_HINT in k.lower() and v.dim() >= 1)


def root_keyed(sd, old_n, extra=()):
    """Tables that live in the source root-id space and must therefore be extended."""
    names = set(all_root_named(sd)) | {n for n in extra if n in sd}
    missing = [n for n in extra if n not in sd]
    if missing:
        raise SystemExit(f'--tensor {missing} not present in the checkpoint')
    return sorted(n for n in names if int(sd[n].shape[0]) == old_n)


def _generalise(name):
    """backbone.layers.17.self_attn.root_embed.weight -> backbone.layers.N.self_attn.…"""
    out = []
    for part in name.split('.'):
        out.append('N' if part.isdigit() else part)
    return '.'.join(out)


def make_rows(t, n_new, init, noise_frac, seed):
    """Deterministic appended rows for tensor `t`.  Stateless: same args -> same rows."""
    shape = (n_new,) + tuple(t.shape[1:])
    if init == 'zeros':
        return torch.zeros(shape, dtype=t.dtype)
    gen = torch.Generator().manual_seed(int(seed))
    if init == 'small-random':
        return (torch.randn(shape, generator=gen, dtype=torch.float32) * 0.02).to(t.dtype)
    base = t.detach().to(torch.float32)
    mu = base.mean(0, keepdim=True)
    sigma = base.std(0, keepdim=True)
    noise = torch.randn(shape, generator=gen, dtype=torch.float32)
    return (mu + float(noise_frac) * sigma * noise).to(t.dtype)


def tensor_seed(seed, name):
    """Per-tensor, order-independent seed (so adding/removing a table cannot reshuffle others).

    Uses crc32, NOT the builtin `hash()`: str hashing is salted per process (PYTHONHASHSEED),
    which would make the appended rows non-reproducible across runs.
    """
    return (int(seed) + zlib.crc32(name.encode('utf-8')) * 7919) % (2 ** 31 - 1)


def transform(sd, plan, init, noise_frac, seed, report):
    """plan: {tensor_name: (old_width, new_width)}.  Append-only by construction."""
    out = dict(sd)
    for k in sorted(plan):
        old_n, new_n = plan[k]
        t = sd[k]
        if int(t.shape[0]) != old_n:
            raise SystemExit(f'{k}: shape {tuple(t.shape)} does not have the planned '
                             f'{old_n} rows')
        rows = make_rows(t, new_n - old_n, init, noise_frac, tensor_seed(seed, k))
        out[k] = torch.cat([t, rows], dim=0).contiguous()
        report.append(
            f'APPEND  {k}: {tuple(t.shape)} -> {tuple(out[k].shape)}  '
            f'(rows 0..{old_n - 1} verbatim, {old_n}..{new_n - 1} {init})')
    return out


def verify(src_path, dst_path, plan, init, noise_frac, seed, against=None):
    print(f'\n[*] verifying {Path(dst_path).name} against {Path(src_path).name} ...')
    src = load_file(str(src_path))
    dst = load_file(str(dst_path))
    bad = []
    if set(src) != set(dst):
        bad.append('tensor name set changed')

    for k in sorted(plan):
        old_n, new_n = plan[k]
        n_new = new_n - old_n
        a, b = src[k], dst[k]
        want = (new_n,) + tuple(a.shape[1:])
        if tuple(b.shape) != want:
            bad.append(f'{k}: shape {tuple(a.shape)} -> {tuple(b.shape)} != {want}')
            continue
        if not torch.equal(b[:old_n], a):
            bad.append(f'{k}: rows 0..{old_n - 1} are NOT bit-identical')
            continue
        expect = make_rows(a, n_new, init, noise_frac, tensor_seed(seed, k))
        if not torch.equal(b[old_n:], expect):
            bad.append(f'{k}: appended rows differ from the deterministic {init} init')
            continue
        new_rows = b[old_n:]
        n_zero = int((new_rows.abs().sum(-1) == 0).sum())
        distinct = int(torch.unique(new_rows, dim=0).shape[0])
        print(f'    OK  {k}: {tuple(a.shape)} -> {tuple(b.shape)}  '
              f'rows 0..{old_n - 1} bit-identical; {n_new} appended rows == deterministic '
              f'{init}(seed={seed}); {distinct}/{n_new} distinct, {n_zero} all-zero; '
              f'finite={bool(torch.isfinite(new_rows).all())}')

    # every other tensor verbatim
    n_other = 0
    for k in src:
        if k in plan:
            continue
        if not torch.equal(src[k], dst[k]):
            bad.append(f'{k}: unrelated tensor changed')
        else:
            n_other += 1

    if against is not None:
        ref = load_file(str(against))
        print(f'    --against {Path(against).name}:')
        for k in sorted(plan):
            old_n = plan[k][0]
            if k not in ref:
                bad.append(f'{k}: absent from --against reference')
                continue
            n = min(old_n, int(ref[k].shape[0]))
            if not torch.equal(dst[k][:n], ref[k][:n]):
                bad.append(f'{k}: rows 0..{n - 1} differ from --against '
                           f'{Path(against).name}')
            else:
                print(f'      OK  {k}: rows 0..{n - 1} bit-identical to the reference '
                      f'({int(ref[k].shape[0])} rows)')

    if bad:
        print('\n[!] VERIFICATION FAILED:')
        for b_ in bad:
            print('    ', b_)
        return False
    print(f'    OK  all {len(src)} tensors accounted for; {n_other} unrelated tensors '
          f'bit-identical')
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', required=True, help='source checkpoint (never modified)')
    ap.add_argument('--dst', default=None,
                    help='output checkpoint; default <src stem>.roots<target>.safetensors')
    ap.add_argument('--target-roots', type=int, required=True,
                    help='root count to grow the root tables to (e.g. 9313, then 9490)')
    ap.add_argument('--old-roots', type=int, default=None,
                    help='source root count; default: inferred from the checkpoint')
    ap.add_argument('--init', choices=INITS, default='mean+noise')
    ap.add_argument('--noise-frac', type=float, default=1.0,
                    help='mean+noise only: noise std as a fraction of each dimension\'s std '
                         '(1.0 = diagonal-Gaussian fit to the trained rows)')
    ap.add_argument('--seed', type=int, default=DEFAULT_SEED)
    ap.add_argument('--tensor', action='append', default=[],
                    help='force an extra tensor name to extend (repeatable); for root-keyed '
                         'tables whose name does not mention "root"')
    ap.add_argument('--append', action='append', default=[], metavar='NAME=NEWWIDTH',
                    help='append-only extension of any other table, e.g. '
                         'backbone.embed_tokens.weight=10052 (repeatable)')
    ap.add_argument('--dry-run', action='store_true',
                    help='print the plan and exit without writing anything')
    ap.add_argument('--against', default=None,
                    help='optional older checkpoint: also assert rows 0..old-1 match it')
    ap.add_argument('--force', action='store_true', help='allow overwriting an existing --dst')
    ap.add_argument('--no-verify', action='store_true')
    args = ap.parse_args()

    src = Path(args.src)
    if not src.exists():
        raise SystemExit(f'missing source checkpoint {src}')
    dst = Path(args.dst) if args.dst else src.with_name(
        f'{src.stem}.roots{args.target_roots}.safetensors')
    if dst.resolve() == src.resolve():
        raise SystemExit('refusing to overwrite the source checkpoint in place; pass --dst')
    if dst.exists() and not args.force and not args.dry_run:
        raise SystemExit(f'{dst} already exists; pass --force to overwrite (never clobber)')

    src_md5 = md5_file(src)
    print(f'[*] src {src}')
    print(f'    md5 {src_md5}  ({src.stat().st_size} bytes)')
    print(f'[*] loading ...', flush=True)
    sd = load_file(str(src))
    old_n = detect_old_roots(sd, args.old_roots)
    print(f'    {len(sd)} tensors; source root count = {old_n}')

    # full inventory of every root-named tensor, and why each is or is not extended
    keys = root_keyed(sd, old_n, args.tensor)
    print(f'[*] root-keyed tensor inventory (name mentions "root"); source root count {old_n}:')
    groups = {}
    for k in all_root_named(sd):
        groups.setdefault((_generalise(k), tuple(sd[k].shape)), []).append(k)
    for (gen, shape), names in sorted(groups.items()):
        suffix = f'  x{len(names)}' if len(names) > 1 else ''
        if names[0] in keys and args.target_roots != old_n:
            print(f'      EXTEND  {gen:<52} {shape}{suffix}')
        elif names[0] in keys:
            print(f'      leave   {gen:<52} {shape}{suffix}  '
                  f'(already at the target root count {args.target_roots}; use --append to '
                  f'extend it explicitly)')
        else:
            print(f'      leave   {gen:<52} {shape}{suffix}  '
                  f'(rows {shape[0]} != {old_n}: different root-id space)')
    if not keys and args.target_roots != old_n:
        raise SystemExit(f'no root-id-keyed tensor with {old_n} rows found; '
                         f'root-named tensors are {all_root_named(sd)}')

    # ---- build the append-only plan ------------------------------------------------------
    if args.target_roots < old_n and keys:
        raise SystemExit(f'--target-roots {args.target_roots} is SMALLER than the source '
                         f'{old_n}; this script is append-only and will not truncate')
    plan = {k: (old_n, args.target_roots) for k in keys if args.target_roots != old_n}
    for spec in args.append:
        name, sep, neww = spec.rpartition('=')
        if not sep or not name:
            raise SystemExit(f'--append expects NAME=NEWWIDTH, got {spec!r}')
        if name not in sd:
            raise SystemExit(f'--append {name}: not present in the checkpoint')
        o, n = int(sd[name].shape[0]), int(neww)
        if n < o:
            raise SystemExit(f'--append {name}: refusing to shrink {o} -> {n} '
                             f'(this script is append-only)')
        if n == o:
            print(f'[=] --append {name}: already {o} rows, nothing to do')
            continue
        plan[name] = (o, n)
        print(f'      APPEND  {_generalise(name):<52} {tuple(sd[name].shape)} -> '
              f'{n} rows  (--append)')

    if not plan:
        print(f'[=] nothing to extend (source root count {old_n}, target '
              f'{args.target_roots})')
        return 0
    total = sum(n - o for o, n in plan.values())
    print(f'[*] plan: {len(plan)} tensor(s), {total} appended rows total, '
          f'--init {args.init} (seed {args.seed}, noise_frac {args.noise_frac})')
    if args.dry_run:
        print('[=] --dry-run: nothing written')
        return 0

    report = []
    out = transform(sd, plan, args.init, args.noise_frac, args.seed, report)
    for line in report:
        print('    ' + line)
    print(f'[*] writing {dst} ...', flush=True)
    save_file(out, str(dst))
    dst_md5 = md5_file(dst)
    print(f'    wrote {dst} ({dst.stat().st_size} bytes)')
    print(f'    md5 {dst_md5}')

    ok = True
    if not args.no_verify:
        ok = verify(src, dst, plan, args.init, args.noise_frac, args.seed, args.against)
        if not ok:
            print(f'[!] leaving the bad output in place for inspection: {dst}')

    manifest = {
        'script': Path(__file__).name,
        'src': str(src), 'src_md5': src_md5,
        'dst': str(dst), 'dst_md5': dst_md5,
        'old_roots': old_n, 'target_roots': args.target_roots,
        'appended': args.target_roots - old_n,
        'plan': {k: list(v) for k, v in plan.items()},
        'init': args.init, 'noise_frac': args.noise_frac, 'seed': args.seed,
        'extended_tensors': sorted(plan),
        'verified': bool(ok),
        'against': args.against,
        'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'note': ('appended rows are UNTRAINED by construction; they receive gradient only once '
                 'the token cache is rebuilt with the current nrmp_vocab root id space'),
    }
    mp = dst.with_suffix('.extend_root.json')
    mp.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'    manifest {mp}')

    if not ok:
        return 1
    print('\n[OK] root extension complete; source left untouched.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
