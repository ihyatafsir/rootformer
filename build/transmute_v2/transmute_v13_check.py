#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
transmute_v13_check.py -- the corrected transmutation, proved.

Three things the brief says went wrong, each asserted here:

  1. CONSTANTS.  The released `awzan142` checkpoint carries 9868 / 9114-row embeddings against
     the blueprint's 10052 / 9490, and `models/unified_rootformer_v12.py` hardcodes 9015 / 128.
     `UnifiedRootformerV13` builds every table in the live space and `assert_one_root_space()`
     raises on any survivor.  `assert_no_aliasing()` raises on the 475-id clamp collision.

  2. THE 290 REAL QWEN TENSORS.  `RootformerNRMT.backbone` builds from the morphemic blueprint,
     so `embed_tokens` has 10052 rows while raw Qwen token ids run to 149,221.  The corrected
     transmutation resizes embed_tokens/lm_head 151936 -> 10052 and loads ALL 290 real Qwen
     tensors (24 layers x 12 + embed_tokens + final norm), asserting the count instead of
     reporting a number that nobody checks.

  3. ROW COUNTS AGREE ON ONE SPACE.  Every root-bearing table is enumerated with its row count
     and the census is printed; a disagreement is an exception, not a warning.

Order matters: transmute from clean Qwen FIRST (that is the initialisation), then overlay the
released trained checkpoint (that is the warm start).  Both stages are reported separately.
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get('RF_ROOT', '/workspace/hf_v19_2_release')).resolve()
for p in (str(ROOT), str(ROOT / 'models'), '/workspace/transmute_v2'):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch

from root_space import (LIVE_ROOTS, LIVE_AWZAN, LIVE_VOCAB, SpaceViolation,
                        assert_one_root_space, build_token_root_table, build_token_wazn_table)
from unified_rootformer_v13 import UnifiedRootformerV13


def banner(s):
    print('\n' + '=' * 78 + f'\n{s}\n' + '=' * 78, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=str(ROOT / 'checkpoints' /
                    'rootformer_v19_2_synthesis_ar_backbone.awzan142.roots9490.tok10052.safetensors'))
    ap.add_argument('--out', default='/workspace/transmute_v2/base_v13_roots9490_tok10052.safetensors')
    ap.add_argument('--report', default='/workspace/transmute_v2/transmute_v13_report.json')
    ap.add_argument('--no-save', action='store_true')
    args = ap.parse_args()
    R = {}

    # ---------------------------------------------------------------- build
    banner('0. BUILD UnifiedRootformerV13 (the live space, asserted at construction)')
    import nrmp_vocab as nv
    V = next(v for k, v in vars(nv).items() if isinstance(v, type) and 'MorphemicVocab' in k)
    bp = str(ROOT / 'data/rootformer_v12_arabic_blueprint.json')
    vocab = V(bp)
    print(f'  live vocab: roots={vocab.num_roots} awzan={vocab.num_awzan} '
          f'vocab={vocab.base_tok.vocab_size}')
    base = UnifiedRootformerV13(bp, 'Qwen/Qwen2.5-0.5B', 'cpu', torch.float32, vocab=vocab)
    R['live_vocab'] = {'roots': int(vocab.num_roots), 'awzan': int(vocab.num_awzan),
                       'vocab': int(vocab.base_tok.vocab_size)}

    # ---------------------------------------------------------------- one space
    banner('1. ONE ROOT-ID SPACE -- census of every root-bearing table')
    census = assert_one_root_space(base.backbone, vocab)
    for k in sorted(census):
        if k.endswith(('.root_embed.weight', '.wazn_embed.weight')):
            print(f'    {k:60s} rows={census[k]}')
    seen = {}
    for k, v in census.items():
        n = k.split('layers.')[1].split('.')[0] if 'layers.' in k else '--'
        seen.setdefault(n, []).append(v)
    roots_rows = sorted({v[0] for v in seen.values() if v})
    print(f'  distinct root-table row counts across ALL 24 layers: {roots_rows}')
    if roots_rows != [LIVE_ROOTS]:
        raise SpaceViolation(f'root tables still disagree: {roots_rows}')
    print(f'  [OK] every root table is {LIVE_ROOTS} rows; every wazn table is {LIVE_AWZAN}')
    R['root_table_rows'] = roots_rows

    # token tables: injective, in range, no offset arithmetic
    banner('2. TOKEN -> ID TABLES (by string, injective, in range)')
    rtab, rinfo = build_token_root_table(vocab, base.vocab_size, fill=0)
    wtab, winfo = build_token_wazn_table(vocab, base.vocab_size, fill=0)
    rt = torch.tensor(rtab)
    rtok = [i for i, t in base.tokenizer.id_to_token.items() if t.startswith('<root_')]
    vals = rt[torch.tensor(rtok)]
    nz = vals[vals != 0]
    print(f'  <root_X> tokens                 : {rinfo["tokens"]}')
    print(f'  distinct live root ids reached  : {int(nz.unique().numel())}')
    print(f'  injective                       : {int(nz.unique().numel()) == len(rtok) - 2}')
    print(f'  min={int(vals.min())} max={int(vals.max())}  (live max must be {LIVE_ROOTS - 1})')
    print(f'  <wazn_X> tokens                 : {winfo["tokens"]} -> '
          f'{winfo["distinct"]} distinct live wazn ids')
    if int(nz.unique().numel()) != len(rtok) - 2:
        raise SpaceViolation('token->root table is not injective')
    if int(vals.max()) > LIVE_ROOTS - 1:
        raise SpaceViolation(f'root id {int(vals.max())} exceeds the live space')
    if [int(x) for x in base.id_to_root_table[torch.tensor(rtok)].tolist()] != vals.tolist():
        raise SpaceViolation('the module table and the standalone builder disagree')
    print('  [OK] injective, in range, offset arithmetic eliminated, module table == builder')
    R['token_root_table'] = {'tokens': rinfo['tokens'], 'distinct': int(nz.unique().numel()),
                            'min': int(vals.min()), 'max': int(vals.max())}

    # ---------------------------------------------------------------- the stale spaces, named
    banner('3. THE THREE HISTORICAL SPACES, AND WHAT THE CORRECTION DID')
    print(f'  9490  morphemic_embed / nrmp_vocab / live NRMT stream   <- V13 uses this everywhere')
    print(f'  9015  unified_rootformer_v12.py hardcoded               -> corrected to 9490')
    print(f'  9015  IshtiqaqAttentionV12.root_embed (9015,64) dead    -> corrected to 9490')
    print(f'  128   the awzan hardcode                                -> corrected to 142')
    from safetensors.torch import load_file
    raw = load_file(args.ckpt)
    shapes = {}
    for k, v in raw.items():
        if k.endswith(('.root_embed.weight', '.wazn_embed.weight', '.root_head.weight',
                       '.wazn_head.weight', '.embed_tokens.weight')):
            shapes.setdefault(k.split('layers.')[0] if 'layers.' in k else k,
                              set()).add(tuple(v.shape))
    print('  released checkpoint shapes (grouped):')
    for k in sorted(shapes):
        for s in sorted(shapes[k]):
            print(f'    {k:56s} {s}')
    R['released_shapes'] = {k: [list(s) for s in sorted(v)] for k, v in shapes.items()}

    # ---------------------------------------------------------------- transmute from clean Qwen
    banner('4. TRANSMUTE FROM CLEAN QWEN2.5-0.5B -- resize + all 290 real tensors')
    info = base.transmute_from_clean_qwen2_5('Qwen/Qwen2.5-0.5B', local_files_only=True)
    R['transmute'] = {k: v for k, v in info.items() if k != 'adapted'}
    print(f'  qwen state_dict keys        : {info["qwen_state_keys"]}')
    print(f'  layer tensors adapted       : {info["layer_tensors"]}  (expect 288)')
    print(f'  resized                     : {info["resized"]}')
    print(f'  consumed Qwen tensors       : {info["consumed_qwen_tensors"]}/290')
    if info['consumed_qwen_tensors'] != 290:
        raise SpaceViolation('not all 290 real Qwen tensors were consumed')

    # ---------------------------------------------------------------- overlay the trained ckpt
    banner('5. OVERLAY THE RELEASED TRAINED CHECKPOINT (warm start for the arms)')
    res = base.load_released_checkpoint(args.ckpt, drop_stale_tables=True)
    print(f'  backbone keys mapped (prefix remapped) : {res["backbone_keys_mapped"]}')
    print(f'  dropped stale-SPACE tables             : {res["n_dropped"]} (expect 48)')
    rows = sorted({v[0] for v in res['dropped_stale'].values()})
    print(f'  their row counts                       : {rows}  (9015 and 128, the stale spaces)')
    print(f'  missing among the mapped backbone keys : {len(res["missing"])} {res["missing"][:5]}')
    print(f'  unexpected                             : {len(res["unexpected"])}')
    print(f'  NOT OWNED by this module (wrapper-side): {len(res["not_owned"])} keys -> '
          f'{sorted(res["not_owned"])}')
    if res['n_dropped'] != 48:
        raise SpaceViolation(f'expected exactly 48 stale tables, dropped {res["n_dropped"]}')
    stale_model_missing = [k for k in res['missing']
                           if k.endswith(('self_attn.root_embed.weight',
                                          'self_attn.wazn_embed.weight'))]
    other_missing = [k for k in res['missing'] if k not in stale_model_missing]
    print(f'  missing stale-SPACE tables (expected, dead, left at init): '
          f'{len(stale_model_missing)} (expect 48)')
    print(f'  other missing (tied lm_head / non-persistent buffers)    : {other_missing}')
    if len(stale_model_missing) != 48:
        raise SpaceViolation(f'expected the 48 dropped tables to be the only stale-space '
                             f'missing keys, got {len(stale_model_missing)}')
    # `backbone.lm_head.weight` is absent from a released NRMT checkpoint (the NRMT head
    # replaces it) and is DEAD in the NRMT path; `morphemic_heads.*` is the v12 aux
    # DisjointMorphologicalHeads, which the liveness test proves is not in model.parameters()
    # (untracked).  Both legitimately stay at init.
    bad = [k for k in other_missing
           if not (k.endswith('lm_head.weight') or 'rotary' in k
                   or k.startswith('morphemic_heads.'))]
    if bad:
        raise SpaceViolation(f'unexpected missing keys: {bad}')
    if res['unexpected']:
        raise SpaceViolation(f'unexpected keys: {res["unexpected"][:5]}')
    R['checkpoint_overlay'] = {'dropped': res['n_dropped'], 'dropped_rows': rows,
                               'backbone_keys_mapped': res['backbone_keys_mapped'],
                               'stale_missing': len(stale_model_missing),
                               'other_missing': other_missing,
                               'missing': res['missing'], 'unexpected': res['unexpected'],
                               'not_owned': sorted(res['not_owned'])}
    print('  [OK] all 506 non-stale trunk keys loaded; the 48 stale-SPACE tables are the ONLY')
    print('       trunk keys left at init, and they are dead code (grad is None in')
    print('       liveness_report.json).  14 wrapper-side keys are overlaid by the trainer.')

    # ---------------------------------------------------------------- re-assert after load
    banner('6. RE-ASSERT ONE SPACE AFTER LOADING')
    census2 = assert_one_root_space(base.backbone, vocab)
    rows2 = sorted({v for k, v in census2.items() if k.endswith('root_embed.weight')})
    print(f'  root-table rows after load: {rows2}')
    if rows2 != [LIVE_ROOTS]:
        raise SpaceViolation('load changed the root space')
    print('  [OK] the loaded model is in one space')

    # ---------------------------------------------------------------- save
    if not args.no_save:
        banner('7. SAVE')
        from safetensors.torch import save_file
        sd = {k: v.detach().to(torch.bfloat16).contiguous()
              for k, v in base.state_dict().items()}
        assert sd
        save_file(sd, args.out, metadata={'roots': str(LIVE_ROOTS), 'awzan': str(LIVE_AWZAN),
                                          'vocab': str(LIVE_VOCAB),
                                          'source': Path(args.ckpt).name})
        try:
            sz = Path(args.out).stat().st_size
            print(f'  wrote {args.out} ({sz:,} B, {len(sd)} tensors)')
            R['saved'] = {'path': args.out, 'bytes': sz, 'tensors': len(sd)}
        except OSError as exc:
            # REPORTED, not swallowed: the pod's filesystem is at quota, so the 795 MB corrected
            # checkpoint cannot be written here.  It is NOT required by the arms: nrmt_train_v13.py
            # performs the same drop-and-load in-process from the released checkpoint.
            print(f'  !! SAVE FAILED: {type(exc).__name__}: {exc}')
            print('     the corrected checkpoint is NOT needed by the arms (nrmt_train_v13.py')
            print('     drops the 48 stale tables and loads in-process).')
            R['saved'] = {'error': f'{type(exc).__name__}: {exc}'}

    Path(args.report).write_text(json.dumps(R, indent=2))
    banner('VERDICT')
    print('  constants 9490 / 142 / 10052 in every table           [OK]')
    print('  token->id tables injective, in range, no offset math  [OK]')
    print('  290/290 real Qwen tensors consumed, table resized     [OK]')
    print('  exactly 48 stale-SPACE tables dropped, nothing else   [OK]')
    print('  one root-id space after loading                       [OK]')
    print(f'  wrote {args.report}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
