#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
model_build.py -- the ONE place the early-root experiment builds its model.

Mirrors the trainer's construction exactly (`nrmt_train_v13.py` lines 382-401 + the v13 stale-table
drop at 438-457) so a proof/benchmark runs the SAME object graph as the arms:

    UnifiedRootformerV13 -> UnifiedRootformerV17_DeepSeekFlash -> RootformerNRMT

`morphemic_embed` does NOT live on UnifiedRootformerV13: `RootformerNRMT` creates it when the base
model has none (nrmt_arch.py:463-470), which is why the released `morphemic_embed.*` rows load into
the NRMT container and not into the V13 model.  Any script that reaches for `base.morphemic_embed`
is reaching for an attribute that has never existed.
"""
import sys

import torch

for _p in ('/workspace/hf_v19_2_release', '/workspace/hf_v19_2_release/models',
           '/workspace/transmute_v2', '/workspace/root_arch', '/workspace/ishtiqaq_check',
           '/workspace/root_attn'):
    if _p not in sys.path:
        sys.path.insert(0, _p)

CK = ('/workspace/hf_v19_2_release/checkpoints/'
      'rootformer_v19_2_synthesis_ar_backbone.awzan142.roots9490.tok10052.safetensors')
BP = '/workspace/hf_v19_2_release/data/rootformer_v12_arabic_blueprint.json'
CACHE = '/workspace/head_fix/nrmp_cache_9490_aligned'
DEV = 'cuda'
WIN = 128


def vocab_of():
    import nrmp_vocab as nv
    V = next(v for k, v in vars(nv).items() if isinstance(v, type) and 'MorphemicVocab' in k)
    return V(BP)


def assert_root_count(vocab, verbose=True):
    """The SILENT SHRINK trap: a missing release module makes nrmp_vocab report 9445, not 9490.

    Nothing downstream raises on 9445 -- the tables are simply built 45 rows short and every
    number computed afterwards is quietly wrong.  Check it before anything else.
    """
    got = (int(vocab.num_roots), int(vocab.num_awzan))
    if got != (9490, 142):
        raise RuntimeError(
            f'SILENT VOCAB SHRINK: nrmp_vocab reports {got[0]} roots / {got[1]} awzan, expected '
            f'9490 / 142.  Known cause: a missing release module (validated_segmentation.py) on '
            f'sys.path.  Refusing to train on a shrunken root space.')
    if verbose:
        print(f'[*] PREFLIGHT root count OK: roots={got[0]} awzan={got[1]}')
    return got


def detach_flash_hooks(model, verbose=False):
    """Exactly the detachment the trainer's `--flash-hooks off` performs."""
    n = 0
    for l in model.backbone.layers:
        n += len(l._forward_hooks)
        l._forward_hooks.clear()
    if verbose:
        print(f'    detached {n} forward hooks')
    return n


def build(ckpt=CK, vocab=None, flash_hooks='off', verbose=True):
    """Returns (vocab, base_v13, flash_wrapper, nrmt_model).  Loads the released checkpoint.

    `flash_hooks='off'` detaches the 151.95 M dead parameters registered by the flash wrapper;
    `'on'` leaves the shipped hooks in place (needed by the perturbation proof of Phase 0d).
    """
    from safetensors.torch import load_file
    import unified_rootformer_v13 as V13
    from deepseek_v4_1_flash_model import UnifiedRootformerV17_DeepSeekFlash
    from nrmt_arch import RootformerNRMT

    vocab = vocab or vocab_of()
    assert_root_count(vocab, verbose)
    base = V13.UnifiedRootformerV13(BP, 'Qwen/Qwen2.5-0.5B', DEV, torch.bfloat16,
                                    vocab=vocab).to(DEV)
    flash = UnifiedRootformerV17_DeepSeekFlash(base, DEV, torch.bfloat16).to(DEV)
    model = RootformerNRMT(flash, vocab, DEV, torch.bfloat16, hist=3, dropout=0.1,
                           use_features=True, feat_gate=True).to(DEV)
    if flash_hooks == 'off':
        n = detach_flash_hooks(model, verbose)
        if verbose:
            print(f'[*] flash hooks DETACHED: {n} hooks cleared (0d decision)')

    sd = load_file(ckpt)
    drop = [k for k in list(sd) if ('self_attn.root_embed.weight' in k
                                    or 'self_attn.wazn_embed.weight' in k)
            and k in model.state_dict()
            and tuple(sd[k].shape) != tuple(model.state_dict()[k].shape)]
    for k in drop:
        del sd[k]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # The 48 `self_attn.root_embed/wazn_embed` keys are DELIBERATELY absent: the checkpoint
    # carries the stale 9015/128 tables and v13's are 9490/142, so there is nothing to load and
    # the per-layer table keeps its init.  It is dead code in the NRMT forward either way --
    # the attached score bias reads `root_source='shared'` (morphemic_embed.root_embed, 9490x448).
    def _expected_missing(k):
        if k.startswith('nrmt_head.'):
            return True
        return ('backbone.layers' in k
                and (k.endswith('self_attn.root_embed.weight')
                     or k.endswith('self_attn.wazn_embed.weight')))

    stale_missing = [k for k in missing if not _expected_missing(k)]
    if verbose:
        print(f'[*] checkpoint loaded: dropped {len(drop)} stale-SPACE self_attn tables; '
              f'missing={len(missing)} ({len(stale_missing)} unexpected-missing) '
              f'unexpected={len(unexpected)}')
    if stale_missing:
        raise RuntimeError(
            f'checkpoint does not cover {len(stale_missing)} non-head tensors -- this is the '
            f'"555 missing / 520 unexpected, model is silently random" trap: {stale_missing[:8]}')
    return vocab, base, flash, model


def win_batch(B, win=WIN):
    """A real (P,R,W,S) batch from the frozen-cache corpus, on DEV."""
    P, R, W, S = (t.long()[:B * win].view(B, win).to(DEV)
                  for t in torch.load(f'{CACHE}/train.pt', map_location='cpu'))
    return P, R, W, S


def set_active_root_ids(model, root_ids):
    """The ONLY wiring the shipped native root path needs (trainer line 746-747)."""
    for l in model.backbone.layers:
        l.self_attn.active_root_ids = root_ids
