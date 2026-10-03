#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rca_compat.py -- keep `root_cross_attn.py` BYTE-IDENTICAL while defusing its in-place clamp.

THE BUG (found by `torch.autograd.set_detect_anomaly(True)`, not by reading)
--------------------------------------------------------------------------
`/workspace/root_attn/root_cross_attn.py`:

    line 136   safe = root_ids.clamp_(0, root_weight.shape[0] - 1)     # IN PLACE
    line 137   r = root_weight.index_select(0, safe.reshape(-1)) ...

`clamp_` returns `self`, so `safe` IS the caller's tensor.  `index_select` then saves a VIEW of it
in `IndexSelectBackward0`, and the NEXT attached layer's `clamp_` bumps the same version counter.
The first backward therefore dies:

    RuntimeError: one of the variables needed for gradient computation has been modified by an
    inplace operation: [torch.cuda.LongTensor [512]] is at version 24; expected version 23
    (Error detected in IndexSelectBackward0 ... root_cross_attn.py line 137)

With 24 attached layers the versions are 1..24 and layer 22's saved tensor is the one that fails --
exactly the "expected 23" in the message.

WHY IT NEVER BIT BEFORE
-----------------------
`index_select` only builds a grad node when `root_weight.requires_grad`.  Every previous arm froze
`morphemic_embed`, so `root_embed.weight.requires_grad` was False, no `IndexSelectBackward0` was
created, the index was never saved, and the version check never ran.  It fires the moment the
INPUT STAGE is trained -- which is precisely what the early-root experiment requires.

THE WORKAROUND
--------------
Do NOT edit the verified module (an earlier draft did; it was reverted).  Wrap the one method so
each layer clamps a tensor it owns.  Numerically identical -- `clamp_` and `clamp` produce the same
values, verified with `torch.equal` -- and the only cost is 24 tiny int64 clones per step.

`install()` is idempotent and asserts that the bug it works around is still present, so a future
upstream fix turns this into a no-op with a clear message instead of silently double-cloning.
"""
from typing import Any

_BUGGY = 'root_ids.clamp_(0, root_weight.shape[0] - 1)'


def install(verbose: bool = True) -> dict:
    import root_cross_attn as RCA

    cls = RCA.RootHistoryCrossAttention
    if getattr(cls, '_rca_clamp_compat', False):
        return {'installed': False, 'reason': 'already installed'}

    import inspect
    src = inspect.getsource(cls.forward)
    has_bug = _BUGGY in src
    if not has_bug:
        # Someone fixed it upstream (`clamp_` -> `clamp`).  Nothing to wrap; say so loudly.
        if verbose:
            print('[rca_compat] upstream root_cross_attn.RootHistoryCrossAttention already uses '
                  'an OUT-OF-PLACE clamp -- no workaround needed, module left as found')
        cls._rca_clamp_compat = True
        return {'installed': False, 'reason': 'upstream already fixed'}

    orig = cls.forward

    def forward(self, h, root_ids, root_weight, exclude_current: bool = False):
        # each layer gets its own tensor, so the in-place clamp cannot invalidate the previous
        # layer's saved index
        rid = None if root_ids is None else root_ids.clone()
        return orig(self, h, rid, root_weight, exclude_current=exclude_current)

    forward.__doc__ = (orig.__doc__ or '') + \
        '\n\n[rca_compat] wrapped: `root_ids` is cloned per layer because the shipped body does' \
        '\n`safe = root_ids.clamp_(...)` IN PLACE on the shared tensor.'
    forward.__name__ = 'forward'
    cls.forward = forward
    cls._rca_clamp_compat = True
    if verbose:
        print('[*] rca_compat: RootHistoryCrossAttention.forward wrapped (per-layer root_ids.clone)'
              ' -- works around the in-place `clamp_` at root_cross_attn.py:136 WITHOUT editing it',
              flush=True)
    return {'installed': True, 'reason': 'in-place clamp_ found', 'bug_line': _BUGGY}
