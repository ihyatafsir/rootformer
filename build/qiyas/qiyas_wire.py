#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qiyas_wire.py -- wire the computed qiyas into the realisation pathway.

WHY A RUNTIME WRAPPER AND NOT A SOURCE EDIT
-------------------------------------------
`nrmp_vocab.py` and the blueprint are READ-ONLY for this task, and the shipped realiser
`PureArabicMorphemicTokenizerV12.realize_root_and_wazn` is the thing the seven frozen test
suites certify.  A wrapper installed on the INSTANCE gives the same experimental power as an
edit (the procedure is on the live pathway, and a flag turns it on and off) with three
properties an in-place edit cannot offer:

  * flag OFF is provably inert -- the wrapper calls the original and returns its value
    unchanged, so no existing byte can move;
  * the wiring is removable (`uninstall`) and the object returns to its exact prior state;
  * every call is logged, so "it acts" is measurable rather than asserted.

MODES
-----
  replace   qiyas wins wherever it has an analogue; the original is the fallback.
  fallback  the original wins UNLESS it is degenerate -- i.e. it returned the bare root
            because no wazn branch fired (`return r` in the shipped realiser), or it
            returned a control token.  This is the conservative mode: it can only touch
            forms the shipped realiser did not actually realise.
  overlay   qiyas wins whenever its analogue support >= min_support AND it disagrees with
            the original.  Aggressive; used to bound the headroom.

The counters (n_calls / n_replaced / n_delegated / n_degenerate) are the evidence for the
ablation in the report.
"""

from __future__ import annotations

import collections
import os
from typing import Any, Dict, List, Optional

ENV_FLAG = 'QIYAS_ENABLE'


class NullEngine:
    """An engine with no analogues: used to prove that switching the flag ON is inert when
    there is nothing to extend by."""

    def realize(self, root: str, wazn: str):
        return None


def _is_degenerate(base: str, root: str, wazn: str) -> bool:
    if base is None:
        return True
    if not base:
        return True
    if base == root:
        return True                     # the shipped realiser's `return r` fall-through
    if base.startswith('<') or base.endswith('>'):
        return True                     # a control token leaked to the surface
    return False


def _dispatch(tok, root: str, wazn: str) -> str:
    """The hook.  Reads the mode FIRST so that `replace` genuinely bypasses the shipped
    realiser when qiyas has an analogue -- that bypass is what the ablation measures."""
    st = tok._qiyas_state
    st['n_calls'] += 1
    orig = tok._qiyas_orig

    if not st['enabled'] or st['engine'] is None:
        st['n_inert'] += 1
        return orig(root, wazn)

    mode = st['mode']
    want_base_first = (mode != 'replace') or st['need_base']
    base = orig(root, wazn) if want_base_first else None

    r = st['engine'].realize(root, wazn)
    if r is None or r[0] is None:
        st['n_delegated'] += 1
        st['n_no_analogue'] += 1
        return orig(root, wazn) if base is None else base
    q, prov, sup, tot, dist = r

    if mode == 'replace':
        # qiyas decides.  The shipped realiser is consulted ONLY when the ablation log is
        # switched on; with log_cap=0 it is genuinely bypassed (this is what "it acts"
        # means for the replace arm).
        if st['need_base'] and base is None:
            base = orig(root, wazn)
        if q != base:
            _note(st, root, wazn, base, q, prov, sup, tot, dist, mode)
            return q
        st['n_delegated'] += 1
        st['n_agree'] += 1
        return base

    if mode == 'fallback':
        if _is_degenerate(base, root, wazn):
            st['n_degenerate'] += 1
            if q != base:
                _note(st, root, wazn, base, q, prov, sup, tot, dist, mode)
                return q
        st['n_delegated'] += 1
        return base

    if mode == 'overlay':
        if sup >= st['min_support'] and q != base:
            _note(st, root, wazn, base, q, prov, sup, tot, dist, mode)
            return q
        st['n_delegated'] += 1
        if q == base:
            st['n_agree'] += 1
        return base

    raise ValueError('unknown qiyas mode %r' % mode)


def _note(st, root, wazn, base, q, prov, sup, tot, dist, mode):
    st['n_replaced'] += 1
    if len(st['log']) < st['log_cap']:
        st['log'].append({
            'root': root, 'wazn': wazn, 'base': base, 'qiyas': q,
            'provenance': prov, 'support': sup, 'cell_n': tot,
            'cell_distinct': dist, 'mode': mode,
        })


def install(tok, enabled: bool = False, engine: Any = None, mode: str = 'fallback',
            min_support: int = 1, log_cap: int = 200000) -> Any:
    """Install the qiyas hook on a tokenizer (or vocab) instance.  Idempotent."""
    if not hasattr(tok, '_qiyas_orig'):
        tok._qiyas_orig = tok.realize_root_and_wazn

        def wrapped(root, wazn, _tok=tok):
            return _dispatch(_tok, root, wazn)

        tok.realize_root_and_wazn = wrapped
        tok._qiyas_state = {
            'enabled': bool(enabled), 'engine': engine, 'mode': mode,
            'min_support': min_support, 'log_cap': log_cap,
            'need_base': log_cap > 0,
            'log': [],
            'n_calls': 0, 'n_inert': 0, 'n_replaced': 0, 'n_delegated': 0,
            'n_degenerate': 0, 'n_no_analogue': 0, 'n_agree': 0,
        }
    else:
        set_state(tok, enabled=enabled, engine=engine, mode=mode,
                  min_support=min_support, log_cap=log_cap)
    return tok


def set_state(tok, **kw) -> Any:
    st = tok._qiyas_state
    for k, v in kw.items():
        if v is not None:
            st[k] = v
    if 'log_cap' in kw and kw['log_cap'] is not None:
        st['need_base'] = kw['log_cap'] > 0
    return tok


def set_enabled(tok, flag: bool) -> Any:
    tok._qiyas_state['enabled'] = bool(flag)
    return tok


def uninstall(tok) -> Any:
    """Restore the exact prior state: the original bound method is put back."""
    if hasattr(tok, '_qiyas_orig'):
        tok.realize_root_and_wazn = tok._qiyas_orig
        del tok._qiyas_orig
        del tok._qiyas_state
    return tok


def stats(tok) -> Dict[str, Any]:
    """Counters only.  'log' and 'engine' are deliberately dropped: the engine is an object,
    not data, and letting it leak into a results structure breaks JSON serialisation."""
    st = tok._qiyas_state
    d = {k: v for k, v in st.items() if k not in ('log', 'engine')}
    d['n_logged'] = len(st['log'])
    return d


def reset_stats(tok, keep_log: bool = False) -> Any:
    st = tok._qiyas_state
    for k in ('n_calls', 'n_inert', 'n_replaced', 'n_delegated', 'n_degenerate',
              'n_no_analogue', 'n_agree'):
        st[k] = 0
    if not keep_log:
        st['log'] = []
    return tok


def env_enabled(default: bool = False) -> bool:
    v = os.environ.get(ENV_FLAG, '')
    if v == '':
        return default
    return v.strip().lower() in ('1', 'on', 'yes', 'true')


def smoke():
    """Non-destructive self-check used by the report's ablations."""
    class FakeTok:
        def __init__(self):
            self.calls = 0

        def realize_root_and_wazn(self, root, wazn):
            self.calls += 1
            return 'ORIG(%s,%s)' % (root, wazn)

    t = FakeTok()
    install(t, enabled=False)
    off = [t.realize_root_and_wazn('كتب', 'فَاعِل') for _ in range(3)]
    assert off == ['ORIG(كتب,فَاعِل)'] * 3, off
    assert t.calls == 3
    assert stats(t)['n_calls'] == 3 and stats(t)['n_replaced'] == 0
    install(t, enabled=True, engine=NullEngine())
    assert t.realize_root_and_wazn('كتب', 'فَاعِل') == 'ORIG(كتب,فَاعِل)'
    assert stats(t)['n_replaced'] == 0 and stats(t)['n_no_analogue'] == 1
    uninstall(t)
    assert t.realize_root_and_wazn('كتب', 'فَاعِل') == 'ORIG(كتب,فَاعِل)'
    return True


if __name__ == '__main__':
    print('qiyas_wire smoke:', smoke())
