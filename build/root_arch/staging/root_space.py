#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
root_space.py -- ONE root-id space, asserted, for every table in the transmutation.

WHY THIS FILE EXISTS
--------------------
Three distinct root-id spaces have been confused three times by three different agents:

    9,490   morphemic_embed.root_embed / nrmp_vocab / the live NRMT stream   <- THE ONLY LIVE ONE
    9,015   models/unified_rootformer_v12.py hardcoded (lines 26,80,91,172)
    9,015   IshtiqaqAttentionV12.root_embed (9015,64) -- dead code, 0/24 calls carry root_ids

and two awzan spaces (142 live vs 128 hardcoded).  The consequence of mixing them is silent:
`clamp(root_ids, 0, 9014)` puts 475 live root ids (5.01 %) onto row 9014, so 476 different
roots share one row, and 475 ids have no row at all.

Everything downstream imports the numbers from HERE, and `assert_one_root_space()` is called
before a model is trained and before a checkpoint is written.  Nothing in this file trusts a
live value for a *specification*: the spec is the literal below, and a live object that
disagrees raises.
"""
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

LIVE_ROOTS = 9490
LIVE_AWZAN = 142
LIVE_VOCAB = 10052

# historical values, named so a mismatch can be REPORTED instead of silently accepted
STALE_ROOTS_9015 = 9015
STALE_AWZAN_128 = 128
STALE_VOCAB_9868 = 9868
PRE_LISAN_ROOTS_9114 = 9114


class SpaceViolation(AssertionError):
    pass


# --------------------------------------------------------------------------- token <-> id tables
def build_token_root_table(vocab, vocab_size: int, fill: int = 0) -> Tuple[List[int], Dict[str, Any]]:
    """token_id -> LIVE root id, built by STRING from the live vocab.  Never by offset.

    `unified_rootformer_v12.py` did `root_part = token_id - 349` and clamped to 9014.  That is
    offset arithmetic over a partition that has been re-cut twice (Lisan extension appended at
    the END of the vocab, `<P:...>` roots inserted after the specials), so it is wrong for at
    least 199 ids and aliases the rest.  This builds the map from the token STRING through the
    live `root2id`, which is the only mapping that cannot drift.

    Every token that is not a concrete root maps to `fill` (0 = <PAD> = "no root"), which is the
    value the attention mechanism already treats as "no root" (`has_root = ids != 0`).
    """
    token_to_id = getattr(vocab, 'token_to_id', None)
    if token_to_id is None:
        tok = getattr(vocab, 'base_tok', None)
        token_to_id = getattr(tok, 'token_to_id', None)
    id_to_token = {}
    if token_to_id is not None:
        id_to_token = {int(i): t for t, i in token_to_id.items()}
    else:
        tok = getattr(vocab, 'base_tok', None)
        id_to_token = dict(getattr(tok, 'id_to_token', {}))

    table = [fill] * int(vocab_size)
    mapped, unmapped, markers, missing_root = [], [], [], []
    for tid in range(int(vocab_size)):
        t = id_to_token.get(tid, '')
        if not (t.startswith('<root_') and t.endswith('>')):
            continue
        surf = t[len('<root_'):-1]
        if surf in ('start', 'end'):
            markers.append((tid, surf))
            continue
        rid = vocab.root2id.get(surf)
        if rid is None:
            missing_root.append((tid, surf))
            continue
        table[tid] = int(rid)
        mapped.append(tid)
    info = {'tokens': len(mapped), 'markers': markers, 'missing_root': missing_root,
            'fill': fill, 'distinct': len({table[t] for t in mapped})}
    if missing_root:
        raise SpaceViolation(
            f'{len(missing_root)} <root_X> vocab tokens have NO live root id, e.g. '
            f'{missing_root[:5]} -- the tokenizer vocab and the NRMP root space disagree')
    if info['distinct'] != len(mapped):
        dup = {}
        for t in mapped:
            dup.setdefault(table[t], []).append(t)
        clash = {k: v for k, v in dup.items() if len(v) > 1}
        raise SpaceViolation(f'token->root table is NOT injective: {len(clash)} root ids are '
                             f'reached by >1 token, e.g. '
                             f'{list(clash.items())[:3]}')
    return table, info


def build_token_wazn_table(vocab, vocab_size: int, fill: int = 0) -> Tuple[List[int], Dict[str, Any]]:
    """token_id -> LIVE wazn id, by string.  Replaces the `torch.zeros_like` STUB at
    `unified_rootformer_v12.py:121`, which returned all-zero wazn ids and was mistaken for the
    architecture even after the dead code was found."""
    tok = getattr(vocab, 'base_tok', None)
    token_to_id = getattr(tok, 'token_to_id', None) or getattr(vocab, 'token_to_id', {})
    id_to_token = {int(i): t for t, i in token_to_id.items()}
    table = [fill] * int(vocab_size)
    mapped, markers, missing = [], [], []
    for tid in range(int(vocab_size)):
        t = id_to_token.get(tid, '')
        if not (t.startswith('<wazn_') and t.endswith('>')):
            continue
        surf = t[len('<wazn_'):-1]
        if surf in ('start', 'end'):
            markers.append((tid, surf))
            continue
        wid = vocab.wazn2id.get(surf)
        if wid is None:
            missing.append((tid, surf))
            continue
        table[tid] = int(wid)
        mapped.append(tid)
    if missing:
        raise SpaceViolation(f'{len(missing)} <wazn_X> tokens have no live wazn id, e.g. '
                             f'{missing[:5]}')
    return table, {'tokens': len(mapped), 'markers': markers,
                   'distinct': len({table[t] for t in mapped})}


# --------------------------------------------------------------------------- model assertions
def assert_one_root_space(model, vocab=None, expect_roots: int = LIVE_ROOTS,
                          expect_awzan: int = LIVE_AWZAN) -> Dict[str, Any]:
    """Assert that every root/wazn-bearing table in `model` lives in ONE space.

    Raises SpaceViolation listing every offender.  Returns the census.
    """
    off, census = [], {}
    for name, p in model.named_parameters():
        if not name.endswith('.weight') or p.dim() != 2:
            continue
        rows = int(p.shape[0])
        if name.endswith('root_embed.weight') or name.endswith('root_head.weight'):
            census[name] = rows
            if rows != expect_roots:
                off.append((name, rows, expect_roots))
        elif name.endswith('wazn_embed.weight') or name.endswith('wazn_head.weight'):
            census[name] = rows
            if rows != expect_awzan:
                off.append((name, rows, expect_awzan))
    if vocab is not None:
        for attr, want in (('num_roots', expect_roots), ('num_awzan', expect_awzan)):
            got = getattr(vocab, attr, None)
            if got is not None and int(got) != want:
                off.append((f'vocab.{attr}', int(got), want))
    if off:
        raise SpaceViolation(
            'ROOT-ID SPACE VIOLATION -- every table must live in the live space '
            f'(roots={expect_roots}, awzan={expect_awzan}):\n  ' +
            '\n  '.join(f'{n}: rows={g} want={w}' for n, g, w in off))
    return census


def assert_no_aliasing(table: Sequence[int], n_ids: int, table_rows: int, what: str = 'root'):
    """The clamp that caused the 5.01 % collision, restated as an assertion."""
    if table_rows >= n_ids:
        return {'ok': True, 'table_rows': table_rows, 'live_ids': n_ids, 'aliased': 0}
    aliased = n_ids - table_rows
    raise SpaceViolation(
        f'{what} table has {table_rows} rows for {n_ids} live ids: {aliased} ids '
        f'({100.0 * aliased / n_ids:.2f} %) alias onto row {table_rows - 1}, which then '
        f'represents {aliased + 1} different {what}s')
