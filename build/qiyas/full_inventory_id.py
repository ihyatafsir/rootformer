#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
full_inventory_id.py -- TASK B over the FULL root inventory, not 37 roots.

WHY
---
QIYAS_REPORT.md / STATE.md flagged the honest completion as never run:
"Task B's candidate set was restricted to 37 roots -- the full 9,220-root inverse search is the
honest completion and was never run."

What the stored result already tells us (control_74_roots in deriv3588_results.json):
37 roots -> top1 3031/3588 = 84.48 %   (chance 2.70 %)
74 roots -> top1 3031/3588 = 84.48 %   (chance 1.35 %)
IDENTICAL, while chance halved.  So the restriction is probably not inflating the rate.  This
script tests that properly, over every root the vocabulary holds.

METHOD
------
The existing `identify()` is O(positions x candidates) -- 3,588 x 9,220 is 33 M inner steps and
rebuilds realizations per position.  Here the grid is inverted ONCE into
    surface -> [(root, wazn, provenance, support, total), ...]
so each position is a dict lookup.  Grid construction is the only cost and is parallelised.

The candidate semantics are reproduced EXACTLY from qiyas_deriv3588.py:
  * a candidate (r, wz) matches position word w iff  (p + qy + s) == w  OR  qy == w
  * candidates are ranked by rank_key(rootfreq): provenance tier, then purity, then support,
    then root frequency, then root string
  * top-1 / top-5 are computed over the deduplicated root order, exactly as `identify` does

CPU ONLY.  Reads /workspace.  Writes /workspace/qiyas/full_inventory_id.json.
"""

from __future__ import annotations

import collections
import json
import os
import sys
import time
import multiprocessing as mp

sys.path.insert(0, '/workspace/hf_v19_2_release')
sys.path.insert(0, '/workspace/hf_v19_2_release/models')
sys.path.insert(0, '/workspace/qiyas')

from qiyas_engine import Qiyas          # noqa: E402
import nrmp_vocab as nv                 # noqa: E402

BLUEPRINT = '/workspace/hf_v19_2_release/data/rootformer_v12_arabic_blueprint.json'
DU_AFTER = '/workspace/lisan_root_fix/du_after.json'
GOLD = '/workspace/qiyas/deriv_gold.json'
OUT = '/workspace/qiyas/full_inventory_id.json'

AF = ('<NONE>', '<PAD>', '<UNK>')


def affix(x):
    return '' if (x is None or x in AF) else x


def log(*a):
    print(*a, flush=True)


def load_vocab():
    return nv.FarāhīdianMorphemicVocab(BLUEPRINT)


def harvest(vocab, du, hold):
    obs = []
    n_skip = collections.Counter()
    for w in du:
        p_id, r_id, wz_id, s_id = vocab.encode_word(w)
        r = vocab.id2root[r_id]
        wz = vocab.id2wazn[wz_id]
        p, s = affix(vocab.id2prefix[p_id]), affix(vocab.id2suffix[s_id])
        if not r or r.startswith('<') or r in hold:
            n_skip['held_or_bad_root'] += 1
            continue
        if not wz or wz in AF:
            n_skip['bad_wazn'] += 1
            continue
        if len(p) + len(s) > len(w):
            n_skip['affix_too_long'] += 1
            continue
        stem = w[len(p):len(w) - len(s)] if s else w[len(p):]
        if not stem or p + stem + s != w:
            n_skip['bad_split'] += 1
            continue
        obs.append((r, wz, stem))
    return obs, dict(n_skip)


def rank_key(rootfreq):
    order = {'positional': 0, 'strict7': 1, 'coarse_weak': 2, 'wazn_only': 3, 'GLOBAL': 4}

    def k(c):
        r, wz, qy, pv, sup, tot = c
        purity = sup / tot if tot else 0.0
        return (order.get(pv.split(':')[0], 5), -purity, -sup, -rootfreq.get(r, 0), r)
    return k


# ---------------------------------------------------------------- parallel grid build
_ENGINE = None
_WAZAN = None
_ROOTS = None


def _init_worker(obs, wazan, roots):
    """Each worker induces its own engine from the same asl (CPU; ~seconds)."""
    global _ENGINE, _WAZAN, _ROOTS
    q = Qiyas('strict7', backoff=True)
    for r, w, s in obs:
        q.observe(r, w, s)
    q.induce()
    _ENGINE = q
    _WAZAN = wazan
    _ROOTS = roots


def _realize_root(root):
    """All (wazn -> realization) for one root; returns (root, wazn, qy, pv, sup, tot) rows."""
    q = _ENGINE
    rows = []
    for wz in _WAZAN:
        res = q.realize(root, wz)
        if res is None:
            continue
        qy, pv, sup, tot, dist = res
        rows.append((root, wz, qy, pv, sup, tot))
    return rows


def build_index_chunk(roots):
    out = []
    for r in roots:
        out.extend(_realize_root(r))
    return out


def rank_key(rootfreq):
    order = {'positional': 0, 'strict7': 1, 'coarse_weak': 2, 'wazn_only': 3, 'GLOBAL': 4}

    def k(c):
        r, wz, qy, pv, sup, tot = c
        purity = sup / tot if tot else 0.0
        return (order.get(pv.split(':')[0], 5), -purity, -sup, -rootfreq.get(r, 0), r)
    return k


# ---------------------------------------------------------------- parallel grid build
_ENGINE = None
_WAZAN = None
_ROOTS = None


def _init_worker(obs, wazan, roots):
    """Each worker induces its own engine from the same asl (CPU; ~seconds)."""
    global _ENGINE, _WAZAN, _ROOTS
    q = Qiyas('strict7', backoff=True)
    for r, w, s in obs:
        q.observe(r, w, s)
    q.induce()
    _ENGINE = q
    _WAZAN = wazan
    _ROOTS = roots


def _realize_root(root):
    """All (wazn -> realization) for one root; returns (root, wazn, qy, pv, sup, tot) rows."""
    q = _ENGINE
    rows = []
    for wz in _WAZAN:
        res = q.realize(root, wz)
        if res is None:
            continue
        qy, pv, sup, tot, dist = res
        rows.append((root, wz, qy, pv, sup, tot))
    return rows


def build_index_chunk(roots):
    out = []
    for r in roots:
        out.extend(_realize_root(r))
    return out


def matches(qy, word, prefix, suffix):
    """EXACTLY the candidate test of qiyas_deriv3588.identify:
         (prefix + qy + suffix) == word   OR   qy == word
    With clean stripping the second disjunct is `qy == stem`, which is what makes the bare form
    (فَعَلَ etc.) findable.  The first disjunct is what makes prefixed/suffixed forms findable.
    Both realizations the engine can emit -- the affixed surface and the free stem -- are indexed
    so that either route can fire.
    """
    return (prefix + qy + suffix) == word or qy == word


def main():
    t0 = time.time()
    vocab = load_vocab()
    if (int(vocab.num_roots), int(vocab.num_awzan)) != (9490, 142):
        raise SystemExit('VOCAB SHRINK TRAP: got %s/%s'
                         % (vocab.num_roots, vocab.num_awzan))
    log('[*] vocab roots=%d awzan=%d' % (vocab.num_roots, vocab.num_awzan))

    gold = json.load(open(GOLD, encoding='utf-8'))
    pos = [{'word': x['word'], 'root': x['root'], 'wazn': x['wazn'],
            'prefix': affix(x['prefix']), 'suffix': affix(x['suffix'])}
           for x in gold['positions']]
    hold = sorted({x['root'] for x in pos})
    log('[*] gold: %d positions across %d held-out roots' % (len(pos), len(hold)))

    du = json.load(open(DU_AFTER, encoding='utf-8'))['words']
    obs, skipped = harvest(vocab, du, set(hold))
    rootfreq = collections.Counter(r for r, _, _ in obs)
    log('[*] asl: %d realisations, %d distinct roots; skipped %s'
        % (len(obs), len(rootfreq), skipped))

    wazn_all = [w for w in vocab.awzan_list if isinstance(w, str) and not w.startswith('<')]
    all_roots = [r for r in vocab.roots_list
                 if isinstance(r, str) and not r.startswith('<') and r]
    log('[*] full inventory: %d roots x %d awzan = %d realizations to compute'
        % (len(all_roots), len(wazn_all), len(all_roots) * len(wazn_all)))

    nproc = min(int(os.environ.get('NPROC', '30')), mp.cpu_count())
    log('[*] building the inverted index on %d processes ...' % nproc)
    chunks = [all_roots[i::nproc] for i in range(nproc)]
    t1 = time.time()
    with mp.Pool(nproc, initializer=_init_worker, initargs=(obs, wazn_all, all_roots)) as pool:
        pieces = pool.map(build_index_chunk, chunks)
    rows = [r for piece in pieces for r in piece]
    log('    %d realizations in %.0fs (%.1f min)' % (len(rows), time.time() - t1,
                                                     (time.time() - t1) / 60))

    # Inverted index, bucketed by surface length.  A realization qy can match a position word w
    # ONLY if len(qy) is (len(w) - len(prefix) - len(suffix)) or len(w); bucketing by length keeps
    # the per-position scan tiny without re-deriving the stem (which is what broke the first
    # attempt: realizations are already the causal stem, so the test must run as written).
    index = collections.defaultdict(list)
    for r, wz, qy, pv, sup, tot in rows:
        index[len(qy)].append((r, wz, qy, pv, sup, tot))
    lens = sorted(index)
    log('    distinct realization lengths indexed: %d (n=%s)'
        % (len(lens), {L: len(index[L]) for L in lens[:6]}))

    rk = rank_key(rootfreq)

    def rank_of(cands, goldr):
        cands = sorted(cands, key=rk)
        rr = []
        for c in cands:
            if c[0] not in rr:
                rr.append(c[0])
        if goldr not in rr:
            return None, rr
        return rr.index(goldr) + 1, rr

    def identify(positions):
        """identify() as a length-bucketed scan, same match + rank semantics."""
        n1 = n5 = nocand = 0
        sizes = []
        for x in positions:
            w, goldr, p, s = x['word'], x['root'], x['prefix'], x['suffix']
            La = len(w) - len(p) - len(s)
            Lb = len(w)
            pool = index.get(La, [])
            if Lb != La:
                pool = list(pool) + list(index.get(Lb, []))
            cands = [c for c in pool if (p + c[2] + s) == w or c[2] == w]
            sizes.append(len(cands))
            if not cands:
                nocand += 1
                continue
            rnk, _ = rank_of(cands, goldr)
            if rnk == 1:
                n1 += 1
            if rnk is not None and rnk <= 5:
                n5 += 1
        return {'top1': n1, 'top5': n5, 'no_candidate': nocand,
                'mean_candidate_instances': sum(sizes) / len(sizes) if sizes else 0.0}

    log('[*] identifying over the FULL inventory ...')
    t2 = time.time()
    full = identify(pos)
    n = len(pos)
    log('    FULL %d roots: top1 %d/%d = %.4f %% (chance %.4f %%), top5 %.4f %%  [%.0fs]'
        % (len(all_roots), full['top1'], n, 100 * full['top1'] / n,
           100 / len(all_roots), 100 * full['top5'] / n, time.time() - t2))
    log('    no admissible candidate: %d; mean candidate instances %.3f'
        % (full['no_candidate'], full['mean_candidate_instances']))

    # the 37-root subset, rebuilt the same way, as an internal control on this harness.
    # Stored deriv3588 result: top1 3031 / 3588 = 84.4760 %.  If this harness reproduces that,
    # the full-inventory number is comparable to it.
    hset = set(hold)
    index_backup = index
    index = collections.defaultdict(list)
    for L, lst in index_backup.items():
        k37 = [c for c in lst if c[0] in hset]
        if k37:
            index[L] = k37
    sub37 = identify(pos)
    index = index_backup
    log('    control 37 roots (same harness): top1 %d/%d = %.4f %% (stored: 3031 = 84.4760 %%)'
        % (sub37['top1'], n, 100 * sub37['top1'] / n))

    out = {
        'harness': 'full_inventory_id.py',
        'built': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'vocab': {'roots': int(vocab.num_roots), 'awzan': int(vocab.num_awzan)},
        'n_positions': n,
        'n_roots_in_inventory': len(all_roots),
        'n_awzan_tried': len(wazn_all),
        'n_realizations_computed': len(rows),
        'n_distinct_surfaces': len(index_backup),
        'full_inventory': dict(full, chance_top1=1 / len(all_roots),
                               chance_top5=5 / len(all_roots),
                               top1_rate=full['top1'] / n, top5_rate=full['top5'] / n),
        'control_37_same_harness': dict(sub37, chance_top1=1 / 37, chance_top5=5 / 37,
                                        top1_rate=sub37['top1'] / n,
                                        top5_rate=sub37['top5'] / n),
        'stored_37_from_deriv3588': {'top1': 3031, 'top1_rate': 3031 / 3588,
                                     'top5': 3032, 'top5_rate': 3032 / 3588},
        'seconds': round(time.time() - t0, 1),
    }
    json.dump(out, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    log('[*] wrote %s  [total %.0fs]' % (OUT, time.time() - t0))


if __name__ == '__main__':
    main()
