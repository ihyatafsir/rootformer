#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exception_cellwise.py -- al-Zajjaji's shadh licence, measured PER CELL.

Al_Zajjaji_Al_Idah_Fi_Ilal_Al_Nahw.txt:1203-1205 (md5 663e493f1d2e7a3223dcc8c6b702e7d0):
  «إن الشيء إذا اطرد عليه باب، فصح في القياس وقام في المعقول، ثم اعترض عليه شيء شاذ نزر
   قليل، لعلة تلحقه، لم يكن ذلك مبطلا للأصل، والمتفق عليه في القياس المطرد»

The licence is conditional: the exception must be «شاذ نزر قليل» AND «لعلة تلحقه» -- rare and
few, and it must have a cause attaching to it.  This script tests the SECOND condition cell by
cell: for each impure cell of an accepted ʿilla, do its exception members have a determinate
own-ʿilla (the finer ʿilla reproduces them from its own cell), or are they idiosyncratic?

Reports per-cell classification and the counts, so the number of "exception cells that get
their own ʿilla" is measured, not asserted.

CPU only.  Writes /workspace/qiyas_v2/exception_cellwise.json.
"""
from __future__ import annotations

import collections
import json
import sys
import time

sys.path.insert(0, '/workspace/hf_v19_2_release')
sys.path.insert(0, '/workspace/hf_v19_2_release/models')
sys.path.insert(0, '/workspace/qiyas')

from qiyas_engine import Qiyas, ILLAS, align          # noqa: E402
import nrmp_vocab as nv                               # noqa: E402

OUT = '/workspace/qiyas_v2'
BLUEPRINT = '/workspace/hf_v19_2_release/data/rootformer_v12_arabic_blueprint.json'
DU_AFTER = '/workspace/lisan_root_fix/du_after.json'
SPECIALS = ('<NONE>', '<PAD>', '<UNK>', '')


def log(*a):
    print(*a, flush=True)


def af(x):
    return '' if (x is None or x in SPECIALS) else x


def main():
    t0 = time.time()
    V = next(v for k, v in vars(nv).items() if isinstance(v, type) and 'MorphemicVocab' in k)
    vocab = V(BLUEPRINT)
    raw = json.load(open(DU_AFTER, encoding='utf-8'))['words']
    types = list(raw.keys())
    baseline_ok = {w: bool(v[0]) for w, v in raw.items()}
    obs = []
    for w in types:
        p_id, r_id, wz_id, s_id = vocab.encode_word(w)
        r = vocab.id2root[r_id]
        wz = vocab.id2wazn[wz_id]
        p = af(vocab.id2prefix[p_id])
        s = af(vocab.id2suffix[s_id])
        if r.startswith('<') or not r:
            continue
        if not wz or wz in ('<NONE>', '<PAD>', '<UNK>'):
            continue
        if not baseline_ok.get(w):
            continue
        if len(p) + len(s) > len(w):
            continue
        stem = w[len(p):len(w) - len(s)] if s else w[len(p):]
        if not stem or p + stem + s != w:
            continue
        obs.append((r, wz, stem))
    log('[*] asl %d / %d roots' % (len(obs), len({o[0] for o in obs})))
    assert len(obs) == 106664

    def engine(name, isolated):
        q = Qiyas(name, backoff=not isolated)
        for r, w, s in obs:
            q.observe(r, w, s)
        q.induce()
        if isolated:
            q.chain = [name]
            q.tables = {name: q.tables[name]}
            q.wazn_cells = collections.defaultdict(collections.Counter)
            q.cells = q.tables[name]
        return q

    result = {'citations': {
        'shadh_licence': 'Al_Zajjaji_Al_Idah_Fi_Ilal_Al_Nahw.txt:1203-1205',
        'edition_md5': '663e493f1d2e7a3223dcc8c6b702e7d0',
        'mundabit_required': 'Ghazali_Al_Mustasfa.txt:12851',
        'muttarid_required': 'Ghazali_Al_Mustasfa.txt:13254',
    }, 'per_illa': {}}

    for name, finer in (('coarse_weak', 'positional'), ('strict7', 'positional'),
                        ('positional', 'strict7'), ('wazn_only', 'positional'),
                        ('identity', 'strict7')):
        qi = engine(name, True)
        qf = engine(finer, False)
        lab = {r: ILLAS[name](r) for r, _, _ in obs}
        by_cell = collections.defaultdict(list)
        for i, (r, w, s) in enumerate(obs):
            by_cell[(lab[r], w)].append(i)
        rows = []
        all_expl = part_expl = none_expl = 0
        n_exc_tot = 0
        for (l, wazn), c in qi.cells.items():
            n = sum(c.values())
            if not n:
                continue
            modal, sup = c.most_common(1)[0]
            if sup == n:
                continue
            exc = []
            for i in by_cell.get((l, wazn), ()):
                r, w, s = obs[i]
                a = align(s, r)
                if a is None or a == modal:
                    continue
                rr = qf.realize(r, wazn)
                expl = (rr is not None and rr[0] == s)
                exc.append({'root': r, 'surface': s, 'alignment': a,
                            'cell_n': n, 'own_illa': finer, 'explained': bool(expl)})
            if not exc:
                continue
            n_exc_tot += len(exc)
            k = sum(1 for e in exc if e['explained'])
            if k == len(exc):
                all_expl += 1
            elif k == 0:
                none_expl += 1
            else:
                part_expl += 1
            rows.append({'cell': '%s|%s' % (l, wazn), 'n': n, 'modal_support': sup,
                         'n_exceptions': len(exc), 'n_explained_by_%s' % finer: k,
                         'examples': exc[:4]})
        rec = {'n_cells': len([1 for c in qi.cells.values() if c]),
               'n_exception_cells': len(rows),
               'n_exception_observations': n_exc_tot,
               'cells_all_exceptions_explained': all_expl,
               'cells_partly_explained': part_expl,
               'cells_no_exception_explained': none_expl,
               'rows': rows}
        result['per_illa'][name] = rec
        log('  %-11s exception-cells=%-4d  all-explained-by-%-11s=%-4d  partly=%-4d  none=%-4d  '
            'exception-obs=%d'
            % (name, len(rows), finer, all_expl, part_expl, none_expl, n_exc_tot))
        del qi, qf

    # identity's iṭṭirād failures: which roots, and are they singleton-cell collapses?
    qi = engine('identity', True)
    sizes = collections.Counter()
    for k, c in qi.cells.items():
        sizes[sum(c.values())] += 1
    result['identity_cell_size_histogram'] = dict(sorted(sizes.items())[:12])
    log('  identity cell-size histogram (first sizes): %s'
        % dict(sorted(sizes.items())[:12]))
    result['identity_n_impure_cells'] = sum(
        1 for c in qi.cells.values() if c and sum(c.values()) != c.most_common(1)[0][1])
    del qi

    json.dump(result, open(OUT + '/exception_cellwise.json', 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    log('[*] wrote %s/exception_cellwise.json  (%.0fs)' % (OUT, time.time() - t0))


if __name__ == '__main__':
    main()
