#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exception_loo_cells.py -- the LAST candidate reading of "exception cells".

An iṭṭirād (leave-one-out) failure can occur in two different situations:
  (a) the observation's own cell is a SINGLETON: under leave-one-out the cell vanishes and the
      prediction falls to the context-free template -> a failure that is arithmetic, not a
      real exception;
  (b) the cell has >= 2 observations and the observation is a MINORITY member: a genuine
      shadhdh exception inside a populated cell.
Only (b) is an exception in al-Zajjaji's sense («شاذ نزر قليل، لعلة تلحقه»).  This script
separates them and counts distinct cells in each class, so the number of "exception cells that
need their own ʿilla" is measured under the LOO definition too.

CPU only.  Writes /workspace/qiyas_v2/exception_loo_cells.json.
"""
from __future__ import annotations

import collections
import json
import sys

sys.path.insert(0, '/workspace/hf_v19_2_release')
sys.path.insert(0, '/workspace/hf_v19_2_release/models')
sys.path.insert(0, '/workspace/qiyas')

from qiyas_engine import Qiyas, ILLAS, align          # noqa: E402
import nrmp_vocab as nv                               # noqa: E402

OUT = '/workspace/qiyas_v2'
BLUEPRINT = '/workspace/hf_v19_2_release/data/rootformer_v12_arabic_blueprint.json'
DU_AFTER = '/workspace/lisan_root_fix/du_after.json'
SPECIALS = ('<NONE>', '<PAD>', '<UNK>', '')
ILLA_NAMES = ('wazn_only', 'coarse_weak', 'strict7', 'positional', 'identity')


def log(*a):
    print(*a, flush=True)


def af(x):
    return '' if (x is None or x in SPECIALS) else x


def main():
    V = next(v for k, v in vars(nv).items() if isinstance(v, type) and 'MorphemicVocab' in k)
    vocab = V(BLUEPRINT)
    raw = json.load(open(DU_AFTER, encoding='utf-8'))['words']
    types = list(raw.keys())
    bok = {w: bool(v[0]) for w, v in raw.items()}
    obs = []
    for w in types:
        p_id, r_id, wz_id, s_id = vocab.encode_word(w)
        r = vocab.id2root[r_id]
        wz = vocab.id2wazn[wz_id]
        p = af(vocab.id2prefix[p_id])
        s = af(vocab.id2suffix[s_id])
        if r.startswith('<') or not r or not wz or wz in ('<NONE>', '<PAD>', '<UNK>'):
            continue
        if not bok.get(w) or len(p) + len(s) > len(w):
            continue
        stem = w[len(p):len(w) - len(s)] if s else w[len(p):]
        if not stem or p + stem + s != w:
            continue
        obs.append((r, wz, stem))
    assert len(obs) == 106664
    log('[*] asl %d / %d roots' % (len(obs), len({o[0] for o in obs})))

    out = {}
    for name in ILLA_NAMES:
        q = Qiyas(name, backoff=False)          # the stored, comparable definition
        for r, w, s in obs:
            q.observe(r, w, s)
        q.induce()
        cell_size = {k: sum(c.values()) for k, c in q.tables[name].items()}
        fail_cells = collections.Counter()
        fail_singleton = collections.Counter()
        fails = 0
        for _label, wazn, root, surface in q.obs:
            a = align(surface, root)
            if a is None:
                continue
            touched = []
            for nm in q.chain:
                key = (q._label(nm, root), wazn)
                q.tables[nm][key][a] -= 1
                touched.append((nm, key, a))
            pred = None
            for nm in q.chain:
                m = q._modal(q.tables[nm], (q._label(nm, root), wazn))
                if m:
                    pred = m[0]
                    break
            if pred is None:
                m = q._modal(q.wazn_cells, wazn)
                pred = m[0] if m else None
            for nm, key, aa in touched:
                q.tables[nm][key][aa] += 1
            if pred != a:
                fails += 1
                k0 = (q._label(name, root), wazn)
                fail_cells[k0] += 1
                if cell_size.get(k0, 0) == 1:
                    fail_singleton[k0] += 1
        multi = {k: v for k, v in fail_cells.items() if k not in fail_singleton}
        roots_in_multi = {r for r, w, _ in obs if (ILLAS[name](r), w) in multi}
        rec = {
            'ittirad_failures': fails,
            'distinct_cells_with_failure': len(fail_cells),
            'cells_failing_only_because_singleton': len(fail_singleton),
            'distinct_NON_SINGLETON_cells_with_failure': len(multi),
            'distinct_roots_in_non_singleton_failure_cells': len(roots_in_multi),
            'non_singleton_failure_examples': ['%s|%s' % k for k in list(multi)[:10]],
        }
        out[name] = rec
        log('  %-11s ittirad-fail=%-6d cells-with-failure=%-6d (singleton-only: %-6d, '
            'NON-singleton exception cells: %d)'
            % (name, fails, len(fail_cells), len(fail_singleton), len(multi)))
        del q
    json.dump(out, open(OUT + '/exception_loo_cells.json', 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    log('[*] wrote %s/exception_loo_cells.json' % OUT)


if __name__ == '__main__':
    main()
