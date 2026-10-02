#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qiyas_corpus.py -- the computed qiyas measured against the shipped realiser, on CPU only.

Runs on the pod against the CURRENT release vocabulary (the one the lisan_root_fix produced:
9,490 roots / 142 awzan, blueprint md5 5bd3e828b41fb1a6a6039d47c2f359fc) and the identical
188,722-type / 3,298,362-form held-out sample that eval_heldout.py scored.

WHAT IT MEASURES
----------------
  1. the attested asl set: (root, wazn) -> stem, harvested from the held-out corpus itself
  2. the three usuli validity conditions per 'illa, with their rates, and a REJECT verdict
  3. the derivational holdout: the 37 roots held out of the neural training are held out of
     qiyas's INDUCTION too; qiyas is then asked to realise the real corpus forms built from
     those roots
  4. the same holdout repeated over random 37-root folds, to show the result is not special
     to those roots
  5. the ablation: the qiyas hook installed on the live tokenizer, flag OFF vs flag ON
  6. the effect on type-level round-trip

Writes JSON to /workspace/qiyas/.  Nothing outside /workspace/qiyas/ is written.
"""

from __future__ import annotations

import collections
import json
import os
import random
import sys
import time

sys.path.insert(0, '/workspace/hf_v19_2_release')
sys.path.insert(0, '/workspace/hf_v19_2_release/models')
sys.path.insert(0, '/workspace/qiyas')

from qiyas_engine import Qiyas, ILLAS, align, instantiate, alignment_slots   # noqa: E402
import qiyas_wire                                                            # noqa: E402

OUT = '/workspace/qiyas'
DU_AFTER = '/workspace/lisan_root_fix/du_after.json'

# the 37 roots held out of the neural training in scratch_comp/RESULTS_COMP.md
HOLD37 = ('بدا بدل بني بوب ثان جاب ختم دال دبب دبر دفع دول ذهن زاد سعع سما سير شخص شرر '
          'شقق صبح صرح طعع عبس عرق عزل عنا غفل قبح قلد كرر كره كلف لغا مهم نفي وهم').split()

# rejection thresholds (declared BEFORE the numbers are seen)
TH = {
    'mundabit_min_purity': 0.98,     # a cell whose hukm is not determinate fails mundabit
    'muttarid_min_rate': 0.98,       # a cause with > 2 % exceptions fails ittirad
    'munakis_min_lift': 0.02,        # a cause that does not discriminate fails mun'akis
    'min_cell_for_munakis': 30,
}


def log(*a):
    print(*a, flush=True)


def load_vocab():
    import nrmp_vocab as nv
    V = next(v for k, v in vars(nv).items() if isinstance(v, type) and 'MorphemicVocab' in k)
    bp = '/workspace/hf_v19_2_release/data/rootformer_v12_arabic_blueprint.json'
    return V(bp)


def harvest(vocab, types, baseline_ok):
    """Build the attested asl set from the held-out corpus.

    A type is an admissible asl only when the shipped pipeline REPRODUCES it exactly
    (baseline_ok): then the split p | stem | s is unambiguous and the (root, wazn) -> stem
    triple is an attested realisation.  Forms the shipped pipeline cannot reproduce are not
    used as asl -- they are exactly the far' the procedure is afterwards tested on.
    """
    obs = []
    skipped = collections.Counter()
    metas = {}
    t0 = time.time()
    for i, w in enumerate(types):
        try:
            p_id, r_id, wz_id, s_id = vocab.encode_word(w)
        except Exception:
            skipped['encode_error'] += 1
            continue
        r = vocab.id2root.get(r_id, '')
        wz = vocab.id2wazn.get(wz_id, '')
        p = vocab.id2prefix.get(p_id, '')
        s = vocab.id2suffix.get(s_id, '')
        p = '' if p in ('<NONE>', '<PAD>', '<UNK>') else p
        s = '' if s in ('<NONE>', '<PAD>', '<UNK>') else s
        metas[w] = (p, r, wz, s)
        if r.startswith('<') or not r:
            skipped['special_root'] += 1
            continue
        if wz in ('<NONE>', '<PAD>', '<UNK>', '') or not wz:
            skipped['no_wazn'] += 1
            continue
        if not baseline_ok.get(w, False):
            skipped['baseline_fail'] += 1
            continue
        if len(p) + len(s) > len(w):
            skipped['bad_split'] += 1
            continue
        stem = w[len(p):len(w) - len(s)] if s else w[len(p):]
        if p + stem + s != w:
            skipped['bad_split'] += 1
            continue
        if not stem:
            skipped['empty_stem'] += 1
            continue
        obs.append((r, wz, stem))
        if (i + 1) % 40000 == 0:
            log('    harvest %d/%d (%.0fs)' % (i + 1, len(types), time.time() - t0))
    return obs, metas, dict(skipped)


def measure_illa(name, obs, backoff=True):
    # The STRICT form of the test: the ʿilla's own rule, with no fall-back crutches.
    q_own = Qiyas(name, backoff=False)
    for r, w, s in obs:
        q_own.observe(r, w, s)
    q_own.induce()
    it_own = q_own.ittirad()
    md_own = q_own.mundabit()
    own = {
        'ittirad_rate': it_own['ittirad_rate'],
        'n': it_own['n'], 'fail': it_own['fail'],
        'purity': md_own['purity'],
        'impure_cells': md_own['impure_cells'],
        'n_cells': md_own['n_cells'],
        'examples': it_own['examples'][:20],
    }
    q_own.cells = None
    q_own.obs = None

    q = Qiyas(name, backoff=backoff)
    n_bad = 0
    for r, w, s in obs:
        if not q.observe(r, w, s):
            n_bad += 1
    q.induce()
    md = q.mundabit()
    it = q.ittirad()
    mk = q.munakis(min_cell=TH['min_cell_for_munakis'])
    res = {
        'illa': name,
        'backoff': backoff,
        'n_observed': len(obs),
        'n_admissible_asl': len(obs) - n_bad,
        'n_unalignable': n_bad,
        'ittirad_own_rule': own,
        'mundabit': {
            'determinacy': md['determinacy'],
            'n_roots': md['n_roots'],
            'n_cells': md['n_cells'],
            'n_observations': md['n_observations'],
            'purity': md['purity'],
            'impure_cells': md['impure_cells'],
            'impure_cell_share': md['impure_cells'] / max(md['n_cells'], 1),
            'distinct_alignment_histogram': md['distinct_alignment_histogram'],
        },
        'muttarid': {
            'n': it['n'], 'ok': it['ok'], 'fail': it['fail'],
            'ittirad_rate': it['ittirad_rate'],
            'examples': it['examples'][:20],
        },
        'munakis': {
            'n_cells': mk['n_cells'], 'in_n': mk['in_n'], 'in_acc': mk['in_acc'],
            'out_n': mk['out_n'], 'out_acc': mk['out_acc'], 'lift': mk['lift'],
            'worst_cells': mk['worst_cells'],
        },
    }
    # The verdict is taken on the ʿilla's OWN rule (no fall-back): that is the classical
    # question -- does the cause hold without exception?  The with-back-off rate is the
    # deployed behaviour and is reported alongside, not substituted for it.
    verdict_md = own['purity'] >= TH['mundabit_min_purity']
    verdict_it = own['ittirad_rate'] >= TH['muttarid_min_rate']
    if name == 'wazn_only':
        verdict_mk = False
        res['munakis']['note'] = ('a constant cause has no outside; every cell is universal, '
                                  'so the lift is 0 by construction')
        res['munakis']['lift'] = 0.0
    else:
        verdict_mk = mk['lift'] >= TH['munakis_min_lift']
    res['verdict'] = {
        'mundabit_pass': bool(verdict_md),
        'muttarid_pass': bool(verdict_it),
        'munakis_pass': bool(verdict_mk),
        'ACCEPTED': bool(verdict_md and verdict_it and verdict_mk),
        'thresholds': TH,
        'basis': ("muttarid/mundabit are judged on the ʿilla's own rule "
                  "(ittirad_own_rule), NOT on the back-off lattice"),
    }
    # free the tables
    q.cells = None
    q.obs = None
    return res


def holdout_experiment(obs, metas, hold_roots, types, baseline_ok, illa='strict7',
                       mode='replace', min_support=1, backoff=True):
    """Induce from asl whose root is NOT held out; realise the far' whose root IS held out."""
    hold = set(hold_roots)
    train_obs = [o for o in obs if o[0] not in hold]
    q = Qiyas(illa, backoff=backoff)
    n_bad = 0
    for r, w, s in train_obs:
        if not q.observe(r, w, s):
            n_bad += 1
    q.induce()

    far = [w for w in types if metas.get(w) and metas[w][1] in hold]
    stats = collections.Counter()
    prov_hist = collections.Counter()
    rows = []
    for w in far:
        p, r, wz, s = metas[w]
        base_ok = bool(baseline_ok.get(w, False))
        res = q.realize(r, wz)
        if res is None:
            stats['no_analogue'] += 1
            prov_hist['NONE'] += 1
            qy_ok = -1                       # not attempted
            prov, sup = 'NONE', 0
        else:
            qy, prov, sup, tot, dist = res
            prov_hist[prov.split(':')[0]] += 1
            pred = p + qy + s
            qy_ok = 1 if pred == w else 0
            stats['attempted'] += 1
            stats['qiyas_ok' if qy_ok else 'qiyas_fail'] += 1
            if qy_ok and base_ok:
                stats['both_ok'] += 1
            elif qy_ok and not base_ok:
                stats['qiyas_fixes'] += 1
            elif base_ok and not qy_ok:
                stats['qiyas_breaks'] += 1
            if len(rows) < 4000:
                rows.append({'word': w, 'prefix': p, 'root': r, 'wazn': wz, 'suffix': s,
                             'base_ok': base_ok, 'qiyas_ok': bool(qy_ok),
                             'qiyas': pred, 'provenance': prov, 'support': sup})
        if base_ok:
            stats['base_ok'] += 1
    return {
        'illa': illa, 'mode': mode, 'backoff': backoff,
        'n_asl_train': len(train_obs) - n_bad,
        'n_far_types': len(far),
        'far_types_with_analogue': stats['attempted'],
        'far_types_no_analogue': stats['no_analogue'],
        'provenance_histogram': dict(prov_hist),
        'baseline_default_ok': stats['base_ok'],
        'baseline_default_rate': stats['base_ok'] / max(len(far), 1),
        'qiyas_ok': stats['qiyas_ok'],
        'qiyas_rate_over_all_far': stats['qiyas_ok'] / max(len(far), 1),
        'qiyas_rate_over_attempted': (stats['qiyas_ok'] / stats['attempted']) if stats['attempted'] else None,
        'both_ok': stats['both_ok'],
        'qiyas_fixes_baseline_failures': stats['qiyas_fixes'],
        'qiyas_breaks_baseline_successes': stats['qiyas_breaks'],
        'examples_repaired': [r for r in rows if r['qiyas_ok'] and not r['base_ok']][:25],
        'examples_broken': [r for r in rows if r['base_ok'] and not r['qiyas_ok']][:25],
        'n_hold_roots': len(hold),
    }


STATIC_BASE_STEM = {}


def random_folds(obs, metas, types, baseline_ok, k=8, fold_size=37, seed=11):
    """The same root-holdout over random folds: is the result special to the 37 roots?"""
    rng = random.Random(seed)
    counts = collections.Counter()
    for w, (p, r, wz, s) in metas.items():
        if not r.startswith('<') and baseline_ok.get(w):
            counts[r] += 1
    pool = [r for r, n in counts.items() if n >= 3]
    rng.shuffle(pool)
    out = []
    for i in range(k):
        hold = pool[i * fold_size:(i + 1) * fold_size]
        if len(hold) < fold_size:
            break
        res = holdout_experiment(obs, metas, hold, types, baseline_ok)
        out.append({kk: res[kk] for kk in
                    ('n_far_types', 'baseline_default_ok', 'baseline_default_rate',
                     'far_types_with_analogue', 'far_types_no_analogue',
                     'qiyas_ok', 'qiyas_rate_over_all_far',
                     'qiyas_fixes_baseline_failures', 'qiyas_breaks_baseline_successes')})
        log('    fold %d: base %.3f  qiyas %.3f  (n=%d)'
            % (i, res['baseline_default_rate'], res['qiyas_rate_over_all_far'],
               res['n_far_types']))
    if out:
        import statistics as st
        agg = {m: {'mean': st.mean(x[m] for x in out),
                   'min': min(x[m] for x in out),
                   'max': max(x[m] for x in out)} for m in out[0]}
        return {'k': len(out), 'folds': out, 'agg': agg}
    return {'k': 0, 'folds': [], 'agg': {}}


def roundtrip_effect(vocab, types, baseline_ok, q, mode, min_support=1):
    """Type-level round-trip with the qiyas hook installed on the live tokenizer."""
    tok = vocab.base_tok
    qiyas_wire.install(tok, enabled=False, engine=q, mode=mode,
                       min_support=min_support, log_cap=400000)
    # flag OFF: must reproduce the recorded baseline exactly
    off_ok = 0
    off_mismatch = []
    for w in types:
        try:
            dec = vocab.decode_word(*vocab.encode_word(w))
        except Exception:
            dec = None
        ok = (dec == w)
        if ok:
            off_ok += 1
        if ok != bool(baseline_ok.get(w, False)):
            off_mismatch.append({'word': w, 'decoded': dec,
                                 'recorded': base_ok_val(baseline_ok, w)})
    qiyas_wire.reset_stats(tok, keep_log=False)
    qiyas_wire.set_enabled(tok, True)
    on_ok = 0
    changed = []
    changed_fix = changed_break = 0
    for w in types:
        try:
            dec = vocab.decode_word(*vocab.encode_word(w))
        except Exception:
            dec = None
        if dec == w:
            on_ok += 1
        b = bool(baseline_ok.get(w, False))
        if (dec == w) != b:
            if dec == w:
                changed_fix += 1
            else:
                changed_break += 1
            if len(changed) < 60:
                changed.append({'word': w, 'decoded': dec, 'baseline_ok': b})
    st = qiyas_wire.stats(tok)
    qiyas_wire.set_enabled(tok, False)
    off_ok2 = 0
    for w in types:
        try:
            if vocab.decode_word(*vocab.encode_word(w)) == w:
                off_ok2 += 1
        except Exception:
            pass
    qiyas_wire.uninstall(tok)
    return {
        'mode': mode, 'n_types': len(types),
        'baseline_recorded_ok': sum(1 for w in types if baseline_ok.get(w)),
        'flag_OFF_ok': off_ok,
        'flag_OFF_mismatches_vs_recorded': len(off_mismatch),
        'flag_OFF_mismatch_examples': off_mismatch[:20],
        'flag_ON_ok': on_ok,
        'flag_ON_after_uninstall_ok': off_ok2,
        'newly_fixed': changed_fix,
        'newly_broken': changed_break,
        'delta_types': on_ok - off_ok,
        'hook_stats': st,
        'changed_examples': changed,
    }


def base_ok_val(d, w):
    v = d.get(w)
    return bool(v) if v is not None else None


def main():
    os.makedirs(OUT, exist_ok=True)
    log('[*] loading the shipped vocabulary ...')
    vocab = load_vocab()
    log('    roots=%d awzan=%d prefixes=%d suffixes=%d'
        % (vocab.num_roots, vocab.num_awzan, vocab.num_prefixes, vocab.num_suffixes))

    du = json.load(open(DU_AFTER, encoding='utf-8'))
    raw = du['words']
    baseline_ok = {w: bool(v[0]) for w, v in raw.items()}
    types = list(raw.keys())
    log('[*] %d distinct held-out types; shipped default round-trips %d (%.4f %%)'
        % (len(types), sum(baseline_ok.values()),
           100 * sum(baseline_ok.values()) / len(types)))

    t0 = time.time()
    obs, metas, skipped = harvest(vocab, types, baseline_ok)
    log('[*] harvested %d attested asl in %.0fs; skipped %s'
        % (len(obs), time.time() - t0, skipped))

    roots_with_asl = {r for r, _, _ in obs}
    log('    distinct roots in the asl set: %d' % len(roots_with_asl))

    result = {
        'harness': 'qiyas_corpus.py',
        'built': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'vocab': {'roots': vocab.num_roots, 'awzan': vocab.num_awzan},
        'types': len(types),
        'baseline_note': ('baseline_ok is the per-type round-trip flag recorded by '
                          'eval_heldout.py AFTER the lisan_root_fix (du_after.json); the '
                          'in-harness flag-OFF pass reproduces it exactly (checked below)'),
        'n_asl': len(obs),
        'harvest_skipped': skipped,
        'n_roots_with_asl': len(roots_with_asl),
    }

    log('[*] measuring the three validity conditions per ʿilla ...')
    illas = {}
    for name in ('wazn_only', 'coarse_weak', 'strict7', 'positional', 'identity'):
        t = time.time()
        r = measure_illa(name, obs)
        illas[name] = r
        mm, tt, kk = r['mundabit'], r['muttarid'], r['munakis']
        log('    %-11s cells=%-6d purity=%.4f impure=%.3f  ittirad=%.4f (fail %d)  '
            'in=%.4f out=%.4f lift=%+.4f  -> %s  [%.0fs]'
            % (name, mm['n_cells'], mm['purity'], mm['impure_cell_share'],
               tt['ittirad_rate'], tt['fail'], kk['in_acc'], kk['out_acc'], kk['lift'],
               'ACCEPTED' if r['verdict']['ACCEPTED'] else 'REJECTED', time.time() - t))
    result['illas'] = illas
    accepted = [n for n, r in illas.items() if r['verdict']['ACCEPTED']]
    result['illas_accepted'] = accepted
    result['illas_rejected'] = [n for n in illas if n not in accepted]

    log('[*] derivational holdout: the 37 roots held out of the neural training ...')
    result['holdout_hold37'] = {
        'roots': HOLD37,
        'strict7_replace': holdout_experiment(obs, metas, HOLD37, types, baseline_ok,
                                              illa='strict7', mode='replace'),
        'positional_replace': holdout_experiment(obs, metas, HOLD37, types, baseline_ok,
                                                 illa='positional', mode='replace'),
        'coarse_weak_replace': holdout_experiment(obs, metas, HOLD37, types, baseline_ok,
                                                  illa='coarse_weak', mode='replace'),
        'identity_replace': holdout_experiment(obs, metas, HOLD37, types, baseline_ok,
                                               illa='identity', mode='replace'),
        'identity_NO_backoff': holdout_experiment(obs, metas, HOLD37, types, baseline_ok,
                                                  illa='identity', backoff=False),
    }
    for k, v in result['holdout_hold37'].items():
        if isinstance(v, dict):
            log('    %-20s far=%-5d base=%.4f qiyas=%.4f (attempted %s) fixes=%d breaks=%d no-analogue=%d prov=%s'
                % (k, v['n_far_types'], v['baseline_default_rate'],
                   v['qiyas_rate_over_all_far'],
                   ('%.4f' % v['qiyas_rate_over_attempted']) if v.get('qiyas_rate_over_attempted') is not None else 'n/a',
                   v['qiyas_fixes_baseline_failures'],
                   v['qiyas_breaks_baseline_successes'],
                   v['far_types_no_analogue'],
                   v.get('provenance_histogram')))

    log('[*] random 37-root folds ...')
    result['random_folds'] = random_folds(obs, metas, types, baseline_ok, k=8, fold_size=37)
    if 'agg' in result['random_folds']:
        for m, a in sorted(result['random_folds']['agg'].items()):
            log('    %-34s mean=%.4f  [%.4f, %.4f]' % (m, a['mean'], a['min'], a['max']))

    log('[*] building the full-corpus engine for the ablation ...')
    qfull = Qiyas('strict7', backoff=True)
    nb = 0
    for r, w, s in obs:
        if not qfull.observe(r, w, s):
            nb += 1
    qfull.induce()
    log('    engine induced over %d asl (%d unalignable)' % (len(obs) - nb, nb))
    for mode in ('fallback', 'replace'):
        t = time.time()
        rt = roundtrip_effect(vocab, types, baseline_ok, qfull, mode)
        result.setdefault('roundtrip', {})[mode] = rt
        log('    mode=%-9s OFF=%d ON=%d delta=%+d  (fixes %d / breaks %d)  hook=%s  [%.0fs]'
            % (mode, rt['flag_OFF_ok'], rt['flag_ON_ok'], rt['delta_types'],
               rt['newly_fixed'], rt['newly_broken'],
               {k: v for k, v in rt['hook_stats'].items()
                if k.startswith('n_') and k != 'n_logged'}, time.time() - t))
        log('      flag-OFF mismatches vs the recorded baseline: %d'
            % rt['flag_OFF_mismatches_vs_recorded'])

    with open(os.path.join(OUT, 'qiyas_results.json'), 'w', encoding='utf-8') as fh:
        json.dump(result, fh, ensure_ascii=False, indent=1)
    log('[*] wrote %s/qiyas_results.json' % OUT)


if __name__ == '__main__':
    main()
