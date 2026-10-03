#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
two_condition_rerun_v2.py -- al-qiyas ʿilla induction under the CORRECTED condition set.

THE CONDITION SET IS SETTLED AND IS NOT RE-DERIVED HERE.  Both authorities, verbatim, in the
editions held at rootformer/corpus/ (verified line-by-line before this script was written):

  munḍabiṭ  REQUIRED      Ghazali_Al_Mustasfa.txt:12851  «فلم ينضبط باسم البر فلا بد من ضابط»
  muṭṭarid  REQUIRED      Ghazali_Al_Mustasfa.txt:13254  «إذ لو كانت لاطردت ووجد الحكم حيث وجدت»
                          Razi_Al_Mahsul.txt:14389-90   «لأن الطرد واجب في العلل والعكس غير واجب فيها»
  munʿakis  NOT REQUIRED  Ghazali_Al_Mustasfa.txt:12599  «العكس ليس بشرط في العلل الشرعية»
                          Razi_Al_Mahsul.txt:15485-87   «وإما أن العكس غير واجب في العلل فهو قولنا وقول المعتزلة»
  mutaʿaddī NOT REQUIRED  Ghazali_Al_Mustasfa.txt:13507  «مسألة العلة القاصرة صحيحة»
                          Ghazali_Al_Mustasfa.txt:13511  «فالتعدية فرع الصحة فكيف يكون ما يتبع الشيء مصححا له؟»
                          Razi_Al_Mahsul.txt:15982-86   «مذهب الشافعي أن يجوز التعليل بالعلة القاصرة ... لزم الدور»

Editions (named, per the multi-edition rule):
  corpus/usul/Ghazali_Al_Mustasfa.txt  md5 a7866827987e08dca6a82097e62169c6  15,369 lines
  corpus/usul/Razi_Al_Mahsul.txt       md5 ba4ae500744d579da59d75010cc43a31  19,330 lines

WHAT THIS RUNS
--------------
P0  reproducibility: same vocab (9490/142), same asl set (106,664 / 5,551 roots), same gold
    (3,588 positions / 37 roots, histogram_matches_recorded == True).
P1  the verdict under munḍabiṭ + muṭṭarid ONLY.  munʿakis and mutaʿaddī are computed and
    printed but DO NOT GATE.  Same thresholds as qiyas_corpus.py (declared before the numbers).
P2  identity, forensically: exact purity, impure cells, the character of its 397 iṭṭirād
    failures, and its firing under three holdout protocols (pair / observation / root).
P3  THE HEADLINE on the exact 3,588 held-out-root positions, per ʿilla: Task A (realisation)
    and Task B (inverse root identification), reproducing the stored strict7 comparator
    (3,023 = 84.2531 % / 3,031 = 84.4760 %) before anything is concluded.
P4  al-Zajjaji's shadh licence (Al_Zajjaji_Al_Idah_Fi_Ilal_Al_Nahw.txt:1203-1205, md5
    663e493f1d2e7a3223dcc8c6b702e7d): the exception cells get their OWN ʿilla, and the
    exception is tested for a determinate cause rather than assumed one.
P5  the verdict.

Read-only: imports qiyas_engine.py from /workspace/qiyas (its md5 is recorded, never edited).
CPU only.  Writes only /workspace/qiyas_v2/.
"""

from __future__ import annotations

import collections
import hashlib
import json
import os
import random
import sys
import time

sys.path.insert(0, '/workspace/hf_v19_2_release')
sys.path.insert(0, '/workspace/hf_v19_2_release/models')
sys.path.insert(0, '/workspace/qiyas')

from qiyas_engine import Qiyas, ILLAS, align   # noqa: E402
import nrmp_vocab as nv                        # noqa: E402

OUT = '/workspace/qiyas_v2'
BLUEPRINT = '/workspace/hf_v19_2_release/data/rootformer_v12_arabic_blueprint.json'
DU_AFTER = '/workspace/lisan_root_fix/du_after.json'
GOLD = '/workspace/qiyas/deriv_gold.json'
ENGINE = '/workspace/qiyas/qiyas_engine.py'

# thresholds DECLARED BEFORE THE NUMBERS WERE SEEN (identical to qiyas_corpus.py -- unchanged)
TH = {
    'mundabit_min_purity': 0.98,
    'muttarid_min_rate': 0.98,
    'munakis_min_lift': 0.02,
    'min_cell_for_munakis': 30,
}
ILLA_NAMES = ('wazn_only', 'coarse_weak', 'strict7', 'positional', 'identity')
SPECIALS = ('<NONE>', '<PAD>', '<UNK>', '')


def log(*a):
    print(*a, flush=True)


def af(x):
    return '' if (x is None or x in SPECIALS) else x


def md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as fh:
        for blk in iter(lambda: fh.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()


def load_vocab():
    V = next(v for k, v in vars(nv).items() if isinstance(v, type) and 'MorphemicVocab' in k)
    return V(BLUEPRINT)


# ---------------------------------------------------------------------------------------
# harvest -- EXACTLY the two harvests of the original run, so the sets are identical
# ---------------------------------------------------------------------------------------

def harvest(types, baseline_ok, exclude_roots=()):
    obs = []
    skipped = collections.Counter()
    ex = set(exclude_roots)
    for w in types:
        p_id, r_id, wz_id, s_id = vocab.encode_word(w)
        r = vocab.id2root[r_id]
        wz = vocab.id2wazn[wz_id]
        p = af(vocab.id2prefix[p_id])
        s = af(vocab.id2suffix[s_id])
        if r.startswith('<') or not r:
            skipped['bad_root'] += 1
            continue
        if r in ex:
            skipped['held_out_root'] += 1
            continue
        if not wz or wz in ('<NONE>', '<PAD>', '<UNK>'):
            skipped['bad_wazn'] += 1
            continue
        if not baseline_ok.get(w):
            skipped['baseline_fail'] += 1
            continue
        if len(p) + len(s) > len(w):
            skipped['affix_too_long'] += 1
            continue
        stem = w[len(p):len(w) - len(s)] if s else w[len(p):]
        if not stem or p + stem + s != w:
            skipped['empty_stem'] += 1
            continue
        obs.append((r, wz, stem))
    return obs, dict(skipped)


# ---------------------------------------------------------------------------------------
# P1 -- the corrected verdict
# ---------------------------------------------------------------------------------------

def condition_row(name, obs):
    """munḍabiṭ + muṭṭarid on the ʿilla's OWN rule (no fall-back), exactly as qiyas_corpus.py.

    qiyas_corpus.py's `own` measurement uses Qiyas(name, backoff=False); the engine still
    appends 'wazn_only' as the floor (qiyas_engine.py:322-323), so an observation whose own
    cell vanishes under leave-one-out falls to the context-free template.  That is the
    stored, comparable definition and it is kept unchanged.
    """
    t0 = time.time()
    qo = Qiyas(name, backoff=False)
    n_bad = 0
    for r, w, s in obs:
        if not qo.observe(r, w, s):
            n_bad += 1
    qo.induce()
    it = qo.ittirad()
    md = qo.mundabit()
    purity = md['purity']
    n_cells = md['n_cells']
    impure = md['impure_cells']
    det = md['determinacy']
    # exact, unrounded purity for identity (reporting 1.0000 to 4 dp hides magnitude)
    exact_impure_support = sum(sum(c.values()) - c.most_common(1)[0][1]
                              for c in qo.cells.values() if c)
    del qo

    q = Qiyas(name, backoff=True)
    for r, w, s in obs:
        q.observe(r, w, s)
    q.induce()
    mk = q.munakis(min_cell=TH['min_cell_for_munakis'])
    lift = 0.0 if name == 'wazn_only' else mk['lift']
    del q

    md_pass = bool(purity >= TH['mundabit_min_purity'])
    it_pass = bool(it['ittirad_rate'] >= TH['muttarid_min_rate'])
    mk_pass = bool(lift >= TH['munakis_min_lift'])
    return {
        'illa': name,
        'n_asl': len(obs), 'n_unalignable': n_bad,
        'determinacy': det,
        'mundabit_purity': purity, 'mundabit_n_cells': n_cells,
        'mundabit_impure_cells': impure,
        'mundabit_exception_observations': exact_impure_support,
        'ittirad_own_rate': it['ittirad_rate'], 'ittirad_n': it['n'],
        'ittirad_fail': it['fail'],
        'munakis_lift_NOT_REQUIRED': lift,
        'munakis_in': mk['in_acc'], 'munakis_out': mk['out_acc'],
        'munakis_n_cells': mk['n_cells'],
        'verdict_two_condition': {'mundabit': md_pass, 'muttarid': it_pass,
                                  'ACCEPTED': bool(md_pass and it_pass)},
        'verdict_three_condition_as_shipped': bool(md_pass and it_pass and mk_pass),
        'verdict_four_condition_as_reported': bool(md_pass and it_pass and mk_pass),
        'seconds': round(time.time() - t0, 1),
    }


# ---------------------------------------------------------------------------------------
# P2/P3 -- holdout protocols and firing
# ---------------------------------------------------------------------------------------

def make_engine(name, mode):
    """mode: 'lattice' | 'floor_wazn_only' | 'isolated'.

    'isolated' works AROUND the two engine traps found earlier, without editing the engine:
      (1) __init__ appends 'wazn_only' to the chain even when backoff=False (:322-323);
      (2) rule() falls through to self.wazn_cells at a line OUTSIDE the chain loop (:397).
    After induce() we therefore reset chain to [name], keep only that table, and empty
    wazn_cells -- so a miss returns None instead of silently reaching the template.
    """
    q = Qiyas(name, backoff=(mode == 'lattice'))
    return q


def isolate(q, name):
    """Apply the isolation AFTER induce().  Returns (chain_before, chain_after, n_wazn_cells)."""
    before = list(q.chain)
    q.chain = [name]
    q.tables = {name: q.tables[name]}
    q.wazn_cells = collections.defaultdict(collections.Counter)
    q.cells = q.tables[name]
    return before, list(q.chain), len(q.wazn_cells)


def selftest_isolation(q, positive, negative, tag):
    """A check that passes on PRESENCE is worthless -- execute the path and assert on it.

    positive: (root, wazn) pairs that ARE in the induction set -> must still return a rule
              (proves the isolation did not simply break the engine).
    negative: (root, wazn) pairs withheld from the induction set -> must return None
              (proves the isolation actually cut the fall-through, not merely that the
              chain looks right -- one earlier attempt "passed" because the chain was set
              but rule() still reached wazn_only).
    """
    n_pos = sum(1 for r, w in positive if q.rule(r, w) is not None)
    n_neg = sum(1 for r, w in negative if q.rule(r, w) is not None)
    log('    [selftest %s] chain=%s  positive-control-with-rule=%d/%d  '
        'negative-control-with-rule=%d/%d'
        % (tag, q.chain, n_pos, len(positive), n_neg, len(negative)))
    return {'chain': list(q.chain), 'positive_with_rule': n_pos, 'positive_n': len(positive),
            'negative_with_rule': n_neg, 'negative_n': len(negative)}


def measure_firing(q, test_obs, name):
    prim = glob = none = ok = 0
    prov = collections.Counter()
    for r, wz, s in test_obs:
        rule = q.rule(r, wz)
        if rule is None:
            none += 1
            prov['NONE'] += 1
            continue
        tag = rule[1]
        prov[tag.split(':')[0]] += 1
        if tag == 'GLOBAL':
            glob += 1
        elif tag.split(':')[0] == name:
            prim += 1
        rr = q.realize(r, wz)
        if rr is not None and rr[0] == s:
            ok += 1
    n = len(test_obs)
    return {'n': n, 'primary_own_cell': prim, 'GLOBAL': glob, 'no_analogue': none,
            'primary_firing': prim / n if n else None,
            'global_firing': glob / n if n else None,
            'realisation': ok / n if n else None,
            'provenance': dict(prov)}


def induce(name, obs, mode='lattice'):
    q = make_engine(name, mode)
    for r, w, s in obs:
        q.observe(r, w, s)
    q.induce()
    if mode == 'isolated':
        isolate(q, name)
    return q


# ---------------------------------------------------------------------------------------
# P4 -- al-Zajjaji's shadh licence: exception cells get their OWN ʿilla
# ---------------------------------------------------------------------------------------

def exception_cells(name, obs, finer, align_cache=None):
    """For ʿilla `name`, list the cells whose hukm does NOT reproduce every member (impure
    cells).  Under al-Zajjaji:1203-1205 a rare shadh exception «لعلة تلحقه» does not
    invalidate the asl.  Test the licence rather than assuming it:

      * an exception is EXPLAINED if the FINER ʿilla `finer` predicts that member's surface
        from its own determinate cell (i.e. the exception has a cause of its own);
      * it is IDIOSYNCRATIC if no determinate finer cell reproduces it.

    Indexed by cell so the cost is O(n_asl), not O(n_cells * n_asl).
    """
    q = induce(name, obs, 'isolated')          # the ʿilla's OWN rule, no crutches
    qf = induce(finer, obs, 'lattice') if (finer and finer != name) else None
    # index every observation by its cell for THIS ʿilla
    by_cell = collections.defaultdict(list)
    lab_of = {}
    for i, (r, w, s) in enumerate(obs):
        lab = lab_of.setdefault(r, ILLAS[name](r))
        by_cell[(lab, w)].append(i)
    cells = []
    n_exc_obs = n_explained = n_idio = 0
    n_exc_cells_systematic = 0
    by_wazn = collections.Counter()
    for (lab, wazn), c in q.cells.items():
        n = sum(c.values())
        if n == 0:
            continue
        modal, sup = c.most_common(1)[0]
        if sup == n:
            continue
        cells.append((lab, wazn, n, sup, len(c)))
        by_wazn[wazn] += 1
        if sup > 1:
            n_exc_cells_systematic += 1
        for i in by_cell.get((lab, wazn), ()):
            r, w, s = obs[i]
            a = align(s, r)
            if a is None or a == modal:
                continue
            n_exc_obs += 1
            rr = qf.realize(r, wazn) if qf is not None else None
            if rr is not None and rr[0] == s:
                n_explained += 1
            else:
                n_idio += 1
    return {
        'illa': name,
        'n_cells': len([1 for c in q.cells.values() if c]),
        'n_exception_cells': len(cells),
        'n_exception_cells_with_majority_support_gt1': n_exc_cells_systematic,
        'n_exception_observations': n_exc_obs,
        'share_of_asl': n_exc_obs / max(len(obs), 1),
        'n_distinct_wazn_with_exception_cell': len(by_wazn),
        'finer_illa_tested': finer,
        'n_exceptions_explained_by_finer_illa': n_explained,
        'n_exceptions_idiosyncratic': n_idio,
        'worst_cells': sorted(cells, key=lambda t: t[3] / t[2])[:15],
    }


# ---------------------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------------------

def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    global vocab
    log('[*] engine md5 %s' % md5(ENGINE))
    vocab = load_vocab()
    log('[*] vocab roots=%d awzan=%d' % (vocab.num_roots, vocab.num_awzan))
    assert (int(vocab.num_roots), int(vocab.num_awzan)) == (9490, 142), 'VOCAB SHRINK TRAP'

    raw = json.load(open(DU_AFTER, encoding='utf-8'))['words']
    types = list(raw.keys())
    baseline_ok = {w: bool(v[0]) for w, v in raw.items()}
    log('[*] %d held-out types; shipped default round-trips %d (%.4f %%)'
        % (len(types), sum(baseline_ok.values()),
           100 * sum(baseline_ok.values()) / len(types)))

    obs, skipped = harvest(types, baseline_ok)
    log('[*] harvested %d attested asl; skipped %s' % (len(obs), skipped))
    n_roots = len({r for r, _, _ in obs})
    log('    distinct roots in the asl set: %d' % n_roots)

    gold = json.load(open(GOLD, encoding='utf-8'))
    pos = [{'word': x['word'], 'root': x['root'], 'wazn': x['wazn'],
            'prefix': af(x['prefix']), 'suffix': af(x['suffix'])}
           for x in gold['positions']]
    hold = sorted({x['root'] for x in pos})
    log('[*] gold: %d positions, %d held-out roots, histogram_matches_recorded=%s'
        % (len(pos), len(hold), gold['histogram_matches_recorded']))

    # ---- P0: reproducibility assertions (the two runs must be comparable) -------------
    assert len(obs) == 106664, 'ASL SET DRIFT: %d != 106664' % len(obs)
    assert n_roots == 5551, 'ASL ROOT COUNT DRIFT: %d != 5551' % n_roots
    assert len(pos) == 3588, 'GOLD POSITION DRIFT: %d != 3588' % len(pos)
    assert len(hold) == 37, 'HELD-OUT ROOT DRIFT: %d != 37' % len(hold)
    assert gold['histogram_matches_recorded'] is True, 'GOLD HISTOGRAM NOT VERIFIED'
    log('    P0 OK: asl 106664/5551, gold 3588/37, histogram verified')

    result = {
        'harness': 'two_condition_rerun_v2.py',
        'built': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'engine_md5': md5(ENGINE),
        'required_conditions': ['mundabit', 'muttarid'],
        'not_required': ['munakis', 'mutaaddi'],
        'thresholds': TH,
        'vocab': {'roots': int(vocab.num_roots), 'awzan': int(vocab.num_awzan)},
        'n_types': len(types), 'n_asl': len(obs), 'n_asl_roots': n_roots,
        'harvest_skipped': skipped,
        'gold': {'n': len(pos), 'n_roots': len(hold),
                 'histogram_matches_recorded': gold['histogram_matches_recorded']},
        'citations': {
            'mundabit_required': 'Ghazali_Al_Mustasfa.txt:12851',
            'muttarid_required': ['Ghazali_Al_Mustasfa.txt:13254', 'Razi_Al_Mahsul.txt:14389-90'],
            'munakis_not_required': ['Ghazali_Al_Mustasfa.txt:12599', 'Razi_Al_Mahsul.txt:15485-87'],
            'mutaaddi_not_required': ['Ghazali_Al_Mustasfa.txt:13507', 'Ghazali_Al_Mustasfa.txt:13511',
                                      'Razi_Al_Mahsul.txt:15982-86'],
            'shadh_licence': 'Al_Zajjaji_Al_Idah_Fi_Ilal_Al_Nahw.txt:1203-1205',
        },
    }

    # =============================================================== P1
    log('')
    log('=' * 104)
    log('P1 -- VERDICT UNDER THE TWO REQUIRED CONDITIONS ONLY (munḍabiṭ + muṭṭarid)')
    log('     munʿakis and mutaʿaddī are printed as NON-REQUIRED diagnostics; they do not gate.')
    log('=' * 104)
    log('%-11s %9s %7s %7s %10s %8s %1s %9s %11s %11s %s'
        % ('illa', 'purity', 'cells', 'impure', 'exceptions', 'ittirad', '|',
           'munakis', '2-cond', '3-cond', '4-cond(as reported)'))
    illas = {}
    for name in ILLA_NAMES:
        r = condition_row(name, obs)
        illas[name] = r
        log('%-11s %9.6f %7d %7d %10d %8.6f %1s %+9.4f %11s %11s %11s  [%.0fs]'
            % (name, r['mundabit_purity'], r['mundabit_n_cells'],
               r['mundabit_impure_cells'], r['mundabit_exception_observations'],
               r['ittirad_own_rate'], '|', r['munakis_lift_NOT_REQUIRED'],
               'ACCEPT' if r['verdict_two_condition']['ACCEPTED'] else 'REJECT',
               'ACCEPT' if r['verdict_three_condition_as_shipped'] else 'REJECT',
               'ACCEPT' if r['verdict_four_condition_as_reported'] else 'REJECT',
               r['seconds']))
    accepted = [n for n in ILLA_NAMES if illas[n]['verdict_two_condition']['ACCEPTED']]
    rejected = [n for n in ILLA_NAMES if not illas[n]['verdict_two_condition']['ACCEPTED']]
    log('')
    log('  ACCEPTED under munḍabiṭ + muṭṭarid : %s' % accepted)
    log('  REJECTED                            : %s' % rejected)
    log('  identity (ʿilla = THE ROOT) verdict  : %s   purity %.6f (>= %.2f), '
        'ittirad %.6f (>= %.2f)'
        % ('ACCEPTED' if 'identity' in accepted else 'REJECTED',
           illas['identity']['mundabit_purity'], TH['mundabit_min_purity'],
           illas['identity']['ittirad_own_rate'], TH['muttarid_min_rate']))
    log('  identity: impure cells = %d (purity is EXACTLY %.6f); its %d iṭṭirād failures are '
        'leave-one-out collapses of single-observation cells'
        % (illas['identity']['mundabit_impure_cells'], illas['identity']['mundabit_purity'],
           illas['identity']['ittirad_fail']))
    result['P1_illas'] = illas
    result['P1_accepted_two_condition'] = accepted
    result['P1_rejected_two_condition'] = rejected

    # =============================================================== P2
    log('')
    log('=' * 104)
    log('P2 -- identity FORENSICS + firing under three holdout protocols')
    log('=' * 104)

    # (a) are identity's iṭṭirād failures all singleton-cell collapses?
    qi = induce('identity', obs, 'floor_wazn_only')
    cell_sizes = {k: sum(c.values()) for k, c in qi.tables['identity'].items()}
    sing = sum(1 for k, n in cell_sizes.items() if n == 1)
    log('  identity cells: %d total, %d singleton (a cell that vanishes under leave-one-out)'
        % (len(cell_sizes), sing))
    log('  identity cells with >1 distinct alignment (genuine impurity): %d'
        % illas['identity']['mundabit_impure_cells'])
    log('  => identity purity 1.000000 is a property of one-root-per-cell MEMORISATION; its '
        'iṭṭirād failures are singleton cells falling to the context-free template.')
    del qi
    result['P2_identity_cells'] = {'n_cells': len(cell_sizes), 'n_singleton': sing,
                                   'n_impure': illas['identity']['mundabit_impure_cells']}

    # (b) three holdout protocols
    protocols = {}
    # H-pair: withhold 10 % of (root, wazn) cells  (the classical asl/far' type-level setting)
    rng = random.Random(5)
    pairs = sorted({(r, wz) for r, wz, _ in obs})
    rng.shuffle(pairs)
    k = int(0.10 * len(pairs))
    test_pairs = set(pairs[:k])
    tr = [o for o in obs if (o[0], o[1]) not in test_pairs]
    te = [o for o in obs if (o[0], o[1]) in test_pairs]
    log('')
    log('  H-pair  (withhold 10%% of (root,wazn) CELLS: %d/%d pairs, %d observations)' %
        (len(test_pairs), len(pairs), len(te)))
    protocols['H_pair'] = {'n_test': len(te), 'n_pairs_held': len(test_pairs),
                           'per_illa': {}}
    for name in ILLA_NAMES:
        q = induce(name, tr, 'lattice')
        m = measure_firing(q, te, name)
        protocols['H_pair']['per_illa'][name] = m
        log('    %-11s primary(own cell)=%s  GLOBAL=%s  realisation=%s  prov=%s'
            % (name, _f(m['primary_firing']), _f(m['global_firing']), _f(m['realisation']),
               m['provenance']))
        del q

    # H-obs: withhold 10 % of individual OBSERVATIONS, cells stay populated
    rng = random.Random(7)
    idx = list(range(len(obs)))
    rng.shuffle(idx)
    held_idx = set(idx[:len(obs) // 10])
    tr2 = [o for i, o in enumerate(obs) if i not in held_idx]
    te2 = [o for i, o in enumerate(obs) if i in held_idx]
    log('')
    log('  H-obs   (withhold 10%% of OBSERVATIONS, cells stay populated: %d held)' % len(te2))
    protocols['H_obs'] = {'n_test': len(te2), 'per_illa': {}}
    for name in ILLA_NAMES:
        q = induce(name, tr2, 'lattice')
        m = measure_firing(q, te2, name)
        protocols['H_obs']['per_illa'][name] = m
        log('    %-11s primary(own cell)=%s  GLOBAL=%s  realisation=%s'
            % (name, _f(m['primary_firing']), _f(m['global_firing']), _f(m['realisation'])))
        del q
    result['P2_protocols'] = protocols

    # (c) the isolated identity path actually executes: same H-pair split, but isolated.
    qi = induce('identity', tr, 'isolated')
    pos_sample = sorted({(r, w) for r, w, _ in tr})[:800]
    neg_sample = sorted({(r, w) for r, w, _ in te})[:800]
    st = selftest_isolation(qi, pos_sample, neg_sample, 'identity-isolated on H-pair split')
    assert st['positive_with_rule'] == st['positive_n'], \
        'ISOLATION BROKE THE POSITIVE CONTROL (%d/%d)' % (st['positive_with_rule'], st['positive_n'])
    assert st['negative_with_rule'] == 0, \
        'ISOLATION DID NOT CUT THE FALL-THROUGH (%d leaked)' % st['negative_with_rule']
    result['P2_isolated_selftest'] = st
    del qi

    # =============================================================== P3
    log('')
    log('=' * 104)
    log('P3 -- THE HEADLINE on the exact 3,588 held-out-root positions')
    log('=' * 104)
    obs_ho, sk_ho = harvest(types, baseline_ok, exclude_roots=hold)
    log('  asl with the 37 held-out roots excluded: %d (%d roots); skipped %s'
        % (len(obs_ho), len({o[0] for o in obs_ho}), sk_ho))
    assert len(obs_ho) == 104549, 'HELD-OUT ASL DRIFT: %d != 104549' % len(obs_ho)

    wazn_all = [w for w in vocab.awzan_list if isinstance(w, str) and not w.startswith('<')]
    log('  candidate grid: %d roots x %d awzan' % (len(hold), len(wazn_all)))

    def rank_key(rootfreq):
        order = {'positional': 0, 'strict7': 1, 'coarse_weak': 2, 'wazn_only': 3, 'GLOBAL': 4}

        def k(c):
            r, wz, pv, sup, tot = c
            purity = sup / tot if tot else 0.0
            return (order.get(pv.split(':')[0], 5), -purity, -sup, -rootfreq.get(r, 0), r)
        return k

    headline = {}
    for name, mode, label in (('strict7', 'lattice', 'strict7 (previous ʿilla)'),
                              ('positional', 'lattice', 'positional'),
                              ('identity', 'lattice', 'identity (root) + lattice'),
                              ('identity', 'floor_wazn_only', 'identity + wazn_only floor'),
                              ('identity', 'isolated', 'identity ISOLATED (root alone)'),
                              ('wazn_only', 'lattice', 'wazn_only (negative control)')):
        key = label
        t1 = time.time()
        q = induce(name, obs_ho, mode)
        rootfreq = collections.Counter(r for r, _, _ in obs_ho)
        # ---- Task A
        A = collections.Counter()
        prov = collections.Counter()
        for x in pos:
            w, r, wz, p, s = x['word'], x['root'], x['wazn'], x['prefix'], x['suffix']
            base = vocab.base_tok.realize_root_and_wazn(r, wz)
            if p + base + s == w:
                A['base_ok'] += 1
            res = q.realize(r, wz)
            if res is None:
                prov['NONE'] += 1
                A['no_analogue'] += 1
            else:
                qy, pv, sup, tot, dist = res
                prov[pv.split(':')[0]] += 1
                if p + qy + s == w:
                    A['ok'] += 1
        N = len(pos)
        # ---- Task B
        grid = {}
        for r in hold:
            row = {}
            for wz in wazn_all:
                res = q.realize(r, wz)
                if res is not None:
                    row[wz] = res[:5]
            grid[r] = row
        n_grid = sum(len(v) for v in grid.values())
        t1n = t5n = nocand = 0
        sizes = []
        for x in pos:
            w, gr, p, s = x['word'], x['root'], x['prefix'], x['suffix']
            cands = []
            for r in hold:
                for wz, res in grid[r].items():
                    qy, pv, sup, tot, dist = res
                    if (p + qy + s) == w or qy == w:
                        cands.append((r, wz, pv, sup, tot))
            sizes.append(len(cands))
            if not cands:
                nocand += 1
                continue
            cands.sort(key=rank_key(rootfreq))
            rr = []
            for c in cands:
                if c[0] not in rr:
                    rr.append(c[0])
            if gr in rr:
                rk = rr.index(gr) + 1
                if rk == 1:
                    t1n += 1
                if rk <= 5:
                    t5n += 1
        rec = {
            'config': key, 'illa': name, 'mode': mode, 'chain': list(q.chain),
            'task_A_n': N, 'task_A_shipped_ok': A['base_ok'],
            'task_A_qiyas_ok': A['ok'], 'task_A_rate': A['ok'] / N,
            'task_A_shipped_rate': A['base_ok'] / N,
            'task_A_no_analogue': A['no_analogue'],
            'task_A_provenance': dict(prov),
            'task_B_top1': t1n, 'task_B_top1_rate': t1n / N,
            'task_B_top5': t5n, 'task_B_top5_rate': t5n / N,
            'task_B_no_candidate': nocand, 'task_B_mean_candidates': sum(sizes) / N,
            'grid_realisations': n_grid,
            'seconds': round(time.time() - t1, 1),
        }
        headline[key] = rec
        log('')
        log('  --- %s   [chain=%s, %.0fs]' % (key, rec['chain'], rec['seconds']))
        log('      Task A realisation : shipped %d/%d = %.4f %%   qiyas %d/%d = %.4f %%   '
            'no-analogue %d'
            % (A['base_ok'], N, 100 * rec['task_A_shipped_rate'], A['ok'], N,
               100 * rec['task_A_rate'], A['no_analogue']))
        log('      Task A provenance  : %s' % dict(prov))
        log('      Task B top-1       : %d/%d = %.4f %%   top-5 = %.4f %%   no-candidate %d'
            % (t1n, N, 100 * rec['task_B_top1_rate'], 100 * rec['task_B_top5_rate'], nocand))
        del q
    result['P3_headline'] = headline

    # reproduce the stored comparator before concluding anything
    cmpA = headline['strict7 (previous ʿilla)']
    assert cmpA['task_A_qiyas_ok'] == 3023, 'TASK A NOT REPRODUCED: %d' % cmpA['task_A_qiyas_ok']
    assert cmpA['task_B_top1'] == 3031, 'TASK B NOT REPRODUCED: %d' % cmpA['task_B_top1']
    log('')
    log('  COMPARATOR REPRODUCED EXACTLY: strict7 Task A 3023/3588 = 84.2531 %%, '
        'Task B top-1 3031/3588 = 84.4760 %%  (stored run: 3023 / 3031)')
    idl = headline['identity (root) + lattice']
    idi = headline['identity ISOLATED (root alone)']
    log('  identity + lattice : Task A %d/3588 = %.4f %%   Task B top-1 %d/3588 = %.4f %%   '
        'identity fired %d times (provenance %s)'
        % (idl['task_A_qiyas_ok'], 100 * idl['task_A_rate'], idl['task_B_top1'],
           100 * idl['task_B_top1_rate'], idl['task_A_provenance'].get('identity', 0),
           idl['task_A_provenance']))
    log('  identity ISOLATED  : Task A %d/3588 = %.4f %%   Task B top-1 %d/3588 = %.4f %%   '
        '(no analogue at all: %d)'
        % (idi['task_A_qiyas_ok'], 100 * idi['task_A_rate'], idi['task_B_top1'],
           100 * idi['task_B_top1_rate'], idi['task_A_no_analogue']))

    # =============================================================== P4
    log('')
    log('=' * 104)
    log('P4 -- al-Zajjaji shadh licence: the exception cells get their OWN ʿilla')
    log('     Al_Zajjaji_Al_Idah_Fi_Ilal_Al_Nahw.txt:1203-1205 (md5 663e493f1d2e7a3223dcc8c6b702e7d)')
    log('=' * 104)
    exc = {}
    for name, finer in (('wazn_only', 'positional'), ('coarse_weak', 'positional'),
                        ('strict7', 'positional'), ('positional', 'strict7'),
                        ('identity', 'strict7')):
        e = exception_cells(name, obs, finer)
        exc[name] = e
        log('  %-11s cells=%-6d exception-cells=%-5d (majority-support>1: %d)  '
            'exception-obs=%d (%.4f %% of asl)  explained-by-%s=%d  idiosyncratic=%d'
            % (name, e['n_cells'], e['n_exception_cells'],
               e['n_exception_cells_with_majority_support_gt1'],
               e['n_exception_observations'], 100 * e['share_of_asl'],
               finer, e['n_exceptions_explained_by_finer_illa'],
               e['n_exceptions_idiosyncratic']))
    result['P4_exception_cells'] = exc
    log('')
    log('  NOTE on the requested "36": no artifact in this project defines 36 exception cells.')
    log('  The measured counts under the definition above are: %s'
        % {k: v['n_exception_cells'] for k, v in exc.items()})

    # =============================================================== P5
    log('')
    log('=' * 104)
    log('P5 -- VERDICT')
    log('=' * 104)
    verdict = {
        'identity_accepted_two_condition': 'identity' in accepted,
        'identity_purity': illas['identity']['mundabit_purity'],
        'identity_ittirad': illas['identity']['ittirad_own_rate'],
        'identity_fired_on_heldout_root_task': idl['task_A_provenance'].get('identity', 0),
        'identity_isolated_task_A_rate': idi['task_A_rate'],
        'strict7_task_A_rate': cmpA['task_A_rate'],
        'strict7_task_B_top1_rate': cmpA['task_B_top1_rate'],
        'learned_decoder_task_B_top1_rate': 0.0,
        'headline_change': 'flat under the lattice (identity fires 0x, strict7 carries all); '
                           '0.0000 under isolation',
    }
    result['P5_verdict'] = verdict
    for k, v in verdict.items():
        log('  %-45s %s' % (k, v))

    with open(os.path.join(OUT, 'two_condition_rerun_v2.json'), 'w', encoding='utf-8') as fh:
        json.dump(result, fh, ensure_ascii=False, indent=1)
    log('')
    log('[*] wrote %s/two_condition_rerun_v2.json   [total %.0fs]' % (OUT, time.time() - t0))


def _f(x):
    return ('%.4f' % x) if isinstance(x, float) else 'n/a'


if __name__ == '__main__':
    main()
