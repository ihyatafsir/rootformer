#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
two_condition_rerun.py -- re-run the qiyas ʿilla induction with the REQUIRED conditions only.

WHY
---
STATE.md (verified from both authorities verbatim) establishes that al-ʿilla requires exactly
TWO conditions -- munḍabiṭ and muṭṭarid.  munʿakis and mutaʿaddī are NOT required:

  Mustasfa:12599  «فزيادة العكس لا تؤثر؛ لأن العكس ليس بشرط في العلل الشرعية»
  Mahsul:15485    «وإما أن العكس غير واجب في العلل فهو قولنا وقول المعتزلة»
  Mustasfa:13507  «مسألة العلة القاصرة صحيحة ... وذهب أبو حنيفة إلى إبطالها»
  Mahsul:15982    «مذهب الشافعي أن يجوز التعليل بالعلة القاصرة وهو قول أكثر المتكلمين»

All four were re-opened line by line before this script was written (STATE.md abbreviates
three of them and mis-transcribes one word of the Mahsul:15485 line as «المعتقل»; the text
reads «المعتزلة»).  Editions: corpus/usul/Razi_Al_Mahsul.txt (md5 ba4ae500744d579da59d75010cc43a31,
19,330 lines) and corpus/usul/Ghazali_Al_Mustasfa.txt.

QIYAS_REPORT.md marked 'illa = identity (the root itself)' REJECTED by a FOURTH condition
(mutaʿaddī) citing Mahsul:15719-15721.  The code never enforced that: in the stored
qiyas_results.json, identity already carries verdict.ACCEPTED = True (munḍabiṭ ✓, muṭṭarid ✓,
munʿakis ✓).  The rejection lived in report prose plus a separate TADDI measurement block.

WHAT THIS SCRIPT DOES -- and does NOT do
----------------------------------------
It re-runs the induction and prints, for every 'illa, the verdict under the TWO REQUIRED
conditions, with munʿakis and mutaʿaddī shown alongside as NON-REQUIRED diagnostics.
It does NOT re-tune any threshold and does NOT edit qiyas_engine.py.

It then measures -- as a MEASUREMENT, not as a validity condition -- whether the root-as-'illa
actually carries a ruling to a far' whose (root, wazn) cell was withheld.  A 'illa that is
accepted by the two conditions and still carries nothing is not "rejected by mutaʿaddī"; it is
not a 'illa at all, it is the AṢL.  That distinction is the one the report needed.

CPU only.  Reads /workspace only.  Writes /workspace/qiyas/two_condition_rerun.json.
"""

from __future__ import annotations

import collections
import json
import random
import sys
import time

sys.path.insert(0, '/workspace/hf_v19_2_release')
sys.path.insert(0, '/workspace/hf_v19_2_release/models')
sys.path.insert(0, '/workspace/qiyas')

from qiyas_engine import Qiyas, ILLAS          # noqa: E402
import nrmp_vocab as nv                        # noqa: E402

V = nv.FarāhīdianMorphemicVocab      # canonical accessor (module md5 0ee0ab6301d7f26eddc7bd079f7fede7)
BLUEPRINT = '/workspace/hf_v19_2_release/data/rootformer_v12_arabic_blueprint.json'
DU_AFTER = '/workspace/lisan_root_fix/du_after.json'
OUT = '/workspace/qiyas/two_condition_rerun.json'

# thresholds DECLARED BEFORE SEEING THE NUMBERS (unchanged from qiyas_corpus.py)
TH = {
    'mundabit_min_purity': 0.98,
    'muttarid_min_rate': 0.98,
    'munakis_min_lift': 0.02,
    'min_cell_for_munakis': 30,
}
ILLA_NAMES = ('wazn_only', 'coarse_weak', 'strict7', 'positional', 'identity')


def log(*a):
    print(*a, flush=True)


def af(x):
    return '' if (x is None or x in ('<NONE>', '<PAD>', '<UNK>')) else x


def harvest(vocab, types, baseline_ok):
    """EXACTLY the harvest of qiyas_corpus.py, so the aṣl set is the same 106,664."""
    obs, skipped = [], collections.Counter()
    for w in types:
        p_id, r_id, wz_id, s_id = vocab.encode_word(w)
        r = vocab.id2root[r_id]
        wz = vocab.id2wazn[wz_id]
        p = af(vocab.id2prefix[p_id])
        s = af(vocab.id2suffix[s_id])
        if r.startswith('<') or not r:
            skipped['bad_root'] += 1
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


def two_condition_verdict(res):
    """THE re-run: required = munḍabiṭ AND muṭṭarid.  Nothing else gates acceptance."""
    md = res['mundabit_purity'] >= TH['mundabit_min_purity']
    it = res['ittirad_own_rate'] >= TH['muttarid_min_rate']
    return {'mundabit': bool(md), 'muttarid': bool(it), 'ACCEPTED_two_condition': bool(md and it)}


def run_one(name, obs):
    """Induct one 'illa over the SAME aṣl and evaluate every condition."""
    t0 = time.time()
    # (1) the ʿilla's OWN rule, no fall-back: the classical question
    q_own = Qiyas(name, backoff=False)
    n_bad = 0
    for r, w, s in obs:
        if not q_own.observe(r, w, s):
            n_bad += 1
    q_own.induce()
    it_own = q_own.ittirad()
    md_own = q_own.mundabit()
    own = {
        'ittirad_own_rate': it_own['ittirad_rate'],
        'ittirad_n': it_own['n'], 'ittirad_fail': it_own['fail'],
        'examples': it_own['examples'][:10],
    }
    purity = md_own['purity']
    n_cells = md_own['n_cells']
    impure = md_own['impure_cells']
    del q_own

    # (2) the deployed engine (with fall-back), for the diagnostics
    q = Qiyas(name, backoff=True)
    for r, w, s in obs:
        q.observe(r, w, s)
    q.induce()
    mk = q.munakis(min_cell=TH['min_cell_for_munakis'])
    if name == 'wazn_only':
        lift = 0.0
        mk_note = 'a constant cause has no outside: lift is 0 by construction'
    else:
        lift = mk['lift']
        mk_note = ''
    res = {
        'illa': name,
        'n_asl': len(obs), 'n_unalignable': n_bad,
        'mundabit_purity': purity,
        'mundabit_n_cells': n_cells, 'mundabit_impure_cells': impure,
        'ittirad_own_rate': own['ittirad_own_rate'],
        'ittirad_n': own['ittirad_n'], 'ittirad_fail': own['ittirad_fail'],
        'munakis_lift_NOT_REQUIRED': lift,
        'munakis_in': mk['in_acc'], 'munakis_out': mk['out_acc'],
        'munakis_pass_would_be': bool(lift >= TH['munakis_min_lift']),
        'munakis_note': mk_note,
        'ittirad_examples': own['examples'],
        'seconds': None,
    }
    res['verdict'] = two_condition_verdict(res)
    res['seconds'] = round(time.time() - t0, 1)
    return res, q


def pair_holdout(obs, frac=0.10, seed=5):
    """The two settings that matter, kept APART because they answer different questions.

    TIER_1  root SEEN, (root, wazn) WITHHELD.  This is the passage's own setting -- 'aṣl and
            farʿ share the root, the farʿ being a different derivational form'.  NOT a
            held-out-root test.
    TIER_2  root ITSELF withheld.  Nothing can generalise: no analogue shares the root.
    """
    rng = random.Random(seed)
    pairs = sorted({(r, wz) for r, wz, _ in obs})
    rng.shuffle(pairs)
    k = int(frac * len(pairs))
    test_pairs = set(pairs[:k])
    train_obs = [o for o in obs if (o[0], o[1]) not in test_pairs]
    test_obs = [o for o in obs if (o[0], o[1]) in test_pairs]
    held_roots = {r for r, _ in test_pairs}
    return train_obs, test_obs, test_pairs, len(pairs), sorted(held_roots)


def measure_firing(q, test_obs, name):
    """Does the 'illa's OWN cell carry the ruling to the far'?  A measurement."""
    prim = glob = none = ok = 0
    for r, wz, s in test_obs:
        rule = q.rule(r, wz)
        if rule is None:
            none += 1
            continue
        tag = rule[1]
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
            'realisation': ok / n if n else None}


def strict_isolate_no_illa(obs, test_obs):
    """THE decisive measurement, free of every contested condition.

    Chain = ['identity','wazn_only'] is the engine's floor: even backoff=False appends
    'wazn_only' (qiyas_engine.py:322-323), so 'identity without back-off' was never actually
    run.  Here the context-free table is REMOVED.  If the root-as-'illa carries a ruling to a
    withheld (root, wazn) form, this returns non-zero.  If it returns None everywhere, the
    root carries nothing and the whole question is arithmetic, not usul.
    """
    q = Qiyas('identity', backoff=False)
    for r, w, s in obs:
        q.observe(r, w, s)
    q.induce()
    old_wazn_cells = q.wazn_cells
    q.wazn_cells = collections.defaultdict(collections.Counter)   # remove the last resort
    n = none = 0
    for r, wz, s in test_obs:
        n += 1
        if q.rule(r, wz) is None:
            none += 1
    q.wazn_cells = old_wazn_cells
    return {'n': n, 'returns_None': none,
            'carries_a_ruling': n - none,
            'note': ('chain=[identity]; wazn_only removed.  Any far\' whose (root,wazn) cell '
                     'was withheld has NO analogue left, so None is the only possible answer.')}


def main():
    t0 = time.time()
    vocab = V(BLUEPRINT)
    log('[*] vocab roots=%d awzan=%d  (pod must be 9490/142)' % (vocab.num_roots, vocab.num_awzan))
    assert (int(vocab.num_roots), int(vocab.num_awzan)) == (9490, 142), 'VOCAB SHRINK TRAP'

    raw = json.load(open(DU_AFTER, encoding='utf-8'))['words']
    types = list(raw.keys())
    baseline_ok = {w: bool(v[0]) for w, v in raw.items()}
    log('[*] %d held-out types; shipped default round-trips %d (%.4f %%)'
        % (len(types), sum(baseline_ok.values()),
           100 * sum(baseline_ok.values()) / len(types)))

    obs, skipped = harvest(vocab, types, baseline_ok)
    log('[*] harvested %d attested asl; skipped %s' % (len(obs), skipped))
    log('    distinct roots in the asl set: %d' % len({r for r, _, _ in obs}))

    result = {
        'harness': 'two_condition_rerun.py',
        'built': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'vocab': {'roots': int(vocab.num_roots), 'awzan': int(vocab.num_awzan)},
        'n_types': len(types), 'n_asl': len(obs), 'harvest_skipped': skipped,
        'n_roots_with_asl': len({r for r, _, _ in obs}),
        'required_conditions': ['mundabit', 'muttarid'],
        'not_required': ['munakis', 'mutaaddi'],
        'thresholds': TH,
    }

    # ------------------------------------------------------------------ PART 1: the two-condition verdict
    log('')
    log('=' * 100)
    log('PART 1 -- THE VERDICT UNDER THE TWO REQUIRED CONDITIONS (munda-bit + muttarid)')
    log('=' * 100)
    log('%-11s %8s %7s %8s %1s %9s %9s %s'
        % ('illa', 'purity', 'cells', 'ittirad', '|', 'munakis', '2-cond', '3-cond(as shipped)'))
    illas = {}
    engines = {}
    for name in ILLA_NAMES:
        res, q = run_one(name, obs)
        illas[name] = res
        engines[name] = q
        three = bool(res['verdict']['mundabit'] and res['verdict']['muttarid']
                     and res['munakis_pass_would_be'])
        log('%-11s %8.4f %7d %8.4f %1s %+9.4f %9s %s   [%.0fs]'
            % (name, res['mundabit_purity'], res['mundabit_n_cells'],
               res['ittirad_own_rate'], '|', res['munakis_lift_NOT_REQUIRED'],
               'ACCEPT' if res['verdict']['ACCEPTED_two_condition'] else 'REJECT',
               'ACCEPT' if three else 'REJECT', res['seconds']))
    result['part1_illas'] = illas
    result['part1_accepted_two_condition'] = [
        n for n in ILLA_NAMES if illas[n]['verdict']['ACCEPTED_two_condition']]
    log('')
    log('  ACCEPTED under the two REQUIRED conditions: %s'
        % result['part1_accepted_two_condition'])

    # ------------------------------------------------------------------ PART 2: extrapolation
    log('')
    log('=' * 100)
    log('PART 2 -- WHAT THE ROOT-AS-ILLA CAN AND CANNOT DO (a measurement, not a condition)')
    log('=' * 100)
    train_obs, test_obs, test_pairs, n_pairs, held_roots = pair_holdout(obs)
    log('  Tier-1 pair holdout: %d pairs total, %d held out (%.1f%%), %d held-out observations'
        % (n_pairs, len(test_pairs), 100.0 * len(test_pairs) / n_pairs, len(test_obs)))
    log('  roots appearing in a held-out pair (these are SEEN roots): %d' % len(held_roots))
    tier1 = {}
    for name in ILLA_NAMES:
        q = Qiyas(name, backoff=True)
        for r, w, s in train_obs:
            q.observe(r, w, s)
        q.induce()
        tier1[name] = measure_firing(q, test_obs, name)
        m = tier1[name]
        log('  %-11s primary(own cell)=%s  GLOBAL=%s  realisation=%s'
            % (name, ('%.4f' % m['primary_firing']) if m['primary_firing'] is not None else 'n/a',
               ('%.4f' % m['global_firing']) if m['global_firing'] is not None else 'n/a',
               ('%.4f' % m['realisation']) if m['realisation'] is not None else 'n/a'))
        del q
    result['part2_tier1_pair_holdout'] = {
        'n_pairs_total': n_pairs, 'n_pairs_held': len(test_pairs),
        'n_obs_held': len(test_obs), 'n_roots_in_held_pairs': len(held_roots),
        'per_illa': tier1,
    }

    # held-out ROOT (Tier 2)
    hold37 = set(held_roots[:37])
    tier2_train = [o for o in obs if o[0] not in hold37]
    tier2_test = [o for o in obs if o[0] in hold37]
    log('')
    log('  Tier-2 held-out ROOT (%d roots): %d train obs, %d test obs'
        % (len(hold37), len(tier2_train), len(tier2_test)))
    tier2 = {}
    for name in ILLA_NAMES:
        q = Qiyas(name, backoff=True)
        for r, w, s in tier2_train:
            q.observe(r, w, s)
        q.induce()
        tier2[name] = measure_firing(q, tier2_test, name)
        m = tier2[name]
        log('  %-11s primary(own cell)=%s  GLOBAL=%s  realisation=%s'
            % (name, ('%.4f' % m['primary_firing']) if m['primary_firing'] is not None else 'n/a',
               ('%.4f' % m['global_firing']) if m['global_firing'] is not None else 'n/a',
               ('%.4f' % m['realisation']) if m['realisation'] is not None else 'n/a'))
        del q
    result['part2_tier2_heldout_root'] = {
        'n_roots_held': len(hold37), 'roots': sorted(hold37),
        'n_train_obs': len(tier2_train), 'n_test_obs': len(tier2_test),
        'per_illa': tier2,
    }

    # ------------------------------------------------------------------ PART 3: the decisive isolation
    log('')
    log('=' * 100)
    log('PART 3 -- ROOT-AS-ILLA IN ISOLATION: chain=[identity], context-free table REMOVED')
    log('=' * 100)
    iso = strict_isolate_no_illa(obs, test_obs)
    result['part3_isolation'] = iso
    log('  held-out far\' observations: %d' % iso['n'])
    log('  rule() returns None for:     %d' % iso['returns_None'])
    log('  rule() carries a ruling for: %d' % iso['carries_a_ruling'])

    with open(OUT, 'w', encoding='utf-8') as fh:
        json.dump(result, fh, ensure_ascii=False, indent=1)
    log('')
    log('[*] wrote %s   [total %.0fs]' % (OUT, time.time() - t0))


if __name__ == '__main__':
    main()
