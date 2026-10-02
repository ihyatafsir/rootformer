#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qiyas_deriv3588.py -- the decisive test on the EXACT 3,588 target positions that the
compositional neural decoder scored 0/3,588 on.

Gold: deriv_gold.json, whose reconstruction is independently verified -- 3,627 sentences and
3,987 held-out-root occurrences match audit_comp.json, and the per-root histogram of the
3,588 target positions matches `holdout_records` in comp24k_eval.json exactly on all 37 roots.

Two tasks on the identical gold set (same 3,588 positions, same 37 roots held out of the
neural training, same gold words):

  TASK A -- REALISATION (forward qiyas).
      Given the 'illa (the held-out root) and the wazn, derive the surface.  qiyas's rules
      are induced ONLY from asl whose root is not one of the 37.  The shipped hand-coded
      realiser is scored on exactly the same words.
      THIS IS NOT THE DECODER'S TASK.  The decoder had to INFER the root from context; qiyas
      is given it.  What is comparable is the property under test: whether a shared
      root-level structure supports forms of a root never seen.

  TASK B -- IDENTIFICATION (inverse qiyas).
      Given the surface word only, recover the root.  Candidates are the 37 held-out roots;
      every wazn in the inventory is tried; (R, W) is admissible iff qiyas's realisation
      equals the gold word; roots are ranked by the 'illa level that fired, then the purity
      of its cell, then support, then asl frequency.  Chance is 1/37 = 2.70 % top-1 and
      5/37 = 13.51 % top-5 -- directly comparable to the decoder's 0/3,588 over the same roots.
      PROTOCOL CAVEAT, stated because it matters: the gold prefix/suffix segmentation is
      supplied.  Prefixes and suffixes are orthographic clitics; they do not name the root.
      Only the root is hidden.

Writes /workspace/qiyas/deriv3588_results.json.
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

from qiyas_engine import Qiyas   # noqa: E402

OUT = '/workspace/qiyas'
DU_AFTER = '/workspace/lisan_root_fix/du_after.json'
GOLD = '/workspace/qiyas/deriv_gold.json'
SPECIALS = ('<NONE>', '<PAD>', '<UNK>', '')


def log(*a):
    print(*a, flush=True)


def affix(x):
    """The per-word dump and deriv_gold.json store the RAW vocabulary token, so '<NONE>' has
    to be mapped to the empty string.  Treating it literally was a real bug: it made every
    affixed gold word unmatchable and depressed the measured realisation rate."""
    return '' if (x is None or x in SPECIALS) else x


def load_vocab():
    import nrmp_vocab as nv
    V = next(v for k, v in vars(nv).items() if isinstance(v, type) and 'MorphemicVocab' in k)
    return V('/workspace/hf_v19_2_release/data/rootformer_v12_arabic_blueprint.json')


def harvest(vocab, du, exclude_roots):
    """asl = realisations the shipped pipeline reproduces exactly, held-out roots excluded."""
    obs = []
    n_skip = collections.Counter()
    for w, v in du.items():
        p_id, r_id, wz_id, s_id = vocab.encode_word(w)
        r = vocab.id2root.get(r_id, '')
        wz = vocab.id2wazn.get(wz_id, '')
        p = affix(vocab.id2prefix.get(p_id, ''))
        s = affix(vocab.id2suffix.get(s_id, ''))
        if r.startswith('<') or not r:
            n_skip['special_root'] += 1
            continue
        if r in exclude_roots:
            n_skip['held_out_root'] += 1
            continue
        if not wz or wz in ('<NONE>', '<PAD>', '<UNK>'):
            n_skip['no_wazn'] += 1
            continue
        if not v[0]:
            n_skip['baseline_fail'] += 1
            continue
        if len(p) + len(s) > len(w):
            n_skip['bad_split'] += 1
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
        r, wz, pv, sup, tot = c
        purity = sup / tot if tot else 0.0
        return (order.get(pv.split(':')[0], 5), -purity, -sup, -rootfreq.get(r, 0), r)
    return k


def main():
    t0 = time.time()
    vocab = load_vocab()
    gold = json.load(open(GOLD, encoding='utf-8'))
    pos = [{'word': x['word'], 'root': x['root'], 'wazn': x['wazn'],
            'prefix': affix(x['prefix']), 'suffix': affix(x['suffix'])}
           for x in gold['positions']]
    log('[*] gold: %d target positions (reconstruction verified: %s)'
        % (len(pos), gold['histogram_matches_recorded']))
    hold = sorted({x['root'] for x in pos})
    log('[*] distinct held-out roots: %d' % len(hold))

    du = json.load(open(DU_AFTER, encoding='utf-8'))['words']
    obs, skipped = harvest(vocab, du, set(hold))
    log('[*] asl: %d attested realisations, %d distinct roots; skipped %s'
        % (len(obs), len({o[0] for o in obs}), skipped))

    q = Qiyas('strict7', backoff=True)
    for r, w, s in obs:
        q.observe(r, w, s)
    q.induce()
    rootfreq = collections.Counter(r for r, _, _ in obs)
    log('[*] engine induced in %.0fs' % (time.time() - t0))

    # ------------------------------------------------------------------ TASK A: realisation
    stat = collections.Counter()
    prov = collections.Counter()
    ex_fixed, ex_broken = [], []
    for x in pos:
        w, r, wz, p, s = x['word'], x['root'], x['wazn'], x['prefix'], x['suffix']
        base = vocab.base_tok.realize_root_and_wazn(r, wz)     # the SHIPPED hand-coded rule
        b_ok = (p + base + s == w)
        res = q.realize(r, wz)
        if res is None:
            stat['no_analogue'] += 1
            q_ok, pv, qy = False, 'NONE', None
        else:
            qy, pv, sup, tot, dist = res
            q_ok = (p + qy + s == w)
            prov[pv.split(':')[0]] += 1
        stat['qiyas_ok' if q_ok else 'qiyas_fail'] += 1
        if b_ok:
            stat['base_ok'] += 1
        if q_ok and not b_ok:
            stat['qiyas_fixes'] += 1
            if len(ex_fixed) < 30:
                ex_fixed.append({'word': w, 'root': r, 'wazn': wz, 'shipped': p + base + s,
                                 'qiyas': p + (qy or '') + s, 'provenance': pv})
        if b_ok and not q_ok:
            stat['qiyas_breaks'] += 1
            if len(ex_broken) < 30:
                ex_broken.append({'word': w, 'root': r, 'wazn': wz, 'shipped': p + base + s,
                                  'qiyas': p + (qy or '') + s, 'provenance': pv})
    N = len(pos)
    taskA = {
        'n_positions': N,
        'shipped_default_ok': stat['base_ok'],
        'shipped_default_rate': stat['base_ok'] / N,
        'qiyas_ok': stat['qiyas_ok'],
        'qiyas_rate': stat['qiyas_ok'] / N,
        'delta_positions': stat['qiyas_ok'] - stat['base_ok'],
        'qiyas_fixes_baseline_failures': stat['qiyas_fixes'],
        'qiyas_breaks_baseline_successes': stat['qiyas_breaks'],
        'no_analogue': stat['no_analogue'],
        'provenance_histogram': dict(prov),
        'examples_fixed': ex_fixed,
        'examples_broken': ex_broken,
    }
    log('[*] TASK A (realisation, given the ʿilla): shipped %d/%d = %.4f %%; '
        'qiyas %d/%d = %.4f %%  (delta %+d; fixes %d, breaks %d, no-analogue %d)'
        % (stat['base_ok'], N, 100 * taskA['shipped_default_rate'],
           stat['qiyas_ok'], N, 100 * taskA['qiyas_rate'], taskA['delta_positions'],
           stat['qiyas_fixes'], stat['qiyas_breaks'], stat['no_analogue']))
    log('    provenance: %s' % dict(prov))

    # cross-check the baseline against the recorded per-type flags, where the word exists
    ov = [(x, du[x['word']]) for x in pos if x['word'] in du]
    agree = sum(1 for x, v in ov
                if (x['prefix'] + vocab.base_tok.realize_root_and_wazn(x['root'], x['wazn'])
                    + x['suffix'] == x['word']) == bool(v[0]))
    taskA['baseline_crosscheck'] = {'overlap_words': len(ov), 'agree': agree,
                                    'agree_rate': agree / max(len(ov), 1)}
    log('    baseline cross-check vs du_after on %d overlapping words: %d agree (%.4f)'
        % (len(ov), agree, agree / max(len(ov), 1)))

    # --------------------------------------------------------------- TASK B: identification
    wazn_all = [w for w in vocab.awzan_list if isinstance(w, str) and not w.startswith('<')]
    log('[*] TASK B: candidate grid')

    def build_grid(cand_roots):
        grid = {}
        for r in cand_roots:
            row = {}
            for wz in wazn_all:
                res = q.realize(r, wz)
                if res is not None:
                    row[wz] = res[:5]
            grid[r] = row
        return grid

    def identify(grid, cand_roots, pos_list):
        t1 = t5 = 0
        nocand = 0
        sizes = []
        detail = []
        for x in pos_list:
            w, goldr, p, s = x['word'], x['root'], x['prefix'], x['suffix']
            cands = []
            for r in cand_roots:
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
            if goldr in rr:
                rk = rr.index(goldr) + 1
                if rk == 1:
                    t1 += 1
                if rk <= 5:
                    t5 += 1
                if len(detail) < 40:
                    detail.append({'word': w, 'gold_root': goldr, 'rank': rk,
                                   'n_candidate_roots': len(rr), 'top5_roots': rr[:5],
                                   'top5_wazn': [c[1] for c in cands[:5]]})
            elif len(detail) < 40:
                detail.append({'word': w, 'gold_root': goldr, 'rank': None,
                               'n_candidate_roots': len(rr), 'top5_roots': rr[:5],
                               'top5_wazn': [c[1] for c in cands[:5]]})
        return t1, t5, nocand, sizes, detail

    for r in hold:
        if r not in vocab.root2id:
            raise SystemExit('held-out root %r is not in the inventory' % r)
    grid37 = build_grid(hold)
    log('    grid %d roots x %d awzan' % (len(hold), len(wazn_all)))
    top1, top5, no_cand, sizes, detail = identify(grid37, hold, pos)
    n = len(pos)
    taskB = {
        'n_positions': n,
        'n_candidate_roots': len(hold),
        'n_awzan_tried': len(wazn_all),
        'chance_top1': 1 / len(hold),
        'chance_top5': min(5, len(hold)) / len(hold),
        'qiyas_top1': top1, 'qiyas_top1_rate': top1 / n,
        'qiyas_top5': top5, 'qiyas_top5_rate': top5 / n,
        'positions_with_no_admissible_candidate': no_cand,
        'mean_candidate_instances': sum(sizes) / n,
        'examples': detail,
    }
    log('[*] TASK B: top-1 %d/%d = %.4f %% (chance %.4f %%), top-5 %d/%d = %.4f %% '
        '(chance %.4f %%)' % (top1, n, 100 * taskB['qiyas_top1_rate'],
                              100 * taskB['chance_top1'], top5, n,
                              100 * taskB['qiyas_top5_rate'], 100 * taskB['chance_top5']))
    log('    positions with no admissible candidate: %d; mean candidate instances %.2f'
        % (no_cand, taskB['mean_candidate_instances']))

    rng = random.Random(23)
    hset = set(hold)
    content = [r for r in vocab.roots_list
               if isinstance(r, str) and not r.startswith('<') and r and r not in hset]
    extra = rng.sample(content, 37)
    hold2 = hold + extra
    grid74 = build_grid(hold2)
    t1b, t5b, noc2, _, _ = identify(grid74, hold2, pos)
    taskB['control_74_roots'] = {
        'n_candidate_roots': len(hold2),
        'chance_top1': 1 / len(hold2), 'chance_top5': 5 / len(hold2),
        'qiyas_top1': t1b, 'qiyas_top1_rate': t1b / n,
        'qiyas_top5': t5b, 'qiyas_top5_rate': t5b / n,
        'positions_with_no_admissible_candidate': noc2,
    }
    log('    control (%d roots): top-1 %.4f %% (chance %.4f %%), top-5 %.4f %% (chance %.4f %%)'
        % (len(hold2), 100 * t1b / n, 100 / len(hold2), 100 * t5b / n, 500 / len(hold2)))

    out = {
        'harness': 'qiyas_deriv3588.py',
        'built': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'gold_source': GOLD,
        'gold_n': gold['n'],
        'gold_histogram_matches_recorded': gold['histogram_matches_recorded'],
        'n_asl': len(obs),
        'n_asl_roots': len({o[0] for o in obs}),
        'hold_roots': hold,
        'neural_reference': {
            'source': 'rootformer/build/scratch_comp/RESULTS_COMP.md (comp 6k and comp 24k)',
            'held_out_root_top1': 0.0, 'held_out_root_top5': 0.0, 'n_positions': 3588,
            'note': 'context-conditioned root prediction; exact search over all letter '
                    'sequences, leak-free holdout; 0/3,588 top-1 and top-5',
        },
        'task_A_realisation': taskA,
        'task_B_identification': taskB,
    }
    json.dump(out, open(os.path.join(OUT, 'deriv3588_results.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    log('[*] wrote %s/deriv3588_results.json  (%.0fs)' % (OUT, time.time() - t0))


if __name__ == '__main__':
    main()
