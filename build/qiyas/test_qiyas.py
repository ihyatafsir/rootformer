#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_qiyas.py -- CPU-first unit tests for the computed qiyas procedure.

Everything here runs with no torch, no GPU, no network, no corpus.  Where a result can be
checked by exhaustive search, it is (see the brute-force section).

Run:  python3 test_qiyas.py
"""

import itertools
import random
import sys
import traceback

from qiyas_engine import (
    ILLAS, BACKOFF, Qiyas,
    align, instantiate, parse_alignment, alignment_arity, alignment_slots,
    brute_force_alignments, brute_force_cell,
    illa_strict7, illa_coarse_weak, illa_positional, illa_wazn_only, illa_identity,
)

PASS = []
FAIL = []


def chk(name, got, exp):
    ok = got == exp
    (PASS if ok else FAIL).append(name)
    if not ok:
        print('  FAIL %-58s got %r expected %r' % (name, got, exp))
    return ok


def chk_true(name, cond):
    return chk(name, bool(cond), True)


def _ittirad_no_backoff(obs, illa='strict7'):
    q = Qiyas(illa, backoff=False)
    for r, w, s in obs:
        q.observe(r, w, s)
    q.induce()
    return q.ittirad()['ittirad_rate']


# =======================================================================================
# 1. alignment algebra
# =======================================================================================
def test_alignment():
    print('[1] alignment algebra')
    chk('align faa3il/sound', align('كاتب', 'كتب'), '^0ا^1^2')
    chk('align ajwaf fa3ala (iʿlāl elides r2)', align('قال', 'قول'), '^0ا^2')
    chk('align mudaaf fa3ala (idghām)', align('كر', 'كرر'), '^0^1')
    chk('align naqis fi3aal', align('شفاء', 'شفي'), '^0^1اء')
    chk('align maF3uul', align('مكتوب', 'كتب'), 'م^0^1و^2')
    chk('align idgham 2-of-3 is admissible', align('كت', 'كتب'), '^0^1')
    chk('align ignores the root -> None', align('ك', 'كتب'), None)
    chk('align non-radical surface -> None', align('ىى', 'كتب'), None)
    chk('align caret surface -> None', align('ك^ت', 'كتب'), None)
    chk('align anchors an imperfective prefix correctly',
        align('يشفي', 'شفي'), 'ي^0^1^2')
    chk('align anchors a definite article correctly',
        align('الكتب', 'كتب'), 'ال^0^1^2')

    chk('instantiate sound', instantiate('^0ا^1^2', 'ثور'), 'ثاور')
    chk('instantiate ajwaf', instantiate('^0ا^2', 'بيع'), 'باع')
    chk('instantiate mudaaf', instantiate('^0^1', 'ردد'), 'رد')
    chk('instantiate too-short root', instantiate('^0ا^1^2', 'قد'), None)

    # instantiate/align are mutually consistent over a generated set: align always
    # reproduces the surface it was given (a section of instantiate)
    roots = ['كتب', 'قول', 'بيع', 'كرر', 'شفي', 'دعا', 'حقق', 'منع', 'ثور', 'وعد', 'يسر']
    alns = ['^0^1^2', '^0ا^1^2', '^0ا^2', '^0^1', '^0^1اء', 'م^0^1و^2', '^0^1ا^2',
            'ا^0^1ا^2', 'ا^0^1^2', '^0^1ا^2ة', 'ي^0^1^2']
    bad = []
    for r, a in itertools.product(roots, alns):
        s = instantiate(a, r)
        if s is None:
            continue
        b = align(s, r)
        if b is None or instantiate(b, r) != s:
            bad.append((r, a, s, b))
    chk('instantiate/align are mutually consistent', bad, [])

    # align uses a MAXIMUM-cardinality matching (the LCS property)
    import random as _rnd
    rng = _rnd.Random(3)
    subopt = []
    for _ in range(300):
        r = ''.join(rng.choice('بتثجحخ') for _ in range(3))
        s = ''.join(rng.choice('بتثجحخايو') for _ in range(rng.randint(3, 6)))
        a = align(s, r)
        if a is None:
            continue
        got = len(alignment_slots(a))
        # brute-force the maximum number of root radicals embeddable in s, in order
        best = [0]
        def rec(ri, si, cnt):
            if cnt > best[0]:
                best[0] = cnt
            for j in range(ri, len(r)):
                for i in range(si, len(s)):
                    if r[j] == s[i]:
                        rec(j + 1, i + 1, cnt + 1)
        rec(0, 0, 0)
        if got != best[0]:
            subopt.append((r, s, a, got, best[0]))
    chk('align attains the maximum radical cardinality', subopt, [])

    chk('parse_alignment literal', parse_alignment('م^0^1و^2'), ['م', '^0', '^1', 'و', '^2'])
    chk('alignment_arity', alignment_arity('م^0^1و^2'), 3)
    chk('alignment_arity with a gap', alignment_arity('^0ا^2'), 3)


# =======================================================================================
# 2. classical cases -- the hukm the classical morphophonology states, re-derived
# =======================================================================================
def test_classical_cases():
    print('[2] classical realisation cases')
    # a sound triliteral paradigm, as an attested asl set.  Every ('illa-label, wazn) cell
    # carries at least two asl with ONE alignment, so ittirad has something to be exact about.
    asl = [
        ('كتب', 'فَعَلَ', 'كتب'), ('نصر', 'فَعَلَ', 'نصر'), ('ضرب', 'فَعَلَ', 'ضرب'),
        ('كتب', 'فَاعِل', 'كاتب'), ('نصر', 'فَاعِل', 'ناصر'), ('ضرب', 'فَاعِل', 'ضارب'),
        ('كتب', 'مَفْعُول', 'مكتوب'), ('نصر', 'مَفْعُول', 'منصور'),
        ('كتب', 'فِعَال', 'كتاب'), ('حسب', 'فِعَال', 'حساب'), ('عتب', 'فِعَال', 'عتاب'),
        # ajwaf
        ('قول', 'فَعَلَ', 'قال'), ('بيع', 'فَعَلَ', 'باع'), ('صوم', 'فَعَلَ', 'صام'),
        ('خوف', 'فَعَلَ', 'خاف'),
        # mudaaf
        ('كرر', 'فَعَلَ', 'كر'), ('ردد', 'فَعَلَ', 'رد'), ('مدد', 'فَعَلَ', 'مد'),
        # mahmuz
        ('سأل', 'فَعَلَ', 'سأل'), ('قرأ', 'فَعَلَ', 'قرأ'),
        # naqis
        ('شفي', 'فِعَال', 'شفاء'), ('دعو', 'فِعَال', 'دعاء'),
        ('دعا', 'فَعَلَ', 'دعا'), ('رضي', 'فَعَلَ', 'رضي'), ('هوي', 'فَعَلَ', 'هوي'),
    ]
    q = Qiyas('strict7')
    for r, w, s in asl:
        q.observe(r, w, s)
    q.induce()

    chk('SAHIH+fa3ala', q.realize('فتح', 'فَعَلَ')[0], 'فتح')
    chk('SAHIH+fa3il', q.realize('فتح', 'فَاعِل')[0], 'فاتح')
    chk('SAHIH+maf3ul', q.realize('فتح', 'مَفْعُول')[0], 'مفتوح')
    chk('SAHIH+fi3aal', q.realize('فتح', 'فِعَال')[0], 'فتاح')
    chk('AJWAF+fa3ala (unseen root)', q.realize('ثور', 'فَعَلَ')[0], 'ثار')
    chk('AJWAF+fa3ala (unseen root, ya)', q.realize('خيف', 'فَعَلَ')[0], 'خاف')
    chk('MUDAAF+fa3ala', q.realize('مدد', 'فَعَلَ')[0], 'مد')
    chk('NAQIS+fi3aal', q.realize('بكي', 'فِعَال')[0], 'بكاء')

    md = q.mundabit()
    chk('mundabit determinacy == 1.0', md['determinacy'], 1.0)
    chk('mundabit purity == 1.0 on a single-alignment-per-cell set', md['purity'], 1.0)

    it = q.ittirad()
    chk('ittirad == 1.0 on the regular set (the ʿilla\'s OWN rule, no back-off)',
        round(_ittirad_no_backoff(asl), 6), 1.0)
    chk_true('ittirad with the fall-back lattice is still >= 0.95 (%.4f)'
             % it['ittirad_rate'], it['ittirad_rate'] >= 0.95)

    mk = q.munakis(min_cell=1)
    chk_true('munakis lift > 0 for a discriminating ʿilla', mk['lift'] > 0.0)

    # ---- the refinement demonstration: fi3laa for a naqis root depends on WHICH weak
    # ---- radical it is.  strict7 cannot see that; positional can.
    obs2 = [('دعو', 'فِعْلَى', 'دعوى'), ('عدو', 'فِعْلَى', 'عدوى'),
            ('رضي', 'فِعْلَى', 'رضى'), ('حمي', 'فِعْلَى', 'حمى'),
            ('دعو', 'فَعَلَ', 'دعو'), ('عدو', 'فَعَلَ', 'عدو'),
            ('رضي', 'فَعَلَ', 'رضي'), ('حمي', 'فَعَلَ', 'حمي')]
    q7 = Qiyas('strict7', backoff=False)
    qp = Qiyas('positional', backoff=False)
    for r, w, s in obs2:
        q7.observe(r, w, s)
        qp.observe(r, w, s)
    q7.induce(); qp.induce()
    it7 = q7.ittirad(); itp = qp.ittirad()
    chk_true('strict7 fails ittirad on fi3laa (%d exceptions)' % it7['fail'], it7['fail'] >= 1)
    chk('positional is exact where strict7 is not', itp['fail'], 0)
    chk_true('positional purity == 1.0 on this set', qp.mundabit()['purity'] == 1.0)
    chk_true('strict7 purity < 1.0 on this set', q7.mundabit()['purity'] < 1.0)


# =======================================================================================
# 3. the 'illa functions
# =======================================================================================
def test_illas():
    print('[3] ʿilla functions are total, deterministic and single-valued')
    cases = {
        'كتب': 'SAHIH', 'قول': 'AJWAF', 'بيع': 'AJWAF', 'كرر': 'MUDAAF', 'ردد': 'MUDAAF',
        'شفي': 'NAQIS', 'دعا': 'NAQIS', 'وعد': 'MITHAL', 'يسر': 'MITHAL',
        'سأل': 'MAHMUZ', 'قرأ': 'MAHMUZ', 'وفي': 'LAFIF', 'وري': 'LAFIF',
    }
    ok = all(illa_strict7(r) == e for r, e in cases.items())
    chk_true('strict7 on the classical divisions', ok)
    for r, e in cases.items():
        if illa_strict7(r) != e:
            print('     strict7(%s) = %s expected %s' % (r, illa_strict7(r), e))

    chk('coarse_weak lumps ajwaf+naqis+lafif+mithal',
        {illa_coarse_weak(r) for r in ['قول', 'شفي', 'وفي', 'وعد']}, {'WEAK'})
    chk('coarse_weak separates geminate', illa_coarse_weak('كرر'), 'MUDAAF')
    chk('wazn_only is constant', len({illa_wazn_only(r) for r in cases}), 1)
    chk('identity is injective', len({illa_identity(r) for r in cases}), len(cases))
    chk('positional separates waw/ya ajwaf',
        len({illa_positional(r) for r in ['قول', 'بيع']}), 2)

    # determinacy: a total single-valued function returns a str for every root
    bad = [r for r in list(cases) + ['', 'ا', 'ءءء'] if not isinstance(illa_positional(r), str)]
    chk('positional total (no exceptions)', bad, [])


# =======================================================================================
# 4. brute force -- the modal hukm is the true argmax over the alignment space
# =======================================================================================
def test_brute_force():
    print('[4] brute-force validation of the induced hukm')
    rng = random.Random(7)
    alphabet = ['ا', 'و', 'ي']
    letters = 'بتثجحخدذرزسشصضطظعغفقكلمنه'
    words = [''.join(rng.choice(letters) for _ in range(3)) for _ in range(6)]
    # three sound triliterals realised by faa3il
    cell = [(w, w[0] + 'ا' + w[1] + w[2]) for w in words]
    bf, score, n_cand = brute_force_cell(cell, alphabet, max_extra=2)
    chk('brute force finds the faa3il alignment', bf, '^0ا^1^2')
    chk('brute force scores all observations', score, len(cell))

    q = Qiyas('strict7')
    for w, s in cell:
        q.observe(w, 'فَاعِل', s)
    q.induce()
    got = q.cells[('SAHIH', 'فَاعِل')].most_common(1)[0][0]
    chk('induced hukm == brute-force hukm', got, bf)

    # an irregular cell: 3 of 4 take faa3il, 1 has a stray hamza inserted
    irr = list(cell) + [(words[0], words[0][0] + 'أا' + words[0][1] + words[0][2])]
    bf2, score2, _ = brute_force_cell(irr, alphabet, max_extra=2)
    q2 = Qiyas('strict7')
    for w, s in irr:
        q2.observe(w, 'فَاعِل', s)
    q2.induce()
    got2 = q2.cells[('SAHIH', 'فَاعِل')].most_common(1)[0][0]
    chk('induced hukm == brute-force hukm (irregular cell)', got2, bf2)
    chk('brute force picks the majority alignment', score2, len(cell))

    # the brute-force space is not trivially small
    chk_true('brute-force space is real (%d candidates)' % n_cand, n_cand > 50)

    # exhaustive agreement with the aligner over a generated corpus: alignment is an
    # ambiguous relation, so the contract is (i) the aligned surface is reproduced exactly
    # and (ii) the aligner never uses FEWER radicals than the alignment that generated it
    bad = []
    for _ in range(400):
        r = ''.join(rng.choice('بتثجحخد') for _ in range(3))
        a = rng.choice(['^0^1^2', '^0ا^1^2', '^0ا^2', '^0^1', 'م^0^1و^2', '^0^1اء',
                        'ا^0^1^2', '^0^1ا^2', '^0ا^1ا^2', 'ي^0^1^2'])
        s = instantiate(a, r)
        if s is None:
            continue
        b = align(s, r)
        if b is None or instantiate(b, r) != s or len(alignment_slots(b)) < len(alignment_slots(a)):
            bad.append((r, a, s, b))
    chk('align reproduces its input and is arity-maximal', bad, [])


# =======================================================================================
# 5. the three validity conditions actually bite
# =======================================================================================
def test_validity_conditions():
    print('[5] the usuli validity conditions')
    # a corpus where one cell is perfectly regular and another has a planted exception
    base = ['كتب', 'نصر', 'ضرب', 'فتح', 'جمع', 'قطع', 'سمع', 'علم', 'فهم', 'حفظ']
    obs = []
    for w in base:
        obs.append((w, 'فَعَلَ', w))
        obs.append((w, 'فَاعِل', w[0] + 'ا' + w[1] + w[2]))
    # regular weak class: ajwaf fa3ala
    for w in ['قول', 'بيع', 'صوم', 'خوف', 'ذوق', 'عوم']:
        obs.append((w, 'فَعَلَ', w[0] + 'ا' + w[2]))
    # planted exception inside the sound class
    obs.append((base[0], 'فَاعِل', 'م' + base[0]))

    q = Qiyas('strict7')
    for r, w, s in obs:
        q.observe(r, w, s)
    q.induce()
    it = q.ittirad()
    chk_true('ittirad < 1 when an exception is planted', it['ittirad_rate'] < 1.0)
    chk('exactly one observation fails ittirad', it['fail'], 1)

    # muttarid failure is LOCALISED to the impure cell, not smeared
    md = q.mundabit()
    chk_true('impure cell is counted', md['impure_cells'] >= 1)
    chk_true('purity < 1 with an exception', md['purity'] < 1.0)

    # mun'akis: wazn_only must have lift exactly 0 (a constant cause is present everywhere)
    qw = Qiyas('wazn_only')
    for r, w, s in obs:
        qw.observe(r, w, s)
    qw.induce()
    mkw = qw.munakis(min_cell=1)
    chk('wazn_only has no discriminating cells (lift undefined -> 0 rows)', mkw['n_cells'], 0)

    # coarse_weak lumps the weak classes into one cause.  It only *matters* where a single
    # wazn requires different alignments for different weak classes -- which is precisely the
    # case for fa3ala: an ajwaf root elides its middle radical (قال) while a naqis/mithal
    # root realises all three (دعا / وعد).  One cause, two rulings -> ittirad must fail.
    obs2 = []
    for w in base:
        obs2.append((w, 'فَعَلَ', w))
    for w in ['قول', 'بيع', 'صوم', 'خوف', 'ذوق', 'عوم']:      # ajwaf: ^0ا^2
        obs2.append((w, 'فَعَلَ', w[0] + 'ا' + w[2]))
    for w in ['دعا', 'رضي', 'شفي', 'هوي']:                    # naqis: ^0^1^2
        obs2.append((w, 'فَعَلَ', w))
    for w in ['وعد', 'يسر', 'وصل', 'وهب']:                    # mithal: ^0^1^2
        obs2.append((w, 'فَعَلَ', w))
    qc = Qiyas('coarse_weak', backoff=False)
    for r, w, s in obs2:
        qc.observe(r, w, s)
    qc.induce()
    itc = qc.ittirad()
    qs = Qiyas('strict7', backoff=False)
    for r, w, s in obs2:
        qs.observe(r, w, s)
    qs.induce()
    its = qs.ittirad()
    print('      strict7  ittirad %.4f (%d exceptions)' % (its['ittirad_rate'], its['fail']))
    print('      coarse   ittirad %.4f (%d exceptions)' % (itc['ittirad_rate'], itc['fail']))
    chk_true('strict7 ittirad (%.4f) > coarse_weak ittirad (%.4f)'
             % (its['ittirad_rate'], itc['ittirad_rate']),
             its['ittirad_rate'] > itc['ittirad_rate'])
    chk('strict7 has no exception on this set', its['fail'], 0)
    chk_true('coarse_weak has exceptions on this set', itc['fail'] > 0)


# =======================================================================================
# 6. no back-off leakage: an unseen (root, wazn) pair is a real far'
# =======================================================================================
def test_holdout_discipline():
    print('[6] holdout discipline')
    obs = []
    for w in ['كتب', 'نصر', 'ضرب', 'فتح', 'جمع', 'قطع']:
        obs.append((w, 'فَعَلَ', w))
        obs.append((w, 'فَاعِل', w[0] + 'ا' + w[1] + w[2]))
    q = Qiyas('strict7')
    for r, w, s in obs:
        q.observe(r, w, s)
    q.induce()
    # 'سفر' and 'زهد' never observed: the ruling still extends
    got, prov, sup, tot, dist = q.realize('سفر', 'فَاعِل')
    chk('unseen sound root is realised', got, 'سافر')
    chk('provenance names the ʿilla that fired', prov, 'strict7:SAHIH')
    chk('support is the number of asl analogues', sup, 6)
    # a root of a class with NO attested cell must back off, and must SAY it backed off
    got2, prov2, sup2, tot2, dist2 = q.realize('زيد', 'فَاعِل')
    chk('unseen ajwaf root backs off to the global rule', prov2, 'GLOBAL')
    chk('the backed-off ruling is still a real form', got2, 'زايد')
    # a wazn never observed anywhere -> no analogue, honest None
    chk('unobserved wazn -> None (no invention)', q.realize('سفر', 'فَوْعَل'), None)
    cov, miss = q.coverage([('سفر', 'فَاعِل'), ('سفر', 'فَوْعَل')])
    chk('coverage counts the miss', miss, 1)


# =======================================================================================
# 7. flag-off inertness (the wiring contract)
# =======================================================================================
def test_wiring_inertness():
    print('[7] wiring: flag-off inertness')
    try:
        import qiyas_wire
    except Exception as exc:
        print('  SKIP qiyas_wire not importable: %r' % (exc,))
        return

    class FakeTok:
        def __init__(self):
            self.calls = 0

        def realize_root_and_wazn(self, root, wazn):
            self.calls += 1
            return 'ORIGINAL(%s,%s)' % (root, wazn)

    # OFF: byte-identical to the original, and the original is still called
    t = FakeTok()
    qiyas_wire.install(t, enabled=False)
    out_off = [t.realize_root_and_wazn('كتب', 'فَاعِل') for _ in range(5)]
    chk('flag OFF -> original output', out_off, ['ORIGINAL(كتب,فَاعِل)'] * 5)
    chk('flag OFF -> original invoked', t.calls, 5)

    # ON with no induced rules: still inert (no analogue -> delegate to the original)
    t2 = FakeTok()
    qiyas_wire.install(t2, enabled=True, engine=qiyas_wire.NullEngine())
    chk('flag ON, no analogues -> original output',
        t2.realize_root_and_wazn('كتب', 'فَاعِل'), 'ORIGINAL(كتب,فَاعِل)')
    chk('flag ON, no analogues -> original invoked', t2.calls, 1)

    # ON with an engine: output changes -- this is the ablation that proves it acts
    q = Qiyas('strict7')
    for w in ['كتب', 'نصر', 'ضرب', 'فتح', 'جمع', 'قطع']:
        q.observe(w, 'فَاعِل', w[0] + 'ا' + w[1] + w[2])
    q.induce()
    t3 = FakeTok()
    qiyas_wire.install(t3, enabled=True, engine=q, mode='replace', log_cap=0)
    out_on = t3.realize_root_and_wazn('زيد', 'فَاعِل')
    chk('flag ON -> qiyas output (ablation changes the result)', out_on, 'زايد')
    chk_true('flag ON bypassed the original for a realised form', t3.calls == 0)

    # uninstall restores exact original behaviour
    qiyas_wire.uninstall(t3)
    chk('uninstall restores original',
        t3.realize_root_and_wazn('زيد', 'فَاعِل'), 'ORIGINAL(زيد,فَاعِل)')
    qiyas_wire.uninstall(t)
    qiyas_wire.uninstall(t2)


# =======================================================================================
def main():
    tests = [
        test_alignment,
        test_classical_cases,
        test_illas,
        test_brute_force,
        test_validity_conditions,
        test_holdout_discipline,
        test_wiring_inertness,
    ]
    for t in tests:
        try:
            t()
        except Exception:
            FAIL.append(t.__name__)
            print('  EXCEPTION in %s' % t.__name__)
            traceback.print_exc()
    print('')
    print('=' * 72)
    print('PASS %d   FAIL %d' % (len(PASS), len(FAIL)))
    if FAIL:
        print('failing checks:')
        for f in FAIL:
            print('  -', f)
        return 1
    print('ALL CHECKS PASSED')
    return 0


if __name__ == '__main__':
    sys.exit(main())
