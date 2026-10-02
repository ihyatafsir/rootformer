#!/usr/bin/env python3
"""test_ewc_unit.py -- CPU unit tests for the EWC penalty and for the patch's inertness.

Pure arithmetic, no model, no GPU.  Run:
    /workspace/venvs/rootformer/bin/python test_ewc_unit.py

Tests
  T1  penalty(theta0) == 0 exactly
  T1b penalty(theta0 + d) == sum_i F_i d_i^2 exactly (the objective as specified)
  T2  gradient == 2*lambda*F_i*(theta_i - theta0_i), against the analytic value
  T3  finite-difference check of d(penalty)/d(theta) on a scalar parameter
  T4  no NaN / no Inf, including F containing zeros and F identically zero
  T5  the closed-form effect: one plain-GD step with task grad g moves by -eta*g;
      with the penalty the retained-structure displacement is damped to -eta*g/(1+2*eta*lambda*F)
      (the ordinary arithmetic that makes "lambda large preserves, lambda = 0 collapses" true)
  T6  missing Fisher tensors are refused, not silently skipped
  T7  --ewc-lambda 0.0 is structurally inert: the patched file must equal the original byte for
      byte once the inserted blocks are removed, and the two insertions in the loop are guarded
"""
import ast, hashlib, os, re, sys
import torch

POD = '/workspace/ghazali_forget'
ORIG = '/workspace/hf_v19_2_release/nrmt_train.py'
PATCHED = os.path.join(POD, 'nrmt_train_ewc.py')
sys.path.insert(0, POD)

PASS, FAIL = [], []


def check(name, cond, detail=''):
    (PASS if cond else FAIL).append(name)
    print(f'  [{"PASS" if cond else "FAIL"}] {name}{(" :: " + detail) if detail else ""}')


def main():
    import nrmt_train_ewc as M
    EWC = M._EWCPenalty

    print('T1/T1b: penalty value')
    torch.manual_seed(0)
    p1 = torch.nn.Parameter(torch.randn(64, dtype=torch.float32) * 0.1)
    p2 = torch.nn.Parameter(torch.randn(16, 8, dtype=torch.float32) * 0.1)
    F1 = torch.rand(64, dtype=torch.float32) * 1e-3
    F2 = torch.rand(16, 8, dtype=torch.float32) * 1e-3
    fisher = {'backbone.layers.20.a': F1, 'backbone.layers.20.b': F2}
    pen = EWC([('backbone.layers.20.a', p1), ('backbone.layers.20.b', p2)], fisher)
    check('T1 penalty(theta0) == 0.0 exactly', pen.value() == 0.0, repr(pen.value()))

    d1 = torch.randn(64) * 0.01
    d2 = torch.randn(16, 8) * 0.01
    with torch.no_grad():
        p1.add_(d1); p2.add_(d2)
    want = float((F1.double() * d1.double() ** 2).sum() + (F2.double() * d2.double() ** 2).sum())
    got = pen.value()
    check('T1b penalty == sum F_i d_i^2', abs(got - want) <= 1e-6 * max(abs(want), 1e-9),
          f'got {got:.10e} want {want:.10e}')

    print('T2: gradient')
    lam = 0.5
    p1.grad = None; p2.grad = None
    loss = lam * pen.penalty()
    loss.backward()
    g1_ok = torch.allclose(p1.grad, 2 * lam * F1 * d1, rtol=1e-3, atol=1e-9)
    g2_ok = torch.allclose(p2.grad, 2 * lam * F2 * d2, rtol=1e-3, atol=1e-9)
    check('T2 grad == 2*lambda*F*(theta-theta0)', g1_ok and g2_ok,
          f'max|dgrad|={(p1.grad - 2*lam*F1*d1).abs().max():.3e} '
          f'(float32 round-trip of d through a param 10x larger)')

    print('T3: finite difference')
    s = torch.nn.Parameter(torch.tensor([0.3]))
    Fs = torch.tensor([0.7])
    pen3 = EWC([('backbone.layers.21.s', s)], {'backbone.layers.21.s': Fs})
    s.grad = None
    (lam * pen3.penalty()).backward()
    ana = float(s.grad)
    eps = 1e-6
    with torch.no_grad():
        v0 = pen3.value(); s.add_(eps); v1 = pen3.value(); s.sub_(eps)
    fd = lam * (v1 - v0) / eps
    check('T3 analytic grad == finite difference', abs(ana - fd) < 1e-6,
          f'analytic {ana:.10f} fd {fd:.10f}')

    print('T4: NaN / Inf')
    pz = torch.nn.Parameter(torch.randn(32))
    Fz = torch.zeros(32)
    penz = EWC([('backbone.layers.22.z', pz)], {'backbone.layers.22.z': Fz})
    with torch.no_grad():
        pz.add_(torch.randn(32))
    lz = lam * penz.penalty()
    lz.backward()
    check('T4 all-zero Fisher -> penalty 0, grad 0, finite',
          float(lz) == 0.0 and bool(torch.isfinite(pz.grad).all()) and float(pz.grad.abs().sum()) == 0.0,
          f'pen {float(lz)}')
    Fmix = torch.zeros(32); Fmix[::3] = 1e-4
    pm = torch.nn.Parameter(torch.randn(32))
    pmix = EWC([('backbone.layers.22.m', pm)], {'backbone.layers.22.m': Fmix})
    with torch.no_grad():
        pm.add_(torch.randn(32) * 1e3)          # pathological displacement
    lm = lam * pmix.penalty()
    lm.backward()
    check('T4 zero rows + huge displacement -> finite', bool(torch.isfinite(lm)) and
          bool(torch.isfinite(pm.grad).all()), f'pen {float(lm):.4e}')

    print('T5: closed-form cap (why lambda large preserves, lambda=0 collapses)')
    # Plain gradient descent on a task loss whose gradient on one parameter is the constant g,
    # plus the penalty lam*F*t^2 (t = theta - theta0).  The iteration is
    #     t <- t - eta*(g + 2*lam*F*t)
    # Exact solution: t_N = t* (1 - (1-2*eta*lam*F)^N) with the fixed point t* = -g/(2*lam*F).
    # lambda = 0 -> t_N = -N*eta*g, i.e. the parameter walks away without bound.
    # lambda > 0 -> the displacement is CAPPED at g/(2*lambda*F): retained structure wins.
    g, eta, Fv, N = 1.0, 0.005, 1.0, 1000
    pt = torch.nn.Parameter(torch.tensor([0.0]))
    pent = EWC([('backbone.layers.23.t', pt)], {'backbone.layers.23.t': torch.tensor([Fv])})
    for lv in (0.0, 0.5, 50.0):
        with torch.no_grad():
            pt.zero_()
        for _ in range(N):
            pt.grad = None
            (lv * pent.penalty() + pt.sum() * g).backward()   # d/dtheta = 2*lv*F*t + g
            with torch.no_grad():
                pt.add_(pt.grad, alpha=-eta)
        tN = float(pt)
        if lv == 0.0:
            want = -N * eta * g
        else:
            tstar = -g / (2 * lv * Fv)
            want = tstar * (1 - (1 - 2 * eta * lv * Fv) ** N)
        check(f'T5 lambda={lv:g}: t_{N} == closed form', abs(tN - want) < 1e-3,
              f't{N} {tN:.8f} want {want:.8f}')
    print(f'  T5 summary: lambda=0 walks to -{N*eta*g:.3f}, lambda=0.5 is capped near -1.0, '
          f'lambda=50 is capped near -0.01 (all three against the same task gradient)')

    print('T6: missing Fisher tensors')
    try:
        EWC([('backbone.layers.20.a', torch.nn.Parameter(torch.zeros(4)))], {})
        check('T6 refuses missing tensors', False, 'no exception')
    except SystemExit:
        check('T6 refuses missing tensors', True)

    print('T7: structural inertness at --ewc-lambda 0.0')
    src = open(ORIG, encoding='utf-8').read()
    dst = open(PATCHED, encoding='utf-8').read()
    sys.path.insert(0, POD)
    import make_ewc_trainer as G
    # strip the inserted blocks back out and demand byte identity with the original
    try:
        back = G.unpatch(dst)
        check('T7 un-patch round-trip == original byte for byte', back == src,
              f'md5 {hashlib.md5(back.encode()).hexdigest()} vs {hashlib.md5(src.encode()).hexdigest()}')
    except SystemExit as e:
        check('T7 un-patch round-trip == original byte for byte', False, str(e))
    check('T7 original nrmt_train.py md5 unchanged',
          hashlib.md5(src.encode()).hexdigest() == '56ef3f2fd9ddc352dee596f32590e7b5',
          hashlib.md5(src.encode()).hexdigest())
    check('T7 construction is guarded by --ewc-lambda != 0.0',
          dst.count("    if args.ewc_lambda != 0.0:\n") == 1)
    check('T7 loss addition is guarded by `if ewc is not None:`',
          dst.count('        if ewc is not None:\n') == 1)
    n_guard = len(re.findall(r'^\s*if ewc is not None:', dst, re.M))
    check('T7 exactly one ewc guard in the file', n_guard == 1, f'{n_guard}')
    ast.parse(dst)
    check('T7 patched file parses', True)

    print(f'\n{len(PASS)} passed, {len(FAIL)} failed')
    if FAIL:
        print('FAILED:', FAIL)
    return 1 if FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
