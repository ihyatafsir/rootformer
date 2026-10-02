#!/usr/bin/env python3
"""CPU unit test for the metric/aggregation paths of echo_identity.py (no model, no GPU)."""
import sys
import torch

sys.path.insert(0, '/workspace/echo_test')
import echo_identity as E  # noqa: E402

torch.manual_seed(0)
N, T, C = 5, 7, 64
H = torch.randn(N, T, 8)
R = torch.randint(0, C, (N, T)); O = torch.zeros(N, T, dtype=torch.long)
W = torch.randint(0, C, (N, T)); P = torch.randint(0, C, (N, T)); S = torch.randint(0, C, (N, T))
tgt_id = torch.randint(0, C, (N, T)); tgt_nw = torch.randint(0, C, (N, T))
keep_id = torch.rand(N * T) > 0.1
keep_nw = torch.rand(N * T) > 0.1
FULL = torch.randn(N, T, C)          # the logits every batch must reproduce


class FakeHead:
    def __init__(self):
        self.i = 0

    def __call__(self, h, r, o, w, p, s, cond_roots=None):
        b = h.shape[0]
        out = FULL[self.i:self.i + b]
        self.i += b
        return {'root_logits': out}


m = E.window_metrics(FakeHead(), H, R, O, W, P, S, tgt_id, tgt_nw, keep_id, keep_nw,
                     torch.device('cpu'), C, bs=2)
assert m['identity']['all'].n == int(keep_id.sum()), (m['identity']['all'].n, int(keep_id.sum()))
assert m['next_word']['all'].n == int(keep_nw.sum())
assert m['identity']['le'].n + m['identity']['gt'].n == m['identity']['all'].n

# brute force
flat = FULL.reshape(-1, C)
for nm, tgt, keep in (('identity', tgt_id, keep_id), ('next_word', tgt_nw, keep_nw)):
    t = tgt.reshape(-1)[keep]
    lg = flat[keep]
    exp_a1 = float((lg.argmax(-1) == t).float().mean())
    exp_a5 = float((lg.topk(5, -1).indices == t.unsqueeze(-1)).any(-1).float().mean())
    got = m[nm]['all']
    assert abs(got.a1 / got.n - exp_a1) < 1e-6, (nm, got.a1 / got.n, exp_a1)
    assert abs(got.a5 / got.n - exp_a5) < 1e-6, (nm, got.a5 / got.n, exp_a5)
    print(f'OK window_metrics {nm}: n={got.n} acc@1={100*exp_a1:.2f}% acc@5={100*exp_a5:.2f}%')

# ---- eval_ridge sanity: features that literally contain the target root id
X = torch.zeros(40, C + 3)
y = torch.randint(0, C, (40,))
X[torch.arange(40), y] = 5.0
keepA = torch.ones(40, dtype=torch.bool)
Wr, lam = E.fit_ridge(X, y, C, torch.device('cpu'), bs=16)
r = E.eval_ridge(Wr, X, y, keepA, C, torch.device('cpu'), C, bs=16)
assert r['all']['acc@1'] == 1.0, r
print('OK eval_ridge: separable features -> acc@1 100%')

# ---- group split bookkeeping
ygt = torch.full((20,), E.CUT + 5)
subs = {'all': E.Sub(), 'le': E.Sub(), 'gt': E.Sub()}
lg = torch.randn(20, 9490)
top, a1, a5 = E.topk_acc(lg, ygt)
E.split_update(subs, top, a1, a5, ygt, {'all': torch.ones(20, dtype=torch.bool),
                                        'le': torch.zeros(20, dtype=torch.bool),
                                        'gt': torch.ones(20, dtype=torch.bool)})
assert subs['all'].n == 20 and subs['gt'].n == 20 and subs['le'].n == 0
print('OK split/subsets:', subs['gt'].out())
print('ALL TESTS PASSED')
