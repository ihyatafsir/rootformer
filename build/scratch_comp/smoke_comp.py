#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CPU smoke test for scratch_lm_comp.py.

(1) unit-test CompRootHead.teacher_forced against a manual chain-rule computation;
(2) unit-test best_first_content against BRUTE-FORCE enumeration of all 2..5-letter roots
    over a small alphabet (exactness of the top-k and of the ordering);
(3) run a few training steps + eval_bits + eval_heads on a synthetic dataset to exercise
    every code path end to end.
"""
import itertools
import json
import math
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, '/workspace/scratch_comp')
import scratch_lm_comp as S  # noqa: E402

torch.manual_seed(0)
dev = torch.device('cpu')

# ---------------------------------------------------------------- (1)+(2) unit tests
n_alpha, d, n_gate = 3, 16, 4
head = S.CompRootHead(d, n_gate, n_alpha + 1, d_dec=8, d_emb=4)
h = torch.randn(1, 2, d)
letters = torch.tensor([[[0, 1, 2, 0, 0], [1, 1, 0, 0, 0]]])       # (1,2,5)
lens = torch.tensor([[3, 2]])
slot_logp, slot_valid, eos = head.teacher_forced(h, letters, lens)
print('slot_valid[0,0] =', slot_valid[0, 0].tolist(), '(expect T,T,T,T,F)')
print('slot_valid[0,1] =', slot_valid[0, 1].tolist(), '(expect T,T,T,F,F)')
assert slot_valid[0, 0].tolist() == [True, True, True, True, False]
assert slot_valid[0, 1].tolist() == [True, True, True, False, False]

# manual chain rule for position 0
state = torch.tanh(head.h0(h))[0, 0]
inp = torch.tensor(eos)
manual = 0.0
seq = [0, 1, 2, eos]
for s in range(4):
    state = head.gru(head.inp(inp), state)
    logp = F.log_softmax(head.out(state), -1)
    manual += float(logp[seq[s]])
    inp = torch.tensor(seq[s])
auto = float(slot_logp[0, 0].sum())
print(f'teacher-forced NLL sum={-auto:.6f}  manual={-manual:.6f}')
assert abs(auto - manual) < 1e-6, (auto, manual)

# brute-force top-5
theta = torch.full((2,), float('-inf'))
found, exact = S.best_first_content(head, h.reshape(-1, d), theta, k=5, max_rounds=5000)
print('best-first exact flags:', exact)
brute = []
for L in range(2, 6):
    for combo in itertools.product(range(n_alpha), repeat=L):
        st = torch.tanh(head.h0(h))[0, 0]
        inp = torch.tensor(eos)
        sc = 0.0
        for c in combo:
            st = head.gru(head.inp(inp), st)
            lg = F.log_softmax(head.out(st), -1)
            sc += float(lg[c])
            inp = torch.tensor(c)
        if L < 5:
            st = head.gru(head.inp(inp), st)
            lg = F.log_softmax(head.out(st), -1)
            sc += float(lg[eos])
        brute.append((sc, tuple(combo)))
brute.sort(key=lambda x: -x[0])
bf = [(round(s, 6), t) for s, t in found[0]]
br = [(round(s, 6), t) for s, t in brute[:5]]
print('best-first top5:', bf)
print('brute-force top5:', br)
assert [t for _, t in bf] == [t for _, t in br], (bf, br)
assert all(abs(a[0] - b[0]) < 1e-4 for a, b in zip(bf, br)), (bf, br)
assert exact == [True, True]
print('[OK] best-first is exact against brute force\n')

# ---------------------------------------------------------------- (3) end-to-end
meta = json.load(open('/workspace/sf_data/meta.json')) if Path('/workspace/sf_data/meta.json').exists() else None
if meta is None:
    print('[skip] no real meta available for the end-to-end part')
    sys.exit(0)

alphabet = list('ابتثجحخدذرزسشصضطظعغفقكلمنهوي')
char2i = {c: i for i, c in enumerate(alphabet)}
n_roots = 60
special_ids = [0, 1, 2, 3, 4, 5, 6]
content_ids = [i for i in range(7, n_roots)]
root_letters = {}
for rid in content_ids:
    L = 2 + (rid % 4)
    root_letters[str(rid)] = [(rid * 7 + k * 3) % len(alphabet) for k in range(L)]
smeta = {'n_roots': n_roots, 'n_awzan': 5, 'n_prefixes': 4, 'n_suffixes': 4,
         'hold_roots': [7, 8], 'special_root_ids': special_ids,
         'content_root_ids': content_ids,
         'roots_list': ['<PAD>', '<BOS>', '<EOS>', '<UNK>', '<PARTICLE>', '<P:a>', '<P:b>']
                      + ['r%d' % i for i in content_ids],
         'alphabet': alphabet, 'n_alphabet': len(alphabet), 'n_sym': len(alphabet) + 1,
         'root_letters': root_letters, 'inexpressible_content_roots': [],
         'hold_roots_inexpressible': [], 'train_words': 1000,
         'blueprint_md5': 'x', 'root_len_dist_train_tokens': {'3': 900}}
# make held-out roots expressible
for rid in smeta['hold_roots']:
    smeta['root_letters'][str(rid)] = [0, 1, 2]
# roots_list entries for content must be real strings for tables.root_str / letters_str
smeta['roots_list'] = (smeta['roots_list'][:7]
                       + [''.join(alphabet[c] for c in smeta['root_letters'][str(i)])
                          for i in content_ids])
tables = S.Tables(smeta)
data = {}
rng = torch.Generator().manual_seed(1)
for sp in ('train', 'val', 'test_gen', 'test_deriv'):
    P, R, W, Ss, L = [], [], [], [], []
    for _ in range(30):
        n = 6
        P.append([i % 4 for i in range(n)])
        R.append([int(x) for x in torch.randint(0, n_roots, (n,), generator=rng)])
        W.append([i % 5 for i in range(n)])
        Ss.append([i % 4 for i in range(n)])
        L.append(n)
    data[sp] = {'P': P, 'R': R, 'W': W, 'S': Ss, 'L': L}
# force held-out roots into test_deriv targets
for k in range(15):
    data['test_deriv']['R'][k][3] = smeta['hold_roots'][k % 2]
    data['test_deriv']['R'][k][1] = smeta['hold_roots'][(k + 1) % 2]

model = S.LMComp((n_roots, 5, 4, 4), tables.n_gate, smeta['n_sym'], d=32, nl=2, h=2, ffn=64)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
for step in range(5):
    b = next(S.batches_morph(data, 'train', 4, __import__('random').Random(step)))
    p_in = S.pad([x[:-1] for x in b['p']], 0)
    r_in = S.pad([x[:-1] for x in b['r']], 0)
    w_in = S.pad([x[:-1] for x in b['w']], 0)
    s_in = S.pad([x[:-1] for x in b['s']], 0)
    r_tg = S.pad([x[1:] for x in b['r']], 0)
    w_tg = S.pad([x[1:] for x in b['w']], 0)
    p_tg = S.pad([x[1:] for x in b['p']], 0)
    s_tg = S.pad([x[1:] for x in b['s']], 0)
    hh, gl, lw, lp, ls = model(p_in, r_in, w_in, s_in)
    gate_tg = tables.gate_tab[r_tg]
    loss_gate = F.cross_entropy(gl.reshape(-1, gl.shape[-1]), gate_tg.reshape(-1),
                               ignore_index=-100)
    slot_logp, slot_valid, _ = model.root.teacher_forced(
        hh, tables.letters[r_tg], tables.lens[r_tg])
    loss_letters = -(slot_logp * slot_valid).sum() / slot_valid.sum().clamp(min=1)
    loss = loss_gate + loss_letters
    loss.backward(); opt.step(); opt.zero_grad()
    print(f'  smoke step {step} loss {float(loss):.4f} gate {float(loss_gate):.4f} '
          f'letters {float(loss_letters):.4f}')
    assert math.isfinite(float(loss)), 'NaN/Inf in loss'

for sp in ('val', 'test_gen', 'test_deriv'):
    b = S.eval_bits(model, data, sp, dev, tables, 'comp', max_sent=30)
    print(f'  bits/word {sp}: {b}')
    assert all(math.isfinite(v) for v in b.values() if isinstance(v, float))
    hr, recs = S.eval_heads(model, data, sp, dev, tables, 'comp', max_sent=30,
                            max_rounds=200, max_rounds_free=300, free_max_pos=50)
    print(f'  heads {sp}: joint={hr["acc"]} content_only={hr["content_only_acc"]} '
          f'inexact={hr["n_bfs_inexact_joint"]}/{hr["n_bfs_inexact_free"]}')
    if recs:
        pc = S.partial_credit(recs, tables)
        print(f'  partial credit: n={pc["n"]} exact1={pc["exact_top1_rate"]:.3f} '
              f'content1={pc["content_top1_rate"]:.3f} anagram={pc["anagram_top1_rate"]:.3f} '
              f'slot={pc["slot_acc_top1"]:.3f} edit={pc["mean_edit_top1"]:.2f}')
        print('  example:', json.dumps(pc['examples'][0], ensure_ascii=False))

print('\n[OK] smoke test passed')
