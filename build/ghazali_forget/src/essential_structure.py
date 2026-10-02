#!/usr/bin/env python3
"""essential_structure.py -- turn "essential structure" into a list of named tensors.

No model, no GPU: pure arithmetic on the two saved diagonal Fishers.  Reads out which of the 92
trunk tensors (layers 20-23) the RETAINED CAPABILITY actually depends on (F_head, the FIX head's
root CE), where the text Fisher ranks them (F_ret), and which ones are the "differentia" -- high
capability importance, low text importance.
"""
import json
import torch

O = '/workspace/ghazali_forget'
P = {t: torch.load(f'{O}/fisher_{t}.pt', map_location='cpu') for t in ('retained', 'head')}
names = sorted(P['head'].keys())
rows = [(n, float(P['head'][n].double().mean()), float(P['retained'][n].double().mean()),
         P['head'][n].numel()) for n in names]
rows.sort(key=lambda r: -r[1])
ret_order = {n: i for i, (n, *_r) in enumerate(sorted(rows, key=lambda r: -r[2]))}
tot = sum(r[1] * r[3] for r in rows)
out = {'n_tensors': len(rows), 'top_head': [], 'differentia': [], 'tensor_rows': [
    {'name': n, 'F_head': fh, 'F_ret': fr, 'numel': num} for n, fh, fr, num in rows]}

hdr = f'{"tensor":<56}{"F_head":>12}{"F_ret":>12}{"rank_ret":>9}{"numel":>10}'
print('=== the 18 tensors the RETAINED CAPABILITY depends on most (F_head) ===')
print(hdr)
for n, fh, fr, num in rows[:18]:
    print(f'{n:<56}{fh:>12.3e}{fr:>12.3e}{ret_order[n]:>9}{num:>10}')
    out['top_head'].append({'name': n, 'F_head': fh, 'F_ret': fr,
                            'rank_ret': ret_order[n], 'numel': num})
mass = 100 * sum(r[1] * r[3] for r in rows[:18]) / tot
print(f'\ntop-18 F_head tensors hold {mass:.1f}% of the capability Fisher mass')

print('\n=== the DIFFERENTIA: high F_head, low F_ret ===')
for n, fh, fr, num in sorted(rows[:20], key=lambda r: -(r[1] / (r[2] + 1e-12)))[:10]:
    print(f'  {n:<56} F_head {fh:.3e}  F_ret {fr:.3e}  ratio {fh/(fr+1e-12):.2f}')
    out['differentia'].append({'name': n, 'F_head': fh, 'F_ret': fr, 'ratio': fh / (fr + 1e-12)})

print('\n=== the OPPOSITE: tensors the text Fisher ranks top, capability does not ===')
for n, fh, fr, num in sorted(rows, key=lambda r: r[3] and -r[2])[:8]:
    print(f'  {n:<56} F_ret {fr:.3e}  rank_head {rows.index([r for r in rows if r[0]==n][0])}')
    out.setdefault('genus', []).append({'name': n, 'F_ret': fr})

json.dump(out, open(f'{O}/essential_structure.json', 'w'), indent=2)
print('ESSENTIAL_DONE')
