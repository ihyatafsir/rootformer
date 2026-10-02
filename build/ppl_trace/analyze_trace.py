#!/usr/bin/env python3
"""Analyse a per-step nrmt_train.py --probe-out trace.

Emits a readable per-step table (step, loss, ppl, root/wazn/prefix/suffix CE, grad_norm,
impossible) as TSV, and prints summary statistics + NaN/Inf audit.
"""
import json
import math
import sys
from pathlib import Path


def load(path):
    rows = []
    for ln in Path(path).read_text().splitlines():
        ln = ln.strip()
        if ln:
            rows.append(json.loads(ln))
    return rows


def main():
    src = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_suffix('.tsv')
    rows = load(src)
    print(f'[analyse] {src}: {len(rows)} steps')

    steps = [r['step'] for r in rows]
    expected = list(range(1, len(rows) + 1))
    print(f'[analyse] step numbers contiguous 1..{len(rows)}: {steps == expected}')
    missing = sorted(set(expected) - set(steps))
    if missing:
        print(f'[analyse] MISSING steps: {missing[:50]}')

    bad = []
    for r in rows:
        for k, v in r.items():
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                bad.append((r['step'], k, v))
    print(f'[analyse] NaN/Inf fields: {len(bad)}')
    if bad:
        for b in bad[:50]:
            print(f'    step {b[0]} {b[1]} = {b[2]}')

    losses = [r['loss'] for r in rows]
    ppls = [math.exp(min(l, 700.0)) for l in losses]

    print(f'[analyse] loss  min={min(losses):.6f} @step {steps[losses.index(min(losses))]}'
          f'  max={max(losses):.6f} @step {steps[losses.index(max(losses))]}'
          f'  final={losses[-1]:.6f}')
    print(f'[analyse] ppl   min={min(ppls):.6g} @step {steps[ppls.index(min(ppls))]}'
          f'  max={max(ppls):.6g} @step {steps[ppls.index(max(ppls))]}'
          f'  final={ppls[-1]:.6g}')

    print('[analyse] spikes loss>20: ' +
          ', '.join(f'{s}:{l:.2f}' for s, l in zip(steps, losses) if l > 20))
    print('[analyse] steps grad_norm>100: ' +
          ', '.join(f'{r["step"]}:{r["grad_norm"]:.0f}' for r in rows if r['grad_norm'] > 100))

    # tail means (flattening detection)
    for w in (10, 50, 100, 200, 500):
        if len(losses) >= w:
            seg = losses[-w:]
            print(f'[analyse] last {w:>4} steps: mean loss {sum(seg)/w:.4f} '
                  f'mean ppl {sum(math.exp(min(x,700)) for x in seg)/w:.4g}')

    cols = ['step', 'loss', 'ppl', 'root', 'wazn', 'prefix', 'suffix',
            'impossible', 'grad_norm', 'lr', 'p_ss']
    with out.open('w') as fh:
        fh.write('\t'.join(cols) + '\n')
        for r in rows:
            vals = []
            for c in cols:
                if c == 'ppl':
                    vals.append(f'{math.exp(min(r["loss"], 700.0)):.6g}')
                elif c in r:
                    v = r[c]
                    vals.append(f'{v:.6f}' if isinstance(v, float) else str(v))
                else:
                    vals.append('')
            fh.write('\t'.join(vals) + '\n')
    print(f'[analyse] wrote {out}')


if __name__ == '__main__':
    main()
