#!/usr/bin/env python3
"""cmp_traces.py -- strict field-for-field comparison of two trainer probe traces.

Usage: cmp_traces.py <a.jsonl> <b.jsonl> [--expect-rows N]

Used for the flag-off equivalence proof: the original nrmt_train.py and the generated
nrmt_train_ewc.py run with --ewc-lambda 0.0 (or with the flag absent) must agree on every logged
field of every step, bit for bit.
Also summarises which EWC fields are present (they must be absent at lambda=0).
"""
import json, sys


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def main():
    a = load(sys.argv[1])
    b = load(sys.argv[2])
    expect = None
    if '--expect-rows' in sys.argv:
        expect = int(sys.argv[sys.argv.index('--expect-rows') + 1])
    keys = sorted({k for r in a + b for k in r})
    bad = []
    for i, (x, y) in enumerate(zip(a, b)):
        for k in keys:
            if x.get(k) != y.get(k):
                bad.append((x.get('step', i), k, x.get(k), y.get(k)))
    print(f'rows A={len(a)} B={len(b)} compared_fields={len(keys)}')
    print('fields:', ','.join(keys))
    print('mismatched fields:', len(bad))
    for row in bad[:8]:
        print('   ', row)
    ok = (not bad) and len(a) == len(b) and (expect is None or len(a) == expect)
    print('EQUIVALENCE:', 'IDENTICAL' if ok else 'DIFFERENT')
    ewc_a = [k for k in keys if k.startswith('ewc_')]
    print('ewc_* fields present in A:', ewc_a)
    print('ewc_* fields present in B:', [k for k in keys if k.startswith('ewc_')])
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
