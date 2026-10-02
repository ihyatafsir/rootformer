#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""collect.py -- parse the per-eval and per-step lines of the arm logs into comparison tables.

READ-ONLY.  Usage:
  python collect.py /workspace/root_attn_ctl/logs/train_RCA_FROZEN_CTL.log:CTL \
                    /workspace/root_attn/logs/train_RCA_UNFREEZE_A2.log:A2 ...
(the parent's chain_final.sh writes to $OUT/train_*.log, not $OUT/logs/, so give explicit paths)
"""
import re
import sys

EV = re.compile(r'\[eval @(?P<step>\d+)\]')
A1 = re.compile(r'ALL_val: acc@1 (?P<v>[\d.]+)% acc@5 (?P<a5>[\d.]+)% CE_z (?P<ce>[-\d.]+)')
NOV = re.compile(r'NOVEL_only: acc@1 (?P<v>[\d.]+)% acc@5 (?P<a5>[\d.]+)% CE_z (?P<ce>[-\d.]+)')
OFF = re.compile(r'ALL_val_RCA_OFF: acc@1 (?P<v>[\d.]+)% acc@5 (?P<a5>[\d.]+)% CE_z (?P<ce>[-\d.]+)')
GATE = re.compile(r'RCA gates (?P<g>[+\-\d.,eE ]+?) \(\|mean\| (?P<gm>[\d.]+)\)')
DRIFT = re.compile(r'h_drift (?P<d>[-\d.eE+]+)')
STEP = re.compile(r'^\s*step (?P<step>\d+)/\d+ loss (?P<loss>[\d.]+).*?gnorm head=(?P<gh>[-\d.eE+]+) '
                  r'rca=(?P<gr>[-\d.eE+]+) trunk=(?P<gt>[-\d.eE+]+)(?:.*?h_drift (?P<hd>[-\d.eE+]+))?')
PROOF = re.compile(r'\[proof\] step (?P<step>\d+):.*?max\|dh\|=(?P<dh>[-\d.eE+]+).*?'
                   r'grad_norm head=(?P<gh>[-\d.eE+]+) rca=(?P<gr>[-\d.eE+]+) trunk=(?P<gt>[-\d.eE+]+)')


def parse(path):
    evs, steps, proofs = [], [], []
    try:
        lines = open(path, errors='replace').read().splitlines()
    except FileNotFoundError:
        return None
    for ln in lines:
        if 'WARN' in ln or 'warn' in ln:
            continue
        m = EV.search(ln)
        if m:
            e = {'step': int(m.group('step'))}
            for key, rx in (('all', A1), ('novel', NOV), ('off', OFF)):
                r = rx.search(ln)
                if r:
                    e[key] = (float(r.group('v')), float(r.group('a5')), float(r.group('ce')))
            r = GATE.search(ln)
            if r:
                e['gates'] = [float(x) for x in r.group('g').replace(' ', '').split(',') if x]
                e['gate_absmean'] = float(r.group('gm'))
            r = DRIFT.search(ln)
            if r:
                e['h_drift'] = float(r.group('d'))
            evs.append(e)
            continue
        m = PROOF.search(ln)
        if m:
            proofs.append({k: (float(v) if k != 'step' else int(v)) for k, v in m.groupdict().items()})
            continue
        m = STEP.match(ln)
        if m:
            steps.append({'step': int(m.group('step')), 'loss': float(m.group('loss')),
                          'gh': float(m.group('gh')), 'gr': float(m.group('gr')),
                          'gt': float(m.group('gt')),
                          'hd': (float(m.group('hd')) if m.group('hd') else None)})
    return {'evals': evs, 'steps': steps, 'proofs': proofs}


def main():
    arms = []
    for a in sys.argv[1:]:
        path, _, label = a.partition(':')
        d = parse(path)
        if d is None:
            print(f'## {label or path}: MISSING')
            continue
        arms.append((label or path, d))

    print('\n=== acc@1 / acc@5 / CE_z per eval (ALL_val, then NOVEL_only, then RCA_OFF) ===')
    for label, d in arms:
        print(f'\n## {label}  ({len(d["evals"])} evals)')
        print('| step | ALL acc@1 | ALL acc@5 | ALL CE_z | NOVEL acc@1 | NOVEL CE_z |'
              ' RCA_OFF acc@1 | gate |mean| | h_drift |')
        print('|---|---|---|---|---|---|---|---|---|')
        for e in d['evals']:
            g = f"{e['gate_absmean']:.4f}" if 'gate_absmean' in e else '--'
            hd = f"{e['h_drift']:.4f}" if 'h_drift' in e else '--'
            f = lambda k, i, p=2: (f"{e[k][i]:.{p}f}" if k in e else '--')
            print(f"| {e['step']} | {f('all',0)} | {f('all',1)} | {f('all',2,4)} | "
                  f"{f('novel',0)} | {f('novel',2,4)} | {f('off',0)} | {g} | {hd} |")

    print('\n=== RCA / head / trunk gradient norms at each training step (h_drift on fixed batch) ===')
    for label, d in arms:
        print(f'\n## {label}')
        print('| step | loss | gnorm head | gnorm rca | gnorm trunk | h_drift |')
        print('|---|---|---|---|---|---|')
        for s in d['steps']:
            hd = f"{s['hd']:.4f}" if s.get('hd') is not None else '--'
            print(f"| {s['step']} | {s['loss']:.4f} | {s['gh']:.4e} | {s['gr']:.4e} | {s['gt']:.4e} | {hd} |")

    print('\n=== h_drift proof lines (fixed batch) ===')
    for label, d in arms:
        print(f'\n## {label}: ' + ' | '.join(
            f"step {p['step']} max|dh|={p['dh']:.4f} rca={p['gr']:.3e} trunk={p['gt']:.3e}"
            for p in d['proofs']))

    print('\n=== bare numbers for the report ===')
    for label, d in arms:
        if not d['evals']:
            continue
        print(f"{label}: " + ' '.join(f"{e['step']}:{e['all'][0]:.2f}"
                                      for e in d['evals'] if 'all' in e))


if __name__ == '__main__':
    main()
