#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""depth_report.py -- matched-step comparison of the DEPTH arm(s) against the CTL comparator.

Parses the trainer stdout logs (the only place CTL's trajectory exists: results_RCA_FROZEN_CTL.json
was never written because the run was SIGTERM'd at step 15000) plus the per-step probe jsonl
(gradient norms).

READ-ONLY.  Usage:
  python depth_report.py --ctl-log train_RCA_FROZEN_CTL.log \
      --arm RCA_DEPTH_T8:train_RCA_DEPTH_T8.log:trace_RCA_DEPTH_T8.jsonl \
      --arm RCA_DEPTH_T12:train_RCA_DEPTH_T12.log:trace_RCA_DEPTH_T12.jsonl
"""
import argparse
import json
import re
import statistics

EV = re.compile(r'\[eval @(\d+)\]')
A1 = re.compile(r'acc@1 ([\d.]+)% acc@5 ([\d.]+)% CE_z ([\d.]+)')


def num(pat, line, default=None):
    m = re.search(pat, line)
    return float(m.group(1)) if m else default


def parse_log(path):
    """-> {step: {'ALL_val':(a1,a5,cez), 'NOVEL_only':..., 'ALL_val_RCA_OFF':...,
    'gate':[...], 'gate_absmean':float, 'h_drift':float}}"""
    out = {}
    try:
        fh = open(path, errors='replace')
    except FileNotFoundError:
        return out
    for line in fh:
        m = EV.search(line)
        if not m:
            continue
        step = int(m.group(1))
        rec = {}
        for key in ('ALL_val', 'NOVEL_only', 'ALL_val_RCA_OFF', 'NOVEL_only_RCA_OFF'):
            seg = line.split(key + ':')
            if len(seg) > 1:
                mm = A1.search(seg[1])
                if mm:
                    rec[key] = tuple(float(x) for x in mm.groups())
        g = re.search(r'RCA gates ([-+0-9.,eE]+)', line)
        if g:
            try:
                rec['gate'] = [float(x) for x in g.group(1).rstrip(',').split(',')]
                rec['gate_absmean'] = sum(abs(x) for x in rec['gate']) / len(rec['gate'])
            except ValueError:
                pass
        hd = num(r'h_drift ([-+0-9.eE]+)', line)
        if hd is not None:
            rec['h_drift'] = hd
        fs = num(r'feat_scale ([\d.]+)', line)
        if fs is not None:
            rec['feat_scale'] = fs
        out[step] = rec
    return out


def parse_trace(path, win=200):
    """-> {step: {'gnorm_rca':..,'gnorm_head':..,'gnorm_trunk':..,'h_drift_max':..}} using the
    MEAN gradient norm over the `win` steps ending at that step (instantaneous values are noisy)."""
    rows = []
    try:
        fh = open(path, errors='replace')
    except FileNotFoundError:
        return {}, 0
    for line in fh:
        try:
            d = json.loads(line)
        except (ValueError, TypeError):
            continue
        rows.append(d)
    out = {}
    for i, d in enumerate(rows):
        s = int(d.get('step', i + 1))
        blk = rows[max(0, i - win + 1):i + 1]
        out[s] = {
            'gnorm_rca': statistics.fmean([b.get('grad_norm_rca') or 0.0 for b in blk]),
            'gnorm_head': statistics.fmean([b.get('grad_norm_head') or 0.0 for b in blk]),
            'gnorm_trunk': statistics.fmean([b.get('grad_norm_trunk') or 0.0 for b in blk]),
            'h_drift_max': d.get('h_drift_max'),
            'it_per_s': d.get('it_per_s'),
        }
    return out, len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ctl-log', required=True)
    ap.add_argument('--ctl-trace', default=None)
    ap.add_argument('--arm', action='append', default=[],
                    help='TAG:logpath[:tracepath]')
    ap.add_argument('--max-step', type=int, default=15000,
                    help='matched-comparison horizon (CTL has a reading at 15000)')
    ap.add_argument('--json-out', default=None)
    args = ap.parse_args()

    ctl = parse_log(args.ctl_log)
    ctlt = parse_trace(args.ctl_trace)[0] if args.ctl_trace else {}
    arms = []
    for spec in args.arm:
        parts = spec.split(':')
        tag, log = parts[0], parts[1]
        tr = parts[2] if len(parts) > 2 else None
        arms.append({'tag': tag, 'evals': parse_log(log),
                     'trace': parse_trace(tr)[0] if tr else {},
                     'n_trace_steps': parse_trace(tr)[1] if tr else 0})

    steps = sorted(s for s in ctl if s <= args.max_step)
    print(f'CTL evals available: {sorted(ctl)}')
    for a in arms:
        print(f"{a['tag']} evals available: {sorted(a['evals'])}  trace steps: {a['n_trace_steps']}")
    print()

    for a in arms:
        print(f"===== MATCHED-STEPS vs CTL (up to {args.max_step}) : {a['tag']} vs RCA_FROZEN_CTL =====")
        print(f"{'step':>6} | {'CTL acc@1':>9} {'CTL acc@5':>9} {'CTL CE_z':>8} | "
              f"{'arm acc@1':>9} {'arm acc@5':>9} {'arm CE_z':>8} | {'d acc@1':>8} | "
              f"{'gn_rca':>9} {'gn_head':>9} | {'gate|mean|':>10}")
        for s in steps:
            c = ctl.get(s, {})
            v = a['evals'].get(s, {})
            if not v:
                continue
            cA = c.get('ALL_val', (float('nan'),) * 3)
            vA = v.get('ALL_val', (float('nan'),) * 3)
            g = v.get('gate_absmean')
            t = a['trace'].get(s, {})
            print(f"{s:>6} | {cA[0]:>9.2f} {cA[1]:>9.2f} {cA[2]:>8.3f} | "
                  f"{vA[0]:>9.2f} {vA[1]:>9.2f} {vA[2]:>8.3f} | {vA[0]-cA[0]:>+8.2f} | "
                  f"{t.get('gnorm_rca', float('nan')):>9.2e} {t.get('gnorm_head', float('nan')):>9.2e} | "
                  f"{(g if g is not None else float('nan')):>10.4f}")
        # NOVEL + ablation at the last matched step
        s = [x for x in steps if x in a['evals']]
        if s:
            s = s[-1]
            v = a['evals'][s]
            c = ctl.get(s, {})
            for key in ('NOVEL_only', 'ALL_val_RCA_OFF', 'NOVEL_only_RCA_OFF'):
                if key in v:
                    cc = c.get(key)
                    print(f"  @{s} {key:>20}: arm acc@1 {v[key][0]:.2f}% acc@5 {v[key][1]:.2f}% "
                          f"CE_z {v[key][2]:.3f}"
                          + (f" | CTL {cc[0]:.2f}% / {cc[2]:.3f}" if cc else ' | CTL n/a'))
            if 'gate' in v:
                print(f"  @{s} gates per layer: {[round(x, 4) for x in v['gate']]}")
            print(f"  @{s} h_drift {v.get('h_drift')} feat_scale {v.get('feat_scale')}")
        # plateau statistics over the last 3 matched evals
        got = [(x, a['evals'][x]['ALL_val'][0]) for x in steps if x in a['evals']]
        if len(got) >= 3:
            tail = got[-3:]
            print(f"  plateau check (last 3 matched): " +
                  ' '.join(f'{x}:{y:.2f}%' for x, y in tail) +
                  f" -> spread {max(y for _, y in tail) - min(y for _, y in tail):.2f} pp")
        print()

    if args.json_out:
        blob = {'ctl': {str(k): v for k, v in ctl.items()},
                'arms': [{'tag': a['tag'], 'evals': {str(k): v for k, v in a['evals'].items()},
                          'trace': {str(k): v for k, v in a['trace'].items()}} for a in arms]}
        json.dump(blob, open(args.json_out, 'w'), indent=1)
        print('wrote', args.json_out)


if __name__ == '__main__':
    main()
