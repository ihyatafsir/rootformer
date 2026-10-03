#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_arms.py -- the Phase 1 report: A vs C at matched steps, against all five comparators.

Reads whatever of `/workspace/root_arch/arms/results_*.json` exists and prints, per arm:
  * final-step and BEST `ALL_val.acc@1` (the headline number)
  * `NOVEL_only.acc@1` -- the only number that proves generalisation rather than memorisation
  * `ce_z` (the scale-invariant cross-entropy) and the causal ablations
  * wall-clock and steps/s from the log file's birth/mtime
  * the cumulative it/s the trainer itself printed
  * per-arm PEAK VRAM from the runner's per-process `vram.csv` joined through `queue/pid_<tag>`

Then the verdict, stated against the recorded comparators:

    19.53 %  RCA_UNFREEZE_A2   late-attached (layers 20-23), 0.01x LR, frozen for most of training
    29.30 %  raw-base probe    root decodability, original Qwen2.5-0.5B   (unseen 23.34 %)
    92.34 %  transmuted probe  root decodability, current trunk          (unseen 83.92 %)
     3.551 % marginal
     2.82 %  RCA ablated

THE QUESTION: does embedding root structure from the INPUT STAGE, with a normally-trained trunk,
recover or exceed 19.53 %?  And, because A is mandatory: if A ALSO scores well, a properly trained
Arabic model just learns roots and the root pathway is not doing the work -- which is a real answer.
"""
import argparse
import glob
import json
import os
import re
import sys
from datetime import datetime, timezone

ARMS = '/workspace/root_arch/arms'
QUEUE = '/workspace/root_arch/queue'
SMOKE = '/workspace/root_arch/smoke'

COMPARATORS = [
    ('RCA_UNFREEZE_A2   late-attached (20-23), 0.01x LR, frozen most of training',
     19.53, 19.90),
    ('raw-base probe    root decodability, original Qwen2.5-0.5B',
     29.30, 23.34),
    ('transmuted probe  root decodability, current trunk',
     92.34, 83.92),
    ('marginal', 3.551, None),
    ('RCA ablated', 2.82, None),
]
LABEL = {
    'FLOOR_A':     'A  FLOOR       no root pathway',
    'EARLYROOT_C': 'C  EARLY-ROOT  24 layers, residual + score bias',
    'RESIDUAL_R':  'E  RESIDUAL    24 layers, residual only',
    'SCOREBIAS_D': 'D  SCORE-BIAS  24 layers, score bias only',
    'LATE_X':      'X  LATE-ATTACH both mechanisms, layers 20-23 only',
    'FLOOR_A_V2':  'A2 FLOOR (clean) no root pathway, current attention implementation',
    'FLOOR_A_S2':  'A3 FLOOR seed=1',
    'EARLYROOT_C_S2': 'C2 EARLY-ROOT seed=1',
}


def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        print(f'  ! could not read {path}: {e}')
        return None


def pct(x):
    return 'n/a' if x is None else f'{100.0 * x:.2f}%'


def arm_stats(tag):
    st = {'tag': tag, 'label': LABEL.get(tag, tag)}
    res = load(f'{ARMS}/results_{tag}.json')
    if res is None:
        for alt in (f'{SMOKE}/results_{tag}.json',):
            res = load(alt)
            if res:
                st['source'] = alt
                break
    if res and res.get('history'):
        h = res['history']
        st['final_step'] = h[-1]['step']
        st['final_all'] = h[-1]['ALL_val']['acc@1']
        st['final_novel'] = h[-1]['NOVEL_only']['acc@1']
        st['final_ce_z'] = h[-1]['ALL_val'].get('ce_z')
        st['final_ppl'] = h[-1]['ALL_val'].get('ppl')
        best = max(h, key=lambda r: r['ALL_val']['acc@1'])
        st['best_all'] = best['ALL_val']['acc@1']
        st['best_all_step'] = best['step']
        st['best_novel'] = best['NOVEL_only']['acc@1']
        st['best_novel_step'] = best['step']
        st['curve'] = [(r['step'], round(100 * r['ALL_val']['acc@1'], 2),
                        round(100 * r['NOVEL_only']['acc@1'], 2)) for r in h]
        for k in ('ALL_val_RCA_OFF', 'NOVEL_only_RCA_OFF', 'ALL_val_ISHTIQAQ_OFF',
                  'NOVEL_only_ISHTIQAQ_OFF'):
            if k in h[-1]:
                st[k] = h[-1][k]['acc@1']
        st['args'] = {k: res.get('args', {}).get(k) for k in
                      ('lr', 'trunk_lr_scale', 'steps', 'batch_size', 'root_cross_attn',
                       'ishtiqaq_root_bias', 'ishtiqaq_root_source', 'sdpa', 'pillar_freeze',
                       'flash_hooks', 'unfreeze_trunk_all', 'model_v13')}
    # trace -> it/s
    tr = f'{ARMS}/trace_{tag}.jsonl'
    if not os.path.exists(tr):
        tr = f'{SMOKE}/trace_{tag}.jsonl'
    if os.path.exists(tr):
        n = sum(1 for l in open(tr) if l.strip())
        st['trace_steps'] = n
        los = []
        for l in open(tr):
            if l.strip():
                try:
                    los.append(json.loads(l)['loss'])
                except Exception:
                    pass
        if los:
            st['loss_first'] = los[0]
            st['loss_last'] = los[-1]
    # wall clock from the log file's birth/mtime
    lg = f'{ARMS}/log_{tag}.txt'
    if not os.path.exists(lg):
        lg = f'{SMOKE}/log_{tag}.txt'
    if os.path.exists(lg):
        s = os.stat(lg)
        birth = getattr(s, 'st_birthtime', s.st_ctime)
        st['wall_s'] = max(0.0, s.st_mtime - birth)
        txt = open(lg, errors='replace').read()
        its = re.findall(r'([0-9.]+) it/s', txt)
        if its:
            st['reported_it_s'] = float(its[-1])
        st['peak_vram_G'] = peak_vram(tag)
    return st


def peak_vram(tag):
    """Peak per-process VRAM for an arm, from the runner's per-poll snapshot."""
    pf = f'{QUEUE}/pid_{tag}'
    csv = f'{QUEUE}/vram.csv'
    if not (os.path.exists(pf) and os.path.exists(csv)):
        return None
    pid = open(pf).read().strip()
    peak = 0
    for line in open(csv):
        p = line.strip().split(',')
        if len(p) == 3 and p[1] == pid:
            try:
                peak = max(peak, int(p[2]))
            except ValueError:
                pass
    return round(peak / 1024.0, 2) if peak else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tags', nargs='*',
                    default=['FLOOR_A_V2', 'EARLYROOT_C', 'RESIDUAL_R', 'SCOREBIAS_D',
                             'LATE_X', 'FLOOR_A', 'EARLYROOT_C_S2'])
    ap.add_argument('--json-out', default='/workspace/root_arch/arms/comparison.json')
    args = ap.parse_args()

    print('=' * 100)
    print('PHASE 1 -- A vs C at MATCHED steps, against the recorded comparators')
    print('=' * 100)
    print(f'  generated {datetime.now(timezone.utc).isoformat(timespec="seconds")}')
    print()

    stats = [arm_stats(t) for t in args.tags]
    present = [s for s in stats if 'final_all' in s]

    for s in stats:
        print('-' * 100)
        print(f'  {s["label"]}   [{s["tag"]}]')
        print('-' * 100)
        if 'final_all' not in s:
            print('   NOT PRESENT / no completed history yet')
            if s.get('trace_steps'):
                print(f'   in flight: {s["trace_steps"]} steps logged, '
                      f'loss {s.get("loss_first")} -> {s.get("loss_last")}')
            print()
            continue
        a = s.get('args', {})
        print(f'   recipe   : lr={a.get("lr")} trunk_lr_scale={a.get("trunk_lr_scale")} '
              f'steps={a.get("steps")} batch={a.get("batch_size")}')
        print(f'              model_v13={a.get("model_v13")} unfreeze_trunk_all={a.get("unfreeze_trunk_all")} '
              f'sdpa={a.get("sdpa")} pillar_freeze={a.get("pillar_freeze")} '
              f'flash_hooks={a.get("flash_hooks")}')
        print(f'              root_cross_attn={a.get("root_cross_attn")} '
              f'ishtiqaq_root_bias={a.get("ishtiqaq_root_bias")} '
              f'source={a.get("ishtiqaq_root_source")}')
        print(f'   FINAL @{s["final_step"]:<6d} ALL_val acc@1 {pct(s["final_all"])}   '
              f'NOVEL_only acc@1 {pct(s["final_novel"])}   ce_z {s["final_ce_z"]:.4f}   '
              f'raw PPL {s["final_ppl"]:.1f}')
        print(f'   BEST            ALL_val acc@1 {pct(s["best_all"])} @step {s["best_all_step"]}   '
              f'NOVEL_only {pct(s["best_novel"])} @step {s["best_novel_step"]}')
        if 'ALL_val_RCA_OFF' in s:
            print(f'   ABLATION  RCA off        ALL {pct(s["ALL_val_RCA_OFF"])}  '
                  f'NOVEL {pct(s.get("NOVEL_only_RCA_OFF"))}')
        if 'ALL_val_ISHTIQAQ_OFF' in s:
            print(f'   ABLATION  score-bias off ALL {pct(s["ALL_val_ISHTIQAQ_OFF"])}  '
                  f'NOVEL {pct(s.get("NOVEL_only_ISHTIQAQ_OFF"))}')
        wc = s.get('wall_s')
        print(f'   cost      : wall {wc/60:.1f} min' if wc else '   cost      : wall n/a', end='')
        if s.get('reported_it_s'):
            print(f'   trainer-reported {s["reported_it_s"]:.2f} it/s', end='')
        if s.get('peak_vram_G'):
            print(f'   peak VRAM {s["peak_vram_G"]:.2f} GB', end='')
        print()
        print(f'   loss {s.get("loss_first"):.4f} -> {s.get("loss_last"):.4f}')
        if s.get('curve') and len(s['curve']) <= 25:
            print('   curve (step, ALL%, NOVEL%):')
            for row in s['curve']:
                print(f'      {row[0]:>6d}  {row[1]:>6.2f}  {row[2]:>6.2f}')
        print()

    print('=' * 100)
    print('COMPARATORS (recorded on this pod)')
    print('=' * 100)
    print(f'  {"":<66s} {"ALL%":>8s} {"NOVEL%":>8s}')
    for name, allv, novel in COMPARATORS:
        print(f'  {name:<66s} {allv:>8.2f} {"n/a" if novel is None else f"{novel:>8.2f}"}')
    for s in present:
        print(f'  {s["label"]:<66s} {100*s["final_all"]:>8.2f} {100*s["final_novel"]:>8.2f}')
    print()

    print('=' * 100)
    print('VERDICT')
    print('=' * 100)
    A = next((s for s in present if s['tag'] == 'FLOOR_A'), None)
    C = next((s for s in present if s['tag'] == 'EARLYROOT_C'), None)
    if A and C:
        da = C['final_all'] - A['final_all']
        dn = C['final_novel'] - A['final_novel']
        print(f'  A (floor, no pathway)      ALL {pct(A["final_all"])}   NOVEL {pct(A["final_novel"])}')
        print(f'  C (early root, 24 layers)  ALL {pct(C["final_all"])}   NOVEL {pct(C["final_novel"])}')
        print(f'  C - A                      ALL {100*da:+.2f} pp   NOVEL {100*dn:+.2f} pp')
        print()
        print(f'  vs 19.53 % (RCA_UNFREEZE_A2, late attachment):')
        print(f'      A is {100*(A["final_all"]-0.1953):+.2f} pp   '
              f'C is {100*(C["final_all"]-0.1953):+.2f} pp')
        if A['final_all'] >= C['final_all'] - 0.005:
            print('  => A MATCHES OR BEATS C.  A normally-trained trunk on Arabic learns root')
            print('     structure by itself; the early root pathway is NOT what produces the')
            print('     accuracy.  That is a real answer, not a failed run.')
        else:
            print('  => C BEATS A.  Embedding root structure from the input stage adds something')
            print('     the normally-trained trunk does not learn on its own.')
    elif C:
        print('  C is present but A is NOT -- A is mandatory and without it "the architecture')
        print('  works" and "training works" are indistinguishable.  Re-run A.')
    elif A:
        print('  A is present, C is not yet.  Waiting for the queue.')
    else:
        print('  Neither arm has finished.  Nothing to conclude.')

    out = {'generated': datetime.now(timezone.utc).isoformat(),
           'arms': {s['tag']: s for s in stats},
           'comparators': [{'name': n, 'all': a, 'novel': v} for n, a, v in COMPARATORS]}
    try:
        json.dump(out, open(args.json_out, 'w'), indent=2, default=str)
        print(f'\nwrote {args.json_out}')
    except Exception as e:
        print(f'\ncould not write {args.json_out}: {e}')


if __name__ == '__main__':
    main()
