#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""summarize_rootattn.py -- machine-generate every number quoted in REPORT_root_attn.md.

Reads the mirrored artifacts only (no torch, no GPU).
"""
import glob
import json
import os
import re
import statistics as st

BASE = os.path.dirname(os.path.abspath(__file__))
RA = os.path.join(BASE, 'remote', 'root_attn')
MARGINAL = 3.551
FIX = 6.836610287427902
ALIGNED_FIX = 6.619323045015335
NOISE = 1.0


def arm_from_results(path):
    H = json.load(open(path))['history']
    ev = [h for h in H if 'ALL_val' in h]
    late = [100 * h['ALL_val']['acc@1'] for h in ev if h['step'] >= 10000]
    last = ev[-1]
    a, n = last['ALL_val'], last['NOVEL_only']
    off = last.get('ALL_val_RCA_OFF', {})
    noff = last.get('NOVEL_only_RCA_OFF', {})
    return {
        'n_evals': len(ev), 'step': last['step'],
        'final': 100 * a['acc@1'], 'final5': 100 * a['acc@5'], 'final_cez': a['ce_z'],
        'best': max(100 * h['ALL_val']['acc@1'] for h in ev),
        'best_step': max(ev, key=lambda h: h['ALL_val']['acc@1'])['step'],
        'novel': 100 * n['acc@1'], 'novel5': 100 * n['acc@5'], 'novel_cez': n['ce_z'],
        'off': 100 * off['acc@1'] if off else None,
        'off_novel': 100 * noff['acc@1'] if noff else None,
        'late_sd': st.pstdev(late) if len(late) > 1 else 0.0,
        'late_range': (max(late) - min(late)) if late else 0.0,
        'gates': last.get('rca_gate'), 'h_drift': last.get('h_drift'),
        'nan': last.get('nan_steps'), 'evals': ev,
    }


def ladder():
    out = {}
    for p in sorted(glob.glob(os.path.join(BASE, 'remote', 'baselines', 'results_*.json'))):
        out[os.path.basename(p)[len('results_'):-len('.json')]] = arm_from_results(p)
    return out


def resources():
    out = {}
    for tag in ('RCA_UNFREEZE_A2', 'RCA_FROZEN_B2', 'ROOTATTN', 'RCA_FROZEN'):
        v = os.path.join(RA, f'vram_{tag}.txt')
        out[tag] = int(open(v).read().strip()) if os.path.exists(v) and open(v).read().strip() else None
    txt = open(os.path.join(RA, 'chain_final.out')).read() if os.path.exists(
        os.path.join(RA, 'chain_final.out')) else ''
    out['wall'] = dict(re.findall(r'\[chain\] (\S+) rc=(\d+) wall_s=(\d+) peak_vram_mib=(\d+)', txt)
                       and {})
    out['wall'] = {m[0]: {'rc': int(m[1]), 'wall_s': int(m[2]), 'vram': int(m[3])}
                   for m in re.findall(r'\[chain\] (\S+) rc=(\d+) wall_s=(\d+) peak_vram_mib=(\d+)', txt)}
    out['wall_rootattn_old'] = {m[0]: {'rc': int(m[1]), 'wall_s': int(m[2]), 'vram': int(m[3])}
                                for m in re.findall(
            r'\[rootattn\] rc=(\d+) wall_s=(\d+) peak_vram_mib=(\d+)', txt) and []}
    n = re.findall(r'\[chain\] (\S+) nan/inf steps: (\d+)', txt)
    out['nan'] = {a: int(b) for a, b in n}
    return out


def main():
    print('=' * 112)
    print('1. THE CLOSED LADDER (final step == the arm_table convention; best is over the 20 evals)')
    print('=' * 112)
    print('%-14s %6s %9s %9s %9s %9s %8s %8s %9s' % (
        'arm', 'n_eval', 'final@1', 'best@1', 'best@', 'finalNOV', 'CE_z', 'late_sd', 'x_marg'))
    for tag, v in sorted(ladder().items(), key=lambda kv: kv[1]['final']):
        print('%-14s %6d %9.3f %9.3f %9d %9.3f %8.3f %8.3f %9.3f' % (
            tag, v['n_evals'], v['final'], v['best'], v['best_step'], v['novel'],
            v['final_cez'], v['late_sd'], v['final'] / MARGINAL))
    print('\n  majority-class marginal %.3f %% | single-h linear ceiling 6.48 %% | order-4 lookup '
          'ceiling 53.13 %%' % MARGINAL)
    print('  FIX (the bar) %.3f %% -> a DECISIVE move needs > %.2f %% (> +%.1f pp); within-run '
          'eval sd over the last 10 evals is 0.03-0.11 pp (above), so the ~1 pp band is '
          'CONFIG-to-CONFIG variation, as the brief states.' % (FIX, FIX + NOISE, NOISE))

    arms = {}
    for tag in ('RCA_UNFREEZE_A2', 'RCA_FROZEN_B2'):
        p = os.path.join(RA, f'results_{tag}.json')
        if os.path.exists(p):
            arms[tag] = arm_from_results(p)
    if arms:
        print()
        print('=' * 112)
        print('2. THE TWO ARMS OF THIS WORK (20,000 steps, batch 32, FIX flags, aligned cache, '
              '--head-init remap)')
        print('=' * 112)
        print('%-18s %9s %9s %9s %8s %9s %9s %9s %9s' % (
            'arm', 'ALL@1', 'ALL@5', 'NOVEL@1', 'NOVEL@5', 'CE_z', 'RCA-OFF@1', 'x_marg', 'd_vs_FIX'))
        for tag, v in arms.items():
            print('%-18s %9.3f %9.3f %9.3f %8.3f %9.3f %9.3f %9.3f %+9.2f' % (
                tag, v['final'], v['final5'], v['novel'], v['novel5'], v['final_cez'],
                v['off'] if v['off'] is not None else float('nan'), v['final'] / MARGINAL,
                v['final'] - FIX))
        print('\n  late-run eval sd: ' + ', '.join(f'{t} {v["late_sd"]:.3f} pp' for t, v in arms.items()))
        for tag, v in arms.items():
            print(f'  {tag}: gates {[round(x,4) for x in (v["gates"] or [])]}, h_drift '
                  f'{v["h_drift"]}, nan_steps {v["nan"]}, best {v["best"]:.3f} @{v["best_step"]}')

        print()
        print('3. FULL HELD-OUT TRAJECTORY (ALL_val acc@1 / NOVEL_only acc@1 / CE_z / RCA-OFF)')
        for tag, v in arms.items():
            print(f'  --- {tag} ---')
            for h in v['evals']:
                a, n = h['ALL_val'], h['NOVEL_only']
                off = h.get('ALL_val_RCA_OFF', {})
                print('    %6d  ALL %6.2f%%  NOVEL %6.2f%%  CE_z %7.4f  | RCA-OFF ALL %6.2f%%  '
                      'gates %s' % (h['step'], 100 * a['acc@1'], 100 * n['acc@1'], a['ce_z'],
                                    100 * off.get('acc@1', float('nan')),
                                    [round(x, 3) for x in h.get('rca_gate', [])]))

    print()
    print('=' * 112)
    print('4. RESOURCES')
    print('=' * 112)
    r = resources()
    for tag, d in r['wall'].items():
        print(f'  {tag}: wall {d["wall_s"]}s ({d["wall_s"]/60:.1f} min), peak VRAM {d["vram"]} MiB, '
              f'rc={d["rc"]}')
    print(f'  nan/inf steps per arm: {r["nan"]}')
    if os.path.exists(os.path.join(RA, 'run_ROOTATTN.out')):
        m = re.findall(r'\[rootattn\] rc=(\d+) wall_s=(\d+) peak_vram_mib=(\d+)',
                       open(os.path.join(RA, 'run_ROOTATTN.out')).read())
        if m:
            print(f'  superseded first attempt (RUN A, killed for cause): wall {m[0][1]}s, '
                  f'peak VRAM {m[0][2]} MiB')

    print()
    print('=' * 112)
    print('5. VERDICT')
    print('=' * 112)
    if 'RCA_UNFREEZE_A2' in arms:
        v = arms['RCA_UNFREEZE_A2']
        print(f'  RUN A2 (RCA top4 + unfreeze top4): ALL@1 {v["final"]:.3f}% == '
              f'{v["final"]/MARGINAL:.2f}x the 3.551% marginal, {v["final"]-FIX:+.2f} pp vs FIX '
              f'{FIX:.2f}%; NOVEL {v["novel"]:.3f}%; CE_z {v["final_cez"]:.4f}')
        print(f'  -> {"DECISIVE WIN" if v["final"] > FIX + NOISE else "within the ~1 pp band"} '
              f'(threshold {FIX + NOISE:.2f}%)')
    if 'RCA_FROZEN_B2' in arms:
        v = arms['RCA_FROZEN_B2']
        print(f'  RUN B2 (RCA top4, trunk FROZEN): ALL@1 {v["final"]:.3f}% == '
              f'{v["final"]/MARGINAL:.2f}x marginal, {v["final"]-FIX:+.2f} pp vs FIX; '
              f'NOVEL {v["novel"]:.3f}%; CE_z {v["final_cez"]:.4f}')
        print(f'  -> {"DECISIVE WIN" if v["final"] > FIX + NOISE else "within the ~1 pp band"}')


if __name__ == '__main__':
    main()
