#!/usr/bin/env python3
"""ewc_arm_report.py -- score a finished (or running) EWC arm against the measured endpoints.

For each tag it prints
  1. the held-out trajectory (ALL_val and NOVEL_only acc@1 / ce_z, plus the RCA-off variant),
  2. per-layer ||dW||/||W|| of the arm's saved trunk payload against the released checkpoint,
  3. the pre-registered endpoint comparison.

The head-independent measure (FIX head on this arm's trunk, RCA off) is produced separately by
    ewc_head_probe.py --live --trunk <head_TAG.pt.trunk.pt> --out probe_head_TAG.json
so that the number is computed by the same code that reproduced 6.641 % / 2.433 %.
"""
import os, sys, json, argparse
import torch
from safetensors import safe_open

REL = '/workspace/hf_v19_2_release'
ENDPOINTS = {'CTL_frozen_converged': 0.1654, 'A2_healthy_0.01x': 0.1953,
             'RunA_collapsed_FIXhead_on_trunk': 0.02411, 'FIXhead_on_released_trunk': 0.06635,
             'RunA_own_arm_at_step5000': 0.0990}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('tags', nargs='+')
    ap.add_argument('--out-dir', default='/workspace/ghazali_forget')
    ap.add_argument('--ckpt', default=os.path.join(
        REL, 'checkpoints/rootformer_v19_2_synthesis_ar_backbone.awzan142.roots9490.tok10052.safetensors'))
    ap.add_argument('--layers', default='20,21,22,23')
    args = ap.parse_args()
    layers = [int(x) for x in args.layers.split(',')]
    rep = {}
    for tag in args.tags:
        print(f'\n================ {tag} ================', flush=True)
        res = os.path.join(args.out_dir, f'results_{tag}.json')
        entry = {}
        if os.path.exists(res):
            h = json.load(open(res))['history']
            print(f'{"step":>7} {"ALL acc@1":>10} {"ALL ce_z":>9} {"NOVEL acc@1":>12} '
                  f'{"RCAoff acc@1":>13} {"h_drift":>10}')
            for e in h:
                a = e.get('ALL_val', {}); n = e.get('NOVEL_only', {}); o = e.get('ALL_val_RCA_OFF', {})
                print(f'{e.get("step",-1):>7} {100*a.get("acc@1",float("nan")):>9.3f}% '
                      f'{a.get("ce_z",float("nan")):>9.4f} {100*n.get("acc@1",float("nan")):>11.3f}% '
                      f'{100*o.get("acc@1",float("nan")):>12.3f}% '
                      f'{str(e.get("h_drift"))[:9]:>10}')
            entry['n_evals'] = len(h)
            entry['final'] = h[-1].get('ALL_val', {})
            entry['best'] = max((e.get('ALL_val', {}).get('acc@1', 0) for e in h), default=0)
        else:
            print('  no results json yet (run still in progress)')
            tr = os.path.join(args.out_dir, f'trace_{tag}.jsonl')
            if os.path.exists(tr):
                rows = [json.loads(l) for l in open(tr) if l.strip()]
                print(f'  trace rows: {len(rows)} | last step {rows[-1]["step"]} | '
                      f'loss {rows[-1]["loss"]:.4f} | ewc_pen {rows[-1].get("ewc_pen")}')
                entry['trace_rows'] = len(rows)
                entry['last_step'] = rows[-1]['step']
        pay = os.path.join(args.out_dir, f'head_{tag}.pt.trunk.pt')
        if os.path.exists(pay):
            st = torch.load(pay, map_location='cpu')
            st = st.get('state', st)
            f = safe_open(args.ckpt, 'pt')
            rel = {}
            for i in layers:
                num = den = 0.0
                for k, v in st.items():
                    if k.startswith(f'backbone.layers.{i}.'):
                        t0 = f.get_tensor(k).float().double()
                        num += float((v.float().double() - t0).pow(2).sum())
                        den += float(t0.pow(2).sum())
                rel[i] = (num ** 0.5) / (den ** 0.5)
            entry['rel_disp'] = rel
            print('  ||dW||/||W||: ' + ' | '.join(f'L{i} {100*rel[i]:.4f}%' for i in layers)
                  + '   (Run A measured 3.4585/3.6328/3.7765/3.5856)')
        ph = os.path.join(args.out_dir, f'probe_head_{tag}.json')
        if os.path.exists(ph):
            d = json.load(open(ph))
            m = d.get('live') or d.get('cached')
            entry['fix_head'] = m
            print(f'  FIX head (head-independent, this arm\'s trunk): acc@1 '
                  f'{100*m["acc@1"]:.3f}% acc@5 {100*m["acc@5"]:.3f}% ce_z {m["ce_z"]:.4f} '
                  f'(n={m["n"]})')
            print(f'    vs released 6.641% | vs Run A 2.433%')
        rep[tag] = entry
    print('\n=== ENDPOINTS (measured, pre-registered) ===')
    for k, v in ENDPOINTS.items():
        print(f'  {k}: {100*v:.3f}%')
    json.dump(rep, open(os.path.join(args.out_dir, 'arm_report.json'), 'w'), indent=2)
    print('ARM_REPORT_DONE')


if __name__ == '__main__':
    main()
