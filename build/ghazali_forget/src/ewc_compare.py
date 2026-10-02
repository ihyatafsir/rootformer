#!/usr/bin/env python3
"""ewc_compare.py -- the decisive analysis: which importance profile, if any, separates
essential from accidental, and which one predicts the MEASURED damage.

Profiles compared over the same 62.25M trunk parameters (layers 20-23):
  F_ret   Fisher of next-token CE on held-out heritage Arabic          (the brief's proposal)
  F_syn   Fisher of next-token CE on the v19.2 synthesis corpus        (the abandoned objective)
  F_head  Fisher of the retained CAPABILITY's own loss (FIX head root CE on the aligned cache)

Ground truth to be predicted is not a hypothesis: it is Run A's saved trunk, whose displacement
  * raised the FIX head's root CE by +3.5321 (7.6581 -> 11.1902) and dropped its acc@1
    6.641 % -> 2.433 % (published 6.635 % -> 2.411 %), and
  * LOWERED the heritage LM CE by -5.4621 (15.9876 -> 10.5255).
So the second-order Taylor term 1/2 sum_i F_i dW_i^2 is checked against two measured deltas of
opposite sign -- which is what makes this falsifiable rather than decorative.
"""
import os, sys, json, argparse
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
import torch
from safetensors import safe_open

REL = '/workspace/hf_v19_2_release'


def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-30))


def spear(a, b):
    return corr(a.argsort().argsort().double(), b.argsort().argsort().double())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default='/workspace/ghazali_forget')
    ap.add_argument('--ckpt', default=os.path.join(
        REL, 'checkpoints/rootformer_v19_2_synthesis_ar_backbone.awzan142.roots9490.tok10052.safetensors'))
    ap.add_argument('--damaged', default='/workspace/root_attn/head_ROOTATTN.pt.trunk.pt')
    args = ap.parse_args()
    O = args.out_dir

    prof = {}
    for tag, fn in (('ret', 'fisher_retained.pt'), ('syn', 'fisher_synthesis.pt'),
                    ('head', 'fisher_head.pt')):
        p = os.path.join(O, fn)
        if os.path.exists(p):
            prof[tag] = torch.load(p, map_location='cpu')
    print('profiles loaded:', {k: len(v) for k, v in prof.items()}, flush=True)

    pay = torch.load(args.damaged, map_location='cpu')
    st = pay.get('state', pay)
    f = safe_open(args.ckpt, 'pt')
    names = sorted(prof['head'].keys())
    d = {}
    for n in names:
        t0 = f.get_tensor(n).float().double()
        d[n] = st[n].float().double() - t0
    dd = torch.cat([d[n].reshape(-1).double().pow(2) for n in names])
    th2 = torch.cat([f.get_tensor(n).float().double().reshape(-1).pow(2) for n in names])
    flat = {k: torch.cat([v[n].reshape(-1).double() for n in names]) for k, v in prof.items()}
    eps = 1e-300
    out = {'n_params': int(dd.numel()), 'measured': {}}

    print('\n=== pairwise profile agreement (parameter level) ===')
    for a in ('ret', 'syn', 'head'):
        for b in ('ret', 'syn', 'head'):
            if a >= b:
                continue
            print(f'  F_{a} vs F_{b}: pearson {corr(flat[a], flat[b]):+.4f} | '
                  f'pearson(log) {corr(flat[a].clamp_min(eps).log(), flat[b].clamp_min(eps).log()):+.4f}'
                  f' | spearman {spear(flat[a], flat[b]):+.4f}', flush=True)
            out[f'corr_{a}_{b}'] = {'pearson': corr(flat[a], flat[b]),
                                    'pearson_log': corr(flat[a].clamp_min(eps).log(),
                                                        flat[b].clamp_min(eps).log()),
                                    'spearman': spear(flat[a], flat[b])}
    print('\n=== tensor level (mean F per tensor, 92 tensors) ===')
    tl = {k: torch.stack([v[n].double().mean() for n in names]) for k, v in prof.items()}
    tdd = torch.stack([d[n].double().pow(2).mean() for n in names])
    for a in ('ret', 'syn', 'head'):
        for b in ('ret', 'syn', 'head'):
            if a >= b:
                continue
            k = max(1, len(names) // 10)
            ov = len(set(torch.topk(tl[a], k).indices.tolist()) &
                     set(torch.topk(tl[b], k).indices.tolist())) / k
            print(f'  F_{a} vs F_{b}: spearman {spear(tl[a], tl[b]):+.4f} | top-decile overlap {ov:.3f}',
                  flush=True)
            out[f'tensor_{a}_{b}'] = {'spearman': spear(tl[a], tl[b]), 'top_decile_overlap': ov}

    print('\n=== does each profile predict the MEASURED damage? ===')
    for k in ('ret', 'syn', 'head'):
        tay = 0.5 * float((flat[k] * dd).sum())
        out[f'taylor_{k}'] = tay
        out[f'corr_dW2_{k}'] = corr(dd, flat[k])
        out[f'tensor_spearman_dW2_{k}'] = spear(tdd, tl[k])
        print(f'  F_{k}: 1/2*sum F*dW^2 = {tay:.6f} | corr(dW^2, F) {corr(dd, flat[k]):+.4f} | '
              f'tensor spearman(dW^2, F) {spear(tdd, tl[k]):+.4f}', flush=True)

    # measured deltas, read from the artefacts that produced them
    dr = os.path.join(O, 'damage_report.json')
    if os.path.exists(dr):
        t = json.load(open(dr))['taylor']
        out['measured']['LM_CE_heritage_delta'] = t['measured_delta']
        print(f'\n  measured heritage LM CE delta (theta0 -> damaged): '
              f'{t["measured_delta"]:+.6f}')
    pr = os.path.join(O, 'probe_head_released.json')
    pa = os.path.join(O, 'probe_head_runA.json')
    if os.path.exists(pr) and os.path.exists(pa):
        r = json.load(open(pr))['live']; a = json.load(open(pa))['live']
        out['measured']['capability_ce_raw_delta'] = a['ce_raw'] - r['ce_raw']
        out['measured']['capability_acc_delta'] = a['acc@1'] - r['acc@1']
        print(f'  measured capability root-CE delta: {a["ce_raw"]-r["ce_raw"]:+.6f} '
              f'({r["ce_raw"]:.4f} -> {a["ce_raw"]:.4f})')
        print(f'  measured capability acc@1: {100*r["acc@1"]:.3f}% -> {100*a["acc@1"]:.3f}%')
    # lambda recalibration per profile, same pre-registered rule
    import math
    L1 = 11.093619346618652
    out['lambda'] = {}
    for k in ('ret', 'syn', 'head'):
        S1 = float((flat[k] * th2).sum())
        S = 0.035 ** 2 * S1
        out['lambda'][k] = {'sum_F_theta0_sq': S1, 'penalty_per_unit_lambda_at_dstar': S,
                            'lambda_kappa_1': L1 / S if S else None}
        print(f'  lambda* for F_{k}: {L1/S if S else float("nan"):.6g}  '
              f'(S={S:.6g}, S1={S1:.6g})')
    json.dump(out, open(os.path.join(O, 'compare_report.json'), 'w'), indent=2)
    print('\nCOMPARE_DONE')


if __name__ == '__main__':
    main()
