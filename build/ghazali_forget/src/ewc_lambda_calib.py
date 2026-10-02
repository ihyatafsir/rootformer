#!/usr/bin/env python3
"""ewc_lambda_calib.py -- fix lambda BEFORE the run, by a stated rule, with checkable arithmetic.

AL-TARJIH (the preference rule), pre-registered:

  "The new objective may move the trunk freely until its displacement reaches the scale that was
   MEASURED to destroy the retained capability; at that scale the retained-structure penalty is
   worth one full unit of task loss.  Beyond it, retained structure wins."

Arithmetic (no free parameters beyond the measured quantities):

  S1 = sum_i F_i * theta0_i^2                     [per unit relative displacement]
  S  = d*^2 * S1                                  penalty per unit lambda at displacement d*
  lambda* = kappa * L_task / S                    with kappa = 1

  d*    = 0.035   the MEASURED relative displacement of the arm that collapsed
                  (||dW||/||W|| = 3.46/3.63/3.78/3.59 % on layers 20-23)
  L_task= the task loss actually observed at step 1 of that same collapsed arm (its own trace)
  F     = diagonal Fisher on the RETAINED linguistic distribution (ewc_fisher.py)
  theta0= the pretrained trunk weights in the penalised scope

kappa = 1 is the pre-registered primary value.  kappa = 0.1 and kappa = 10 are reported only as
a pre-stated sensitivity band, NOT as a post-hoc selection.
"""
import os, sys, json, argparse
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
import torch
from safetensors import safe_open

REL = '/workspace/hf_v19_2_release'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default='/workspace/ghazali_forget')
    ap.add_argument('--ckpt', default=os.path.join(
        REL, 'checkpoints/rootformer_v19_2_synthesis_ar_backbone.awzan142.roots9490.tok10052.safetensors'))
    ap.add_argument('--trace', default='/workspace/root_attn/trace_ROOTATTN.jsonl')
    ap.add_argument('--layers', default='20,21,22,23')
    ap.add_argument('--dstar', type=float, default=0.035)
    ap.add_argument('--fisher', default='fisher_retained.pt')
    ap.add_argument('--tag', default='retained')
    args = ap.parse_args()
    layers = tuple(int(x) for x in args.layers.split(','))

    F = torch.load(os.path.join(args.out_dir, args.fisher), map_location='cpu')
    f = safe_open(args.ckpt, 'pt')
    keys = list(F.keys())
    S1 = 0.0
    theta_sq = 0.0
    fsum = 0.0
    n = 0
    for k in keys:
        t0 = f.get_tensor(k).float().double()
        S1 += float((F[k].double() * t0 * t0).sum())
        theta_sq += float(t0.pow(2).sum())
        fsum += float(F[k].double().sum())
        n += t0.numel()
    S = args.dstar ** 2 * S1

    losses = []
    with open(args.trace) as fh:
        for i, line in enumerate(fh):
            if i >= 200:
                break
            try:
                losses.append(json.loads(line)['loss'])
            except Exception:
                pass
    L1 = losses[0]
    L200 = sum(losses) / len(losses)

    out = {'fisher': args.fisher, 'tag': args.tag,
           'layers': list(layers), 'n_params': n, 'sum_F': fsum, 'mean_F': fsum / n,
           'sum_F_theta0_sq': S1, 'sum_theta0_sq': theta_sq,
           'theta0_rms': (theta_sq / n) ** 0.5,
           'dstar': args.dstar, 'penalty_per_unit_lambda_at_dstar': S,
           'L_task_step1': L1, 'L_task_mean_first200': L200,
           'trace': args.trace}
    for kappa in (0.1, 1.0, 10.0):
        out[f'lambda_kappa_{kappa:g}'] = kappa * L1 / S
    out['LAMBDA_PRIMARY'] = out['lambda_kappa_1']
    # what the penalty is worth at the damage scale, for the primary lambda
    out['penalty_value_at_dstar_with_primary_lambda'] = out['LAMBDA_PRIMARY'] * S
    json.dump(out, open(os.path.join(args.out_dir, f'lambda_calib_{args.tag}.json'), 'w'), indent=2)
    for k, v in out.items():
        print(f'  {k}: {v}')
    print('LAMBDA_CALIB_DONE')


if __name__ == '__main__':
    main()
