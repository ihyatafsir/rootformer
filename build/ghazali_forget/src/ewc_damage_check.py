#!/usr/bin/env python3
"""ewc_damage_check.py -- does the retained-Fisher actually model the MEASURED damage?

This is the check that decides whether the essential/accidental criterion is doing real work,
using the artefact that already exists rather than a new training run:

  theta0      = released trunk  (rootformer_v19_2_synthesis_ar_backbone...tok10052.safetensors)
  theta_dmg   = Run A's trunk   (/workspace/root_attn/head_ROOTATTN.pt.trunk.pt, the 0.1x LR arm
                                 that produced the measured collapse: FIX head 6.635 % -> 2.411 %,
                                 ||dW||/||W|| = 3.46/3.63/3.78/3.59 % on layers 20-23)

Measurements, all ordinary arithmetic:
  A. per-layer ||dW||/||W||            (must reproduce the published 3.46-3.78 %)
  B. correlation of dW^2 with F_retained and with F_synthesis (per-tensor and per-parameter)
  C. Taylor prediction  dL_ret ~ 1/2 sum_i F_ret,i * dW_i^2   vs  the MEASURED change in the
     retained (heritage LM) cross-entropy, on the same batches the Fisher was estimated from.
     If the prediction is right, the Fisher is a valid local model of the damage.
"""
import os, sys, json, argparse
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
import torch

REL = '/workspace/hf_v19_2_release'
sys.path.insert(0, REL)
sys.path.insert(0, os.path.join(REL, 'models'))
from ewc_fisher import build_model, lm_loss, tokenize_corpus   # reuse, same construction


def taylor(F, d):
    tot = 0.0
    for n, f in F.items():
        if n in d:
            tot += float((f.double() * d[n].double().pow(2)).sum())
    return 0.5 * tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default='/workspace/ghazali_forget')
    ap.add_argument('--damaged', default='/workspace/root_attn/head_ROOTATTN.pt.trunk.pt')
    ap.add_argument('--heritage', default='/workspace/data/farahidian_heritage_clean.txt')
    ap.add_argument('--batches', type=int, default=48)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--seq', type=int, default=128)
    ap.add_argument('--threads', type=int, default=6)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    layers = (20, 21, 22, 23)
    scope = tuple(f'backbone.layers.{i}.' for i in layers)

    base, flash, model, vocab = build_model(
        os.path.join(REL, 'checkpoints/rootformer_v19_2_synthesis_ar_backbone.'
                           'awzan142.roots9490.tok10052.safetensors'))
    ref = {n: p.detach().clone() for n, p in model.named_parameters() if n.startswith(scope)}

    pay = torch.load(args.damaged, map_location='cpu')
    st = pay['state'] if 'state' in pay else pay
    print(f'[*] damaged payload: {len(st)} tensors | meta unfrozen={pay.get("unfrozen_layers")} '
          f'trunk_lr_scale={pay.get("trunk_lr_scale")} step={pay.get("step")}', flush=True)

    d = {}
    for n, p in model.named_parameters():
        if n.startswith(scope) and n in st:
            d[n] = (st[n].float() - p.detach().float())
    print(f'[*] delta tensors: {len(d)} | params {sum(v.numel() for v in d.values())/1e6:.2f}M',
          flush=True)

    # ---- A. per-layer relative displacement ------------------------------------------------
    rel = {}
    for i in layers:
        num = sum(float(d[n].double().pow(2).sum()) for n in d if n.startswith(f'backbone.layers.{i}.'))
        den = sum(float(ref[n].double().pow(2).sum()) for n in ref if n.startswith(f'backbone.layers.{i}.'))
        rel[i] = (num ** 0.5) / (den ** 0.5)
    print('[A] ||dW||/||W|| per layer: ' +
          ' | '.join(f'L{i} {100*rel[i]:.4f}%' for i in layers), flush=True)

    # ---- B. correlation of dW^2 with the two Fisher profiles --------------------------------
    Fr = torch.load(os.path.join(args.out_dir, 'fisher_retained.pt'), map_location='cpu')
    Fs = torch.load(os.path.join(args.out_dir, 'fisher_synthesis.pt'), map_location='cpu')

    def flat(F):
        return torch.cat([F[n].reshape(-1).double() for n in sorted(F)])
    names = sorted(n for n in Fr if n in d)
    fr = torch.cat([Fr[n].reshape(-1).double() for n in names])
    fs = torch.cat([Fs[n].reshape(-1).double() for n in names])
    dd = torch.cat([d[n].reshape(-1).double().pow(2) for n in names])
    eps = 1e-30
    out = {'layers': list(layers), 'rel_disp': {str(k): v for k, v in rel.items()},
           'n_params': int(dd.numel()),
           'damaged_payload': {'tensors': len(st), 'step': pay.get('step'),
                               'trunk_lr_scale': pay.get('trunk_lr_scale')}}

    def corr(a, b):
        a = a - a.mean(); b = b - b.mean()
        return float((a * b).sum() / (a.norm() * b.norm() + eps))
    def spear(a, b):
        ra = a.argsort().argsort().double(); rb = b.argsort().argsort().double()
        return corr(ra, rb)

    out['corr_Fret_Fsyn_param'] = corr(fr, fs)
    out['corr_logFret_logFsyn_param'] = corr(fr.clamp_min(eps).log(), fs.clamp_min(eps).log())
    out['spearman_Fret_Fsyn_param'] = spear(fr, fs)
    out['corr_dW2_Fret'] = corr(dd, fr)
    out['corr_dW2_Fsyn'] = corr(dd, fs)
    out['corr_logdW2_logFret'] = corr(dd.clamp_min(eps).log(), fr.clamp_min(eps).log())
    out['corr_logdW2_logFsyn'] = corr(dd.clamp_min(eps).log(), fs.clamp_min(eps).log())

    # per-tensor (not per-parameter) view: aggregate each tensor's mean F and mean dW^2
    tfr, tfs, tdd = [], [], []
    for n in names:
        tfr.append(Fr[n].double().mean()); tfs.append(Fs[n].double().mean())
        tdd.append(d[n].double().pow(2).mean())
    tfr, tfs, tdd = torch.stack(tfr), torch.stack(tfs), torch.stack(tdd)
    out['tensor_level'] = {'n_tensors': len(names),
                           'spearman_Fret_Fsyn': spear(tfr, tfs),
                           'pearson_logFret_logFsyn': corr(tfr.clamp_min(eps).log(),
                                                           tfs.clamp_min(eps).log()),
                           'spearman_dW2_Fret': spear(tdd, tfr),
                           'spearman_dW2_Fsyn': spear(tdd, tfs)}
    # top-decile overlap between the two profiles
    k = max(1, len(tfr) // 10)
    ti_r = set(torch.topk(tfr, k).indices.tolist())
    ti_s = set(torch.topk(tfs, k).indices.tolist())
    out['tensor_level']['top_decile_overlap'] = len(ti_r & ti_s) / k

    # ---- C. Taylor prediction vs measured retained-loss change -----------------------------
    tok = base.tokenizer
    stream = tokenize_corpus(tok, 'retained', args.heritage, 40000,
                             os.path.join(args.out_dir, 'corpus_retained.pt'))
    hi = stream.numel() - args.seq - 1
    g = torch.Generator().manual_seed(20261002)
    starts = [torch.randint(0, hi, (args.batch,), generator=g) for _ in range(args.batches)]
    for p in model.parameters():
        p.requires_grad = False
    l0, l1 = [], []
    with torch.no_grad():
        for s in starts:
            ids = torch.stack([stream[a:a + args.seq] for a in s]).long()
            l0.append(float(lm_loss(flash, ids)))
    for n in d:
        model.get_parameter(n).data.copy_(ref[n] + d[n])
    with torch.no_grad():
        for s in starts:
            ids = torch.stack([stream[a:a + args.seq] for a in s]).long()
            l1.append(float(lm_loss(flash, ids)))
    m0 = sum(l0) / len(l0); m1 = sum(l1) / len(l1)
    pred_r = taylor(Fr, d)
    pred_s = taylor(Fs, d)
    out['taylor'] = {'L_ret_theta0': m0, 'L_ret_damaged': m1, 'measured_delta': m1 - m0,
                     'predicted_delta_Fret': pred_r, 'predicted_delta_Fsyn': pred_s,
                     'ratio_measured_over_Fret': (m1 - m0) / pred_r if pred_r else None}
    print(f'[C] retained LM CE: theta0 {m0:.5f} -> damaged {m1:.5f} '
          f'(measured delta {m1-m0:+.5f})', flush=True)
    print(f'[C] Taylor 1/2*sum F_ret*dW^2 = {pred_r:.5f} | 1/2*sum F_syn*dW^2 = {pred_s:.5f}',
          flush=True)
    print(f'[C] ratio measured/predicted(F_ret) = '
          f'{(m1-m0)/pred_r if pred_r else float("nan"):.4f}', flush=True)
    print('[B] corr(F_ret,F_syn) param-level %.4f | Spearman %.4f | tensor-level Spearman %.4f '
          '| top-decile overlap %.3f' % (out['corr_Fret_Fsyn_param'],
                                         out['spearman_Fret_Fsyn_param'],
                                         out['tensor_level']['spearman_Fret_Fsyn'],
                                         out['tensor_level']['top_decile_overlap']), flush=True)
    print('[B] corr(dW^2, F_ret) %.4f | corr(dW^2, F_syn) %.4f | tensor Spearman dW^2~F_ret %.4f '
          '| ~F_syn %.4f' % (out['corr_dW2_Fret'], out['corr_dW2_Fsyn'],
                             out['tensor_level']['spearman_dW2_Fret'],
                             out['tensor_level']['spearman_dW2_Fsyn']), flush=True)
    json.dump(out, open(os.path.join(args.out_dir, 'damage_report.json'), 'w'), indent=2)
    print('DAMAGE_DONE')


if __name__ == '__main__':
    main()
