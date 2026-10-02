#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""curve_lm.py -- learning curve for the ORIGINAL scratch_lm.py morph arm.

Motivation
----------
sf_morph.pt (6,000 steps) has train root top-1 19.19% and held-out root top-1 20.18%: no
overfitting.  But "no overfitting" is compatible with two very different futures:

  * the curve is still CLIMBING at 6,000 steps  -> the 6k number is a snapshot of an
    under-trained model and says nothing about the morphemic hypothesis
  * the curve has PLATEAUED at the unigram marginal -> 6,000 steps was enough to learn
    everything this configuration can learn, and the ceiling is a property of the idea

Nothing in the existing evidence distinguishes those.  This script does, by training the SAME
architecture on the SAME data and evaluating held-out accuracy at many step counts.

Faithfulness: the model, sampler, padding, loss and optimiser are scratch_lm.py's own
(MorphemicLM / batches_morph / pad / OneCycleLR), and the data is /workspace/sf_data, byte
unchanged since the sf_morph.pt run (meta.json n_roots=9114, n_awzan=130, train_words=25000003).
The ONLY additions are observation: a loss moving average (the original prints one noisy
batch), a held-out evaluation every --eval-every steps, and a --steps value long enough to
see the curve bend.

Read-only on every project artefact.  Writes only under /workspace/scratch_curve/.

GPU hygiene: --wait-for-gpu polls until no other nrmt_train is running and GPU memory is low,
and NEVER kills anything.  While waiting it does not allocate CUDA.
"""
import argparse
import json
import math
import os
import random
import sys
import time

import torch

sys.path.insert(0, '/workspace/hf_v19_2_release')
import scratch_lm as S  # noqa: E402

D = '/workspace/sf_data'
OUT = '/workspace/scratch_curve'
CKPT6K = '/workspace/scratch_lm_run/sf_morph.pt'


def gpu_busy():
    """(other_training_processes, memory_used_MiB) -- read-only, never kills."""
    import subprocess
    try:
        n = int(subprocess.run(['pgrep', '-fc', 'python nrmt_train.py'],
                               capture_output=True, text=True).stdout.strip() or 0)
    except Exception:
        n = 0
    try:
        mem = int(subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
            capture_output=True, text=True).stdout.strip().splitlines()[0])
    except Exception:
        mem = 99999
    return n, mem


def wait_for_gpu(max_wait=5400, quiet_mib=1200):
    t0 = time.time()
    while True:
        n, mem = gpu_busy()
        if n == 0 and mem < quiet_mib:
            time.sleep(8)
            n, mem = gpu_busy()
            if n == 0 and mem < quiet_mib:
                print(f'[{time.strftime("%H:%M:%S")}] GPU quiet (mem={mem}MiB) '
                      f'after {time.time()-t0:.0f}s', flush=True)
                return True
        if time.time() - t0 > max_wait:
            print(f'[{time.strftime("%H:%M:%S")}] GAVE UP waiting after '
                  f'{time.time()-t0:.0f}s (nrmt_train={n}, mem={mem}MiB)', flush=True)
            return False
        time.sleep(15)


@torch.no_grad()
def held_out_metrics(model, data, n_roots, special_ids, device, n_sent, seed,
                     hold_ids=None):
    """Per-head top-1/top-5 plus true bits/word on a fixed held-out sentence sample.

    Counts ALL positions, the convention of scratch_lm.root_accuracy.  Root accuracy is
    additionally split held-out-root / real-seen-root / special exactly as
    heldout_root_eval.py does, so the numbers are comparable to the published table.

    hold_ids MUST be passed as a 1-D bool tensor over roots.  (It used to be read from a
    module global named hold_t, which silently produced n=0 for every category because the
    caller's hold_t was a local in main() and this guard short-circuited.  Validated in
    test_heldout_metrics.py: with the tensor passed in, test_deriv real_seen_root top1 and
    special top1 land on the published 1.468% / 39.202%.)
    """
    model.eval()
    spec = torch.zeros(n_roots, dtype=torch.bool)
    for i in special_ids:
        if 0 <= i < n_roots:
            spec[i] = True
    spec = spec.to(device)
    if hold_ids is not None:
        hold_t = hold_ids.to(device)
    else:
        hold_t = torch.zeros(n_roots, dtype=torch.bool, device=device)
    acc = {h: {'n': 0, 'h1': 0, 'h5': 0} for h in ('p', 'r', 'w', 's')}
    cats = {k: {'n': 0, 'h1': 0, 'h5': 0} for k in ('heldout_root', 'real_seen_root', 'special')}
    bits, npos = 0.0, 0
    rng = random.Random(seed)
    for split in ('val', 'test_gen', 'test_deriv'):
        Dd = data[split]
        if len(Dd['L']) == 0:
            continue
        order = rng.sample(range(len(Dd['L'])), min(n_sent, len(Dd['L'])))
        tgt = {h: [Dd[k][j][:S.CTX_WORDS][1:] for j in order]
               for h, k in (('p', 'P'), ('r', 'R'), ('w', 'W'), ('s', 'S'))}
        ctx = {h: [Dd[k][j][:S.CTX_WORDS][:-1] for j in order]
               for h, k in (('p', 'P'), ('r', 'R'), ('w', 'W'), ('s', 'S'))}
        if min((len(x) for x in ctx['r']), default=0) < 1:
            continue
        T = max(len(x) for x in ctx['r'])
        m = torch.zeros((len(order), T), dtype=torch.bool)
        for b, x in enumerate(ctx['r']):
            m[b, :len(x)] = True
        m = m.to(device)
        p_in = S.pad(ctx['p'], 0).to(device)
        r_in = S.pad(ctx['r'], 0).to(device)
        w_in = S.pad(ctx['w'], 0).to(device)
        s_in = S.pad(ctx['s'], 0).to(device)
        lr, lw, lp, ls = model(p_in, r_in, w_in, s_in)
        lg = {'r': lr, 'w': lw, 'p': lp, 's': ls}
        tg = {h: S.pad(tgt[h], 0).to(device) for h in ('p', 'r', 'w', 's')}
        for h in ('p', 'r', 'w', 's'):
            top5 = lg[h].topk(5, -1).indices
            acc[h]['n'] += int(m.sum())
            acc[h]['h1'] += int(((top5[..., 0] == tg[h]) & m).sum())
            acc[h]['h5'] += int(((top5 == tg[h].unsqueeze(-1)).any(-1) & m).sum())
            nll = torch.nn.functional.cross_entropy(
                lg[h].reshape(-1, lg[h].shape[-1]), tg[h].reshape(-1), reduction='none')
            bits += float(nll.reshape(tg[h].shape)[m].sum()) / math.log(2)
        npos += int(m.sum())
        if split == 'test_deriv' and hold_t is not None:
            r_tg = tg['r']
            for key, mm in (('heldout_root', hold_t[r_tg] & m),
                            ('real_seen_root', (~spec[r_tg]) & (~hold_t[r_tg]) & m),
                            ('special', spec[r_tg] & m)):
                cats[key]['n'] += int(mm.sum())
                cats[key]['h1'] += int(((lg['r'].argmax(-1) == r_tg) & mm).sum())
                cats[key]['h5'] += int(((lg['r'].topk(5, -1).indices ==
                                         r_tg.unsqueeze(-1)).any(-1) & mm).sum())
    model.train()
    out = {h: {'top1': acc[h]['h1'] / max(acc[h]['n'], 1),
               'top5': acc[h]['h5'] / max(acc[h]['n'], 1), 'n': acc[h]['n']}
           for h in ('p', 'r', 'w', 's')}
    out['bits_per_word'] = bits / max(npos, 1)
    out['root_cats'] = {k: {'top1': v['h1'] / max(v['n'], 1),
                            'top5': v['h5'] / max(v['n'], 1), 'n': v['n']}
                        for k, v in cats.items()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['curve'])
    ap.add_argument('--steps', type=int, default=120000)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--lr', type=float, default=6e-4)
    ap.add_argument('--eval-every', type=int, default=5000)
    ap.add_argument('--eval-sents', type=int, default=400)
    ap.add_argument('--tag', default='curve120k')
    ap.add_argument('--wait-for-gpu', action='store_true')
    ap.add_argument('--quiet-mib', type=int, default=1200)
    ap.add_argument('--save-every', type=int, default=20000)
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    if args.wait_for_gpu:
        if not wait_for_gpu(quiet_mib=args.quiet_mib):
            sys.exit('GPU never went quiet; not launching. Nothing was modified.')

    device = torch.device('cuda')
    meta = json.load(open(f'{D}/meta.json'))
    data = torch.load(f'{D}/streams.pt', weights_only=False)
    n_roots = meta['n_roots']
    special_ids = meta['special_root_ids']
    hold = meta['hold_roots']

    rng = random.Random(0)
    torch.manual_seed(0)
    model = S.MorphemicLM((n_roots, meta['n_awzan'],
                           meta['n_prefixes'], meta['n_suffixes'])).to(device)
    nparam = sum(p.numel() for p in model.parameters())
    print(f'[*] curve tag={args.tag} params={nparam/1e6:.2f}M steps={args.steps} '
          f'batch={args.batch} lr={args.lr} train_words={meta["train_words"]}', flush=True)
    print(f'[*] data: /workspace/sf_data  n_roots={n_roots} n_awzan={meta["n_awzan"]} '
          f'hold_roots={len(hold)}', flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01,
                            betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.steps,
        pct_start=max(0.02, min(0.3, 200.0 / max(args.steps, 1))))
    torch.cuda.reset_peak_memory_stats()

    # hold_t needs to exist before held_out_metrics is called
    hold_t = torch.zeros(n_roots, dtype=torch.bool)
    for i in hold:
        hold_t[i] = True
    hold_t = hold_t.to(device)

    curve = []
    step = 0
    t0 = time.time()
    ma_loss, ma_n = 0.0, 0
    loss_hist = []
    next_eval = args.eval_every

    def do_eval(step, wall):
        m = held_out_metrics(model, data, n_roots, special_ids, device,
                             args.eval_sents, seed=11, hold_ids=hold_t)
        rec = {'step': step, 'wall_s': wall, 'train_loss_ma': (ma_loss / max(ma_n, 1)),
               'lr': sched.get_last_lr()[0],
               'peak_vram_MiB': torch.cuda.max_memory_allocated() / 2**20,
               **{f'{h}_top1': m[h]['top1'] for h in ('p', 'r', 'w', 's')},
               **{f'{h}_top5': m[h]['top5'] for h in ('p', 'r', 'w', 's')},
               'bits_per_word': m['bits_per_word'],
               'test_deriv_heldout_root_top1': m['root_cats']['heldout_root']['top1'],
               'test_deriv_heldout_root_n': m['root_cats']['heldout_root']['n'],
               'test_deriv_real_seen_root_top1': m['root_cats']['real_seen_root']['top1'],
               'test_deriv_special_top1': m['root_cats']['special']['top1']}
        curve.append(rec)
        print(f'  [EVAL @{step}] loss_ma={rec["train_loss_ma"]:.4f} '
              f'bits/word={rec["bits_per_word"]:.3f} | root t1/t5 '
              f'{rec["r_top1"]*100:.2f}/{rec["r_top5"]*100:.2f} | prefix {rec["p_top1"]*100:.2f} '
              f'wazn {rec["w_top1"]*100:.2f} suffix {rec["s_top1"]*100:.2f} | '
              f'deriv real-seen {rec["test_deriv_real_seen_root_top1"]*100:.2f} '
              f'special {rec["test_deriv_special_top1"]*100:.2f} | '
              f'peakVRAM={rec["peak_vram_MiB"]:.0f}MiB {wall:.0f}s', flush=True)
        json.dump({'tag': args.tag, 'params': nparam, 'steps': args.steps,
                   'batch': args.batch, 'lr': args.lr, 'data': D,
                   'n_roots': n_roots, 'hold_roots': len(hold),
                   'curve': curve},
                  open(f'{OUT}/{args.tag}.json', 'w'), indent=2)
        ma_loss_local = 0.0
        return rec

    while step < args.steps:
        it = S.batches_morph(data, 'train', args.batch, rng)
        for b in it:
            p_in = S.pad([x[:-1] for x in b['p']], 0).to(device)
            r_in = S.pad([x[:-1] for x in b['r']], 0).to(device)
            w_in = S.pad([x[:-1] for x in b['w']], 0).to(device)
            s_in = S.pad([x[:-1] for x in b['s']], 0).to(device)
            r_tg = S.pad([x[1:] for x in b['r']], 0).to(device)
            w_tg = S.pad([x[1:] for x in b['w']], 0).to(device)
            p_tg = S.pad([x[1:] for x in b['p']], 0).to(device)
            s_tg = S.pad([x[1:] for x in b['s']], 0).to(device)
            lr_, lw, lp, ls = model(p_in, r_in, w_in, s_in)
            loss = (torch.nn.functional.cross_entropy(
                        lr_.reshape(-1, lr_.shape[-1]), r_tg.reshape(-1), ignore_index=0)
                    + 0.5 * torch.nn.functional.cross_entropy(
                        lw.reshape(-1, lw.shape[-1]), w_tg.reshape(-1), ignore_index=0)
                    + 0.25 * torch.nn.functional.cross_entropy(
                        lp.reshape(-1, lp.shape[-1]), p_tg.reshape(-1), ignore_index=0)
                    + 0.25 * torch.nn.functional.cross_entropy(
                        ls.reshape(-1, ls.shape[-1]), s_tg.reshape(-1), ignore_index=0))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            ma_loss += float(loss.detach()); ma_n += 1
            loss_hist.append(float(loss.detach()))
            if step == next_eval:
                do_eval(step, time.time() - t0)
                ma_loss, ma_n = 0.0, 0
                next_eval += args.eval_every
            if args.save_every and step % args.save_every == 0:
                torch.save(model.state_dict(), f'{OUT}/{args.tag}_step{step}.pt')
            if step >= args.steps:
                break
        if step % 2000 == 0:
            print(f'  step {step}/{args.steps} loss_ma={ma_loss/max(ma_n,1):.4f} '
                  f'{step/(time.time()-t0):.2f} it/s', flush=True)

    torch.save(model.state_dict(), f'{OUT}/{args.tag}.pt')
    if not curve or curve[-1]['step'] != step:
        do_eval(step, time.time() - t0)
    print(f'\n[*] done: {step} steps in {time.time()-t0:.0f}s, '
          f'peak VRAM {torch.cuda.max_memory_allocated()/2**20:.0f}MiB', flush=True)
    print(f'[*] wrote {OUT}/{args.tag}.json and {OUT}/{args.tag}.pt', flush=True)


if __name__ == '__main__':
    main()
