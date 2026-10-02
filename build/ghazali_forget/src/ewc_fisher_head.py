#!/usr/bin/env python3
"""ewc_fisher_head.py -- Fisher of the RETAINED CAPABILITY, not of a broken LM readout.

Why this exists (measured, not hypothesised): the released trunk's tied-embedding LM readout is
miscalibrated -- mean CE 15.99 on held-out heritage Arabic against ln(10052) = 9.22 for a uniform
distribution -- and Run A's trunk displacement LOWERED that CE (15.99 -> 10.53) while destroying the
capability (FIX head 6.635 % -> 2.411 %).  So the LM loss is not the retained capability and its
Fisher is not a valid importance measure here (corr with the measured dW^2 = -0.014, Taylor ratio
-87).  The retained capability as actually MEASURED by this project is the FIX head's root accuracy
on the aligned held-out cache.  This script measures the diagonal empirical Fisher of exactly that
loss, so the essential/accidental test can be run against a loss that is honest about what is
being kept.

Loss: the FIX head's own root cross-entropy on the aligned val windows (the 6.635 % metric),
differentiated into the trunk parameters of layers 20-23 through the LIVE trunk.

Every val window is used (300 = 75 batches of 4), so the Taylor prediction
   L(theta0+d) - L(theta0) ~ 1/2 sum_i F_i d_i^2
is directly comparable with the measured ce_raw change at step 5000 of Run A.
"""
import os, sys, json, argparse
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
import torch
import torch.nn.functional as F

REL = '/workspace/hf_v19_2_release'
sys.path.insert(0, REL)
sys.path.insert(0, os.path.join(REL, 'models'))
from ewc_fisher import build_model
import ewc_head_probe as HP


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default='/workspace/ghazali_forget')
    ap.add_argument('--head', default='/workspace/head_fix/head_ALIGNED_FIX.pt')
    ap.add_argument('--cache', default='/workspace/head_fix/nrmp_cache_9490_aligned')
    ap.add_argument('--val-windows', type=int, default=300)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--layers', default='20,21,22,23')
    ap.add_argument('--threads', type=int, default=6)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    layers = tuple(int(x) for x in args.layers.split(','))
    scope = tuple(f'backbone.layers.{i}.' for i in layers)

    base, flash, model, vocab = build_model(os.path.join(
        REL, 'checkpoints/rootformer_v19_2_synthesis_ar_backbone.awzan142.roots9490.tok10052.safetensors'),
        dtype=torch.float32)
    for p in model.parameters():
        p.requires_grad = False
    named = [(n, p) for n, p in model.named_parameters() if n.startswith(scope)]
    for _, p in named:
        p.requires_grad = True
    r = model.nrmt_head.load_state_dict(torch.load(args.head, map_location='cpu'), strict=False)
    if r.missing_keys or r.unexpected_keys:
        raise SystemExit(f'FIX head load failed: {r.missing_keys} {r.unexpected_keys}')
    head = model.nrmt_head
    for p in head.parameters():
        p.requires_grad = False
    print(f'[*] scope {len(named)} tensors / '
          f'{sum(p.numel() for _, p in named)/1e6:.2f}M params', flush=True)

    ctx = HP.build_meta(model, args.cache, args.val_windows)
    (Pv, Tv, Wv, Sv), Ova_w, keep_all, _, _ = ctx
    specials_t = None
    print(f'[*] val windows {Tv.shape[0]} | radical positions {int(keep_all.sum())}', flush=True)

    FISH = {n: torch.zeros_like(p, dtype=torch.float32) for n, p in named}
    losses, accs = [], []
    nw = Tv.shape[0]
    nb = 0
    for i in range(0, nw, args.batch):
        sl = slice(i, min(i + args.batch, nw))
        emb = model.morphemic_embed(Pv[sl], Tv[sl], Wv[sl], Sv[sl])
        out = model.backbone(inputs_embeds=emb)
        h = model.final_norm(out.last_hidden_state).float()[:, :-1].contiguous()
        ho = head(h, Tv[sl][:, :-1].contiguous(), Ova_w[sl], Wv[sl][:, :-1].contiguous(),
                  Pv[sl][:, :-1].contiguous(), Sv[sl][:, :-1].contiguous())
        lg = ho['root_logits'].reshape(-1, vocab.num_roots)
        tg = Tv[sl][:, 1:].reshape(-1)
        m = ~torch.isin(tg, torch.tensor(sorted(
            {vocab.PAD_ROOT, vocab.BOS_ROOT, vocab.EOS_ROOT, vocab.UNK_ROOT,
             vocab.root2id['<PARTICLE>']} |
            {j for j, rr in enumerate(vocab.roots_list) if rr.startswith('<P:')})))
        loss = F.cross_entropy(lg[m], tg[m])
        losses.append(float(loss.detach()))
        accs.append(float((lg[m].argmax(-1) == tg[m]).float().mean()))
        for _, p in named:
            p.grad = None
        loss.backward()
        with torch.no_grad():
            for n, p in named:
                if p.grad is not None:
                    FISH[n].add_(p.grad.detach().float().pow(2))
        for _, p in named:
            p.grad = None
        nb += 1
        if nb % 10 == 0:
            print(f'  batch {nb} n_pos {int(m.sum())} loss {sum(losses)/len(losses):.5f} '
                  f'acc@1 {sum(accs)/len(accs):.5f}', flush=True)
    for n in FISH:
        FISH[n].div_(nb)
    torch.save(FISH, os.path.join(args.out_dir, 'fisher_head.pt'))
    rep = {'n_batches': nb, 'batch_windows': args.batch, 'val_windows': nw,
           'loss_mean': sum(losses) / len(losses), 'acc_mean': sum(accs) / len(accs),
           'sum_F': float(sum(v.double().sum() for v in FISH.values())),
           'n_params': int(sum(v.numel() for v in FISH.values())), 'layers': list(layers)}
    json.dump(rep, open(os.path.join(args.out_dir, 'fisher_head_report.json'), 'w'), indent=2)
    print(f'[fisher:head] loss {rep["loss_mean"]:.5f} acc@1 {100*rep["acc_mean"]:.3f}% | {rep}',
          flush=True)
    print('FISHER_HEAD_DONE')


if __name__ == '__main__':
    main()
