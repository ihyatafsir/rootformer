#!/usr/bin/env python3
"""ewc_cpu_endpoints.py -- prove the trivially-correct case on the REAL trunk, on CPU.

Both endpoints, one script, same weights, same data, same optimiser, only lambda differs:

    lambda = 0      -> the retained capability must collapse (that is the measured failure mode)
    lambda large    -> the retained capability must be preserved

"Retained behaviour" is measured twice, exactly as the GPU endpoint table does:
    (i)  FIX baseline's own head, root acc@1 on held-out windows (the capability number), and
    (ii) held-out heritage next-token CE (the linguistic number).

"New objective" is the abandoned synthesis corpus (the bilingual Basran-Andalusian stream),
fine-tuned at a deliberately harsh LR so that 60 CPU steps are enough to show the endpoint.
"""
import os, sys, json, argparse, copy
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
import torch
import torch.nn.functional as F

REL = '/workspace/hf_v19_2_release'
sys.path.insert(0, REL)
sys.path.insert(0, os.path.join(REL, 'models'))
from ewc_fisher import build_model, lm_loss, tokenize_corpus
import ewc_head_probe as HP


def clear_attn_state(model):
    """The trainer's live path never sets active_root_ids: the IshtiqaqAttention root branch is
    inactive there, and the model is built fresh.  Any earlier call through the LM path
    (ewc_fisher.lm_loss) leaves a STALE [batch,seq] root tensor behind, which then gets consumed by
    the next forward at a different batch size.  Clear it so the capability forward reproduces the
    trainer's live path exactly."""
    for layer in model.backbone.layers:
        layer.self_attn.active_root_ids = None
        layer.self_attn.active_wazn_ids = None


def live_h(model, P, T, W, S, bs=8):
    clear_attn_state(model)
    hs = []
    with torch.no_grad():
        for i in range(0, T.shape[0], bs):
            sl = slice(i, min(i + bs, T.shape[0]))
            emb = model.morphemic_embed(P[sl], T[sl], W[sl], S[sl])
            out = model.backbone(inputs_embeds=emb)
            hs.append(model.final_norm(out.last_hidden_state).float()[:, :-1].contiguous())
    return torch.cat(hs)


def fix_acc(model, head, ctx, n_windows):
    (Pv, Tv, Wv, Sv), Ova_w, keep_all, _, _ = ctx
    Pv, Tv, Wv, Sv = Pv[:n_windows], Tv[:n_windows], Wv[:n_windows], Sv[:n_windows]
    keep = keep_all[:n_windows * (HP.WIN - 1)]
    H = live_h(model, Pv, Tv, Wv, Sv)
    m = HP.eval_head(model, head, H, Tv[:, :-1].contiguous(), Ova_w[:n_windows],
                     Wv[:, :-1].contiguous(), Pv[:, :-1].contiguous(), Sv[:, :-1].contiguous(),
                     keep, Tv[:, 1:].contiguous())
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default='/workspace/ghazali_forget')
    ap.add_argument('--head', default='/workspace/head_fix/head_ALIGNED_FIX.pt')
    ap.add_argument('--fisher', default='fisher_retained.pt')
    ap.add_argument('--tag', default='F_ret')
    ap.add_argument('--cache', default='/workspace/head_fix/nrmp_cache_9490_aligned')
    ap.add_argument('--synthesis', default='/workspace/rootformer_v12/v18_next_root_morph/data/'
                                           'unified_basran_andalusian_train.jsonl')
    ap.add_argument('--heritage', default='/workspace/data/farahidian_heritage_clean.txt')
    ap.add_argument('--steps', type=int, default=60)
    ap.add_argument('--batch', type=int, default=2)
    ap.add_argument('--seq', type=int, default=128)
    ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--lambdas', default='0,2421.7928259011846')
    ap.add_argument('--val-windows', type=int, default=200)
    ap.add_argument('--heritage-batches', type=int, default=24)
    ap.add_argument('--threads', type=int, default=6)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    layers = (20, 21, 22, 23)
    scope = tuple(f'backbone.layers.{i}.' for i in layers)

    base, flash, model, vocab = build_model(os.path.join(
        REL, 'checkpoints/rootformer_v19_2_synthesis_ar_backbone.awzan142.roots9490.tok10052.safetensors'),
        dtype=torch.float32)
    for p in model.parameters():
        p.requires_grad = False
    named = [(n, p) for n, p in model.named_parameters() if n.startswith(scope)]
    for _, p in named:
        p.requires_grad = True
    theta0 = {n: p.detach().clone() for n, p in named}

    hsd = torch.load(args.head, map_location='cpu')
    r = model.nrmt_head.load_state_dict(hsd, strict=False)
    if r.missing_keys or r.unexpected_keys:
        raise SystemExit(f'FIX head load failed: {r.missing_keys} {r.unexpected_keys}')
    head = model.nrmt_head
    for p in head.parameters():
        p.requires_grad = False

    # data
    syn = tokenize_corpus(base.tokenizer, 'synthesis', args.synthesis, 40000,
                          os.path.join(args.out_dir, 'corpus_synthesis.pt'))
    her = tokenize_corpus(base.tokenizer, 'retained', args.heritage, 40000,
                          os.path.join(args.out_dir, 'corpus_retained.pt'))
    ctx = HP.build_meta(model, args.cache, args.val_windows)
    hi = her.numel() - args.seq - 1
    g = torch.Generator().manual_seed(7)
    herb_ids = [torch.stack([her[a:a + args.seq] for a in
                             torch.randint(0, hi, (args.batch,), generator=g)]).long()
                for _ in range(args.heritage_batches)]

    Fisher = torch.load(os.path.join(args.out_dir, args.fisher), map_location='cpu')
    Fb = {n: Fisher[n].to(torch.float32) for n, _ in named}

    def measure(tag):
        with torch.no_grad():
            ces = [float(lm_loss(flash, ids)) for ids in herb_ids]
        acc = fix_acc(model, head, ctx, args.val_windows)
        out = {'tag': tag, 'heritage_ce': sum(ces) / len(ces), 'fix_acc@1': acc['acc@1'],
               'fix_acc@5': acc['acc@5'], 'fix_ce_z': acc['ce_z'], 'n_pos': acc['n']}
        print(f'  [{tag}] heritage CE {out["heritage_ce"]:.4f} | FIX head acc@1 '
              f'{100*out["fix_acc@1"]:.3f}% acc@5 {100*out["fix_acc@5"]:.3f}% ce_z '
              f'{out["fix_ce_z"]:.4f} (n={acc["n"]})', flush=True)
        return out

    def rel_disp():
        num = sum(float((p.detach().double() - theta0[n].double()).pow(2).sum()) for n, p in named)
        den = sum(float(theta0[n].double().pow(2).sum()) for n, _ in named)
        return (num ** 0.5) / (den ** 0.5)

    def penalty():
        tot = None
        for n, p in named:
            d = p.float() - theta0[n].float()
            t = (Fb[n] * d * d).sum()
            tot = t if tot is None else tot + t
        return tot

    print('[*] ARM lambda=-- (anchor, no training)', flush=True)
    results = {'baseline': measure('theta0_anchor'), 'fisher': args.fisher, 'tag': args.tag,
               'lr': args.lr, 'steps': args.steps,
               'batch': args.batch, 'seq': args.seq, 'val_windows': args.val_windows,
               'heritage_batches': args.heritage_batches, 'arms': {}}

    for lam in [float(x) for x in args.lambdas.split(',')]:
        for n, p in named:
            with torch.no_grad():
                p.copy_(theta0[n])
        opt = torch.optim.AdamW([p for _, p in named], lr=args.lr, weight_decay=0.0)
        hi_s = syn.numel() - args.seq - 1
        gs = torch.Generator().manual_seed(11)
        tl = []
        for step in range(1, args.steps + 1):
            starts = torch.randint(0, hi_s, (args.batch,), generator=gs)
            ids = torch.stack([syn[a:a + args.seq] for a in starts]).long()
            opt.zero_grad(set_to_none=True)
            loss = lm_loss(flash, ids)
            pen = penalty()
            total = loss + lam * pen
            if not torch.isfinite(total):
                raise SystemExit(f'non-finite at lambda={lam} step={step}')
            total.backward()
            opt.step()
            tl.append(float(loss.detach()))
        print(f'[*] ARM lambda={lam:g}: {args.steps} steps done | task loss '
              f'{tl[0]:.3f} -> {tl[-1]:.3f} | rel disp {100*rel_disp():.4f}% | '
              f'penalty {float(penalty()):.4f}', flush=True)
        m = measure(f'lambda_{lam:g}')
        m.update({'lambda': lam, 'rel_disp': rel_disp(), 'task_loss_first': tl[0],
                  'task_loss_last': tl[-1], 'penalty_final': float(penalty())})
        results['arms'][f'{lam:g}'] = m

    json.dump(results, open(os.path.join(args.out_dir, f'cpu_endpoints_{args.tag}.json'), 'w'), indent=2)
    b = results['baseline']
    print('\n=== CPU ENDPOINT TABLE (retained behaviour vs lambda) ===')
    print(f'{"arm":>22} {"rel.disp":>9} {"heritageCE":>11} {"dCE":>8} {"FIXacc@1":>9} {"dAcc":>8}')
    print(f'{"anchor (no train)":>22} {"-":>9} {b["heritage_ce"]:>11.4f} {"-":>8} '
          f'{100*b["fix_acc@1"]:>8.3f}% {"-":>8}')
    for k, m in results['arms'].items():
        print(f'{("lambda=" + k):>22} {100*m["rel_disp"]:>8.3f}% {m["heritage_ce"]:>11.4f} '
              f'{m["heritage_ce"]-b["heritage_ce"]:>+8.4f} {100*m["fix_acc@1"]:>8.3f}% '
              f'{100*(m["fix_acc@1"]-b["fix_acc@1"]):>+7.3f}')
    print('CPU_ENDPOINTS_DONE')


if __name__ == '__main__':
    main()
