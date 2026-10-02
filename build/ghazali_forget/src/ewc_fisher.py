#!/usr/bin/env python3
"""ewc_fisher.py -- measure the essential/accidental split for the trunk, for real.

al-dhatiyyat (essential)   := high diagonal Fisher on the RETAINED linguistic distribution
al-'aradiyyat (accidental) := low Fisher there (matters only for the abandoned objective)

Two distributions, one functional, real data, no hypothesis:

  retained  = held-out Classical Arabic heritage running text
              (/workspace/data/farahidian_heritage_clean.txt, tail slice)
  synthesis = the abandoned v19.2 synthesis corpus (the bilingual Basran-Andalusian
              /workspace/rootformer_v12/v18_next_root_morph/data/unified_basran_andalusian_train.jsonl)

Both are fed through the SAME trained path (token -> IshtiqaqAttention with active_root_ids set
from the token's own root -> tied lm_head) and the SAME functional (next-token CE).  The only
thing that differs is the data distribution.  That is the minimal form of the claim, so if the
two importance profiles come out the same, the distinction is empty and there is nothing to
protect selectively.

Estimator: diagonal EMPIRICAL Fisher, F_i = E_b[ (d loss_b / d theta_i)^2 ], loss_b = the same
batch-mean cross-entropy the trainer optimises.  This keeps the arithmetic consistent with the
EWC quadratic: L(theta0 + d) ~ L(theta0) + 1/2 sum_i F_i d_i^2, so lambda = 1/2 is the Laplace
value rather than a tuned knob.

Model + checkpoint construction is copied verbatim from nrmt_train.py (RootformerNRMT), because
the released checkpoint's keys are RootformerNRMT's names (backbone.layers.*), NOT
UnifiedRootformerV12's (backbone.model.layers.*).

CPU only (CUDA_VISIBLE_DEVICES is cleared).  Nothing on the pod is modified.
"""
import os, sys, json, time, argparse
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
import torch

REL = '/workspace/hf_v19_2_release'
sys.path.insert(0, REL)
sys.path.insert(0, os.path.join(REL, 'models'))

LAYERS = (20, 21, 22, 23)


def build_model(ckpt, dtype=torch.float32):
    from models.unified_rootformer_v12 import UnifiedRootformerV12
    from deepseek_v4_1_flash_model import UnifiedRootformerV17_DeepSeekFlash
    from safetensors.torch import load_file
    import nrmp_vocab as nv
    from nrmt_arch import RootformerNRMT
    bp = os.path.join(REL, 'data/rootformer_v12_arabic_blueprint.json')
    V = next(v for k, v in vars(nv).items() if isinstance(v, type) and 'MorphemicVocab' in k)
    vocab = V(bp)
    base = UnifiedRootformerV12(bp, 'Qwen/Qwen2.5-0.5B', 'cpu', dtype)
    flash = UnifiedRootformerV17_DeepSeekFlash(base, 'cpu', dtype)
    model = RootformerNRMT(flash, vocab, 'cpu', dtype, hist=3, dropout=0.0,
                           use_features=True, feat_gate=True)
    sd = load_file(ckpt)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    bb_missing = [k for k in missing if k.startswith('backbone.')]
    real_unexp = [k for k in unexpected if not k.startswith('nrmp_head.')]
    print(f'[*] load: missing={len(missing)} (backbone missing={len(bb_missing)}) '
          f'unexpected={len(unexpected)} (non-legacy-head={len(real_unexp)})', flush=True)
    if bb_missing or real_unexp:
        raise SystemExit(f'CHECKPOINT DID NOT LOAD CLEANLY: bb_missing={bb_missing[:4]} '
                         f'unexpected={real_unexp[:4]}')
    return base, flash, model, vocab


def lm_loss(flash, ids):
    """The trunk's own linguistic objective on its trained path (identical to
    UnifiedRootformerV12.forward's loss_lm, without the layer-14 auxiliary heads)."""
    m = flash.model
    root_ids, wazn_ids = m.extract_morphemic_ids(ids)
    for layer in m.backbone.model.layers:
        layer.self_attn.active_root_ids = root_ids
        layer.self_attn.active_wazn_ids = wazn_ids
    return m.backbone(input_ids=ids, labels=ids).loss


def tokenize_corpus(tok, kind, path, want_tokens, cache):
    if os.path.exists(cache):
        t = torch.load(cache, map_location='cpu')
        print(f'[corpus:{kind}] cache {cache} -> {tuple(t.shape)} tokens', flush=True)
        return t
    ids = []
    n_tok = 0
    t0 = time.time()
    if kind == 'retained':
        # held-out slice: start 60 % into the file so this is not the training prefix
        size = os.path.getsize(path)
        with open(path, encoding='utf-8', errors='ignore') as f:
            f.seek(int(size * 0.60))
            f.readline()
            for line in f:
                line = line.strip()
                if len(line) < 20:
                    continue
                e = tok.encode(line)
                if len(e) < 6:
                    continue
                ids.extend(e)
                ids.append(10)          # the stream's own separator token (see probe: id 10)
                n_tok += len(e)
                if n_tok >= want_tokens:
                    break
    else:
        lines = []
        with open(path, encoding='utf-8', errors='ignore') as f:
            for line in f:
                lines.append(line)
        tail = lines[int(len(lines) * 0.5):]        # held-out half of the synthesis corpus
        for line in tail:
            try:
                it = json.loads(line)
            except Exception:
                continue
            ar = (it.get('arabic') or '').strip()
            if len(ar.split()) < 2:
                continue
            e = tok.encode(ar)
            if len(e) < 4:
                continue
            ids.extend(e)
            ids.append(10)
            n_tok += len(e)
            if n_tok >= want_tokens:
                break
    t = torch.tensor(ids, dtype=torch.long)
    torch.save(t, cache)
    print(f'[corpus:{kind}] {path.split("/")[-1]} -> {tuple(t.shape)} tokens in '
          f'{time.time()-t0:.1f}s', flush=True)
    return t


def fisher_profile(flash, model, stream, n_batches, batch, seq, layers, tag):
    """Diagonal empirical Fisher over the trunk params of `layers`."""
    scope = tuple(f'backbone.layers.{i}.' for i in layers)
    params = [(n, p) for n, p in model.named_parameters() if n.startswith(scope)]
    n_par = sum(p.numel() for _, p in params)
    print(f'[fisher:{tag}] {len(params)} tensors / {n_par/1e6:.2f}M params in layers {layers}',
          flush=True)
    F = {n: torch.zeros_like(p, dtype=torch.float32) for n, p in params}
    hi = stream.numel() - seq - 1
    g = torch.Generator().manual_seed(20261002)
    t0 = time.time()
    losses = []
    for b in range(n_batches):
        starts = torch.randint(0, hi, (batch,), generator=g)
        ids = torch.stack([stream[s:s + seq] for s in starts]).long()
        for _, p in params:
            p.grad = None
        loss = lm_loss(flash, ids)
        if not torch.isfinite(loss):
            print(f'[warn] non-finite loss at batch {b}; skipped', flush=True)
            continue
        loss.backward()
        losses.append(float(loss.detach()))
        with torch.no_grad():
            for n, p in params:
                if p.grad is not None:
                    F[n].add_(p.grad.detach().float().pow(2))
        for _, p in params:
            p.grad = None
        if (b + 1) % 8 == 0:
            print(f'  [{tag}] batch {b+1}/{n_batches} mean_ce {sum(losses)/len(losses):.4f} '
                  f'{time.time()-t0:.0f}s', flush=True)
    for n in F:
        F[n].div_(max(len(losses), 1))
    mean_ce = sum(losses) / max(len(losses), 1)
    print(f'[fisher:{tag}] done in {time.time()-t0:.0f}s | mean CE {mean_ce:.4f} '
          f'(ppl {torch.exp(torch.tensor(mean_ce)).item():.1f})', flush=True)
    return F, mean_ce, len(losses)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default=os.path.join(
        REL, 'checkpoints/rootformer_v19_2_synthesis_ar_backbone.awzan142.roots9490.tok10052.safetensors'))
    ap.add_argument('--out-dir', default='/workspace/ghazali_forget')
    ap.add_argument('--heritage', default='/workspace/data/farahidian_heritage_clean.txt')
    ap.add_argument('--synthesis', default='/workspace/rootformer_v12/v18_next_root_morph/data/'
                                           'unified_basran_andalusian_train.jsonl')
    ap.add_argument('--tokens', type=int, default=40000)
    ap.add_argument('--batches', type=int, default=48)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--seq', type=int, default=128)
    ap.add_argument('--layers', default='20,21,22,23')
    ap.add_argument('--threads', type=int, default=6)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    layers = tuple(int(x) for x in args.layers.split(','))
    os.makedirs(args.out_dir, exist_ok=True)

    base, flash, model, vocab = build_model(args.checkpoint)
    for p in model.parameters():
        p.requires_grad = False
    scope = tuple(f'backbone.layers.{i}.' for i in layers)
    for n, p in model.named_parameters():
        if n.startswith(scope):
            p.requires_grad = True

    # -- consistency check: our fast path must equal the shipped UnifiedRootformerV12 path
    chk = torch.load(os.path.join(args.out_dir, 'corpus_retained.pt'), map_location='cpu') \
        if os.path.exists(os.path.join(args.out_dir, 'corpus_retained.pt')) else None
    if chk is None:
        tok = base.tokenizer
        chk = tokenize_corpus(tok, 'retained', args.heritage, args.tokens,
                              os.path.join(args.out_dir, 'corpus_retained.pt'))
    ids_chk = chk[5000:5000 + args.batch * args.seq].reshape(args.batch, args.seq).long()
    l_fast = float(lm_loss(flash, ids_chk))
    with torch.no_grad():
        l_ship = float(base.forward(input_ids=ids_chk, labels=ids_chk)['loss_lm'])
    print(f'[*] path check: fast={l_fast:.8f} shipped={l_ship:.8f} '
          f'|d|={abs(l_fast-l_ship):.3e} -> '
          f'{"IDENTICAL" if abs(l_fast-l_ship) < 1e-6 else "MISMATCH"}', flush=True)

    tok = base.tokenizer
    rep = {'layers': list(layers), 'n_batches': args.batches, 'batch': args.batch,
           'seq': args.seq, 'checkpoint': args.checkpoint,
           'path_check': {'fast': l_fast, 'shipped': l_ship, 'absdiff': abs(l_fast - l_ship)}}

    profiles = {}
    for tag, kind, path in (('retained', 'retained', args.heritage),
                            ('synthesis', 'synthesis', args.synthesis)):
        stream = tokenize_corpus(tok, kind, path, args.tokens,
                                 os.path.join(args.out_dir, f'corpus_{kind}.pt'))
        F, mean_ce, nb = fisher_profile(flash, model, stream, args.batches, args.batch,
                                        args.seq, layers, tag)
        torch.save(F, os.path.join(args.out_dir, f'fisher_{tag}.pt'))
        rep[tag] = {'mean_ce': mean_ce, 'n_batches': nb,
                    'sum_F': float(sum(v.double().sum() for v in F.values())),
                    'n_params': int(sum(v.numel() for v in F.values()))}
        profiles[tag] = F
    json.dump(rep, open(os.path.join(args.out_dir, 'fisher_report.json'), 'w'), indent=2)
    print('[fisher] wrote fisher_retained.pt / fisher_synthesis.pt / fisher_report.json',
          flush=True)
    print('FISHER_DONE')


if __name__ == '__main__':
    main()
