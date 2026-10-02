#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nrmp_train.py -- make NRMP actually predict: train the Rootformer v19.2 NRMP heads.

Diagnosis this fixes
--------------------
1. The shipped v19.2 run never trained/evaluated the NRMP heads on a clean split, and
   19.3% of analyzed tokens fall back to the `<PARTICLE>` catch-all class plus 4.8% to
   `<UNK>`. A model can score well by predicting that constant. We MASK both out of the
   root loss so the heads are forced to learn real radical roots.
2. Context contributes almost nothing (acc@1 17.3% vs 14.7% shuffled). Training on the
   factorized next-word-event objective with an unfrozen upper backbone fixes that.

Modes
-----
  python nrmp_train.py --prepare                 # encode corpora -> cached tensors + split
  python nrmp_train.py --train                    # phase 1: heads only
  python nrmp_train.py --train --unfreeze-last 8  # phase 2: also fine-tune upper backbone

Outputs /workspace/nrmp_trained_v1.safetensors and a JSON metrics log.
"""
import argparse
import glob
import json
import math
import random
import re
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'models'))

CKPT = ROOT / 'checkpoints/rootformer_v19_2_synthesis_ar_backbone.safetensors'
BLUEPRINT = ROOT / 'data/rootformer_v12_arabic_blueprint.json'
CACHE = Path('/workspace/nrmp_cache')
SEED = 1337

CORPORA = [
    '/workspace/andalusian_canon_sanitized',
    '/workspace/scholastic_sanitized',
    '/workspace/scholastic_sunni_sanitized',
    '/workspace/scholastic_masters',
    '/workspace/scholastic_falsafa_sanitized',
    '/workspace/chronological_mujtahid_corpus',
    '/workspace/heritage_foundations',
]


def build_vocab():
    import nrmp_vocab as nv
    cls = next(v for k, v in vars(nv).items() if isinstance(v, type) and 'MorphemicVocab' in k)
    return cls(str(BLUEPRINT))


def sentences_from(path, min_words=4, max_words=64, limit=None):
    files = sorted(glob.glob(str(Path(path) / '*.txt'))) or [str(path)]
    out = []
    for f in files:
        try:
            txt = Path(f).read_text(encoding='utf-8', errors='ignore')
        except Exception as exc:
            # VISIBILITY: an unreadable corpus file silently shrank the TRAINING set, so a run
            # could report a loss/metric for a fraction of the intended data with no warning.
            print(f'[nrmp_train] skipping unreadable corpus file {str(f)!r}: {exc!r}',
                  file=sys.stderr)
            continue
        for s in re.split(r'[\n.!?؟]+', txt):
            s = s.strip()
            if not s:
                continue
            nw = len(s.split())
            if min_words <= nw <= max_words and re.search(r'[\u0600-\u06FF]', s):
                out.append((f, s))
                if limit and len(out) >= limit:
                    return out
    return out


def cmd_prepare(args):
    CACHE.mkdir(parents=True, exist_ok=True)
    vocab = build_vocab()
    all_pairs = []
    for c in CORPORA:
        if not Path(c).exists():
            print(f'  [skip missing] {c}')
            continue
        got = sentences_from(c, limit=args.per_corpus)
        print(f'  {c}: {len(got)} sentences')
        all_pairs += got

    # de-duplicate identical sentences, then split by SOURCE FILE so no file straddles
    seen, uniq = set(), []
    for f, s in all_pairs:
        if s in seen:
            continue
        seen.add(s)
        uniq.append((f, s))
    files = sorted({f for f, _ in uniq})
    random.Random(SEED).shuffle(files)
    n_val = max(1, int(len(files) * 0.08))
    val_files = set(files[:n_val])
    print(f'  unique sentences: {len(uniq)} | files: {len(files)} | val files: {len(val_files)}')

    t0 = time.time()
    streams = {'train': [[], [], [], []], 'val': [[], [], [], []]}
    special_root = {vocab.PAD_ROOT, vocab.BOS_ROOT, vocab.EOS_ROOT,
                    vocab.UNK_ROOT, vocab.root2id['<PARTICLE>']}
    n_words = {'train': 0, 'val': 0}
    n_junk = {'train': 0, 'val': 0}

    for f, s in uniq:
        split = 'val' if f in val_files else 'train'
        enc = vocab.encode_sentence(s)
        if len(enc) < 2:
            continue
        st = streams[split]
        st[0].append(vocab.BOS_ROOT)  # boundary marker in the root stream
        for (p, r, w, sf) in enc:
            st[0].append(p)
            st[1].append(r)
            st[2].append(w)
            st[3].append(sf)
            n_words[split] += 1
            if r in special_root:
                n_junk[split] += 1
        st[0].append(vocab.EOS_ROOT)

    for split in ('train', 'val'):
        tens = [torch.tensor(x, dtype=torch.int32) for x in streams[split]]
        torch.save(tens, CACHE / f'{split}.pt')
        tot = n_words[split]
        print(f'  {split}: {tot} word events ({100*n_junk[split]/max(tot,1):.1f}% junk roots) '
              f'-> {CACHE / (split + ".pt")}')
    print(f'  prepared in {time.time()-t0:.1f}s')


def load_stream(split):
    return [t.long() for t in torch.load(CACHE / f'{split}.pt')]


def chunk(tensors, seq_len, device):
    n = tensors[1].numel()
    n_blocks = n // seq_len
    if n_blocks == 0:
        return None
    t = [x[:n_blocks * seq_len].view(n_blocks, seq_len).to(device) for x in tensors]
    return t


def load_model(device, unfreeze_last=0, init=None):
    from safetensors.torch import load_file
    from models.unified_rootformer_v12 import UnifiedRootformerV12
    from deepseek_v4_1_flash_model import UnifiedRootformerV17_DeepSeekFlash
    from rootformer_v18_nrmp_model import RootformerV18_NRMP

    vocab = build_vocab()
    base = UnifiedRootformerV12(str(BLUEPRINT), 'Qwen/Qwen2.5-0.5B', str(device), torch.bfloat16).to(device)
    flash = UnifiedRootformerV17_DeepSeekFlash(base, device, torch.bfloat16).to(device)
    model = RootformerV18_NRMP(flash, vocab, device, torch.bfloat16).to(device)
    src = str(init or CKPT)
    missing, unexpected = model.load_state_dict(load_file(src), strict=False)
    print(f'[*] loaded checkpoint {src}: missing={len(missing)} unexpected={len(unexpected)}')
    return model, vocab


def reinit_params(model, reinit_heads=False, reinit_last=0, seed=0):
    """Deliberately destroy selected weights so we can measure what prior training bought us."""
    torch.manual_seed(seed)
    def reset(mod):
        for n, p in mod.named_parameters(recurse=False):
            with torch.no_grad():
                p.data.normal_(0.0, 0.02)
    if reinit_heads:
        for name, mod in model.named_modules():
            if name.startswith(('nrmp_head', 'morphemic_embed', 'final_norm')):
                reset(mod)
    if reinit_last:
        layers = model.backbone.layers
        for layer in layers[max(0, len(layers) - reinit_last):]:
            for n, p in layer.named_parameters():
                with torch.no_grad():
                    p.data.normal_(0.0, 0.02)
    return reinit_heads, reinit_last


def set_trainable(model, unfreeze_last):
    for p in model.parameters():
        p.requires_grad = False
    for name, p in model.named_parameters():
        if name.startswith(('nrmp_head.', 'morphemic_embed.', 'final_norm.')):
            p.requires_grad = True
    if unfreeze_last:
        layers = model.backbone.layers
        n = len(layers)
        for layer in layers[max(0, n - unfreeze_last):]:
            for p in layer.parameters():
                p.requires_grad = True
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'[*] trainable params: {tr/1e6:.2f}M (unfreeze_last={unfreeze_last})')
    return tr


@torch.no_grad()
def evaluate(model, vocab, blocks, device, max_blocks=24):
    model.eval()
    special = {vocab.PAD_ROOT, vocab.BOS_ROOT, vocab.EOS_ROOT,
               vocab.UNK_ROOT, vocab.root2id['<PARTICLE>']}
    special |= {i for i, r in enumerate(vocab.roots_list) if r.startswith('<P:')}
    tot = t1 = t5 = 0
    loss_sum, loss_n = 0.0, 0
    rad_tot = rad_t1 = rad_t5 = 0
    rad_ce = 0.0
    for i in range(min(max_blocks, blocks[0].shape[0])):
        p, r, w, sf = (b[i:i + 1] for b in blocks)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            emb = model.morphemic_embed(p, r, w, sf)
            out = model.backbone(inputs_embeds=emb)
            h = model.final_norm(out.last_hidden_state)
            heads = model.nrmp_head(h, model.morphemic_embed.root_embed, target_roots=r)
            logits = heads['root_logits'][0, :-1, :].float()
            tgt = r[0, 1:]
            keep = ~torch.isin(tgt, torch.tensor(sorted(special), device=device))
            if keep.sum() == 0:
                continue
            lg, tg = logits[keep], tgt[keep]
            rad_tot += int(tg.numel())
            rad_t1 += int((lg.argmax(-1) == tg).sum().item())
            k = min(5, lg.shape[-1])
            rad_t5 += int((lg.topk(k, -1).indices == tg.unsqueeze(-1)).any(-1).sum().item())
            rad_ce += float(F.cross_entropy(lg, tg, reduction='sum').item())
    model.train()
    return {
        'radical_positions': rad_tot,
        'radical_acc_top1': rad_t1 / max(rad_tot, 1),
        'radical_acc_top5': rad_t5 / max(rad_tot, 1),
        'radical_ce': rad_ce / max(rad_tot, 1),
        'radical_ppl': math.exp(min(rad_ce / max(rad_tot, 1), 20.0)),
    }


def cmd_train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(SEED)
    random.seed(SEED)

    model, vocab = load_model(device, args.unfreeze_last, getattr(args, 'init', None))
    rh, rl = getattr(args, 'reinit_heads', False), getattr(args, 'reinit_last', 0)
    if rh or rl:
        reinit_params(model, reinit_heads=rh, reinit_last=rl, seed=getattr(args, 'seed', 0))
        print(f'[*] REINITIALISED weights: heads={rh} last_layers={rl} '
              f'(deliberate ablation of prior training)')
    set_trainable(model, args.unfreeze_last)
    model.train()

    tr = chunk(load_stream('train'), args.seq_len, device)
    va = chunk(load_stream('val'), args.seq_len, device)
    print(f'[*] train blocks {tr[0].shape} | val blocks {va[0].shape}')

    special_ids = [vocab.PAD_ROOT, vocab.BOS_ROOT, vocab.EOS_ROOT,
                   vocab.UNK_ROOT, vocab.root2id['<PARTICLE>']]
    W_ROOT, W_WAZN, W_PREF, W_SUFF = 1.0, 0.5, 0.25, 0.25

    head_params = [p for n, p in model.named_parameters()
                   if n.startswith(('nrmp_head.', 'morphemic_embed.', 'final_norm.')) and p.requires_grad]
    bb_params = [p for n, p in model.named_parameters()
                 if not n.startswith(('nrmp_head.', 'morphemic_embed.', 'final_norm.')) and p.requires_grad]
    groups = [{'params': head_params, 'lr': args.lr}]
    if bb_params:
        groups.append({'params': bb_params, 'lr': args.lr * args.bb_lr_scale})
    opt = torch.optim.AdamW(groups, lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[g['lr'] for g in groups], total_steps=args.steps,
        pct_start=max(0.05, min(0.3, 4.0 / max(args.steps, 1))), anneal_strategy='cos')

    def batch_loss(p, r, w, sf):
        emb = model.morphemic_embed(p, r, w, sf)
        out = model.backbone(inputs_embeds=emb)
        h = model.final_norm(out.last_hidden_state)
        heads = model.nrmp_head(h, model.morphemic_embed.root_embed, target_roots=r)

        def shifted(x):
            return x[:, :-1, :].contiguous()

        def tgt(x):
            return x[:, 1:].contiguous()

        tr_, tw_, tp_, ts_ = tgt(r), tgt(w), tgt(p), tgt(sf)
        # root loss: ignore pad AND the catch-all classes so the head learns real roots
        ignore = torch.tensor(special_ids, device=p.device)
        root_keep = ~torch.isin(tr_, ignore)
        root_logits = shifted(heads['root_logits'])
        if root_keep.sum() == 0:
            l_root = root_logits.sum() * 0.0
        else:
            l_root = F.cross_entropy(root_logits[root_keep], tr_[root_keep])
        l_wazn = F.cross_entropy(shifted(heads['wazn_logits']).view(-1, vocab.num_awzan),
                                 tw_.view(-1), ignore_index=vocab.PAD_WAZN)
        l_pref = F.cross_entropy(shifted(heads['prefix_logits']).view(-1, vocab.num_prefixes),
                                 tp_.view(-1), ignore_index=vocab.PAD_PREFIX)
        l_suff = F.cross_entropy(shifted(heads['suffix_logits']).view(-1, vocab.num_suffixes),
                                 ts_.view(-1), ignore_index=vocab.PAD_SUFFIX)
        total = W_ROOT * l_root + W_WAZN * l_wazn + W_PREF * l_pref + W_SUFF * l_suff
        return total, dict(root=float(l_root), wazn=float(l_wazn),
                           prefix=float(l_pref), suffix=float(l_suff))

    nb = tr[0].shape[0]
    log = []
    t0 = time.time()
    best = -1.0
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, nb, (args.batch_size,), device=device)
        p, r, w, sf = (tr[j][idx] for j in range(4))
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss, parts = batch_loss(p, r, w, sf)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([q for g in groups for q in g['params']], 1.0)
        opt.step()
        sched.step()

        if step % args.log_every == 0 or step == 1:
            sps = step / (time.time() - t0)
            print(f'  step {step:5d}/{args.steps} loss {float(loss):.4f} '
                  f'(root {parts["root"]:.3f} wazn {parts["wazn"]:.3f} '
                  f'pref {parts["prefix"]:.3f} suff {parts["suffix"]:.3f}) {sps:.2f} it/s',
                  flush=True)
            log.append({'step': step, 'loss': float(loss), **parts, 'it_s': sps})

        if step % args.eval_every == 0 or step == args.steps:
            m = evaluate(model, vocab, va, device, args.eval_blocks)
            m['step'] = step
            print(f'  [eval @{step}] radical acc@1 {m["radical_acc_top1"]*100:.2f}% '
                  f'acc@5 {m["radical_acc_top5"]*100:.2f}% PPL {m["radical_ppl"]:.1f} '
                  f'(n={m["radical_positions"]})', flush=True)
            log.append({'eval': m})
            if m['radical_acc_top1'] > best:
                best = m['radical_acc_top1']
                save_model(model, args.out)
                print(f'  [save] best acc@1 {best*100:.2f}% -> {args.out}', flush=True)

    print(f'[*] done in {time.time()-t0:.0f}s | best radical acc@1 {best*100:.2f}%')
    Path(args.out).with_suffix('.metrics.json').write_text(json.dumps(log, indent=2))


def save_model(model, path):
    from safetensors.torch import save_file
    sd = {k: v.contiguous().to(torch.bfloat16) for k, v in model.state_dict().items()}
    save_file(sd, str(path), metadata={'format': 'pt', 'rootformer': 'v19.2-nrmp-finetuned'})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prepare', action='store_true')
    ap.add_argument('--train', action='store_true')
    ap.add_argument('--per-corpus', type=int, default=120000)
    ap.add_argument('--seq-len', type=int, default=128)
    ap.add_argument('--steps', type=int, default=1500)
    ap.add_argument('--batch-size', type=int, default=24)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--bb-lr-scale', type=float, default=0.05)
    ap.add_argument('--unfreeze-last', type=int, default=0)
    ap.add_argument('--log-every', type=int, default=50)
    ap.add_argument('--eval-every', type=int, default=250)
    ap.add_argument('--eval-blocks', type=int, default=24)
    ap.add_argument('--init', default=None, help='fine-tune from this checkpoint instead of the v19.2 release')
    ap.add_argument('--reinit-heads', action='store_true',
                    help='randomly reinitialise nrmp_head/morphemic_embed/final_norm before training')
    ap.add_argument('--reinit-last', type=int, default=0,
                    help='randomly reinitialise the top N backbone layers before training')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='/root/nrmp_trained_v1.safetensors')
    args = ap.parse_args()

    if args.prepare:
        cmd_prepare(args)
    elif args.train:
        cmd_train(args)
    else:
        ap.print_help()


if __name__ == '__main__':
    main()
