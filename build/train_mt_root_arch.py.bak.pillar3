#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_mt_root_arch.py -- ARCHITECTURAL root conditioning for Arabic->English.

Round 1 showed that TEXTUAL root conditioning (prefixing a Farahidian concept string) gives no gain
(-0.28 chrF++). This script tests the architectural form: continuous vectors derived from the
morphemic (prefix, root, wazn, suffix) stream are PREPENDED to the encoder input embeddings and the
decoder attends to them.

Three matched arms, all continuing from the SAME in-domain fine-tuned checkpoint, same data slice,
same schedule. The only difference is the conditioning:

  A. arch_off      -- no root vectors                          (matched control)
  B. arch_real     -- root vectors from the sentence's own tuples
  C. arch_shuffled -- root vectors taken from a DIFFERENT sentence (null control)

If B > A, architectural root conditioning helps. If B ~= C, it does not come from the roots being
correct. Evaluation is chrF++ / BLEU on mt_split/test_clean.jsonl (works absent from training).

Usage:
  python train_mt_root_arch.py --arm arch_real --base /workspace/mt_subword_model --steps 1500
"""
import argparse
import json
import math
import pathlib
import random
import sys
import time

import torch
import torch.nn as nn

SPLIT = pathlib.Path('/workspace/mt_split')
TUPLE_CACHE = SPLIT / 'root_tuples.jsonl'
MAX_WORDS = 96
K_VEC = 8                      # number of prepended root vectors
D_ROOT = 256
sys.path.insert(0, '/workspace/hf_v19_2_release')
sys.path.insert(0, '/workspace/hf_v19_2_release/models')


def read_jsonl(p, limit=None):
    rows = [json.loads(l) for l in open(p, encoding='utf-8') if l.strip()]
    return rows[:limit] if limit else rows


def build_tuple_cache(sentences):
    """Cache the morphemic tuple ids for each sentence (analyzer is CPU and deterministic)."""
    cache = {}
    if TUPLE_CACHE.exists():
        for l in open(TUPLE_CACHE, encoding='utf-8'):
            try:
                d = json.loads(l)
                cache[d['arabic']] = d['tuples']
            except Exception:
                pass
    missing = [s for s in sentences if s not in cache]
    if missing:
        import nrmp_vocab as nv
        V = next(v for k, v in vars(nv).items() if isinstance(v, type) and 'MorphemicVocab' in k)
        vocab = V('/workspace/hf_v19_2_release/data/rootformer_v12_arabic_blueprint.json')
        t0 = time.time()
        with open(TUPLE_CACHE, 'a', encoding='utf-8') as f:
            for i, s in enumerate(missing):
                t = vocab.encode_sentence(s)[:MAX_WORDS]
                cache[s] = t
                f.write(json.dumps({'arabic': s, 'tuples': t}, ensure_ascii=False) + '\n')
                if (i + 1) % 20000 == 0:
                    print(f'    tuples {i+1}/{len(missing)} ({time.time()-t0:.0f}s)', flush=True)
        print(f'    built {len(missing)} tuple sequences in {time.time()-t0:.0f}s', flush=True)
    return cache


class RootEncoder(nn.Module):
    """(prefix, root, wazn, suffix) id stream -> K continuous vectors of width d_model."""

    def __init__(self, vocab_sizes, d_model, d_root=D_ROOT, k=K_VEC, n_layers=2):
        super().__init__()
        n_roots, n_wazn, n_pref, n_suff = vocab_sizes
        self.k = k
        # same sub-dimension ratios as the project's FarahidianWordEmbedding:
        # root 1/2, wazn 1/4, prefix 1/8, suffix the remainder -> concat == d_root
        d_r, d_w = d_root // 2, d_root // 4
        d_p = d_root // 8
        d_s = d_root - d_r - d_w - d_p
        self.emb_root = nn.Embedding(n_roots, d_r)
        self.emb_wazn = nn.Embedding(n_wazn, d_w)
        self.emb_pref = nn.Embedding(n_pref, d_p)
        self.emb_suff = nn.Embedding(n_suff, d_s)
        self.proj = nn.Linear(d_root, d_model)
        layer = nn.TransformerEncoderLayer(d_model, nhead=4, dim_feedforward=d_model * 2,
                                          dropout=0.1, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.query = nn.Parameter(torch.randn(k, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(d_model, 4, batch_first=True)

    def forward(self, tuples, mask):
        # tuples: [B, T, 4] ; mask: [B, T] 1=real
        p, r, w, s = tuples[..., 0], tuples[..., 1], tuples[..., 2], tuples[..., 3]
        x = torch.cat([self.emb_root(r), self.emb_wazn(w),
                       self.emb_pref(p), self.emb_suff(s)], dim=-1)
        x = self.proj(x)
        x = self.enc(x, src_key_padding_mask=(mask == 0))
        q = self.query.unsqueeze(0).expand(x.shape[0], -1, -1)
        out, _ = self.attn(q, x, x, key_padding_mask=(mask == 0))
        return out                                       # [B, K, d_model]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', choices=['arch_off', 'arch_real', 'arch_shuffled'], required=True)
    ap.add_argument('--base', default='/workspace/mt_subword_model')
    ap.add_argument('--steps', type=int, default=1500)
    ap.add_argument('--batch-size', type=int, default=24)
    ap.add_argument('--lr', type=float, default=1e-5)
    ap.add_argument('--train-n', type=int, default=40000)
    ap.add_argument('--test-n', type=int, default=1500)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    out_path = args.out or f'/workspace/mt_{args.arm}.json'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    from transformers import MarianMTModel, MarianTokenizer
    tok = MarianTokenizer.from_pretrained(args.base)
    model = MarianMTModel.from_pretrained(args.base).to(device)
    d_model = model.config.d_model
    print(f'[*] ARM={args.arm} base={args.base} d_model={d_model} scale_embedding='
          f'{getattr(model.config, "scale_embedding", False)}', flush=True)

    import nrmp_vocab as nv
    V = next(v for k, v in vars(nv).items() if isinstance(v, type) and 'MorphemicVocab' in k)
    vocab = V('/workspace/hf_v19_2_release/data/rootformer_v12_arabic_blueprint.json')

    train = read_jsonl(SPLIT / 'train.jsonl', args.train_n)
    test = read_jsonl(SPLIT / 'test_clean.jsonl', args.test_n)
    tup = build_tuple_cache([r['arabic'] for r in train] + [r['arabic'] for r in test])

    root_enc = None
    if args.arm != 'arch_off':
        root_enc = RootEncoder((vocab.num_roots, vocab.num_awzan, vocab.num_prefixes,
                                vocab.num_suffixes), d_model).to(device)
        print(f'[*] RootEncoder params: {sum(p.numel() for p in root_enc.parameters())/1e6:.2f}M')

    def enc_inputs(batch_rows, shuffle_seed=None):
        """Return (inputs_embeds, attention_mask) or (None, None) for arch_off."""
        ids = tok([r['arabic'] for r in batch_rows], return_tensors='pt', padding=True,
                  truncation=True, max_length=256).to(device)
        emb = model.get_input_embeddings()(ids.input_ids)
        if getattr(model.config, 'scale_embedding', False):
            emb = emb * (d_model ** 0.5)
        if root_enc is None:
            return emb, ids.attention_mask
        seqs, m = [], []
        rows = list(batch_rows)
        if args.arm == 'arch_shuffled':
            rows = rows[1:] + rows[:1]                    # rotate -> roots belong to another sentence
        for r in rows:
            t = tup.get(r['arabic'], [])
            t = t[:MAX_WORDS]
            seqs.append(t if t else [(0, 0, 0, 0)])
        T = max(len(x) for x in seqs)
        tt = torch.zeros(len(seqs), T, 4, dtype=torch.long)
        mm = torch.zeros(len(seqs), T, dtype=torch.long)
        for i, x in enumerate(seqs):
            tt[i, :len(x)] = torch.tensor(x)
            mm[i, :len(x)] = 1
        tt, mm = tt.to(device), mm.to(device)
        rv = root_enc(tt, mm)                             # [B, K, d]
        return (torch.cat([rv, emb], dim=1),
                torch.cat([torch.ones(emb.shape[0], K_VEC, dtype=ids.attention_mask.dtype,
                                      device=device), ids.attention_mask], dim=1))

    params = list(model.parameters()) + (list(root_enc.parameters()) if root_enc else [])
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.steps,
        pct_start=max(0.05, min(0.3, 4.0 / max(args.steps, 1))))
    model.train()
    if root_enc:
        root_enc.train()

    order = torch.randperm(len(train)).tolist()
    step = 0
    t0 = time.time()
    while step < args.steps:
        for i in range(0, len(order), args.batch_size):
            if step >= args.steps:
                break
            rows = [train[j] for j in order[i:i + args.batch_size]]
            labels = tok(text_target=[r['english'] for r in rows], return_tensors='pt',
                         padding=True, truncation=True, max_length=256).input_ids.to(device)
            labels[labels == tok.pad_token_id] = -100
            inputs_embeds, am = enc_inputs(rows)
            out = model(inputs_embeds=inputs_embeds, attention_mask=am, labels=labels)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % 250 == 0:
                print(f'  step {step}/{args.steps} loss {float(out.loss):.4f} '
                      f'{step/(time.time()-t0):.2f} it/s', flush=True)

    # ---- evaluate on the leak-free held-out works -----------------------------------------
    @torch.no_grad()
    def translate(rows, bs=32):
        model.eval()
        hyps = []
        for i in range(0, len(rows), bs):
            chunk = rows[i:i + bs]
            emb, am = enc_inputs(chunk)
            gen = model.generate(inputs_embeds=emb, attention_mask=am, max_new_tokens=200,
                                 num_beams=1, do_sample=False)
            hyps.extend(tok.batch_decode(gen, skip_special_tokens=True))
        model.train()
        return hyps

    refs = [r['english'] for r in test]
    hyps = translate(test)
    from sacrebleu.metrics import BLEU, CHRF
    c = CHRF(word_order=2).corpus_score(hyps, [refs]).score
    b = BLEU(tokenize='13a').corpus_score(hyps, [refs]).score
    print(f'\n  *** {args.arm}: TEST chrF++ {c:.2f}  BLEU {b:.2f}  (n={len(test)})', flush=True)
    json.dump({'arm': args.arm, 'base': args.base, 'steps': args.steps,
               'n_train': len(train), 'n_test': len(test),
               'test_chrf++': c, 'test_bleu': b,
               'samples': [{'src': test[i]['arabic'][:160], 'ref': refs[i][:200],
                            'hyp': hyps[i][:200]} for i in range(min(6, len(test)))]},
              open(out_path, 'w'), ensure_ascii=False, indent=2)
    print(f'wrote {out_path}')


if __name__ == '__main__':
    main()
