#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_mt_inomain.py -- does Farahidian root-awareness measurably help Arabic->English?

Design
------
Both arms fine-tune the SAME pretrained subword NMT model (Helsinki-NLP/opus-mt-ar-en) on the
SAME in-domain data (the 117,264 classical pairs whose works are disjoint from the test set).
The ONLY difference is the source representation:

  subword :  <arabic>
  root    :  <farahidian root-concept string> ||| <arabic>

so any difference on the leak-free held-out works is attributable to the root conditioning, not to
capacity, data or tuning. Evaluation uses chrF++ / BLEU on mt_split/test_clean.jsonl
(17,916 sentences from 31 works that contribute no training pairs and share no 13-gram with train).

Usage:
  python train_mt_inomain.py --arm subword --epochs 2
  python train_mt_inomain.py --arm root    --epochs 2
"""
import argparse
import json
import math
import pathlib
import sys
import time

import torch

SPLIT = pathlib.Path('/workspace/mt_split')
CONCEPT_CACHE = pathlib.Path('/workspace/mt_split/concepts.jsonl')
sys.path.insert(0, '/workspace/hf_v19_2_release')
sys.path.insert(0, '/workspace/hf_v19_2_release/models')


def read_jsonl(p, limit=None):
    rows = [json.loads(l) for l in open(p, encoding='utf-8') if l.strip()]
    return rows[:limit] if limit else rows


def build_concepts(rows):
    """Cache the Farahidian (concept, root) string for every source sentence."""
    cache = {}
    if CONCEPT_CACHE.exists():
        for l in open(CONCEPT_CACHE, encoding='utf-8'):
            try:
                d = json.loads(l)
                cache[d['arabic']] = d['concepts']
            except Exception:
                pass
    missing = [r['arabic'] for r in rows if r['arabic'] not in cache]
    if missing:
        from nmt_eval import ConceptExtractor, concepts_to_str
        ce = ConceptExtractor()
        t0 = time.time()
        with open(CONCEPT_CACHE, 'a', encoding='utf-8') as f:
            for i, a in enumerate(missing):
                c = concepts_to_str(ce.concepts(a))
                cache[a] = c
                f.write(json.dumps({'arabic': a, 'concepts': c}, ensure_ascii=False) + '\n')
                if (i + 1) % 20000 == 0:
                    print(f'    concepts {i+1}/{len(missing)} ({time.time()-t0:.0f}s)', flush=True)
        print(f'    built {len(missing)} concept strings in {time.time()-t0:.0f}s', flush=True)
    return cache


def make_src(rows, concepts, arm):
    if arm == 'subword':
        return [r['arabic'] for r in rows]
    return [f"{concepts.get(r['arabic'], '')} ||| {r['arabic']}" for r in rows]


@torch.no_grad()
def translate(model, tok, texts, device, batch_size=48, max_new=200):
    model.eval()
    out = []
    for i in range(0, len(texts), batch_size):
        enc = tok(texts[i:i + batch_size], return_tensors='pt', padding=True,
                  truncation=True, max_length=320).to(device)
        gen = model.generate(**enc, max_new_tokens=max_new, num_beams=1, do_sample=False)
        out.extend(tok.batch_decode(gen, skip_special_tokens=True))
    return out


def chrf(hyps, refs):
    from sacrebleu.metrics import BLEU, CHRF
    return (CHRF(word_order=2).corpus_score(hyps, [refs]).score,
            BLEU(tokenize='13a').corpus_score(hyps, [refs]).score)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', choices=['subword', 'root'], required=True)
    ap.add_argument('--model', default='Helsinki-NLP/opus-mt-ar-en')
    ap.add_argument('--epochs', type=int, default=2)
    ap.add_argument('--batch-size', type=int, default=48)
    ap.add_argument('--lr', type=float, default=3e-5)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--dev-n', type=int, default=400)
    ap.add_argument('--test-n', type=int, default=1500)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    out_path = args.out or f'/workspace/mt_{args.arm}.json'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    from transformers import MarianMTModel, MarianTokenizer
    tok = MarianTokenizer.from_pretrained(args.model)
    model = MarianMTModel.from_pretrained(args.model).to(device)

    train = read_jsonl(SPLIT / 'train.jsonl', args.limit or None)
    dev = read_jsonl(SPLIT / 'dev.jsonl', args.dev_n)
    test = read_jsonl(SPLIT / 'test_clean.jsonl', args.test_n)
    print(f'[*] ARM={args.arm} train={len(train)} dev={len(dev)} test={len(test)}')

    concepts = build_concepts(train + dev + test) if args.arm == 'root' else {}
    src_tr = make_src(train, concepts, args.arm)
    tgt_tr = [r['english'] for r in train]
    src_dv = make_src(dev, concepts, args.arm)
    ref_dv = [r['english'] for r in dev]
    src_te = make_src(test, concepts, args.arm)
    ref_te = [r['english'] for r in test]
    if args.arm == 'root':
        print('  example source:', src_tr[0][:220])

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps_per_epoch = math.ceil(len(src_tr) / args.batch_size)
    total = steps_per_epoch * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total,
                                               pct_start=0.05)
    model.train()
    hist = []
    step = 0
    t0 = time.time()
    for ep in range(args.epochs):
        order = torch.randperm(len(src_tr)).tolist()
        for i in range(0, len(order), args.batch_size):
            idx = order[i:i + args.batch_size]
            batch = tok([src_tr[j] for j in idx], text_target=[tgt_tr[j] for j in idx],
                        return_tensors='pt', padding=True, truncation=True,
                        max_length=320).to(device)
            loss = model(**batch).loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            step += 1
            if step % 400 == 0:
                print(f'  ep{ep+1} step {step}/{total} loss {float(loss):.4f} '
                      f'{step/(time.time()-t0):.2f} it/s', flush=True)
        hyp = translate(model, tok, src_dv, device)
        c, b = chrf(hyp, ref_dv)
        hist.append({'epoch': ep + 1, 'dev_chrf++': c, 'dev_bleu': b})
        print(f'  == after epoch {ep+1}: dev chrF++ {c:.2f}  BLEU {b:.2f}', flush=True)
        model.train()

    hyp_te = translate(model, tok, src_te, device)
    c, b = chrf(hyp_te, ref_te)
    print(f'\n  *** TEST (leak-free held-out works) chrF++ {c:.2f}  BLEU {b:.2f}  n={len(test)}')
    json.dump({'arm': args.arm, 'model': args.model, 'epochs': args.epochs,
               'n_train': len(train), 'n_test': len(test),
               'test_chrf++': c, 'test_bleu': b, 'history': hist,
               'samples': [{'src': test[i]['arabic'][:160], 'ref': ref_te[i][:200],
                            'hyp': hyp_te[i][:200]} for i in range(min(6, len(test)))]},
              open(out_path, 'w'), ensure_ascii=False, indent=2)
    model.save_pretrained(f'/workspace/mt_{args.arm}_model')
    tok.save_pretrained(f'/workspace/mt_{args.arm}_model')
    print(f'wrote {out_path} and /workspace/mt_{args.arm}_model')


if __name__ == '__main__':
    main()
