#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nrmp_run.py -- Honest Next-Root/Morph Prediction (NRMP) runner for Rootformer v19.2.

What this does (and why it exists):
  The shipped `transmute_quickstart.py` never calls the NRMP heads. It runs ONE encoder
  pass, throws the hidden states away, and returns a hardcoded Arabic->English dictionary
  lookup. See `khalil_students_andalusian_master_engine.py` (SURFACE_SCHOLASTIC_LEXICON).

  This script instead exercises the *actual* trained NRMP capability:
    * loads the v19.2 checkpoint and REPORTS load fidelity (no silent strict=False),
    * generates next words by predicting (prefix, ROOT, wazn, suffix) autoregressively,
    * measures next-root prediction accuracy@1/@5 and root perplexity on real text,
    * compares against uniform and oracle-unigram baselines,
    * runs a shuffled-context control to prove the model uses context at all.

Usage:
  python nrmp_run.py --generate
  python nrmp_run.py --eval --corpus /workspace/andalusian_canon_sanitized --max-sentences 300
"""
import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

from collections import Counter

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'models'))

CKPT = ROOT / 'checkpoints/rootformer_v19_2_synthesis_ar_backbone.safetensors'
BLUEPRINT = ROOT / 'data/rootformer_v12_arabic_blueprint.json'


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------
def load_engine(device=None, dtype=torch.bfloat16, ckpt=None):
    from safetensors.torch import load_file
    import nrmp_vocab as _nv
    from models.unified_rootformer_v12 import UnifiedRootformerV12
    from deepseek_v4_1_flash_model import UnifiedRootformerV17_DeepSeekFlash
    from rootformer_v18_nrmp_model import RootformerV18_NRMP

    # the class name carries Arabic-script diacritics (FarāhīdianMorphemicVocab)
    vocab_cls = next(v for k, v in vars(_nv).items()
                     if isinstance(v, type) and 'MorphemicVocab' in k)

    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[*] device={device} dtype={dtype}', flush=True)

    vocab = vocab_cls(str(BLUEPRINT))
    print(f'[*] vocab: roots={vocab.num_roots} awzan={vocab.num_awzan} '
          f'prefixes={vocab.num_prefixes} suffixes={vocab.num_suffixes}', flush=True)

    base = UnifiedRootformerV12(str(BLUEPRINT), 'Qwen/Qwen2.5-0.5B', str(device), dtype).to(device)
    flash = UnifiedRootformerV17_DeepSeekFlash(base, device, dtype).to(device)
    model = RootformerV18_NRMP(flash, vocab, device, dtype).to(device)
    model.eval()

    sd = load_file(str(ckpt or CKPT))
    model_sd = model.state_dict()

    # --- explicit, non-silent load fidelity report -------------------------------------
    missing, unexpected = model.load_state_dict(sd, strict=False)

    nrmp_sd = {k: v for k, v in sd.items()
               if k.startswith(('nrmp_head.', 'morphemic_embed.'))}
    loaded, mismatched, absent = [], [], []
    for k, v in nrmp_sd.items():
        if k not in model_sd:
            absent.append(k)
        elif tuple(model_sd[k].shape) != tuple(v.shape):
            mismatched.append((k, tuple(v.shape), tuple(model_sd[k].shape)))
        else:
            loaded.append(k)

    report = {
        'checkpoint_tensors': len(sd),
        'model_tensors': len(model_sd),
        'nrmp_tensors_in_ckpt': len(nrmp_sd),
        'nrmp_tensors_loaded': len(loaded),
        'nrmp_shape_mismatch': mismatched,
        'nrmp_absent_from_model': absent,
        'missing_keys_total': len(missing),
        'unexpected_keys_total': len(unexpected),
    }
    print('=' * 78)
    print('NRMP LOAD FIDELITY')
    print('=' * 78)
    print(f"  checkpoint tensors              : {report['checkpoint_tensors']}")
    print(f"  NRMP tensors found in checkpoint: {report['nrmp_tensors_in_ckpt']}")
    print(f"  ... loaded with matching shapes : {report['nrmp_tensors_loaded']}")
    print(f"  ... shape mismatches            : {len(mismatched)}")
    print(f"  ... absent from model           : {len(absent)}")
    print(f"  total missing keys              : {len(missing)}")
    print(f"  total unexpected keys           : {len(unexpected)}")
    if mismatched:
        for k, a, b in mismatched[:10]:
            print(f"    MISMATCH {k}: ckpt{a} vs model{b}")
    if absent:
        for k in absent[:10]:
            print(f"    ABSENT   {k}")
    # non-NRMP missing keys, summarised (these are expected: no separate head files shipped)
    other_missing = [k for k in missing if not k.startswith(('nrmp_head.', 'morphemic_embed.'))]
    print(f"  non-NRMP missing keys           : {len(other_missing)}")
    for k in other_missing[:8]:
        print(f"    - {k}")
    print('=' * 78, flush=True)

    return model, vocab, device, report


# --------------------------------------------------------------------------------------
# Autoregressive NRMP generation
# --------------------------------------------------------------------------------------
@torch.no_grad()
def generate(model, prompt, max_new_words=12, min_new_words=4):
    t0 = time.perf_counter()
    out = model.generate_words(prompt, max_new_words=max_new_words, min_new_words=min_new_words)
    dt = (time.perf_counter() - t0) * 1000
    n = max(out['num_words'], 1)
    out['latency_ms'] = dt
    out['ms_per_word'] = dt / n
    # readable tuple decomposition
    decomp = []
    for (p, r, w, s) in out['generated_tuples']:
        decomp.append({
            'prefix': model.vocab.id2prefix.get(p, '?'),
            'root': model.vocab.id2root.get(r, '?'),
            'wazn': model.vocab.id2wazn.get(w, '?'),
            'suffix': model.vocab.id2suffix.get(s, '?'),
        })
    out['decomposition'] = decomp
    return out


# --------------------------------------------------------------------------------------
# Evaluation of next-root prediction
# --------------------------------------------------------------------------------------
def _tensors(vocab, sentences, device):
    seqs = [vocab.encode_sentence(s) for s in sentences]
    seqs = [s for s in seqs if len(s) >= 2]
    if not seqs:
        raise SystemExit('no usable sentences')
    p = torch.tensor([[t[0] for t in s] for s in seqs], dtype=torch.long, device=device)
    r = torch.tensor([[t[1] for t in s] for s in seqs], dtype=torch.long, device=device)
    w = torch.tensor([[t[2] for t in s] for s in seqs], dtype=torch.long, device=device)
    s_ = torch.tensor([[t[3] for t in s] for s in seqs], dtype=torch.long, device=device)
    return p, r, w, s_


@torch.no_grad()
def evaluate(model, vocab, sentences, device, shuffle_context=False):
    """Per-sentence evaluation (batch=1) so variable-length sentences need no padding."""
    n_pos = 0
    n_top1 = 0
    n_top5 = 0
    ce_sum = 0.0
    root_counts = {}
    r_pos = r_top1 = r_top5 = 0
    r_ce = 0.0
    pairs = []  # (prev_root, target_root) for count-based reference baselines
    specials_set = {vocab.PAD_ROOT, vocab.BOS_ROOT, vocab.EOS_ROOT,
                    vocab.UNK_ROOT, vocab.root2id.get('<PARTICLE>', -1)}
    # particle roots (<P:...>) are function words, not radicals: exclude them too so the
    # "radical only" metric measures real triconsonantal root prediction.
    specials_set |= {i for i, r in enumerate(vocab.roots_list) if r.startswith('<P:')}
    specials_set = {i for i in specials_set if i >= 0}

    for sent in sentences:
        s = vocab.encode_sentence(sent)
        if len(s) < 2:
            continue
        p = torch.tensor([[t[0] for t in s]], dtype=torch.long, device=device)
        r = torch.tensor([[t[1] for t in s]], dtype=torch.long, device=device)
        w = torch.tensor([[t[2] for t in s]], dtype=torch.long, device=device)
        sf = torch.tensor([[t[3] for t in s]], dtype=torch.long, device=device)

        if shuffle_context:
            perm = torch.randperm(r.shape[1], device=device)
            p, r, w, sf = p[:, perm], r[:, perm], w[:, perm], sf[:, perm]

        out = model(p, r, w, sf, target_roots=r, target_awzan=w,
                    target_prefixes=p, target_suffixes=sf)
        logits = out['root_logits'][0, :-1, :]
        targets = r[0, 1:]
        mask = targets != vocab.PAD_ROOT
        logits, targets = logits[mask], targets[mask]
        if targets.numel() == 0:
            continue

        n_pos += int(targets.numel())
        n_top1 += int((logits.argmax(-1) == targets).sum().item())
        k = min(5, logits.shape[-1])
        n_top5 += int((logits.topk(k, dim=-1).indices == targets.unsqueeze(-1)).any(-1).sum().item())
        ce_sum += float(torch.nn.functional.cross_entropy(logits, targets, reduction='sum').item())
        for v in targets.tolist():
            root_counts[v] = root_counts.get(v, 0) + 1

        prevs = r[0, :-1]
        pairs.extend(zip(prevs.tolist(), targets.tolist()))

        # radical-only subset: the catch-all classes are trivially predictable and
        # ~30% of tokens, so they inflate every aggregate metric. Report both.
        specials = torch.tensor(sorted(specials_set), device=targets.device)
        keep = ~torch.isin(targets, specials)
        if keep.any():
            lg, tg = logits[keep], targets[keep]
            r_pos += int(tg.numel())
            r_top1 += int((lg.argmax(-1) == tg).sum().item())
            kk = min(5, lg.shape[-1])
            r_top5 += int((lg.topk(kk, dim=-1).indices == tg.unsqueeze(-1)).any(-1).sum().item())
            r_ce += float(torch.nn.functional.cross_entropy(lg, tg, reduction='sum').item())

    if n_pos == 0:
        raise SystemExit('no supervised positions')

    ce = ce_sum / n_pos
    unigram_counts = {k: v for k, v in root_counts.items() if k not in specials_set}
    unigram = max(unigram_counts.values()) / sum(unigram_counts.values()) if unigram_counts else 0.0
    # ---- count-based reference baselines over the SAME radical positions ----
    rad_pairs = [(pv, tg) for pv, tg in pairs if tg not in specials_set]
    ngram = {}
    if rad_pairs:
        uni = Counter(tg for _, tg in rad_pairs)
        top_uni = uni.most_common(1)[0][0]
        bi = Counter(rad_pairs)
        top_bi = {}
        for (pv, tg), c in bi.items():
            if pv not in top_bi or c > top_bi[pv][1]:
                top_bi[pv] = (tg, c)
        uni_hits = sum(1 for _, tg in rad_pairs if tg == top_uni)
        bi_hits = 0
        for pv, tg in rad_pairs:
            pred = top_bi.get(pv, (top_uni, 0))[0]
            bi_hits += int(pred == tg)
        ngram = {
            'radical_baseline_unigram_top1': uni_hits / len(rad_pairs),
            'radical_baseline_bigram_top1': bi_hits / len(rad_pairs),
            'unigram_most_common_root': vocab.id2root.get(top_uni, '?'),
        }

    return {
        'positions': n_pos,
        'radical_positions': r_pos,
        'radical_acc_top1': r_top1 / max(r_pos, 1),
        'radical_acc_top5': r_top5 / max(r_pos, 1),
        'radical_ce': r_ce / max(r_pos, 1),
        'radical_ppl': float(torch.exp(torch.tensor(min(r_ce / max(r_pos, 1), 20.0)))),
        'root_acc_top1': n_top1 / n_pos,
        'root_acc_top5': n_top5 / n_pos,
        'root_ce': ce,
        'root_ppl': float(torch.exp(torch.tensor(min(ce, 20.0)))),
        'baseline_uniform_top1': 1.0 / vocab.num_roots,
        'baseline_oracle_unigram_top1': unigram,
        'distinct_target_roots': len(root_counts),
        'shuffled_context': shuffle_context,
        **ngram,
    }


def load_sentences(corpus, max_sentences, min_words=5, max_words=60):
    path = Path(corpus)
    files = sorted(path.glob('*.txt')) if path.is_dir() else [path]
    sents = []
    for f in files:
        try:
            txt = f.read_text(encoding='utf-8', errors='ignore')
        except Exception as exc:
            # VISIBILITY: an unreadable corpus file silently reduced the evaluation set (the
            # reported metrics were computed on whatever was left).  The skip is unchanged.
            print(f'[nrmp_run] skipping unreadable corpus file {str(f)!r}: {exc!r}',
                  file=sys.stderr)
            continue
        for line in re.split(r'[\n]+', txt):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            for s in re.split(r'(?<=[.!?؟])\s+', line):
                s = s.strip()
                if not s:
                    continue
                nw = len(s.split())
                if min_words <= nw <= max_words and re.search(r'[\u0600-\u06FF]', s):
                    sents.append(s)
                if len(sents) >= max_sentences:
                    return sents
    return sents


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--generate', action='store_true')
    ap.add_argument('--eval', action='store_true')
    ap.add_argument('--corpus', default='/workspace/andalusian_canon_sanitized')
    ap.add_argument('--corpus2', default='/workspace/scholastic_sanitized')
    ap.add_argument('--max-sentences', type=int, default=250)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--checkpoint', default=None)
    ap.add_argument('--json-out', default=None)
    args = ap.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    model, vocab, device, load_report = load_engine(ckpt=args.checkpoint)
    results = {'load_fidelity': load_report}

    if args.generate or not (args.generate or args.eval):
        prompts = [
            'العلم نور يضيء العقل ويهدي إلى الحق',
            'رأس الحكمة مخافة الله',
            'اليقين لا يزول بالشك',
            'إنما الأعمال بالنيات وإنما لكل امرئ ما نوى',
            'الواجب الوجود بذاته لا علة له وهو مبدأ كل وجود',
            'النظم ليس شيئا غير توخي معاني النحو فيما بين الكلم',
        ]
        gens = []
        print('\n' + '=' * 78)
        print('AUTOREGRESSIVE NRMP GENERATION (trained root/wazn/prefix/suffix heads)')
        print('=' * 78)
        for pr in prompts:
            try:
                g = generate(model, pr)
            except Exception as e:
                print(f'\n  PROMPT : {pr}\n  ERROR  : {type(e).__name__}: {e}', flush=True)
                gens.append({'prompt': pr, 'error': f'{type(e).__name__}: {e}'})
                continue
            print(f'\n  PROMPT : {pr}')
            print(f"  NEXT   : {g['generated_text']}")
            print(f"  WORDS  : {g['num_words']}  latency {g['latency_ms']:.1f} ms "
                  f"({g['ms_per_word']:.1f} ms/word)")
            for d in g['decomposition']:
                print(f"      {d['prefix']:<5}{d['root']:<14} wazn={d['wazn']:<12} {d['suffix']}")
            gens.append({'prompt': pr, 'text': g['generated_text'],
                         'tuples': g['decomposition'], 'num_words': g['num_words'],
                         'latency_ms': g['latency_ms'], 'ms_per_word': g['ms_per_word']})
        results['generation'] = gens

    if args.eval:
        evals = {}
        for tag, corpus in (('corpus1', args.corpus), ('corpus2', args.corpus2)):
            sents = load_sentences(corpus, args.max_sentences)
            print(f'\n[{tag}] {corpus}: {len(sents)} sentences', flush=True)
            if len(sents) < 5:
                continue
            real = evaluate(model, vocab, sents, device, shuffle_context=False)
            shuf = evaluate(model, vocab, sents, device, shuffle_context=True)
            evals[tag] = {
                'corpus': corpus,
                'n_sentences': len(sents),
                'examples': sents[:3],
                'real_context': real,
                'shuffled_context_control': shuf,
            }
            print(f"  ALL classes     : acc@1 {real['root_acc_top1']*100:.2f}%  "
                  f"acc@5 {real['root_acc_top5']*100:.2f}%  PPL {real['root_ppl']:.1f}  "
                  f"(positions={real['positions']}, includes <PARTICLE>/<UNK> catch-alls)")
            print(f"  RADICAL only    : acc@1 {real['radical_acc_top1']*100:.2f}%  "
                  f"acc@5 {real['radical_acc_top5']*100:.2f}%  PPL {real['radical_ppl']:.1f}  "
                  f"(positions={real['radical_positions']})")
            print(f"  RADICAL shuffled: acc@1 {shuf['radical_acc_top1']*100:.2f}%  "
                  f"acc@5 {shuf['radical_acc_top5']*100:.2f}%  PPL {shuf['radical_ppl']:.1f}"
                  f"   <-- context destroyed")
            print(f"  context gain    : {(real['radical_acc_top1']-shuf['radical_acc_top1'])*100:+.2f} pp")
            if 'radical_baseline_bigram_top1' in real:
                print(f"  count-based ref : unigram {real['radical_baseline_unigram_top1']*100:.2f}%  "
                      f"bigram {real['radical_baseline_bigram_top1']*100:.2f}%  "
                      f"(most common root '{real['unigram_most_common_root']}')")
            print(f"  baselines       : uniform {real['baseline_uniform_top1']*100:.4f}%   "
                  f"oracle-unigram(radical) {real['baseline_oracle_unigram_top1']*100:.2f}%")
        results['evaluation'] = evals

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results, ensure_ascii=False, indent=2))
        print(f'\nwrote {args.json_out}')


if __name__ == '__main__':
    main()
