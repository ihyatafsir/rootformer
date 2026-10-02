#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
audit_bilingual_alignment.py -- is the English side actually a translation of the Arabic side?

Method
------
Translate the Arabic with an INDEPENDENT pretrained MT model (Helsinki-NLP/opus-mt-ar-en,
never trained on this project's data) and measure chrF between that translation and the
English actually stored in the corpus.

Absolute chrF alone is hard to interpret, so we also compute the SAME quantity with the
Arabic shuffled across pairs. That gives the null distribution for this corpus/genre:
  * real pairs clearly above shuffled  -> the pairs are genuinely aligned
  * real ~= shuffled                   -> the pairs are effectively randomly matched
"""
import argparse
import json
import os
import random
import re
import statistics
import sys

import torch
from sacrebleu.metrics import CHRF

DATA = '/workspace/rootformer_v12/v18_next_root_morph/data'
DEFAULT_FILES = [
    'pure_gold_bilingual_train.jsonl',
    'pure_gold_bilingual_val.jsonl',
    'grand_scholastic_bilingual_train.jsonl',
    'grand_scholastic_bilingual_val.jsonl',
    'unified_basran_andalusian_train.jsonl',
    'unified_basran_andalusian_val.jsonl',
    'sovereign_classical_transmute_corpus.jsonl',
    'grand_neural_transmute_corpus.jsonl',
    'lisan_3pillar_master_pairs.jsonl',
]
AR = re.compile(r'[\u0600-\u06FF]')


def load_pairs(path, n, seed=0, min_ar_words=4):
    rows = []
    with open(path, encoding='utf-8', errors='ignore') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                # DELIBERATE FALLBACK, kept: the corpus is external JSONL scanned line by line;
                # one malformed record must not abort the whole audit, and the line is skipped
                # exactly like the blank line above.  (A per-file bad-record COUNT would be the
                # stronger fix; not added because no caller consumes one today.)
                continue
            a = (d.get('arabic') or d.get('ar') or '').strip()
            e = (d.get('english') or d.get('en') or '').strip()
            if not a or not e:
                continue
            if len(a.split()) < min_ar_words:
                continue
            # sentences left untranslated (Arabic quoted inside the English) are not usable
            # as clean supervision targets; flag but keep, caller decides
            rows.append((a, e, AR.search(e) is not None))
            if len(rows) >= n * 6:
                break
    random.Random(seed).shuffle(rows)
    return rows[:n]


@torch.no_grad()
def translate(model, tok, texts, device, batch_size=32, max_len=256):
    out = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tok(batch, return_tensors='pt', padding=True, truncation=True,
                  max_length=max_len).to(device)
        gen = model.generate(**enc, max_new_tokens=max_len, num_beams=1, do_sample=False)
        out.extend(tok.batch_decode(gen, skip_special_tokens=True))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=120, help='pairs sampled per corpus')
    ap.add_argument('--files', nargs='*', default=DEFAULT_FILES)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='/workspace/alignment_audit.json')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[*] device={device}', flush=True)
    from transformers import MarianMTModel, MarianTokenizer
    name = 'Helsinki-NLP/opus-mt-ar-en'
    print(f'[*] loading independent MT model {name} (never trained on this data)...', flush=True)
    tok = MarianTokenizer.from_pretrained(name)
    model = MarianMTModel.from_pretrained(name).to(device).eval()

    chrf = CHRF(word_order=2)  # chrF++
    results = {}
    print(f"\n{'corpus':<44}{'n':>5}{'real':>8}{'shuffled':>10}{'gap':>8}{'arKeep%':>9}")
    print('-' * 86)
    for fn in args.files:
        path = os.path.join(DATA, fn)
        if not os.path.exists(path):
            print(f'{fn:<44} MISSING')
            continue
        pairs = load_pairs(path, args.n)
        if len(pairs) < 5:
            print(f'{fn:<44} too few usable pairs')
            continue
        ar = [p[0] for p in pairs]
        en = [p[1] for p in pairs]
        ar_kept = sum(p[2] for p in pairs) / len(pairs)

        mt = translate(model, tok, ar, device)
        real = [chrf.sentence_score(e, [m]).score for e, m in zip(en, mt)]

        sh_ar = ar[:]
        random.Random(args.seed + 1).shuffle(sh_ar)
        mt_sh = translate(model, tok, sh_ar, device)
        shuf = [chrf.sentence_score(e, [m]).score for e, m in zip(en, mt_sh)]

        r, s = statistics.mean(real), statistics.mean(shuf)
        results[fn] = {
            'n': len(pairs), 'chrf_real': r, 'chrf_shuffled': s, 'gap': r - s,
            'arabic_kept_in_english_frac': ar_kept,
            'examples': [{'ar': a[:120], 'en': e[:160], 'mt': m[:160]}
                         for (a, e, _), m in list(zip(pairs, mt))[:3]],
        }
        print(f'{fn:<44}{len(pairs):>5}{r:>8.2f}{s:>10.2f}{r-s:>+8.2f}{100*ar_kept:>8.1f}%')

    json.dump(results, open(args.out, 'w'), ensure_ascii=False, indent=2)
    print(f'\nwrote {args.out}')
    print('\nInterpretation: gap >> 0 means the English really tracks the Arabic;'
          '\ngap ~ 0 means the pairs are effectively randomly matched.')


if __name__ == '__main__':
    main()
