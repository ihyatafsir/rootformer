#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
morph_fertility.py -- does the NRMT-P representation actually escape BPE?

Measures, on the same classical Arabic text:
  * words
  * morphemic tokens  (P,R,W,S) decomposition -- the NRMT-P stream
  * BPE/subword tokens from the tokenizer the project's backbone descends from (Qwen2.5)
  * characters
and reports tokens-per-word for each.

This is the cheap, decisive half of the "root/morph prediction does not need BPE" claim: if the
morphemic stream is ~1 token/word where BPE is ~2-3, sequence length (and therefore the number of
prediction steps needed to cover the same text) drops by that factor.

Also reports, for Arabic, how many BPE tokens a single Arabic word fragments into on average -- that
fragmentation is what the morphemic scheme removes.

Usage:
  python morph_fertility.py --n-words 400000
"""
import argparse
import glob
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, '/workspace/hf_v19_2_release')
sys.path.insert(0, '/workspace/hf_v19_2_release/models')

AR = re.compile(r'[\u0600-\u06FF]')
DIAC = re.compile(r'[\u064b-\u0652\u0670\u0640]')


def corpus_text(limit_files=12, max_words=400000):
    files = (sorted(glob.glob('/workspace/scholastic_sanitized/*.txt'))
             + sorted(glob.glob('/workspace/andalusian_canon_sanitized/*.txt')))[:limit_files]
    out = []
    for f in files:
        out.append(Path(f).read_text(encoding='utf-8', errors='ignore'))
    txt = '\n'.join(out)
    words = [w for w in txt.split() if AR.search(w)]
    return words[:max_words], files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-words', type=int, default=400000)
    ap.add_argument('--bpe-model', default='Qwen/Qwen2.5-0.5B')
    ap.add_argument('--out', default='/workspace/morph_fertility.json')
    args = ap.parse_args()

    words, files = corpus_text(max_words=args.n_words)
    print(f'[*] files: {len(files)}   words (Arabic-only): {len(words)}')
    n_chars = sum(len(w) for w in words)
    print(f'[*] characters: {n_chars}   chars/word: {n_chars/len(words):.2f}\n')

    # ---- BPE (Qwen2.5 -- the lineage the backbone descends from) ----
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.bpe_model)
    bpe_total = 0
    frag = Counter()
    CH = 20000
    for i in range(0, len(words), CH):
        chunk = words[i:i + CH]
        # tokenize each word alone so we can measure per-word fragmentation
        enc = tok(chunk, add_special_tokens=False)['input_ids']
        bpe_total += sum(len(x) for x in enc)
        for x in enc:
            frag[len(x)] += 1
    bpe_tpw = bpe_total / max(len(words), 1)

    # ---- morphemic (NRMT-P): one token per word, plus optional P/W/S heads ----
    import nrmp_vocab as nv
    V = next(v for k, v in vars(nv).items() if isinstance(v, type) and 'MorphemicVocab' in k)
    vocab = V('/workspace/hf_v19_2_release/data/rootformer_v12_arabic_blueprint.json')

    n_tuple, n_root_only, unk = 0, 0, 0
    sample = words[:200000]
    for w in sample:
        enc = vocab.encode_sentence(w)
        if not enc:
            unk += 1
            continue
        n_tuple += len(enc)
        n_root_only += 1
    mor_tpw = n_tuple / max(len(sample), 1)

    res = {
        'files': len(files), 'words': len(words),
        'chars_total': n_chars, 'chars_per_word': n_chars / max(len(words), 1),
        'bpe_model': args.bpe_model,
        'bpe_tokens': bpe_total, 'bpe_tokens_per_word': bpe_tpw,
        'morphemic_sample_words': len(sample),
        'morphemic_tokens': n_tuple, 'morphemic_tokens_per_word': mor_tpw,
        'word_fragmentation': {str(k): v for k, v in sorted(frag.items())},
        'pct_words_single_bpe_token': 100.0 * frag.get(1, 0) / max(sum(frag.values()), 1),
    }
    json.dump(res, open(args.out, 'w'), indent=2)

    print(f'{"scheme":<34}{"tokens/word":>13}')
    print('-' * 47)
    print(f'{"BPE (Qwen2.5, per word)":<34}{bpe_tpw:>13.3f}')
    print(f'{"NRMT-P morphemic (P,R,W,S)":<34}{mor_tpw:>13.3f}')
    print(f'\nsequence-length ratio BPE / morphemic: {bpe_tpw/max(mor_tpw,1e-9):.2f}x')
    print(f'words that are a SINGLE BPE token: {res["pct_words_single_bpe_token"]:.1f}%')
    print('\nhow many BPE tokens one Arabic word becomes:')
    tot = sum(frag.values())
    for k in sorted(frag):
        if k <= 8:
            print(f'   {k} token(s): {100*frag[k]/tot:6.2f}%')
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
