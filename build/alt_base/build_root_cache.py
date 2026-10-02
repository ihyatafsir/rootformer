#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_root_cache.py -- attach the project's MORPHEMIC root annotation to an ARBITRARY base
tokenizer as an EXTERNAL side-annotation, with character-offset alignment and verification.

Why this exists
---------------
`RootHistoryCrossAttention` needs only (hidden states h, one root id per position).  It reads the
already-trained 9490x448 `morphemic_embed.root_embed` table as K/V, so the base model does NOT
need a morphemic vocabulary.  The root stream is therefore built here, OUTSIDE the base:

    sentence text
      -> morphemic side : FarahidianMorphemicVocab.encode_sentence -> one (p,r,w,s) per WORD,
                          with the WORD's character span in the sentence
      -> base side      : base_tokenizer(text, return_offsets_mapping=True) -> one id + char span
                          per BASE TOKEN
      -> align          : each base token gets the root id of the word it overlaps most

The cache is at WORD granularity, deliberately: the published numbers (19.53 %, 2.82 %,
3.551 % marginal) are per-WORD-EVENT root accuracies, so keeping word events as the unit is what
makes a new base comparable to the existing trunk.  Each word additionally carries HOW MANY base
tokens its surface occupies, and the concatenated base token ids per window are stored alongside.

Alignment policy (stated, then measured)
----------------------------------------
* word split across several base tokens  -> ALL those tokens share the word's root id; the head
  reads the hidden state of the word's LAST token.  (A word is one morphological event; giving its
  tokens one root is the only choice that keeps "one root per word".)
* base token spanning a word boundary    -> assigned to the word with which it shares the most
  non-whitespace characters; ties go to the EARLIER word.  A token that shares nothing with any
  word (whitespace/newline tokens, or tokens made entirely of punctuation) inherits the root of
  the nearest PRECEDING word (a "carry" rule), or PAD at the very start of a sentence.
* the base tokenizer's decoded text must equal the sentence exactly, else the sentence is
  rejected and counted (no silent shifts).

Outputs (into --out-dir):
    word_train.pt / word_val.pt   list [P,R,W,S,n_tok]  int32, one entry per WORD event
    tok_train.pt  / tok_val.pt    list [token_ids]      int32, flat base token stream
    tok_off_train.pt ...          list [tok_word]       int32, flat word index of each base token
    manifest_alt.json             the split rule, the alignment audit, and the md5s
"""
import argparse
import glob
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

RELEASE = Path(os.environ.get('NRMP_RELEASE', '/workspace/hf_v19_2_release'))
ALT = Path('/workspace/alt_base')
ARABIC = re.compile(r'[\u0600-\u06FF]')
SPLIT_RE = re.compile(r'[\n.!?؟]+')
WIN = 128


def md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, 'rb') as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def words_with_spans(sentence):
    """(word, char_start, char_end) reproducing `sentence.split()` EXACTLY.

    `str.split()` is not the same as `split(' ')`: it also splits on \\t and \\xa0.  We scan with
    the same rule and then ASSERT that the joined words reproduce the sentence, so a sentence
    whose whitespace cannot be reconstructed is rejected rather than silently misaligned.
    """
    out, i, n = [], 0, len(sentence)
    while i < n:
        while i < n and sentence[i].isspace():
            i += 1
        if i >= n:
            break
        j = i
        while j < n and not sentence[j].isspace():
            j += 1
        out.append((sentence[i:j], i, j))
        i = j
    return out


def split_sentences(uniq, val_files):
    """Reuse the release's own split so the held-out protocol is identical to the morphemic runs."""
    return [('val' if f in val_files else 'train', s) for f, s in uniq]


def align_sentence(sentence, enc, tok, tok_ids, word_spans):
    """Return (tok_word, audit) : per-base-token word index, plus verification counters."""
    n_words = len(word_spans)
    n_tok = len(tok_ids)
    # per-word non-whitespace character set, for the overlap vote
    wchars = []
    for (w, a, b) in word_spans:
        wchars.append({c for c in range(a, b) if not sentence[c].isspace()})
    tok_word = [-1] * n_tok
    n_none = 0
    for k, (a, b) in enumerate(tok.offsets):
        best, best_ov = -1, 0
        for wi, cs in enumerate(wchars):
            ov = sum(1 for c in range(a, b) if c in cs)
            if ov > best_ov:
                best, best_ov = wi, ov
        if best < 0:
            n_none += 1
        tok_word[k] = best
    # carry rule: a token with no overlap inherits the nearest PRECEDING word
    last = -1
    n_carry = 0
    for k in range(n_tok):
        if tok_word[k] < 0:
            tok_word[k] = last
            if last >= 0:
                n_carry += 1
        else:
            last = tok_word[k]
    # a sentence whose FIRST tokens precede any word gets PAD (word index -1) -> root PAD
    audit = {'n_tok': n_tok, 'n_words': n_words, 'n_no_overlap': n_none, 'n_carry': n_carry,
             'n_head_pad': sum(1 for x in tok_word if x < 0)}
    return tok_word, audit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default='Qwen/Qwen2.5-0.5B')
    ap.add_argument('--hf-home', default='/workspace/alt_base/cache')
    ap.add_argument('--out-dir', default='/workspace/alt_base/cache/rootqwen')
    ap.add_argument('--per-corpus', type=int, default=120000)
    ap.add_argument('--seg-mode', default='0', choices=['0', '1', '2'])
    ap.add_argument('--min-words', type=int, default=4)
    ap.add_argument('--max-words', type=int, default=64)
    ap.add_argument('--verify-sentences', type=int, default=0,
                    help='>0: run the alignment on the first N sentences, print/inspect, write nothing')
    ap.add_argument('--verify-examples', type=int, default=8)
    ap.add_argument('--limit-train', type=int, default=0)
    args = ap.parse_args()

    os.environ['HF_HOME'] = args.hf_home
    os.environ['HF_HUB_CACHE'] = str(Path(args.hf_home) / 'hub')
    os.environ['ROOTFORMER_VALIDATED_SEG'] = args.seg_mode
    os.environ.setdefault('HF_TOKEN', Path('/workspace/.hf_token').read_text().strip())

    sys.path.insert(0, str(RELEASE))
    sys.path.insert(0, str(RELEASE / 'models'))
    import numpy as np
    import torch
    import nrmp_train as NP
    import nrmp_vocab as nv
    from transformers import AutoTokenizer

    V = next(v for k, v in vars(nv).items() if isinstance(v, type) and 'MorphemicVocab' in k)
    bp = str(RELEASE / 'data/rootformer_v12_arabic_blueprint.json')
    vocab = V(bp)
    print(f'[*] morphemic vocab : num_roots={vocab.num_roots} prefixes={vocab.num_prefixes} '
          f'awzan={vocab.num_awzan} suffixes={vocab.num_suffixes}', flush=True)
    print(f'[*] PAD/BOS/EOS/UNK = {vocab.PAD_ROOT}/{vocab.BOS_ROOT}/{vocab.EOS_ROOT}/'
          f'{vocab.UNK_ROOT}', flush=True)

    tok = AutoTokenizer.from_pretrained(args.base, cache_dir=str(Path(args.hf_home) / 'hub'))
    print(f'[*] base tokenizer : {args.base} vocab={tok.vocab_size} '
          f'len={len(tok)} model_max={tok.model_max_length}', flush=True)

    # ---- the split, from the release's own code path -------------------------------------
    t0 = time.time()
    all_pairs = []
    for c in NP.CORPORA:
        if not Path(c).exists():
            print(f'  [skip missing] {c}')
            continue
        got = NP.sentences_from(c, min_words=args.min_words, max_words=args.max_words,
                                limit=args.per_corpus)
        print(f'  {c}: {len(got)} sentences')
        all_pairs += got
    seen, uniq = set(), []
    for f, s in all_pairs:
        if s in seen:
            continue
        seen.add(s)
        uniq.append((f, s))
    files = sorted({f for f, _ in uniq})
    random.Random(NP.SEED).shuffle(files)
    n_val = max(1, int(len(files) * 0.08))
    val_files = set(files[:n_val])
    print(f'[*] unique sentences={len(uniq)} files={len(files)} val_files={len(val_files)} '
          f'[{time.time()-t0:.0f}s]', flush=True)

    # ---- encode + align ------------------------------------------------------------------
    W = {'train': [[], [], [], [], []], 'val': [[], [], [], [], []]}   # P,R,W,S,n_tok
    streams = {'train': {'wtok': [], 'wstart': [], 'tok': [], 'soff': [], 'wtokoff': []},
               'val': {'wtok': [], 'wstart': [], 'tok': [], 'soff': [], 'wtokoff': []}}
    aud = Counter()
    aud_examples = []
    n_reject_ws = n_reject_dec = n_skip_short = 0
    t0 = time.time()
    n_words_tot = {'train': 0, 'val': 0}
    n_tok_tot = {'train': 0, 'val': 0}
    wlen_hist = Counter()
    for i, (f, s) in enumerate(uniq):
        if args.verify_sentences and i >= args.verify_sentences:
            break
        split = 'val' if f in val_files else 'train'
        if ''.join(w for w, _, _ in words_with_spans(s)) != s.replace(' ', ''):
            pass  # checked properly below against the split() word list
        words = s.split()
        spans = words_with_spans(s)
        if [w for w, _, _ in spans] != words:
            n_reject_ws += 1
            continue
        enc = vocab.encode_sentence(s)
        if len(enc) < 2:
            n_skip_short += 1
            continue
        if len(enc) != len(words):
            n_reject_dec += 1
            continue
        e = tok(s, return_offsets_mapping=True, add_special_tokens=False)
        tok_ids = e['input_ids']
        tok.offsets = e['offset_mapping']
        if tok.decode(tok_ids, skip_special_tokens=False) != s:
            n_reject_dec += 1
            continue
        tw, a = align_sentence(s, enc, tok, tok_ids, spans)
        for k, v in a.items():
            aud[k] += v
        aud['sentences'] += 1
        aud['words'] += len(words)
        aud['multi_tok_words'] += sum(1 for c in Counter(tw) if c >= 0 and False)
        # words split across base tokens
        cnt = Counter(x for x in tw if x >= 0)
        aud['words_ge2_tokens'] += sum(1 for wi in range(len(words)) if cnt.get(wi, 0) >= 2)
        aud['words_0_tokens'] += sum(1 for wi in range(len(words)) if cnt.get(wi, 0) == 0)
        wlen_hist.update(cnt.get(wi, 0) for wi in range(len(words)))
        if len(aud_examples) < args.verify_examples:
            aud_examples.append({
                'sentence': s,
                'words': [{'w': w, 'root': vocab.id2root[enc[j][1]], 'span': [sp[1], sp[2]],
                           'n_tok': cnt.get(j, 0)} for j, (w, sp) in enumerate(zip(words, spans))][:40],
                'n_tok': len(tok_ids),
                'roundtrip_ok': True,
            })
        st = streams[split]
        st['wstart'].append(len(W[split][0]))
        st['soff'].append(len(st['tok']))
        st['tok'].extend(tok_ids)
        for j, (p, r, w_, sf) in enumerate(enc):
            # index of this word's FIRST base token within the sentence's token list
            st['wtokoff'].append(min(k for k in range(len(tok_ids)) if tw[k] == j)
                                 if any(tw[k] == j for k in range(len(tok_ids))) else -1)
            # keep only this word's own base tokens.  A leading token that belongs to no word
            # (word index -1) is DROPPED rather than attached to word 0, so toffs<n>_<split>[i]
            # is exactly the base-token sequence of word i and the CSR stays clean.
            tids = [tok_ids[k] for k in range(len(tok_ids)) if tw[k] == j]
            if not tids:
                tids = [tok.unk_token_id if tok.unk_token_id is not None else 0]
                aud['words_synthesised_1_unk_token'] += 1
            W[split][0].append(p); W[split][1].append(r)
            W[split][2].append(w_); W[split][3].append(sf)
            W[split][4].append(len(tids))
            st['wtok'].append(tids)
        st['nsent'] = st.get('nsent', 0) + 1
        n_words_tot[split] += len(words)
        n_tok_tot[split] += sum(W[split][4][-len(words):])
        if i and i % 50000 == 0:
            print(f'    {i}/{len(uniq)}  words train={n_words_tot["train"]} val={n_words_tot["val"]} '
                  f'tokens={n_tok_tot["train"]+n_tok_tot["val"]} [{time.time()-t0:.0f}s]', flush=True)

    print(f'[*] rejected: whitespace-reconstruction={n_reject_ws} decode/len-mismatch='
          f'{n_reject_dec} short={n_skip_short}', flush=True)
    tot_w = sum(n_words_tot.values())
    tot_t = sum(n_tok_tot.values())
    audit = {
        'base': args.base, 'base_vocab_size': int(tok.vocab_size),
        'sentences_encoded': aud['sentences'], 'words': aud['words'],
        'base_tokens': aud['n_tok'],
        'rejected_whitespace': n_reject_ws, 'rejected_decode_or_len': n_reject_dec,
        'skipped_short': n_skip_short,
        'words_ge2_base_tokens': aud['words_ge2_tokens'],
        'words_0_base_tokens': aud['words_0_tokens'],
        'tokens_with_no_word_overlap': aud['n_no_overlap'],
        'tokens_assigned_by_carry_rule': aud['n_carry'],
        'tokens_leading_before_first_word': aud['n_head_pad'],
        'fertilite_tokens_per_word': round(tot_t / max(tot_w, 1), 4),
        'words_per_base_token': round(tot_w / max(tot_t, 1), 4),
        'word_len_hist': dict(sorted(wlen_hist.items())[:12]),
        'reconstruction_rule': 'base tokenizer decoded text must equal the sentence exactly; '
                               'morphemic word list must equal sentence.split() exactly',
    }
    print(json.dumps(audit, indent=2, ensure_ascii=False), flush=True)
    for ex in aud_examples[:3]:
        print('--- example:', ex['sentence'][:100], flush=True)
        for w in ex['words'][:6]:
            print('     ', w, flush=True)

    if args.verify_sentences:
        out = ALT / 'align_verify.json'
        out.write_text(json.dumps({'audit': audit, 'examples': aud_examples,
                                   'val_files_n': len(val_files), 'n_files': len(files)},
                                  ensure_ascii=False, indent=2))
        print(f'[*] VERIFY ONLY -> {out}', flush=True)
        return

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        'base': args.base, 'base_repo': args.base,
        'writer': str(Path(NP.__file__)), 'writer_md5': md5(Path(NP.__file__)),
        'corpora': NP.CORPORA, 'seed': NP.SEED, 'val_fraction': 0.08,
        'n_files': len(files), 'n_val_files': n_val, 'val_files': sorted(val_files),
        'sentence_filter': {'min_words': args.min_words, 'max_words': args.max_words,
                            'per_corpus_limit': args.per_corpus,
                            'split_regex': r'[\n.!?؟]+'},
        'dedup': 'identical sentence strings, first occurrence wins',
        'level': 'FILE (no file straddles the split)',
        'seg_mode': args.seg_mode,
        'alignment_audit': audit,
        'counts': {}, 'outputs': {}, 'md5': {},
    }
    for split in ('train', 'val'):
        sd = streams[split]
        nw = len(W[split][0])
        if split == 'train' and args.limit_train:
            nw = min(nw, args.limit_train)
        tok = torch.tensor(sd['tok'], dtype=torch.int32)
        soff = torch.tensor(sd['soff'], dtype=torch.int32)
        wtokoff = torch.tensor(sd['wtokoff'], dtype=torch.int32)
        wstart = torch.tensor(sd['wstart'], dtype=torch.int32)
        wlen = torch.tensor(W[split][4], dtype=torch.int32)
        word_t = [torch.tensor(W[split][k][:nw], dtype=torch.int32) for k in range(5)]
        objs = {'word': word_t, 'tok': tok, 'soff': soff, 'wtokoff': wtokoff,
                'wstart': wstart, 'wlen': wlen}
        for name, obj in objs.items():
            p = out / f'{name}_{split}.pt'
            torch.save(obj, p)
            manifest['md5'][f'{name}_{split}'] = md5(p)
            manifest['outputs'][f'{name}_{split}'] = str(p)
        R = word_t[1]
        ids = torch.unique(R)
        # 128-word block token lengths (block t covers words 128t..128t+127)
        wlb = wlen[:nw].numpy().astype('int64')
        nb = len(wlb) // WIN
        blk = wlb[:nb * WIN].reshape(nb, WIN).sum(axis=1)
        manifest['counts'][split] = {
            'word_events': int(R.numel()), 'base_tokens': int(tok.numel()),
            'sentence_count': int(len(soff)),
            'root_max': int(R.max()), 'root_min': int(R.min()),
            'distinct_root_ids': int(ids.numel()),
            'tokens_per_word': round(float(tok.numel()) / max(int(R.numel()), 1), 4),
            'n_blocks_128': int(nb),
            'block_tokens_p50': int(np.percentile(blk, 50)) if nb else 0,
            'block_tokens_p99': int(np.percentile(blk, 99)) if nb else 0,
            'block_tokens_p999': int(np.percentile(blk, 99.9)) if nb else 0,
            'block_tokens_max': int(blk.max()) if nb else 0,
        }
        print(f'[*] {split}: {R.numel()} word events, {tok.numel()} base tokens, '
              f'{nb} blocks of {WIN}; block tokens p50={manifest["counts"][split]["block_tokens_p50"]} '
              f'p99={manifest["counts"][split]["block_tokens_p99"]} '
              f'max={manifest["counts"][split]["block_tokens_max"]} -> {out}', flush=True)

    (out / 'manifest_alt.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f'[*] manifest -> {out/"manifest_alt.json"}', flush=True)
    print(f'[*] total {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
