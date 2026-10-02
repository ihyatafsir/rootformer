#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mt_baselines.py -- Arabic->English baselines on the leak-free held-out works.

Evaluated on /workspace/mt_split/test_clean.jsonl: sentences drawn from 31 works that contribute
NO training pairs, and whose 13-grams never occur in the training split.

  A. lookup    -- the shipped v19.2 rule path (CLASSICAL_COMPOUND_IDIOMS + SURFACE_SCHOLASTIC_LEXICON
                  + COMPREHENSIVE_SCHOLASTIC_MAP), word by word. This is what the project's own
                  "100% exact" benchmark was actually scoring.
  B. opus-mt   -- Helsinki-NLP/opus-mt-ar-en: a pretrained SUBWORD NMT baseline, never trained here.
  C. identity  -- emits the Arabic unchanged (floor sanity check).
  D. llm_plain / llm_concepts -- an LLM translator without and with Farahidian root-concept
                  conditioning (the v19.2.1 mechanism), optionally on a subsample.

Metrics: chrF++ / BLEU (sacrebleu); COMET added when the `comet` package is importable.

Usage:
  python mt_baselines.py --n 1500
  python mt_baselines.py --n 300 --llm Qwen/Qwen2.5-3B-Instruct
"""
import argparse
import ast
import json
import pathlib
import re
import sys
import time

SPLIT = pathlib.Path('/workspace/mt_split')
ENGINE_SRC = pathlib.Path('/workspace/hf_v19_2_release/khalil_students_andalusian_master_engine.py')
WANTED = {'CLASSICAL_COMPOUND_IDIOMS', 'SURFACE_SCHOLASTIC_LEXICON', 'COMPREHENSIVE_SCHOLASTIC_MAP'}
AR = re.compile(r'[\u0600-\u06FF]')


def load_rows(n=None, path=None):
    p = pathlib.Path(path) if path else (SPLIT / 'test_clean.jsonl')
    rows = [json.loads(l) for l in open(p, encoding='utf-8') if l.strip()]
    return rows[:n] if n else rows


def extract_dicts():
    """Pull the shipped lookup tables straight out of the source with ast (no model load)."""
    if not ENGINE_SRC.exists():
        return {}, {}, {}
    tree = ast.parse(ENGINE_SRC.read_text(encoding='utf-8', errors='ignore'))
    found = {}
    for node in tree.body:
        tgt = None
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            tgt = node.target.id
        elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            tgt = node.targets[0].id
        if tgt in WANTED and node.value is not None:
            try:
                found[tgt] = ast.literal_eval(node.value)
            except Exception:
                # DELIBERATE FALLBACK, kept: this only SNIFFS optional lookup tables out of the
                # engine's source.  A table that is not a static literal is simply not extracted;
                # the caller falls back to {} via .get() below, and the baseline still runs.
                pass
    return (found.get('CLASSICAL_COMPOUND_IDIOMS', {}),
            found.get('SURFACE_SCHOLASTIC_LEXICON', {}),
            found.get('COMPREHENSIVE_SCHOLASTIC_MAP', {}))


def score(hyps, refs, tag, srcs=None):
    from sacrebleu.metrics import BLEU, CHRF
    out = {'n': len(hyps),
           'chrf++': CHRF(word_order=2).corpus_score(hyps, [refs]).score,
           'bleu': BLEU(tokenize='13a').corpus_score(hyps, [refs]).score}
    out['comet'] = None
    try:
        from comet import download_model, load_from_checkpoint
        ck = download_model('Unbabel/wmt22-comet-da')
        cm = load_from_checkpoint(ck)
        data = [{'src': (srcs[i] if srcs else ''), 'mt': h, 'ref': r}
                for i, (h, r) in enumerate(zip(hyps, refs))]
        out['comet'] = float(cm.predict(data, batch_size=16, gpus=1).system_score)
    except Exception as e:
        out['comet_note'] = f'{type(e).__name__}: {str(e)[:90]}'
    c = 'none' if out['comet'] is None else f'{out["comet"]:.4f}'
    print(f'  {tag:<20} chrF++ {out["chrf++"]:6.2f}   BLEU {out["bleu"]:6.2f}   COMET {c}   (n={out["n"]})',
          flush=True)
    return out


# --------------------------------------------------------------------------- A. shipped lookup
def lookup_translate(text, idioms, lexicon, comp_map, vocab=None):
    clean = text.strip()
    for pat, rep in idioms.items():
        if clean.rstrip('.،؛') == pat.rstrip('.،؛'):
            return rep
    out = []
    for w in clean.split():
        cw = re.sub(r'[^\u0600-\u06FF]', '', w)
        if not cw:
            continue
        if cw in lexicon:
            out.append(str(lexicon[cw]))
            continue
        stem = cw[2:] if cw.startswith('ال') else cw
        if stem in lexicon:
            out.append(str(lexicon[stem]))
            continue
        if vocab is not None:
            p, r, wz, s = vocab.base_tok.decompose_arabic_word(cw)
            if r and not r.startswith('<') and (r, wz or '') in comp_map:
                out.append(str(comp_map[(r, wz or '')]))
                continue
            if r and not r.startswith('<'):
                out.append(r)
                continue
        out.append('')
    return ' '.join(x for x in out if x)


# --------------------------------------------------------------------------- B. opus-mt
def opus_translate(texts, model_name, batch_size=48, max_len=256):
    import torch
    from transformers import MarianMTModel, MarianTokenizer
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tok = MarianTokenizer.from_pretrained(model_name)
    mdl = MarianMTModel.from_pretrained(model_name).to(dev).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            b = texts[i:i + batch_size]
            enc = tok(b, return_tensors='pt', padding=True, truncation=True,
                      max_length=max_len).to(dev)
            gen = mdl.generate(**enc, max_new_tokens=max_len, num_beams=1, do_sample=False)
            out.extend(tok.batch_decode(gen, skip_special_tokens=True))
    return out


# --------------------------------------------------------------------------- D. LLM
FEWSHOT = [("العِلْمُ نُورٌ يُضِيءُ العَقْلَ", "knowledge (علم), light (نور), illuminate (ضوء), intellect (عقل)",
            "Knowledge is a light that illuminates the intellect."),
           ("العَدْلُ أَسَاسُ المُلْكِ وَمِيزَانُ الحَقِّ", "justice (عدل), foundation (أسس), sovereignty (ملك), balance (وزن), truth (حق)",
            "Justice is the foundation of sovereignty and the scale of truth.")]


def llm_translate(texts, concepts, model_name, use_concepts, batch_size=16, max_new=110):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.padding_side = 'left'
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16).to(dev).eval()
    sysp = ('You are a master classical Arabic-to-English translator and grammarian. Output ONLY '
            'the final, fluent scholastic English translation with zero explanation.')
    prompts = []
    for t, c in zip(texts, concepts):
        blocks = [f'Arabic: {a}\nRoot Concepts: {cs}\nTranslation: {e}' if use_concepts
                  else f'Arabic: {a}\nTranslation: {e}' for a, cs, e in FEWSHOT]
        tail = (f'Arabic: {t}\nRoot Concepts: {c}\nTranslation:' if use_concepts
                else f'Arabic: {t}\nTranslation:')
        prompts.append(tok.apply_chat_template(
            [{'role': 'system', 'content': sysp},
             {'role': 'user', 'content': '\n\n'.join(blocks + [tail])}],
            tokenize=False, add_generation_prompt=True))
    out = []
    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            enc = tok(prompts[i:i + batch_size], return_tensors='pt', padding=True,
                      truncation=True, max_length=1024).to(dev)
            gen = mdl.generate(**enc, max_new_tokens=max_new, do_sample=False,
                               pad_token_id=tok.pad_token_id)
            for j in range(enc.input_ids.shape[0]):
                s = tok.decode(gen[j][enc.input_ids.shape[1]:], skip_special_tokens=True)
                out.append(s.strip().split('\n')[0].strip().strip('"\''))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=1500)
    ap.add_argument('--opus', default='Helsinki-NLP/opus-mt-ar-en')
    ap.add_argument('--llm', default=None, help='run the LLM arms with this model')
    ap.add_argument('--llm-n', type=int, default=300)
    ap.add_argument('--out', default='/workspace/mt_baselines.json')
    args = ap.parse_args()

    rows = load_rows(args.n)
    srcs = [r['arabic'] for r in rows]
    refs = [r['english'] for r in rows]
    print(f'[*] test_clean: {len(rows)} held-out sentences from works absent from training')

    res = {}
    # C. identity floor
    res['identity'] = score(srcs, refs, 'identity (floor)', srcs)

    # A. shipped lookup
    idioms, lexicon, comp_map = extract_dicts()
    print(f'[*] shipped lookup tables: idioms={len(idioms)} lexicon={len(lexicon)} '
          f'(root,wazn)={len(comp_map)}')
    try:
        import nrmp_vocab as nv
        V = next(v for k, v in vars(nv).items() if isinstance(v, type) and 'MorphemicVocab' in k)
        vocab = V('/workspace/hf_v19_2_release/data/rootformer_v12_arabic_blueprint.json')
    except Exception as e:
        print('  [warn] vocab unavailable, lookup falls back to lexicon only:', e)
        vocab = None
    t0 = time.time()
    hyp_lookup = [lookup_translate(s, idioms, lexicon, comp_map, vocab) for s in srcs]
    print(f'  lookup took {time.time()-t0:.0f}s')
    res['lookup'] = score(hyp_lookup, refs, 'A. shipped lookup', srcs)

    # B. opus-mt
    t0 = time.time()
    hyp_opus = opus_translate(srcs, args.opus)
    print(f'  opus-mt took {time.time()-t0:.0f}s')
    res['opus_mt'] = score(hyp_opus, refs, 'B. opus-mt-ar-en', srcs)

    # D. LLM arms
    if args.llm:
        sub = rows[:args.llm_n]
        s_src = [r['arabic'] for r in sub]
        s_ref = [r['english'] for r in sub]
        try:
            from nmt_eval import ConceptExtractor, concepts_to_str
            ce = ConceptExtractor()
            cons = [concepts_to_str(ce.concepts(a)) for a in s_src]
        except Exception as e:
            print('  [warn] concept extractor unavailable:', e)
            cons = ['' for _ in s_src]
        h_plain = llm_translate(s_src, cons, args.llm, False)
        res['llm_plain'] = score(h_plain, s_ref, f'D1. llm plain ({args.llm.split("/")[-1]})', s_src)
        h_con = llm_translate(s_src, cons, args.llm, True)
        res['llm_concepts'] = score(h_con, s_ref, f'D2. llm + concepts', s_src)

    json.dump({'args': vars(args), 'results': res,
               'samples': [{'src': srcs[i], 'ref': refs[i], 'lookup': hyp_lookup[i],
                            'opus': hyp_opus[i]} for i in range(min(8, len(rows)))]},
              open(args.out, 'w'), ensure_ascii=False, indent=2)
    print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
