#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CPU-only early readout from a comp checkpoint: held-out-root content-only top-1/top-5,
partial credit and concrete examples, on a subset of test_deriv.  No GPU needed."""
import json
import sys

import torch

sys.path.insert(0, '/workspace/scratch_comp')
import scratch_lm_comp as S  # noqa: E402

D = '/workspace/scratch_comp/sf_data_9490'
ck = sys.argv[1] if len(sys.argv) > 1 else f'{D}/comp6k.pt'
ms = int(sys.argv[2]) if len(sys.argv) > 2 else 200
meta = json.load(open(f'{D}/meta.json'))
data = torch.load(f'{D}/streams.pt', weights_only=False)
tables = S.Tables(meta)
model = S.LMComp((meta['n_roots'], meta['n_awzan'], meta['n_prefixes'], meta['n_suffixes']),
                 tables.n_gate, meta['n_sym'])
model.load_state_dict(torch.load(ck, map_location='cpu', weights_only=True))
model.eval()
print(f'[*] {ck} max_sent={ms} hold={len(tables.hold)}')

for sp in ('test_deriv', 'test_gen'):
    hr, recs = S.eval_heads(model, data, sp, torch.device('cpu'), tables, 'comp',
                            max_sent=ms, k=5, max_rounds=200, max_rounds_free=250,
                            free_max_pos=200)
    print(f'[{sp}] joint={json.dumps(hr["acc"])}')
    print(f'[{sp}] content_only={json.dumps(hr["content_only_acc"])} '
          f'per_head={hr["per_head_top1"]} inexact={hr["n_bfs_inexact_joint"]}/'
          f'{hr["n_bfs_inexact_free"]}')
    if recs:
        pc = S.partial_credit(recs, tables)
        print(f'[{sp}] partial credit (n={pc["n"]}): '
              + json.dumps({k: v for k, v in pc.items() if k != 'examples'}, ensure_ascii=False))
        print(f'[{sp}] examples:')
        for e in pc['examples'][:8]:
            print('    true=%-7s joint=%-10s | content=%-7s ctop5=%s | c_anag=%s | ctx=%s'
                  % (e['true'], e['pred'], e['content_top1'], e['content_top5'],
                     e['c_anagram'], e['ctx_tail']))
