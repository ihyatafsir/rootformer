#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Secondary diagnostic for the compositional-root experiment.

The OUTPUT side is what the redesign fixes.  On the INPUT side the root embedding is still a
lookup table, and a held-out root's row received no gradient (it never occurs in training).
That can only affect a target position if a held-out root appears EARLIER in the same sentence
(i.e. inside the model's context).  This measures how often that is true, so the size of the
residual, unfixed confound is known rather than assumed.

Also reports: per-split counts of held-out targets by context contamination, and the length
mix of held-out root targets.
"""
import json
import sys
from collections import Counter

import torch

D = sys.argv[1] if len(sys.argv) > 1 else '/workspace/scratch_comp/sf_data_9490'
meta = json.load(open(f'{D}/meta.json'))
data = torch.load(f'{D}/streams.pt', weights_only=False)
hold = set(meta['hold_roots'])
spec = set(meta['special_root_ids'])
id2root = {i: r for i, r in enumerate(meta['roots_list'])}

out = {}
for sp in ('test_gen', 'test_deriv', 'val', 'train'):
    R = data[sp]['R']
    n_targets = n_hold_targets = n_contaminated = 0
    lenmix = Counter()
    for seq in R:
        seen_hold_so_far = False
        for t, rid in enumerate(seq):
            if t == 0:
                if rid in hold:
                    seen_hold_so_far = True
                continue
            n_targets += 1
            if rid in hold:
                n_hold_targets += 1
                lenmix[len(id2root.get(rid, ''))] += 1
                if seen_hold_so_far:
                    n_contaminated += 1
            if rid in hold:
                seen_hold_so_far = True
    out[sp] = {'target_positions': n_targets, 'heldout_root_targets': n_hold_targets,
               'heldout_targets_with_heldout_root_earlier_in_sentence': n_contaminated,
               'frac_contaminated': n_contaminated / max(n_hold_targets, 1),
               'heldout_target_len_mix': dict(sorted(lenmix.items()))}
    print(f'{sp}: targets={n_targets} heldout_targets={n_hold_targets} '
          f'context_contaminated={n_contaminated} '
          f'({100*n_contaminated/max(n_hold_targets,1):.1f}%) lenmix={dict(sorted(lenmix.items()))}')

json.dump(out, open('/workspace/scratch_comp/context_diag.json', 'w'), indent=2, ensure_ascii=False)
print('\nwrote /workspace/scratch_comp/context_diag.json')
