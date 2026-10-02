#!/usr/bin/env python3
"""make a tiny smoke cache (4 equal streams) so the PATCHED trainer can be run end-to-end on CPU
without touching the real caches or the occupied GPU."""
import sys
from pathlib import Path
import torch

sys.path.insert(0, '/workspace/hf_v19_2_release')
import nrmp_vocab as nv

V = next(v for k, v in vars(nv).items() if isinstance(v, type) and 'MorphemicVocab' in k)
vocab = V('/workspace/hf_v19_2_release/data/rootformer_v12_arabic_blueprint.json')
g = torch.Generator().manual_seed(7)
N = 900
# real radical root ids: skip the low special block and the <P:...> tail
lo, hi = 300, max(400, vocab.num_roots - 400)
streams = [
    torch.randint(0, max(2, vocab.num_prefixes), (N,), generator=g),
    torch.randint(lo, hi, (N,), generator=g),
    torch.randint(0, max(2, vocab.num_awzan), (N,), generator=g),
    torch.randint(0, max(2, vocab.num_suffixes), (N,), generator=g),
]
out = Path('/workspace/discrete_path/smoke_cache')
out.mkdir(parents=True, exist_ok=True)
torch.save(streams, out / 'train.pt')
torch.save([t.clone() for t in streams], out / 'val.pt')
print('num_roots', vocab.num_roots, 'num_awzan', vocab.num_awzan,
      'num_prefixes', vocab.num_prefixes, 'num_suffixes', vocab.num_suffixes)
print('wrote', out)
