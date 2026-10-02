#!/usr/bin/env python3
"""Download the chosen base model into the isolated /workspace/alt_base/cache HF home.

Deliberately does NOT touch the shared /workspace/.hf_home (another agent's cache).
"""
import os, sys, time, json
from pathlib import Path

REPO = os.environ.get('BASE_REPO', 'Qwen/Qwen2.5-0.5B')
CACHE = Path(os.environ.get('ALT_HF_HOME', '/workspace/alt_base/cache'))
CACHE.mkdir(parents=True, exist_ok=True)
os.environ['HF_HOME'] = str(CACHE)
os.environ['HF_HUB_CACHE'] = str(CACHE / 'hub')
tok = Path('/workspace/.hf_token')
if tok.exists():
    os.environ.setdefault('HF_TOKEN', tok.read_text().strip())

t0 = time.time()
from huggingface_hub import snapshot_download

allow = ['*.json', '*.safetensors', '*.txt', '*.model', '*.jinja']
p = snapshot_download(
    repo_id=REPO,
    cache_dir=str(CACHE / 'hub'),
    allow_patterns=allow,
    max_workers=4,
)
print('[*] snapshot ->', p, flush=True)

snap = Path(p)
tot, files = 0, []
for f in sorted(snap.rglob('*')):
    if f.is_file():
        sz = f.stat().st_size
        tot += sz
        files.append({'name': f.name, 'bytes': sz})
print(json.dumps({'repo': REPO, 'path': str(snap), 'total_bytes': tot,
                  'total_mb': round(tot / 1e6, 1), 'files': files,
                  'seconds': round(time.time() - t0, 1)}, indent=2))
