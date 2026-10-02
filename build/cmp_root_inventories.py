import json, re, sys

DIAC = re.compile(r'[\u064b-\u0652\u0670\u0640\u06d6-\u06ed]')


def norm(s):
    s = DIAC.sub('', str(s)).strip()
    for a, b in (('أ', 'ا'), ('إ', 'ا'), ('آ', 'ا'), ('ٱ', 'ا'), ('ى', 'ي')):
        s = s.replace(a, b)
    return s


def roots_from(p, key=None):
    try:
        d = json.load(open(p, encoding='utf-8'))
    except Exception as e:
        return None, str(e)
    if isinstance(d, dict):
        for k in ('roots', 'data', 'lexicon', 'entries', 'root'):
            if k in d and isinstance(d[k], (list, dict)):
                d = d[k]
                break
    xs = list(d.keys()) if isinstance(d, dict) else list(d)
    out = set()
    for x in xs:
        if isinstance(x, dict):
            for k in ('root', 'bare', 'جذر'):
                if k in x:
                    x = x[k]
                    break
        if isinstance(x, str) and x.strip():
            out.add(norm(x))
    return out, None


def roots_from_jsonl(p):
    out = set()
    for line in open(p, encoding='utf-8'):
        line = line.strip()
        if not line:
            continue
        try:
            out.add(norm(json.loads(line)['root']))
        except Exception:
            pass
    return out


sets = {}
for name, p in (
    ('lisan_9195', '/workspace/rootformer_v12/v18_next_root_morph/data/lisan_9195_roots.json'),
    ('lisan_root_lexicon', '/workspace/rootformer/data/lisan_root_lexicon.json'),
):
    s, err = roots_from(p)
    if s is None:
        print(f'{name:22s} ERROR {err}')
    else:
        sets[name] = s
        print(f'{name:22s} {len(s):6d} roots')

p3 = '/workspace/rootformer_v12/v18_next_root_morph/data/lisan_3pillar_master_roots.jsonl'
sets['3pillar_9015'] = roots_from_jsonl(p3)
print(f'{"3pillar_9015":22s} {len(sets["3pillar_9015"]):6d} roots')

bp = json.load(open('/workspace/hf_v19_2_release/data/rootformer_v12_arabic_blueprint.json'))
i2t = bp['vocab_id_to_token']
lo, hi = bp['partitions']['classical_roots']
bpr = {norm(re.sub(r'^<root_|>$', '', i2t[str(i)])) for i in range(lo, hi)}
sets['blueprint'] = bpr
print(f'{"blueprint":22s} {len(bpr):6d} roots')

print()
base = 'lisan_9195'
if base in sets:
    for n, s in sets.items():
        if n == base:
            continue
        missing = sets[base] - s
        print(f'in {base} but NOT in {n:20s}: {len(missing):5d}   e.g. {" ".join(sorted(missing)[:14])}')
print()
for n, s in sets.items():
    if n == base:
        continue
    extra = s - sets[base]
    print(f'in {n:20s} but NOT in {base}: {len(extra):5d}   e.g. {" ".join(sorted(extra)[:14])}')
