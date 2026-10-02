import json, sys
def row(p):
    d = {}
    for l in open(p):
        j = json.loads(l)
        d[j['step']] = j
    return d
a = row('/workspace/root_attn/trace_ROOTATTN.jsonl')
b = row('/workspace/head_fix/trace_ALIGNED_FIX.jsonl')
print('step  | ROOTATTN:  loss    root   gnorm_trunk   lr      | ALIGNED_FIX: loss    root   gnorm     lr')
for s in (1, 100, 300, 600, 1000, 2000, 3000, 5000):
    x = a.get(s); y = b.get(s, {})
    if x is None:
        continue
    print('%5d | %9.4f %7.3f %11.3e %.2e | %11.4f %7.3f %9.3e %.2e' % (
        s, x['loss'], x['root'], x.get('grad_norm_trunk', 0.0), x['lr'],
        y.get('loss', float('nan')), y.get('root', float('nan')),
        y.get('grad_norm', 0.0), y.get('lr', 0.0)))
# late-stage of the baseline for reference
print('ALIGNED_FIX @10000/@20000:', ['%.4f' % b[s]['loss'] for s in (10000, 20000) if s in b])
