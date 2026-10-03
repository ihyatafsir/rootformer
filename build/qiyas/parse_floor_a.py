"""parse_floor_a.py -- tabulate an arm's [eval @N] lines from a saved log excerpt."""
import re, sys
path = sys.argv[1] if len(sys.argv) > 1 else 'build/qiyas/arms_evidence/FLOOR_A_evals.txt'
rows = []
for ln in open(path, encoding='utf-8'):
    m = re.search(r'eval @(\d+)\] ALL_val: acc@1 ([\d.]+)% acc@5 ([\d.]+)% CE_z ([\d.]+)', ln)
    n = re.search(r'NOVEL_only: acc@1 ([\d.]+)% acc@5 ([\d.]+)% CE_z ([\d.]+)', ln)
    w = re.search(r'wazn: ce [\d.]+ ce_z ([\d.]+) acc@1 ([\d.]+)%', ln)
    d = re.search(r'h_drift ([\deE.+-]+)', ln)
    if m:
        rows.append((int(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4)),
                     float(n.group(1)) if n else float('nan'),
                     float(n.group(3)) if n else float('nan'),
                     float(w.group(2)) if w else float('nan'),
                     float(d.group(1)) if d else float('nan')))
print(f"{'step':>6} {'ALL@1':>6} {'ALL@5':>6} {'ALLCEr':>7} {'NOV@1':>6} {'NOVCEr':>7} {'wazn@1':>7} {'h_drift':>9}")
for r in rows:
    print(f"{r[0]:>6} {r[1]:>6.2f} {r[2]:>6.2f} {r[3]:>7.4f} {r[4]:>6.2f} {r[5]:>7.4f} {r[6]:>7.2f} {r[7]:>9.2f}")
a = [r[1] for r in rows]; c = [r[3] for r in rows]
print(f"\nn={len(rows)}  acc@1 min={min(a):.2f} max={max(a):.2f} spread={max(a)-min(a):.2f}pp")
print(f"CE_z first={c[0]:.4f} last={c[-1]:.4f} delta={c[-1]-c[0]:+.4f}")
print("CE_z monotone non-decreasing:", all(c[i] <= c[i+1] for i in range(len(c)-1)))
print("acc@1 monotone increasing  :", all(a[i] <= a[i+1] for i in range(len(a)-1)))
