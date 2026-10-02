#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Post-verification analysis.

  * de-duplicate / re-order cases
  * cross-tabulate the records' OWN hedging language against the verdicts
    (a record that says "Asas al-Balaghah is not attested" while an Asas entry
     exists is an inconsistent adjudication and is flagged)
  * extract every WRONG case with record text + the source quote
  * compute the final rates on CHECKABLE cases only
"""
import json
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, "/workspace/raghib_audit")
from extract_mufradat_index import strip_diac

VD = "/workspace/raghib_audit/verdicts_verified.json"
ENR = "/workspace/raghib_audit/fidelity_sample_enriched.json"
HEDGE = re.compile(
    r"not attested|not directly attested|no literal .{0,40}attested|"
    r"as delineated from|as inferred from|delineated here from|"
    r"though not attested|is not established in the classical|"
    r"no literal physical meaning is attested", re.I)


def main():
    V = json.load(open(VD, encoding="utf-8"))
    cases = V["cases"]
    S = json.load(open(ENR, encoding="utf-8"))
    bundle = {e["root"]: e for e in S["bundle"]}

    # duplicates / order
    rc = Counter(c["root"] for c in cases)
    dup = [r for r, n in rc.items() if n > 1]
    print("cases=%d distinct=%d duplicates=%s" % (len(cases), len(rc), dup))

    print("\n=== rates among CHECKABLE cases only ===")
    for k in ("mufradat", "maqayis", "asas"):
        sub = [c for c in cases if c[k + "_verdict"] != "UNCHECKABLE"]
        tot = len(sub)
        cc = Counter(c[k + "_verdict"] for c in sub)
        print("  %-9s checkable n=%3d : " % (k, tot) + "  ".join(
            "%s=%d(%.1f%%)" % (v, cc[v], 100.0 * cc[v] / tot)
            for v in ("AGREES", "PARTIAL", "WRONG")))
        # agreement at-or-above PARTIAL
        good = cc["AGREES"]
        print("             AGREES rate = %.1f%% ; AGREES+PARTIAL = %.1f%%"
              % (100.0 * good / tot, 100.0 * (good + cc["PARTIAL"]) / tot))

    print("\n=== hedging language in the records vs verdicts ===")
    hedged = [c for c in cases
              if HEDGE.search(json.dumps(bundle[c["root"]]["record"], ensure_ascii=False))]
    print("sampled records containing an explicit non-attestation/delineation hedge: %d / %d"
          % (len(hedged), len(cases)))
    hc = Counter(c["asas_verdict"] for c in hedged)
    print("  their asas_verdict distribution:", hc.most_common())
    bad = [c for c in hedged if c["asas_verdict"] == "AGREES"]
    print("  !! hedged BUT Asas verdict AGREES (inconsistent): %d" % len(bad))
    for c in bad[:12]:
        print("     %s | %s" % (c["root"], c["asas_reason"][:120]))

    print("\n=== every WRONG case ===")
    wrongs = []
    for c in cases:
        e = bundle[c["root"]]
        for k in ("mufradat", "maqayis", "asas"):
            if c[k + "_verdict"] == "WRONG":
                src = e["sources"].get(k)
                if k == "asas" and src:
                    stext = " ||| ".join(x["text"] for x in src["entries"])[:420]
                    sref = "asas:%s" % src["entries"][0]["mode"]
                elif src:
                    stext = src["text"][:420]
                    sref = "%s line %s (hdr %s)" % (k, src["line"], src["header"])
                else:
                    stext = "(NO ENTRY)"
                    sref = k
                wrongs.append({"root": c["root"], "check": k, "src_ref": sref,
                               "record_core": (e["record"].get("ontological_core") or "")[:340],
                               "record_haqiqah": (e["record"].get("haqiqah_literal") or "")[:340],
                               "source_text": stext,
                               "quote": c[k + "_quote"], "reason": c[k + "_reason"]})
    print("total WRONG verdicts: %d across %d distinct roots"
          % (len(wrongs), len(set(w["root"] for w in wrongs))))
    for w in wrongs:
        print("\n  ROOT %s  [%s]  src=%s" % (w["root"], w["check"], w["src_ref"]))
        print("    reason: %s" % w["reason"][:220])
        print("    quote : %s" % w["quote"][:200])
    json.dump(wrongs, open("/workspace/raghib_audit/wrong_cases.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    json.dump(cases, open("/workspace/raghib_audit/verdicts_final.json", "w", encoding="utf-8"),
              ensure_ascii=False)
    print("\nwrote wrong_cases.json and verdicts_final.json")


if __name__ == "__main__":
    main()
