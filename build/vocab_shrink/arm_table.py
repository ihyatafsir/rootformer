import json
paths = [("CONTROL", "/workspace/v18fix/results_CONTROL.json"),
         ("GUARD", "/workspace/v18fix/results_GUARD.json"),
         ("FIX", "/workspace/v18fix/results_FIX.json"),
         ("NOFEAT", "/workspace/v18fix/results_NOFEAT.json"),
         ("GATE2", "/workspace/v18fix/results_GATE2.json"),
         ("FIX2", "/workspace/v18fix/results_FIX2.json"),
         ("ALIGNED", "/workspace/head_fix/results_ALIGNED.json")]
hdr = "%-10s %6s %8s %8s %8s %8s %9s %11s" % ("arm", "step", "ALL@1", "ALL@5", "NOV@1", "NOV@5", "CE_z", "feat_scale")
print(hdr)
out = {}
for tag, p in paths:
    try:
        h = json.load(open(p))["history"][-1]
        a, n = h["ALL_val"], h.get("NOVEL_only", {})
        out[tag] = {"step": h["step"], "ALL_acc1": 100 * a["acc@1"], "ALL_acc5": 100 * a["acc@5"],
                    "NOV_acc1": 100 * n.get("acc@1", float("nan")),
                    "NOV_acc5": 100 * n.get("acc@5", float("nan")), "CE_z": a["ce_z"],
                    "feat_scale": h.get("feat_scale")}
        print("%-10s %6d %8.2f %8.2f %8.2f %8.2f %9.4f %11.3f" % (
            tag, h["step"], out[tag]["ALL_acc1"], out[tag]["ALL_acc5"],
            out[tag]["NOV_acc1"], out[tag]["NOV_acc5"], out[tag]["CE_z"], out[tag]["feat_scale"]))
    except Exception as e:
        print(tag, "ERR", repr(e))
json.dump(out, open("/workspace/head_fix/arm_table.json", "w"), indent=2)
print("wrote /workspace/head_fix/arm_table.json")
