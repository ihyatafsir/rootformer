# root_attn_depth — DEPTH arm of the root cross-attention experiment

**Question (pre-registered).** The RCA pathway carries essentially the whole capability into a
FROZEN trunk (forcing its gates to 0 at inference drops acc@1 to ~3 %, below the 3.551 % marginal),
yet it is attached to only 4 of 24 trunk layers (20–23), and both RCA arms plateaued
(CTL 16.5 %, A2 19.53 %). Is **depth** — how much of the trunk gets to see the root history — the
structural lever that closes the gap to the 53.13 % root-history n-gram ceiling, or is the frozen
trunk itself the ceiling?

**Design.** ONE variable: `--root-cross-attn top4 (layers 20–23) -> top8 (16–23) -> top12 (12–23)`.
Everything else identical to `RCA_FROZEN_CTL`
(`/workspace/root_attn_ctl/run_frozen_ctl.sh`): trunk **FROZEN** (`--unfreeze-last 0`),
`--rca-lr-scale 1.0`, FIX flags (`--feat-gate --margin-ramp 10 --grad-warmup 10 --logit-scale ln`),
`--head-init remap`, aligned cache `/workspace/head_fix/nrmp_cache_9490_aligned`, 20 000 steps,
batch 32, eval every 1000.

**No module was edited.** `top8` / `top12` are already parsed by the shipped
`parse_layer_spec` in `/workspace/root_attn/root_cross_attn.py`
(md5 `b53ae702bde947ac66f933191f904c36`) driven by the shipped
`/workspace/hf_v19_2_release/nrmt_train.py` (md5 `56ef3f2fd9ddc352dee596f32590e7b5`) — the same two
files CTL used. No `.bak` files were needed and no test suite rerun was required
(`cmdline_parity.txt` shows the normalized command lines are identical apart from `<RCA>`/tag/paths).

## Parameter cost

| arm | layers | RCA params | total trainable |
|---|---|---|---|
| CTL | 20–23 (4) | 9.64 M | 22.43 M |
| T8  | 16–23 (8) | **19.27 M** | **32.07 M** |
| T12 | 12–23 (12) | **28.91 M** | **41.71 M** |

Per layer = q 896² + k 448·896 + v 448·896 + o 896² + 2 LayerNorms + gate = 2 408 897 ≈ 2.409 M.
The predicted 8/4 × 9.64 M = 19.3 M is confirmed exactly by the trainer (19.27 M).

## Result — stage 1 (T8), 20 000 steps, rc=0, 0 NaN, 0 WARN

Matched-step `ALL_val` against CTL's log (CTL was SIGTERM'd at 15 000, so 15 000 is the last
strictly matched point; CTL has no `results_RCA_FROZEN_CTL.json`):

| step | CTL acc@1 | T8 acc@1 | Δ | CTL acc@5 | T8 acc@5 | CTL CE_z | T8 CE_z |
|---|---|---|---|---|---|---|---|
| 1000 | 13.57 | 14.92 | +1.35 | 26.00 | 29.93 | 6.473 | 6.294 |
| 2000 | 15.09 | 15.94 | +0.85 | 30.00 | 31.79 | 6.806 | 6.838 |
| 3000 | 16.02 | 16.13 | +0.11 | 31.17 | 32.61 | 7.187 | 7.110 |
| 4000 | 15.94 | 16.11 | +0.17 | 31.83 | 33.13 | 7.459 | 7.264 |
| 5000 | 16.20 | 16.60 | +0.40 | 32.42 | 33.83 | 7.590 | 7.346 |
| 6000 | 16.24 | 16.36 | +0.12 | 32.64 | 34.06 | 7.757 | 7.434 |
| 7000 | 16.33 | 16.49 | +0.16 | 32.84 | 34.29 | 7.841 | 7.504 |
| 8000 | 16.46 | 16.57 | +0.11 | 33.27 | 34.51 | 7.904 | 7.590 |
| 9000 | 16.47 | 16.77 | +0.30 | 33.38 | 34.63 | 7.995 | 7.678 |
| 10000 | 16.68 | 16.89 | +0.21 | 33.53 | 34.87 | 8.078 | 7.744 |
| 11000 | 16.50 | 16.85 | +0.35 | 33.66 | 35.04 | 8.184 | 7.826 |
| 12000 | 16.53 | 16.75 | +0.22 | 33.73 | 35.21 | 8.246 | 7.934 |
| 13000 | 16.52 | 16.86 | +0.34 | 34.18 | 35.40 | 8.360 | 7.979 |
| 14000 | 16.58 | 16.66 | +0.08 | 33.91 | 35.45 | 8.433 | 8.089 |
| **15000** | **16.54** | **16.67** | **+0.13** | **33.97** | **35.46** | **8.534** | **8.159** |
| 16000–20000 (T8 only) | — | 16.72, 16.67, 16.70, 16.72, **16.71** | — | — | 35.47→35.53 | — | 8.23→8.35 |

T8 peak: 16.89 % @10k. Last six evals span 0.05 pp (hard plateau). Final: ALL 16.71 / 35.53 / 8.3525;
NOVEL 17.05 / 36.63 / 8.2928. `RCA_OFF` at 15k 3.33 % (final 3.32 %, NOVEL 3.12 %) — the pathway
still carries the capability. Gates final `[+0.1574,-0.2372,-0.3915,+0.5383,+0.5329,+0.6817,
+0.7058,+0.5869]`, |mean| 0.479 (CTL final `[+0.1807,+0.5117,+1.1172,+0.9844]`, |mean| 0.698): the
four NEW layers 16–19 carry live non-zero gates, so they are not dead — they are redundant.
Gradient norms (mean over the last 200 steps, rca / head / trunk): 6.68e-1 / 9.30e-1 / 0 at 1k →
2.27e-2 / 3.53e-2 / **0** at 15k (trunk exactly zero: it stayed frozen). h_drift 20.31 @15k.

Attention probe at the matched step 15 000 (read-only probe over the saved payloads;
`probe_rca_dynamics.py`, copied verbatim — see `probe_*.json`):

| | T8 (8 layers) | CTL (4 layers) |
|---|---|---|
| injected residual RMS | 11.1 – 14.9 | 16.8 – 22.5 |
| normalised entropy (1.0 = uniform) | 0.437 – 0.496 | 0.463 – 0.513 |
| effective attended positions (of ≤128) | 7.8 – 10.0 | 8.4 – 10.6 |
| max attention weight | 0.414 – 0.483 | 0.369 – 0.458 |
| diagonal mass (j = t) | 0.0284 – 0.0312 | 0.0235 – 0.0286 |
| argmax = current root | 0.034 – 0.039 | 0.025 – 0.054 |
| mean mass within offsets 0–8 | 0.036 – 0.039 | 0.031 – 0.036 |
| max abs Δh live vs gates-off | 23.31 | 22.69 |

Same attention shape, non-degenerate, diffuse over the whole root history (only ~3.7 % of the mass
sits within the last 9 positions), and a large live-vs-ablated hidden-state difference. The 8-layer
stack spreads a similar total injection over twice as many, smaller per-layer residuals.

**Costs.** T8: wall-clock 7 805 s = **130 min** at ~3 it/s (contended: one other GPU arm plus a
CPU-only `F_ret_lowlr` job; alone it is ~70 min at 4.8 it/s). Peak VRAM **11 896 MiB** for the arm,
28 688 MiB for the whole GPU (never OOM; the launch gate kept it to two concurrent 32-batch arms).

## Verdict (stage 1)

**Depth is not the lever.** Doubling the number of trunk layers that see the root history
(4 → 8, +9.64 M params) buys **+0.13 pp acc@1 at the matched step 15 000** and ~+0.3 pp on the
converged plateau, against **+3.03 pp** from the trunk-LR lever (CTL 16.5 → A2 19.53) and +9.7 pp
from adding the RCA at all. `acc@5` (+1.5 pp) and `CE_z` (−0.375) do improve consistently, so the
extra layers compute something real — but it is redundant for top-1. Per the pre-registered rule
("accuracy stays ≈16–17 % → depth is not the lever either"), **the ceiling is the FROZEN TRUNK
itself**, which makes a trunk trained for this objective the only remaining path rather than a
choice.

## Stage 2 (top12) — ABANDONED BY PARENT DECISION, as a settled-negative replication

`top12` (layers 12–23) was set up as the second depth point and started once. Its parameter cost was
confirmed live by the trainer before any training:

```
[*] root cross-attention attached to trunk layers [12..23] (28.91M new params, heads=8x112)
[*] trainable parameters: 41.71M (head 12.80M @lr 0.001 | rca 28.91M @lr 0.001 layers [12..23])
```

Attempt 1 ran 18:58:33 → 19:08:34 UTC (601 s, eval @1000 = 15.13 % acc@1 vs T8 14.92 %, CTL 13.57 %)
and then received **SIGTERM (rc=143)** — no CUDA-OOM traceback, no `dmesg` OOM entry, peak VRAM only
11 042 MiB. **The SIGTERM was deliberate and came from the parent agent**, to free the card for the
`RCA_W2N` width arm; it was not OOM and not a neighbouring process-killer. That attribution was
initially misread here (a gated retry was armed on the assumption of a hostile neighbour); the
correction is recorded in `chain_t12.out` and the retry was stood down.

The parent then chose option (b): **stop top12 and free the queue**, because a second depth point
could only be a settled-negative replication — if 8 layers do not differ from 4, there is nothing in
the q–k form (which reads a frozen embedding) for 12 to do differently — while the card was needed
for the live structural question (`RCA_W2N`) and the EWC 6 000-step arm. This is therefore **not a
failed measurement but a decision not to spend 2 GPU-hours on a replication of an already-decided
negative**; the T8 contrast is the deliverable. Evidence of the one attempt is kept:
`logs/train_RCA_DEPTH_T12_attempt1_killed.log`, `trace_RCA_DEPTH_T12_attempt1_killed.jsonl`,
`chain_t12.out`. (Naming: exactly ONE top12 training run ever executed — the 601 s one above. Its
log was copied twice, once manually as `*_attempt1_killed.*` and once by the retry wrapper as
`*_attempt0_killed.*` on the pod; the two copies are the same run.)

## Files

- `run_depth.sh` — the arm launcher (durable GPU gate: ≤1 other arm and ≥12 GiB free, verified
  twice; peak-VRAM poller; per-eval snapshotter). `chain_depth.sh` ran T8 then T12;
  `chain_depth_t12.sh` was the (stood-down) retrying wrapper for the top12 stage.
- `depth_report.py` — matched-step parser for CTL's log + the arms' logs/traces.
- `probe_depth.sh` — wrapper for the read-only attention probe.
- `cmdline_parity.txt` — normalized command-line diff (empty).
- `logs/train_RCA_DEPTH_T8.log`, `trace_RCA_DEPTH_T8.jsonl`, `results_RCA_DEPTH_T8.json`,
  `chain_depth.out`, `ctl/` (CTL's log and trace), `probe_*.json` (T8 final, T8@9k, T8@15k,
  CTL@9k, CTL@15k).
- Stage-2 evidence: `chain_t12.out`, `logs/train_RCA_DEPTH_T12_attempt1_killed.log`,
  `trace_RCA_DEPTH_T12_attempt1_killed.jsonl`.
- Remote: `/workspace/root_attn_depth/` (same layout; nothing outside it was ever written).

## GPU discipline actually observed

Every launch went through the durable gate and was verified against `nvidia-smi` and `ps` before
starting. Never more than **two** concurrent 32-batch arms were started by this experiment; peak
whole-GPU usage observed while my T8 arm ran was 28 688 MiB of 32 623 MiB (my arm's own peak
11 896 MiB), and no OOM ever occurred. No process belonging to another agent was ever signalled;
the only kills performed here were of this experiment's own launcher, when the parent asked for the
queue to be freed. The 130-min wall-clock for T8 (3 it/s vs 4.8 it/s alone) is the price of sharing
the card, not of the depth change.

