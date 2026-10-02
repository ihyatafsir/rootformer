# discrete_path — status (2026-10-02)

DESIGN: `--ngram-features vocab` (explicit, collision-free order-1..4 root-context tables),
NOT the brief's 2^20 hash (measured hash ceiling at 2^20 = 5.54 % vs 53.13 % collision-free).

CPU PROOF: closed-form construction reproduces the order-k lookup with gap 0.0000 pp at k=1,2,3,4
(53.1295 % ALL / 49.5082 % NOVEL at k=4). 24/24 unit tests, seven suites all pass.

CPU SHORT FIT (h=0, 149.9 M params, 1500 steps = 0.86 epochs): 25.53 % ALL / 23.74 % NOVEL and
still accelerating with the fade-in gate at only 0.278. GPU run has 13.3x the step budget.

GPU: `chain_ngram.sh` armed (durable nohup, polls nvidia-smi compute-apps; never preempts).
It will run ONE 20k arm:
  nrmt_train.py --cache /workspace/head_fix/nrmp_cache_9490_aligned --head-init remap \
    --steps 20000 --batch-size 32 --eval-every 1000 --tag NGRAM \
    --feat-gate --margin-ramp 10 --grad-warmup 10 --logit-scale ln \
    --ngram-features vocab --ngram-orders 1,2,3,4 --ngram-dim 16
Outputs: train_NGRAM.log, results_NGRAM.json, trace_NGRAM.jsonl, head_NGRAM.pt, vram_NGRAM.txt
Gate log: chain_ngram.log
