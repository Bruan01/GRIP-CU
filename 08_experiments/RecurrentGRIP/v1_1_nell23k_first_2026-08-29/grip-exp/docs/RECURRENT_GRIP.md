# RecurrentGRIP v1.1 Runtime Guide

RecurrentGRIP compiles one graph into executor-layer LoRA parameters and reuses the
same decoder block for `K` closed-book execution steps.  v1.1 keeps the fixed-depth
mechanism from v1 and makes NELL23K the first integration benchmark.

## 1. Prepare NELL23K

From the version root:

```bash
bash configs/prepare_nell23k.sh
```

The adapter reads `train.txt`, `valid.txt`, `test.txt`, and `entity2text.json` from
`grip-exp/data/raw_datasets/nell23k/`.  Train triples define both the graph and
training QA; validation and test roles are preserved.

Output:

```text
grip-exp/outputs/data/nell23k/recurrent_relation_prediction.json
grip-exp/outputs/data/nell23k/recurrent_relation_prediction.json.stats.json
```

Each recurrent question contains a stable ID, split, candidate relations, source and
target entities, answer, train-graph shortest path, structural distance, and nullable
`true_hop`.  Unknown distances remain in overall accuracy and are excluded from
hop-specific statistics.

## 2. Run WSL tests

```bash
bash configs/setup_wsl3090.sh
bash configs/test_wsl3090.sh
```

## 3. Smoke

```bash
RUN_ID=wsl3090_nell23k_smoke_20260829_01 \
  bash configs/run_nell23k_smoke_wsl.sh
```

The one-graph NELL23K setup evaluates correct and disabled adapters.  Shuffled
adapter evaluation requires at least two graph adapters and is skipped explicitly.

## 4. Analyze

```bash
RUN_DIR="$(cat results/LAST_NELL23K_SMOKE_RUN.txt)"
RUN_DIR="$RUN_DIR" bash configs/analyze_nell23k.sh
```

The analyzer emits overall/K/control accuracy for every sample and hop/K buckets only
for samples with known structural distance.

## 5. Pilot

After the smoke gate passes, regenerate a larger input and run:

```bash
MAX_TRAIN_QUESTIONS=512 \
MAX_VALIDATION_QUESTIONS=128 \
MAX_TEST_QUESTIONS=512 \
  bash configs/prepare_nell23k.sh

RUN_ID=wsl3090_nell23k_pilot_20260829_01 \
  bash configs/run_nell23k_pilot_wsl.sh
```

CLEGR utilities inherited from v1 remain available for later controlled K-to-hop
mechanism confirmation, but they are not part of the v1.1 smoke critical path.
