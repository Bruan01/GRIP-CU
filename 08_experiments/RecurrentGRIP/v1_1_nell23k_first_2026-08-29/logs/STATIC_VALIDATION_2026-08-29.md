# Static Validation — 2026-08-29

## Scope

Host: macOS development machine. No CUDA/ML training was attempted. Full Torch,
Transformers and PEFT tests remain assigned to WSL2 RTX 3090.

## Checks

```bash
bash configs/check_mac_static.sh
```

Result:

```text
AST syntax check: PASS (85 Python files)
NELL23K preparation unit tests: 2/2 PASS
Bash syntax: PASS
JSON syntax: PASS
macOS static checks: PASS
```

## Real NELL23K Fixture Check

Prepared from the repository-local raw files with:

```text
seed=2026
train=64
validation=32
test=64
num_candidates=10
```

Observed:

```text
records=1
nodes=20799
train graph edges=24321
train/validation/test questions=64/32/64
unknown structural hop questions=9
all answers included in candidates=PASS
all known hops positive=PASS
```

A second generation to a different path produced the same output SHA256:

```text
e9b4f037150c3b58ca3421647fe53b63a15876a9a5a9ed755df7907c99e1c478
```

Deterministic preparation: PASS.

## Isolated Schema/Metric Checks

Because the system macOS Python has no Torch/Transformers environment, the modified
pure-Python output dataclass and metric module were loaded in isolation:

```text
nullable true_hop validation=PASS
known/unknown hop accounting=PASS
split × K × adapter aggregation=PASS
```

Running the complete `unittest discover` under the system Python stops on missing
`torch` and `transformers`; this is an environment absence, not accepted as WSL test
completion. `configs/test_wsl3090.sh` remains the required full integration gate.

## Repository Isolation

```text
Original GRIP submodule: 2835b440bfd2c4de36f0380ae19bc1c22e6cb459
Original GRIP status: clean
v1_fixed_depth_2026-08-28: unchanged
v1_1_nell23k_first_2026-08-29: independent snapshot
```
