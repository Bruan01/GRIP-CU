# Static Validation — 2026-08-28

## Source isolation

- Original repository: `13_base_method/grip-exp`
- Result: `git status --short` empty；
- Conclusion: RecurrentGRIP v1 did not modify the Original GRIP repository.

## Syntax and artifact checks

```text
python3 -m compileall -q arguments data evaluation grip scripts tests
bash -n ../configs/prepare_clegr.sh ../configs/run_qwen05b_pilot.sh ../configs/analyze_pilot.sh
python3 -m json.tool ../configs/pilot_qwen05b.json
```

Result: PASS.

## Pure algorithm smoke

Validated without loading ML dependencies:

- station endpoint recovery；
- undirected BFS shortest path；
- official `StationShortestCount = max(true_hop - 1, 0)` relation；
- train/validation hop 1–2 and test hop 3–4 separation；
- recurrent argument validation；
- recurrent prediction schema。

Result: PASS.

## PEFT path audit

PEFT 0.17.1 matches `layers_to_transform` using a regex equivalent to:

```python
re.match(r".*\.layers\.(\d+)\.", module_key)
```

The wrapped target path has the form:

```text
model.layers.INDEX.block.self_attn.q_proj
```

Therefore the layer index remains syntactically matchable after wrapping. The
actual tiny-model integration test remains in `tests/test_adapter_scope.py` and
must be executed after installing the project's Torch/Transformers/PEFT runtime.

## Pending runtime checks

- `FixedDepthRecurrentBlock` forward on a real Qwen decoder layer；
- PEFT injection and adapter save/load；
- cache-free `generate()`；
- Qwen2.5-0.5B smoke training；
- GPU memory and wall-time behavior。
