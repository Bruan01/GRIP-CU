# WSL2 + RTX 3090 Runbook

## A. 拉取
```bash
cd /mnt/c/Users/Administrator/Desktop/实验/GRIP-CU
git pull --ff-only origin wsl/nell23k-smoke-20260829
git rev-parse HEAD
git status --short
```
HEAD 必须等于本机本次推送 commit。服务器不 reset/clean、不改实验代码。

## B. Preflight
```bash
cd 08_experiments/QueryGraphLoRAGRIP/v0_2_oracle_basis_routing_2026-09-04
bash configs/check_wsl_runtime.sh
```
失败只回传日志。

## C. 复算 E03
```bash
cd /mnt/c/Users/Administrator/Desktop/实验/GRIP-CU
python 09_results_analysis/2026-09-04_entity_constrained_grip_phase_a_audit/scripts/audit_e03_results.py --run-dir 08_experiments/EntityConstrainedGRIP/v0_1_global_entity_decoder_2026-09-04/results/runs/wsl3090_entity_decoder_validation_20260904_02
```

## D. E09 最小矩阵
```bash
cd 08_experiments/QueryGraphLoRAGRIP/v0_2_oracle_basis_routing_2026-09-04
export RUN_ID=wsl3090_e09_phase_a_20260904_01
export MODEL_OVERRIDE=/home/kieran/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775
bash configs/run_wsl_phase_a.sh 2>&1 | tee "results/${RUN_ID}.log"
```
预计 10 runs、约 3–5 GPU 小时。RUN_ID 不复用。

## E. Gate
```bash
cat "results/runs/$RUN_ID/REPORT.md"
cat "results/runs/$RUN_ID/suite_summary.json"
```
`STOP_GRAPH_CONDITIONAL_LORA` 时停止；`GO_LEARNED_QUERY_ROUTER` 也只报告，等待 macOS 开发下一版本。

## F. Manifest 与回传
```bash
python -c 'from pathlib import Path;import hashlib,json,os;r=Path("results/runs")/os.environ["RUN_ID"];xs=[{"path":str(p.relative_to(r)),"size_bytes":p.stat().st_size,"sha256":hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(r.rglob("*")) if p.is_file()];(r/"artifact_manifest.json").write_text(json.dumps({"files":xs},indent=2)+"\\n");print(len(xs))'
find "results/runs/$RUN_ID" -iname '*test*'
```
回传 HEAD/环境、preflight、时长/峰值显存、REPORT、suite_summary/csv、manifest、所有 run_summary、失败日志，并明确 `test_file_opened=false`。
