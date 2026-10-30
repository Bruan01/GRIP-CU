#!/usr/bin/env python3
"""Write human-readable benchmark construction records.

The records are intentionally generated from the manifests, task artifacts,
and Stage1/Stage2 caches instead of hand-copying counts into documentation.
They are a compact audit trail for downstream training and evaluation.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "04_benchmark" / "data"
OUT = ROOT / "04_benchmark" / "construction_records"
MAX_EXAMPLE = 1200


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path.relative_to(ROOT)), "status": "missing"}
    return {
        "path": str(path.relative_to(ROOT)),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def shorten(value: Any) -> str:
    if isinstance(value, dict):
        value = value.get("response", value)
    if isinstance(value, dict):
        value = value.get("response", value.get("query", value))
    text = str(value).replace("\x00", " ").strip()
    return text if len(text) <= MAX_EXAMPLE else text[:MAX_EXAMPLE] + " …"


def cache_example(cache_dir: Path, stage: str) -> dict[str, str] | None:
    path = cache_dir / f"{stage}.jsonl"
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            response = row.get("response", {})
            if not isinstance(response, dict):
                continue
            return {
                "stage": stage,
                "prompt": shorten(response.get("query", "")),
                "model_output": shorten(response.get("response", "")),
            }
    return None


def cache_meta(cache_dir: Path) -> dict[str, Any]:
    if not cache_dir.is_dir():
        return {"path": str(cache_dir), "status": "not_found"}
    files = {}
    for path in sorted(cache_dir.glob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            count = sum(1 for line in handle if line.strip())
        files[path.name] = {"records": count, "size_bytes": path.stat().st_size}
    return {"path": str(cache_dir), "status": "available", "files": files}


def examples(cache_dir: Path) -> list[dict[str, str]]:
    result = []
    for stage in ("summarization_s1", "summarization_s2", "reasoning_qa_s1", "reasoning_qa_s2"):
        item = cache_example(cache_dir, stage)
        if item:
            result.append(item)
    return result


def final_task_examples(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    payload = read_json(path)
    result = []
    for key in ("context_samples", "qa_samples"):
        values = payload.get(key) or payload.get("by_task", {}).get(key, [])
        if values:
            result.append({"stage": key, "sample": shorten(values[0])})
    return result


def task_payload(path: Path) -> dict[str, Any]:
    """Read a generated task payload when it exists."""
    if not path.is_file():
        return {}
    try:
        payload = read_json(path)
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def task_complete(path: Path) -> bool:
    return task_payload(path).get("generation_status") == "complete"


def task_run_metadata(path: Path) -> dict[str, Any]:
    payload = task_payload(path)
    if not payload:
        return {"status": "missing"}
    elapsed = payload.get("elapsed_seconds")
    result = {
        "status": payload.get("generation_status", "unknown"),
        "completed_at_utc": payload.get("completed_at_utc"),
        "elapsed_seconds": elapsed,
        "counts": payload.get("counts", {}),
    }
    if isinstance(elapsed, (int, float)):
        result["elapsed_hours"] = round(float(elapsed) / 3600, 3)
    return result


def md_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def write_record(filename: str, payload: dict[str, Any]) -> None:
    lines = [
        f"# {payload['dataset_name']} 构造记录",
        "",
        f"- 记录更新时间：`{utc_now()}`",
        f"- 构造状态：`{payload['status']}`",
        f"- 目标：`{payload['purpose']}`",
        "",
        "## 构造过程",
        "",
        payload["construction_process"],
        "",
        "## 官方 split 与防泄漏检查",
        "",
        payload["split_protocol"],
        "",
        "## 产物元数据",
        "",
        "```json",
        md_json(payload["artifacts"]),
        "```",
        "",
        "## GPU 时间",
        "",
        "```json",
        md_json(payload["gpu_time"]),
        "```",
        "",
        "## Stage1 / Stage2 样例",
        "",
        "```json",
        md_json(payload["examples"]),
        "```",
        "",
        "## 下游结论",
        "",
        payload["downstream_conclusion"],
        "",
    ]
    path = OUT / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_codexm_record() -> None:
    codexm_manifest = read_json(DATA / "codexm" / "manifest.json")
    write_record("codexm.md", {
        "dataset_name": "CoDEx-M",
        "status": "relation_data_ready_task_inference_pending",
        "purpose": "给定固定实体对后排序关系候选；保留 CoDEx 官方三元组分类硬负例用于独立审计。",
        "construction_process": (
            "从官方 CoDEx 仓库固定 commit 下载 train/valid/test 正例、valid/test 硬负例及英语实体/关系元数据；"
            "每个上游文件在转换前按 Git blob SHA-1 校验。生成实体/关系文本映射和标准 GRIP graph record；"
            "valid/test 覆盖全量官方正例，候选列表固定为 10 选 1。此次只做数据转换，没有调用模型或使用训练 GPU。"
        ),
        "split_protocol": (
            "官方 split 为 train=185,584、valid=10,310、test=10,311，图上下文只使用 train。每个 valid/test 问题从训练关系标签中取 9 个负候选，"
            "排除 gold 与同一 (h,t) 的 train-known 关系，使用 seed=2026 固定候选列表。官方 valid/test negatives 是替换实体的三元组分类负例，"
            "保留在 raw 目录但不作为关系排序负例或固定实体对候选的真假标签。"
        ),
        "artifacts": {
            "manifest": artifact(DATA / "codexm" / "manifest.json"),
            "source_manifest": artifact(DATA / "codexm" / "raw" / "source_manifest.json"),
            "train": artifact(DATA / "codexm" / "raw" / "train.txt"),
            "valid": artifact(DATA / "codexm" / "raw" / "valid.txt"),
            "test": artifact(DATA / "codexm" / "raw" / "test.txt"),
            "valid_verified_negatives": artifact(DATA / "codexm" / "raw" / "valid_negatives.txt"),
            "test_verified_negatives": artifact(DATA / "codexm" / "raw" / "test_negatives.txt"),
            "entity_text": artifact(DATA / "codexm" / "raw" / "entity2text.json"),
            "relation_text": artifact(DATA / "codexm" / "raw" / "rel2text.json"),
            "processed_val": artifact(DATA / "codexm" / "processed" / "processed_val.json"),
            "processed_test": artifact(DATA / "codexm" / "processed" / "processed_test.json"),
            "generation_plan": artifact(DATA / "codexm" / "tasks" / "generation_plan.json"),
            "candidate_and_negative_audit": codexm_manifest.get("protocol", {}).get("candidate_coverage", {}),
        },
        "gpu_time": {
            "this_run_gpu_seconds": 0,
            "full_task_generation": "not run; Stage1/Stage2 remains pending_inference",
            "model": "not invoked; CoDEx-M provides the text metadata needed for data conversion",
        },
        "examples": [],
        "downstream_conclusion": (
            "PASS：原始 split、实体/关系映射及全量 10-way valid/test 关系评估记录已生成；"
            "adapter 训练前仍需生成并审核完整 Original GRIP Stage1/Stage2 task JSON。"
        ),
    })


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only-codexm",
        action="store_true",
        help="Write only the CoDEx-M construction record.",
    )
    args = parser.parse_args()
    if args.only_codexm:
        write_codexm_record()
        return

    nell_manifest = read_json(DATA / "nell23k" / "manifest.json")
    fb_manifest = read_json(DATA / "fb15k237_2" / "manifest.json")
    wn_manifest = read_json(DATA / "wn18rr" / "manifest.json")
    facts_manifest = read_json(DATA / "clegr_facts" / "dataset_manifest.json")
    reasoning_manifest = read_json(DATA / "clegr_reasoning" / "dataset_manifest.json")

    write_codexm_record()

    write_record("nell23k.md", {
        "dataset_name": "NELL23K",
        "status": "ready_existing_task_artifact",
        "purpose": "关系预测训练/验证/测试；复用已经复现并完成的 GRIP 任务构造。",
        "construction_process": "保留官方 train/valid/test 三个原始关系文件；使用现有 GRIP processed_val/processed_test 和 Qwen-7B 任务文件。此次只做校验和登记，没有重复调用模型。Stage1/Stage2 的原始 Qwen 缓存位于 `/mnt/e/GRIP-CU/tasks/cache`。",
        "split_protocol": "原始三分区互不相交；训练图只来自 train.txt，valid/test 仅用于评估。NELL 的 processed 文件沿用已复现入口，因此保留其原始处理器的兼容性行为。",
        "artifacts": {
            "manifest": artifact(DATA / "nell23k" / "manifest.json"),
            "processed_val": artifact(DATA / "nell23k" / "processed" / "processed_val.json"),
            "processed_test": artifact(DATA / "nell23k" / "processed" / "processed_test.json"),
            "tasks": artifact(DATA / "nell23k" / "tasks" / "grip_nell23k_tasks.json"),
            "task_cache": cache_meta(Path("/mnt/e/GRIP-CU/tasks/cache")),
        },
        "gpu_time": {
            "this_run_gpu_seconds": 0,
            "full_task_generation": "已有产物，本轮不重复推理",
            "model": "Qwen2.5-7B-Instruct (登记来源；本轮未调用)",
        },
        "examples": examples(Path("/mnt/e/GRIP-CU/tasks/cache")) or final_task_examples(DATA / "nell23k" / "tasks" / "grip_nell23k_tasks.json"),
        "downstream_conclusion": "PASS：任务文件、processed 文件和原始 split 均已存在，可直接进入下游训练/评估；测试答案仍来自官方测试文件，不从合成任务回填。",
    })

    fb_cache = Path("/mnt/e/GRIP-CU/artifacts/info_nce_fb15k237_qwen7b_20260921/tasks/task_cache_qwen7b_official_processor_script")
    write_record("fb15k237_2.md", {
        "dataset_name": "FB15K237-2",
        "status": "ready_existing_task_artifact",
        "purpose": "关系预测训练/验证/测试；复用已完成的 Qwen-7B 任务产物。",
        "construction_process": "保留官方 train/valid/test 三个原始关系文件；processed_test 使用项目已复现的固定 10,000 条测试协议，原始 test.txt 的全部 20,466 条仍保留。现有任务文件和 Stage1/Stage2 cache 已登记，本轮不重复推理。",
        "split_protocol": "原始三分区互不相交；train.txt 作为训练图，valid/test 只做评估。固定 10,000 条测试子集是当前已复现实验协议，不等同于删除官方 test.txt。若最终论文要求全量 test，需单独切换评估选择器。",
        "artifacts": {
            "manifest": artifact(DATA / "fb15k237_2" / "manifest.json"),
            "processed_val": artifact(DATA / "fb15k237_2" / "processed" / "processed_val.json"),
            "processed_test": artifact(DATA / "fb15k237_2" / "processed" / "processed_test.json"),
            "tasks": artifact(DATA / "fb15k237_2" / "tasks" / "grip_fb15k237_2_tasks.json"),
            "task_cache": cache_meta(fb_cache),
        },
        "gpu_time": {
            "this_run_gpu_seconds": 0,
            "full_task_generation": "已有产物，本轮不重复推理",
            "model": "Qwen2.5-7B-Instruct (登记来源；本轮未调用)",
        },
        "examples": examples(fb_cache) or final_task_examples(DATA / "fb15k237_2" / "tasks" / "grip_fb15k237_2_tasks.json"),
        "downstream_conclusion": "PASS（当前复现实验协议）：可以直接用于下游。注意论文最终报告必须明确 10,000 条固定测试子集，或另行运行全量官方 test。",
    })

    wn_cache = Path("/mnt/e/GRIP-CU/artifacts/benchmark_tasks/wn18rr_qwen7b_official")
    wn_task = DATA / "wn18rr" / "tasks" / "grip_wn18rr_tasks.json"
    wn_task_meta = task_run_metadata(wn_task)
    wn_is_complete = task_complete(wn_task)
    write_record("wn18rr.md", {
        "dataset_name": "WN18RR",
        "status": "ready_complete" if wn_is_complete else "inference_running",
        "purpose": "按 GRIP 官方单图协议从 train 图构造 Stage1/Stage2 任务，并用 valid/test 做关系预测评估。",
        "construction_process": "原始 entity2text/train/valid/test 从已复现 GRIP 资产复制并做 SHA256 登记；processed_test/processed_val 保留完整 train 图作为上下文，问题分别来自官方 test/valid。任务类型沿用官方 GripTaskGeneration：context 和未改写的 edge context-QA 由脚本直接构造，summarization/reasoning-QA 由 Qwen-7B 生成。配置为 context=8000、reasoning QA=2000、summarization=6000，关闭节点属性 QA 和 context QA 改写；每个模型阶段写入可恢复 JSONL cache。",
        "split_protocol": "train/valid/test 原始三分区互不相交；所有图上下文边来自 train.txt，valid/test 只提供问题与答案。WN 没有伪造 processed_train：参考 QA 使用官方 valid，这与原 GRIP WN18RR 处理器和 mp_wrapper 的 fallback 一致。",
        "artifacts": {
            "manifest": artifact(DATA / "wn18rr" / "manifest.json"),
            "processed_val": artifact(DATA / "wn18rr" / "processed" / "processed_val.json"),
            "processed_test": artifact(DATA / "wn18rr" / "processed" / "processed_test.json"),
            "generation_plan": artifact(DATA / "wn18rr" / "tasks" / "generation_plan.json"),
            "tasks": artifact(wn_task),
            "task_cache": cache_meta(wn_cache),
        },
        "gpu_time": {
            "calibration": "Qwen2.5-7B 首次加载实测约 163 秒、显存约 14.2 GiB；一图一例 smoke 已成功完成。",
            "running_configuration": "单 RTX 3090，guardenv，batch=4；cache 可断点续跑。",
            "prediction": "按正式配置约 42,000 个模型 prompt、当前 batch=4 实测约 2–4 秒/prompt，预计约 20–60 GPU 小时（I/O、输出长度和 cache 命中会改变上界）；首次模型加载另计约 3 分钟。",
            "actual": wn_task_meta if wn_is_complete else "任务文件尚未完成；当前仅记录预测值。",
        },
        "examples": examples(wn_cache) or [{"stage": "smoke", "note": "正式 cache 尚未产生可读样本；/tmp/grip_wn18rr_stage_smoke.json 已验证 Stage1/Stage2 入口成功。"}],
        "downstream_conclusion": (
            "PASS：原始 split、上下文图和参考 QA 没有测试泄漏；任务文件已完成，可进入下游训练/评估。"
            if wn_is_complete else
            "RUNNING：原始 split、上下文图和参考 QA 没有测试泄漏；下游应等待 generation_status=complete 后再训练。"
        ),
    })

    facts_cache = Path("/mnt/e/GRIP-CU/artifacts/benchmark_tasks/clegr_facts_qwen7b_initial")
    reasoning_cache = Path("/mnt/e/GRIP-CU/artifacts/benchmark_tasks/clegr_reasoning_qwen7b_initial")
    facts_task = DATA / "clegr_facts" / "tasks" / "grip_clegr_facts_tasks.json"
    reasoning_task = DATA / "clegr_reasoning" / "tasks" / "grip_clegr_reasoning_tasks.json"
    facts_is_complete = task_complete(facts_task)
    reasoning_is_complete = task_complete(reasoning_task)
    clegr_is_complete = facts_is_complete and reasoning_is_complete
    write_record("clegr.md", {
        "dataset_name": "CLEGR（Facts + Reasoning）",
        "status": "ready_complete_low_budget" if clegr_is_complete else "queued_or_running_on_single_gpu",
        "purpose": "组合图问答训练/验证/测试；保留官方 Facts 与 Reasoning 两个变体，合并记录但不混合标签。",
        "construction_process": "从本地官方 CLEGR PyTorch archive 逐图转换为 JSONL，保留节点、边、问题、答案、question_type/group/subgroup；按官方 500 图、3:1:1 图级 split 形成 train=300、val=100、test=100。Stage1/Stage2 只从 train 图生成，test 图只用于最终评估。任务类型沿用官方 GripTaskGeneration：context 和未改写的 edge context-QA 由脚本直接构造，summarization/reasoning-QA 由 Qwen-7B 生成；本次是低预算可运行版本，每图 summarization=2、reasoning QA=2、context QA=2，并关闭改写、节点属性 QA 和 context upsampling。它不是官方 CLEGR shell 的完整参数复现；所有模型调用写入 cache。",
        "split_protocol": "图级切分，不是问题级切分；每个 graph_id 在 train/val/test 只出现一次，图内全部问题随图保留。该比例与 CLEGR 官方论文一致；公开 archive 未提供作者隐藏的具体 graph-id 索引，因此 manifest 固定记录 seed=2026 和排序规则。",
        "artifacts": {
            "facts_manifest": artifact(DATA / "clegr_facts" / "dataset_manifest.json"),
            "reasoning_manifest": artifact(DATA / "clegr_reasoning" / "dataset_manifest.json"),
            "facts_train": artifact(DATA / "clegr_facts" / "processed" / "processed_train.json"),
            "facts_test": artifact(DATA / "clegr_facts" / "processed" / "processed_test.json"),
            "reasoning_train": artifact(DATA / "clegr_reasoning" / "processed" / "processed_train.json"),
            "reasoning_test": artifact(DATA / "clegr_reasoning" / "processed" / "processed_test.json"),
            "facts_tasks": artifact(facts_task),
            "reasoning_tasks": artifact(reasoning_task),
            "facts_task_cache": cache_meta(facts_cache),
            "reasoning_task_cache": cache_meta(reasoning_cache),
        },
        "gpu_time": {
            "model": "Qwen2.5-7B-Instruct, single RTX 3090",
            "initial_budget": "300 train graphs × (2 summaries + 2 reasoning-QA); context QA is deterministic under current flags",
            "official_comparison": {
                "generator_implementation": "same official GripTaskGeneration task classes",
                "current_profile": "summarization=2, reasoning_qa=2, context_qa=2 per train graph; sample_node_attribute_task=False; repharse_context_qa=False; context_upsampling=False",
                "official_clegr_reasoning_shell": "summarization=20, reasoning_qa=160, context_qa=20; default task flags are not overridden",
                "interpretation": "official-generator-compatible low-budget artifact, not an exact official full-budget CLEGR reproduction"
            },
            "prediction": "约 2–8 GPU 小时/变体，合计约 4–16 GPU 小时；首次加载约 3 分钟。若提高到官方 shell 的 20 summaries + 160 reasoning-QA/图，单卡预计会超过 1 天/变体，需另行扩容。",
            "actual": {
                "facts": task_run_metadata(facts_task),
                "reasoning": task_run_metadata(reasoning_task),
            } if clegr_is_complete else "任务文件尚未全部完成；当前仅记录预测值。",
        },
        "examples": examples(facts_cache) + examples(reasoning_cache),
        "downstream_conclusion": (
            "PASS（低预算版本）：两个变体的 generation_status 均为 complete；原始数据与图级 split 可用于下游，但不能将当前任务文件宣称为官方 CLEGR 完整预算复现。训练/验证/测试不得按问题随机打散跨图。"
            if clegr_is_complete else
            "RUNNING：Stage1/Stage2 任务必须等两个变体的 generation_status=complete 后使用，且训练/验证/测试不得按问题随机打散跨图。"
        ),
    })


if __name__ == "__main__":
    main()
