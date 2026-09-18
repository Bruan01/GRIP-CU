#!/usr/bin/env python3
"""从 grip_nell23k_tasks.json 的 qa_samples 中提取 DPO 格式数据集。

输出 JSONL，每行:
  {"prompt": "...", "chosen": "...", "rejected": "..."}
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task_json",
        type=Path,
        required=True,
        help="grip_nell23k_tasks.json 路径",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="输出 JSONL 路径",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="限制输出条数（冒烟用 None=全量）",
    )
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def extract_dpo_pairs(sample: str, rng: random.Random) -> dict | None:
    """从一条 qa_sample 完整 chat 文本中拆分 prompt/chosen/rejected。

    qa_samples 格式:
        <|im_start|>system...<|im_end|>
        <|im_start|>user...<|im_end|>
        <|im_start|>assistant\n<answer>xxx</answer><|im_end|>
    """
    # 提取 assistant 部分作为 chosen
    asst_tag = "<|im_start|>assistant\n"
    asst_end = "<|im_end|>"
    asst_start_idx = sample.find(asst_tag)
    if asst_start_idx == -1:
        return None
    answer_start = asst_start_idx + len(asst_tag)
    answer_end = sample.find(asst_end, answer_start)
    if answer_end == -1:
        return None
    chosen_full = sample[answer_start:answer_end].strip()
    # 从 <answer> 标签中提取纯答案文本
    tag_start = chosen_full.find("<answer>")
    tag_end = chosen_full.find("</answer>")
    chosen_answer = chosen_full[tag_start + len("<answer>"):tag_end] if tag_start != -1 and tag_end != -1 else chosen_full

    # prompt = system + user（去掉 assistant 部分）
    prompt = sample[:asst_start_idx].rstrip()

    # 构造一个 rejected：用随机的无关答案
    # 从常见 relation 词汇池中选取错误的答案
    # 由于我们没有该问的 candidate_relations，我们用固定的全量关系池
    # 注意：这只是一个冒烟测试用的简化方案
    return {
        "prompt": prompt,
        "chosen": chosen_full,
        "rejected": None,  # 将在后面填充
        "_chosen_answer": chosen_answer,
    }


def build_dpo_dataset(
    qa_samples: list[str],
    rng: random.Random,
    max_samples: int | None,
) -> list[dict]:
    """构建 DPO 数据集。"""

    # 先提取所有样本，收集所有答案
    entries = []
    for sample in qa_samples:
        pair = extract_dpo_pairs(sample, rng)
        if pair is not None:
            entries.append(pair)

    if max_samples and len(entries) > max_samples:
        rng.shuffle(entries)
        entries = entries[:max_samples]

    # 收集所有答案词汇用于构造 rejected
    all_answers = [e["_chosen_answer"] for e in entries]
    unique_answers = list(dict.fromkeys(all_answers))  # 去重保序
    # 过滤掉空字符串
    unique_answers = [a for a in unique_answers if a.strip()]

    # 为每个 entry 分配一个 rejected
    for entry in entries:
        chosen = entry["_chosen_answer"]
        # 选一个不同的 answer 作为 rejected
        pool = [a for a in unique_answers if a != chosen]
        if pool:
            entry["rejected"] = rng.choice(pool)
        else:
            entry["rejected"] = "<unknown>"
        # 清理临时字段
        del entry["_chosen_answer"]

    return entries


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    # 读取 task json
    payload = json.loads(args.task_json.read_text(encoding="utf-8"))
    qa_samples = payload.get("qa_samples", [])
    print(f"qa_samples 总量: {len(qa_samples)}")

    if not qa_samples:
        # 尝试其他可能的 key
        for key in ("samples", "data", "items"):
            if key in payload:
                qa_samples = payload[key]
                print(f"fallback 到 key '{key}': {len(qa_samples)}")
                break

    entries = build_dpo_dataset(qa_samples, rng, args.max_samples)
    print(f"生成 DPO pairs: {len(entries)}")
    if entries:
        print(f"第 1 条 prompt 预览 (前 120 字): {entries[0]['prompt'][:120]}")
        print(f"第 1 条 chosen: {entries[0]['chosen'][:80]}")
        print(f"第 1 条 rejected: {entries[0]['rejected'][:80]}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"写入: {args.output}")


if __name__ == "__main__":
    main()