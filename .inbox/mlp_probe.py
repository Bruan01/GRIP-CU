#!/usr/bin/env python3
"""单层 MLP 探针冒烟测试。

目标：检验 GRIP LoRA 是否让模型中间层表示包含“指定图事实是否存在”的信息。
探针样本由 build_mlp_probe_data.py 从 train.txt 构造；正样本是真实有向边，负样本
固定 source/relation 并替换 target，且替换后的三元组不存在。

实验协议：
1. 构造明确询问路径是否存在的文本 x，标签 y∈{0,1} 由训练图 BFS 结果提供；
2. 在指定中间层提取 adapter ON 的最后一个有效 token 隐状态 h；
3. 只使用 ON/train 训练一个小型两层 MLP 探针；
4. 使用同一个冻结探针分别测试 ON/test 和 OFF/test；
5. 以 ON-OFF 的 balanced accuracy 差值作为 LoRA 信息增量的初步证据。

注意：该实验只证明标签能否从某层表示中被小型非线性探针解码，不直接证明模型
执行了图算法。ON/OFF 共享同一个探针，避免两个独立探针能力不同造成混淆。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def load_records(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"输入文件不存在: {path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    payload = json.loads(text)
    return payload if isinstance(payload, list) else [payload]


def make_probe_samples(record: dict) -> list[dict]:
    """从图记录生成路径可达性探针样本。"""
    title = str(record.get("title", "nell23k"))
    samples: list[dict] = []
    for item in record.get("recurrent_questions", []):
        source = item.get("source_node")
        target = item.get("target_node")
        split = item.get("split")
        if source is None or target is None or split not in {"train", "validation", "test"}:
            continue
        label = int(bool(item.get("structural_reachable", False)))
        text = (
            f"In context graph {title}, is there any path between word node "
            f"{source} and word node {target}? Answer yes or no."
        )
        samples.append(
            {
                "question_id": item.get("question_id"),
                "text": text,
                "label": label,
                "split": split,
                "source_node": source,
                "target_node": target,
                "structural_distance": item.get("structural_distance"),
            }
        )
    if not samples:
        raise ValueError("没有可用探针样本")
    return samples


def normalize_probe_samples(records: list[dict]) -> list[dict]:
    """读取构造脚本输出的 JSONL，也兼容旧的图记录 JSON。"""
    if records and {"text", "label", "split"}.issubset(records[0]):
        samples = []
        for item in records:
            sample = dict(item)
            sample["label"] = int(sample["label"])
            if sample["label"] not in {0, 1}:
                raise ValueError(f"探针标签必须是 0/1: {sample.get('question_id')}")
            samples.append(sample)
        return samples
    return make_probe_samples(records[0])


def balanced_subset(
    samples: list[dict], max_per_class: int, seed: int
) -> list[dict]:
    """对每个标签独立采样，避免准确率被多数类支配。"""
    rng = random.Random(seed)
    groups: dict[int, list[dict]] = {0: [], 1: []}
    for sample in samples:
        groups[int(sample["label"])].append(sample)
    if not groups[0] or not groups[1]:
        raise ValueError(
            f"二分类数据不完整: negative={len(groups[0])}, positive={len(groups[1])}"
        )
    selected: list[dict] = []
    for label in (0, 1):
        group = groups[label]
        count = len(group) if max_per_class <= 0 else min(max_per_class, len(group))
        selected.extend(rng.sample(group, count))
    rng.shuffle(selected)
    return selected


def last_valid_token(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """兼容左/右 padding，返回每条序列最后一个有效 token 的表示。"""
    positions = torch.arange(attention_mask.shape[1], device=hidden.device)
    positions = positions.unsqueeze(0).expand_as(attention_mask)
    last_indices = positions.masked_fill(attention_mask == 0, -1).max(dim=1).values
    if torch.any(last_indices < 0):
        raise ValueError("遇到完全为空的 token 序列")
    batch_indices = torch.arange(hidden.shape[0], device=hidden.device)
    return hidden[batch_indices, last_indices]


@torch.inference_mode()
def extract_features(
    model,
    tokenizer,
    samples: list[dict],
    layer_index: int,
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    texts = [sample["text"] for sample in samples]
    feature_batches: list[torch.Tensor] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        # hidden_states[0] 是 embedding；第 i 层输出位于 i+1。
        layer_hidden = outputs.hidden_states[layer_index + 1]
        pooled = last_valid_token(layer_hidden, attention_mask)
        feature_batches.append(pooled.float().cpu())
    features = torch.cat(feature_batches, dim=0)
    labels = torch.tensor([sample["label"] for sample in samples], dtype=torch.long)
    return features, labels


class MLPProbe(nn.Module):
    """容量受限的两层非线性探针。"""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def standardize(
    train_x: torch.Tensor, *others: torch.Tensor
) -> tuple[torch.Tensor, ...]:
    """只用训练集统计量标准化，避免测试集信息泄漏。"""
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True).clamp_min(1e-6)
    return tuple((x - mean) / std for x in (train_x, *others))


def metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    predictions = logits.argmax(dim=-1)
    accuracy = (predictions == labels).float().mean().item()
    recalls: list[float] = []
    for label in (0, 1):
        mask = labels == label
        recalls.append(
            (predictions[mask] == labels[mask]).float().mean().item() if mask.any() else 0.0
        )
    return {
        "accuracy": round(accuracy, 6),
        "balanced_accuracy": round(sum(recalls) / 2.0, 6),
        "negative_recall": round(recalls[0], 6),
        "positive_recall": round(recalls[1], 6),
    }


@torch.inference_mode()
def evaluate(
    probe: nn.Module, features: torch.Tensor, labels: torch.Tensor, device: torch.device
) -> dict[str, float]:
    probe.eval()
    logits = probe(features.to(device)).cpu()
    return metrics(logits, labels)


def train_probe(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    validation_x: torch.Tensor,
    validation_y: torch.Tensor,
    hidden_dim: int,
    dropout: float,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
    device: torch.device,
) -> tuple[MLPProbe, dict]:
    probe = MLPProbe(train_x.shape[1], hidden_dim, dropout).to(device)
    optimizer = torch.optim.AdamW(
        probe.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(2026)
    loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=min(batch_size, len(train_y)),
        shuffle=True,
        generator=generator,
    )

    best_state: dict[str, torch.Tensor] | None = None
    best_score = -1.0
    best_epoch = 0
    stale_epochs = 0
    history: list[dict] = []

    for epoch in range(1, max_epochs + 1):
        probe.train()
        total_loss = 0.0
        seen = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(probe(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_y)
            seen += len(batch_y)

        validation_metrics = evaluate(probe, validation_x, validation_y, device)
        score = validation_metrics["balanced_accuracy"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(total_loss / max(seen, 1), 6),
                "validation": validation_metrics,
            }
        )
        if score > best_score + 1e-6:
            best_score = score
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in probe.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is None:
        raise RuntimeError("MLP 探针没有产生有效 checkpoint")
    probe.load_state_dict(best_state)
    probe.to(device)
    return probe, {
        "best_epoch": best_epoch,
        "best_validation_balanced_accuracy": best_score,
        "epochs_ran": len(history),
        "history": history,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--code-dir", default=None)
    parser.add_argument("--model-name", default="qwen-0.5b")
    parser.add_argument("--model-cache-dir", default="model_cache")
    parser.add_argument("--layer-index", type=int, default=12)
    parser.add_argument("--max-train-per-class", type=int, default=100)
    parser.add_argument("--max-validation-per-class", type=int, default=30)
    parser.add_argument("--max-test-per-class", type=int, default=100)
    parser.add_argument("--extract-batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--probe-hidden-dim", type=int, default=64)
    parser.add_argument("--probe-dropout", type=float, default=0.2)
    parser.add_argument("--probe-learning-rate", type=float, default=1e-3)
    parser.add_argument("--probe-weight-decay", type=float, default=1e-4)
    parser.add_argument("--probe-batch-size", type=int, default=32)
    parser.add_argument("--probe-max-epochs", type=int, default=200)
    parser.add_argument("--probe-patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-file", default=None)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    project_root = Path(__file__).resolve().parent.parent
    code_dir = (
        Path(args.code_dir).resolve()
        if args.code_dir
        else project_root
        / "08_experiments/RecurrentGRIP/v1_1_nell23k_first_2026-08-29/grip-exp"
    )
    if not (code_dir / "constants.py").is_file():
        raise FileNotFoundError(f"GRIP 代码目录无效: {code_dir}")
    sys.path.insert(0, str(code_dir))

    from constants import HF_DECODER_ONLY_LLMS, TORCH_DTYPE
    from models.utils import get_hf_llm_tokenizer
    from peft import PeftModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    records = load_records(Path(args.input_file).resolve())
    samples = normalize_probe_samples(records)
    split_samples = {
        split: [sample for sample in samples if sample["split"] == split]
        for split in ("train", "validation", "test")
    }
    selected = {
        "train": balanced_subset(
            split_samples["train"], args.max_train_per_class, args.seed + 1
        ),
        "validation": balanced_subset(
            split_samples["validation"], args.max_validation_per_class, args.seed + 2
        ),
        "test": balanced_subset(
            split_samples["test"], args.max_test_per_class, args.seed + 3
        ),
    }
    for split, subset in selected.items():
        print(f"{split}: {len(subset)}，标签分布={dict(Counter(x['label'] for x in subset))}")

    model_id = HF_DECODER_ONLY_LLMS[args.model_name]
    base_model, tokenizer = get_hf_llm_tokenizer(
        model_name=model_id,
        model_source="local",
        model_cache_dir=str(code_dir / args.model_cache_dir),
        local_files_only=True,
        dtype=TORCH_DTYPE["bfloat16"],
        peft=False,
    )
    adapter_dir = Path(args.adapter_dir).resolve()
    if not adapter_dir.is_dir():
        raise FileNotFoundError(f"LoRA adapter 不存在: {adapter_dir}")
    model = PeftModel.from_pretrained(base_model, str(adapter_dir)).to(device)
    model.eval()

    num_layers = model.config.num_hidden_layers
    layer_index = args.layer_index if args.layer_index >= 0 else num_layers + args.layer_index
    if not 0 <= layer_index < num_layers:
        raise IndexError(f"layer-index={args.layer_index} 超出 0..{num_layers - 1}")
    print(f"探测层: L{layer_index}/{num_layers - 1}")

    on_features: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    off_features: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    print("提取 adapter ON 隐状态……")
    for split, subset in selected.items():
        on_features[split] = extract_features(
            model,
            tokenizer,
            subset,
            layer_index,
            args.extract_batch_size,
            args.max_length,
            device,
        )
    print("提取 adapter OFF 隐状态……")
    with model.disable_adapter():
        for split, subset in selected.items():
            off_features[split] = extract_features(
                model,
                tokenizer,
                subset,
                layer_index,
                args.extract_batch_size,
                args.max_length,
                device,
            )

    train_on_x, train_y = on_features["train"]
    validation_on_x, validation_y = on_features["validation"]
    test_on_x, test_y = on_features["test"]
    test_off_x, test_off_y = off_features["test"]
    train_on_x, validation_on_x, test_on_x, test_off_x = standardize(
        train_on_x, validation_on_x, test_on_x, test_off_x
    )

    print("训练单个 MLP 探针（仅使用 adapter ON/train）……")
    probe, training_summary = train_probe(
        train_on_x,
        train_y,
        validation_on_x,
        validation_y,
        args.probe_hidden_dim,
        args.probe_dropout,
        args.probe_learning_rate,
        args.probe_weight_decay,
        args.probe_batch_size,
        args.probe_max_epochs,
        args.probe_patience,
        device,
    )

    on_metrics = evaluate(probe, test_on_x, test_y, device)
    off_metrics = evaluate(probe, test_off_x, test_off_y, device)
    delta = round(
        on_metrics["balanced_accuracy"] - off_metrics["balanced_accuracy"], 6
    )
    result = {
        "experiment": "single_layer_mlp_probe_smoke",
        "label": "supplied_by_probe_jsonl",
        "model": args.model_name,
        "adapter_dir": str(adapter_dir),
        "layer_index": layer_index,
        "probe": {
            "architecture": f"Linear({train_on_x.shape[1]},{args.probe_hidden_dim})-ReLU-Dropout-Linear({args.probe_hidden_dim},2)",
            "trained_on": "adapter_ON/train_only",
            "same_probe_for_on_and_off": True,
        },
        "sample_counts": {split: len(value) for split, value in selected.items()},
        "training": training_summary,
        "test_adapter_ON": on_metrics,
        "test_adapter_OFF": off_metrics,
        "balanced_accuracy_delta_ON_minus_OFF": delta,
        "interpretation": (
            "positive_probe_signal" if delta >= 0.05 else "no_clear_probe_signal"
        ),
        "limitations": [
            "A positive delta shows nonlinear decodability, not causal graph execution.",
            "A negative result does not prove that LoRA stores no graph information.",
            "Entity-frequency or prompt-distribution shortcuts require additional controls.",
        ],
    }

    output_path = (
        Path(args.output_file).resolve()
        if args.output_file
        else adapter_dir.parent / f"mlp_probe_layer_{layer_index}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== 单层 MLP 探针结果 ===")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\n结果保存到: {output_path}")


if __name__ == "__main__":
    main()
