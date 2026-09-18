#!/usr/bin/env python3
"""DPO 冒烟测试：在 Stage-2 listed adapter 上做 5 步 DPO 训练，验证流程跑通。

用法:
  # 冒烟（5 步）
  TIME_LIMIT_MIN=3 bash configs/run_dpo_smoke.sh

  # 全量（12k 样本，~10 epoch）
  SCALE=full bash configs/run_dpo_smoke.sh

依赖: pip install trl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from torch.utils.data import Dataset
from transformers import TrainingArguments, AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer

# 复用项目路径
HNG = Path(__file__).resolve().parents[1]
VERSION_DIR = Path(
    "/home/ubuntu2/linkc/ltw-lkc/GRIP-CU/08_experiments/RecurrentGRIP/"
    "v1_1_nell23k_first_2026-08-29"
)
GRIP_EXP = VERSION_DIR / "grip-exp"
sys.path.insert(0, str(GRIP_EXP))
sys.path.insert(0, str(HNG / "src"))

from constants import HF_DECODER_ONLY_LLMS, TORCH_DTYPE  # noqa: E402
from models.utils import get_hf_llm_tokenizer, get_lora_model  # noqa: E402


class DPODataset(Dataset):
    """从 JSONL 加载 DPO 格式数据。"""

    def __init__(self, data_path: Path):
        self.entries = []
        with data_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.entries.append(json.loads(line))
        print(f"[DPO] loaded {len(self.entries)} entries from {data_path}")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        return self.entries[idx]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dpo_data", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--s2_adapter", type=Path, default=None,
                        help="Stage-2 listed adapter 目录。None = 从基座直接挂新 LoRA")
    parser.add_argument("--model_name", default="qwen-0.5b")
    parser.add_argument("--model_cache_dir", default="model_cache")
    parser.add_argument("--lora_r", type=int, default=4)
    parser.add_argument("--lora_alpha", type=int, default=8)
    parser.add_argument("--target_modules", nargs="+", default=["down_proj", "up_proj", "gate_proj"])
    parser.add_argument("--beta", type=float, default=0.1,
                        help="DPO beta 温度参数")
    parser.add_argument("--max_steps", type=int, default=5,
                        help="冒烟步数; 全量设为 0 会用 num_train_epochs")
    parser.add_argument("--num_train_epochs", type=float, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--max_prompt_length", type=int, default=1024)
    parser.add_argument("--save_steps", type=int, default=10)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--skip_train", action="store_true")
    return parser.parse_args()


def release_cuda() -> None:
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def load_model_with_adapter(args: argparse.Namespace):
    """加载模型。如果有 s2_adapter，先加载 adapter 再 merge 后挂新 LoRA。"""
    model_id = HF_DECODER_ONLY_LLMS[args.model_name]
    cache_dir = str(Path(args.model_cache_dir).expanduser().resolve())

    if args.s2_adapter is not None:
        print(f"[load] loading s2 adapter from {args.s2_adapter}")
        # 用 PeftModel 加载已有 adapter
        base, tokenizer = get_hf_llm_tokenizer(
            model_name=model_id,
            model_source="local",
            model_cache_dir=cache_dir,
            local_files_only=True,
            dtype=TORCH_DTYPE["bfloat16"],
            peft=False,
            device_map=None,
        )
        model = PeftModel.from_pretrained(base, str(args.s2_adapter), is_trainable=True)
        # merge adapter 到 base，然后挂新的 LoRA 做 DPO
        model = model.merge_and_unload()
        print("[load] merged s2 adapter. now attaching new LoRA for DPO...")
        model = get_lora_model(
            model,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=args.target_modules,
        )
    else:
        print("[load] loading base model + fresh LoRA for DPO")
        model, tokenizer = get_hf_llm_tokenizer(
            model_name=model_id,
            model_source="local",
            model_cache_dir=cache_dir,
            local_files_only=True,
            dtype=TORCH_DTYPE["bfloat16"],
            peft=True,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=args.target_modules,
        )

    if torch.cuda.is_available():
        model.to("cuda")
    model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    return model, tokenizer


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存配置
    config = vars(args)
    config["executed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    (output_dir / "dpo_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    adapter_dir = output_dir / "dpo_adapter"
    if adapter_dir.is_dir() and args.skip_train:
        print(f"[skip] adapter already exists: {adapter_dir}")
        return

    # 加载模型
    release_cuda()
    model, tokenizer = load_model_with_adapter(args)

    # 加载 DPO 数据集
    dataset = DPODataset(args.dpo_data)
    if len(dataset) == 0:
        print("error: empty DPO dataset", file=sys.stderr)
        sys.exit(1)

    # 确保 tokenizer 有 pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 构建 TrainingArguments
    train_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=0,
        max_steps=args.max_steps if args.max_steps > 0 else 0,
        num_train_epochs=args.num_train_epochs if args.max_steps == 0 else 0,
        logging_steps=1,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        remove_unused_columns=False,
        seed=args.seed,
        data_seed=args.seed,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_pin_memory=False,
        ddp_find_unused_parameters=False,
        report_to="none",
    )

    # DPOTrainer
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None,
        beta=args.beta,
        args=train_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
    )

    # 训练
    print(f"[DPO] starting training — beta={args.beta}, max_steps={args.max_steps}, dataset={len(dataset)}")
    train_result = dpo_trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    print(f"[DPO] training done: {train_result}")

    # 保存 adapter
    adapter_dir.mkdir(parents=True, exist_ok=True)
    dpo_trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"[DPO] adapter saved to {adapter_dir}")


if __name__ == "__main__":
    main()