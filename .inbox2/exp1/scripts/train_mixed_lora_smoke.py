#!/usr/bin/env python3
"""Interleaved two-adapter LoRA smoke test for the GRIP stage-interference study.

This is an engineering smoke test, not a quality experiment: small bounded samples,
short sequence length, deterministic S1/S2 alternation, resumable step checkpoints.
"""
import argparse
import json
import os
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--input", type=Path, default=ROOT / "inputs/nell23k_smoke_16_each.json")
    p.add_argument("--run-id", default="qwen05b_mixed_lora_smoke")
    p.add_argument("--max-steps", type=int, default=8)
    p.add_argument("--save-steps", type=int, default=2)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def get_latest_checkpoint(path: Path):
    checkpoints = []
    for item in path.glob("checkpoint-*"):
        try:
            checkpoints.append((int(item.name.rsplit("-", 1)[1]), item))
        except ValueError:
            pass
    return max(checkpoints, default=(0, None))[1]


def main():
    args = parse_args()
    run_dir = ROOT / "runs" / args.run_id
    if run_dir.exists() and not args.resume:
        raise FileExistsError(f"Run exists; choose another --run-id or use --resume: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "smoke_metrics.jsonl"
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing CPU fallback")
    if not args.model.is_dir() or not (args.model / "config.json").is_file():
        raise FileNotFoundError(f"Expected complete local model directory: {args.model}")

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    samples = {
        "stage1": payload["context_samples"],
        "stage2": payload["qa_samples"],
    }
    if not all(samples.values()):
        raise ValueError("Both stage sample sets must be non-empty")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, local_files_only=True,
        low_cpu_mem_usage=True, device_map={"": 0}
    )
    config = LoraConfig(
        r=4, lora_alpha=8, lora_dropout=0.0, bias="none",
        target_modules=["down_proj", "up_proj", "gate_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config, adapter_name="stage1")
    model.add_adapter("stage2", config)
    # Keep both adapters in the optimizer parameter list; set_adapter toggles gradients
    # so only the selected adapter receives gradients on each alternating step.
    for name, parameter in model.named_parameters():
        if "lora_A.stage1" in name or "lora_B.stage1" in name or "lora_A.stage2" in name or "lora_B.stage2" in name:
            parameter.requires_grad_(True)
    model.config.use_cache = False
    model.print_trainable_parameters()

    ckpt = get_latest_checkpoint(run_dir) if args.resume else None
    start_step = int(ckpt.name.rsplit("-", 1)[1]) if ckpt else 0
    if ckpt:
        state_path = ckpt / "trainer_state.json"
        if state_path.is_file():
            start_step = int(json.loads(state_path.read_text())["global_step"])
        for adapter in ("stage1", "stage2"):
            adapter_state = ckpt / f"{adapter}_adapter.pt"
            if not adapter_state.is_file():
                raise FileNotFoundError(f"Incomplete checkpoint: {adapter_state}")
            set_peft_model_state_dict(
                model, torch.load(adapter_state, map_location="cpu", weights_only=True),
                adapter_name=adapter,
            )

    optim = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
    )
    if ckpt and (ckpt / "optimizer.pt").is_file():
        optim.load_state_dict(torch.load(ckpt / "optimizer.pt", map_location="cpu", weights_only=False))
    began = time.time()
    model.train()
    for step in range(start_step, args.max_steps):
        phase = "stage1" if step % 2 == 0 else "stage2"
        model.set_adapter(phase)
        model.enable_adapter_layers()
        text = samples[phase][(step // 2) % len(samples[phase])]
        encoded = tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=args.max_length, padding=False,
        )
        input_ids = encoded["input_ids"].to("cuda")
        attention_mask = encoded["attention_mask"].to("cuda")
        optim.zero_grad(set_to_none=True)
        output = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = output.loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step + 1}: {loss.item()}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optim.step()
        record = {
            "step": step + 1, "phase": phase, "loss": float(loss.detach().cpu()),
            "elapsed_seconds": round(time.time() - began, 3),
            "peak_cuda_memory_gib": round(torch.cuda.max_memory_allocated() / 2**30, 4),
        }
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        print(json.dumps(record), flush=True)
        if (step + 1) % args.save_steps == 0 or step + 1 == args.max_steps:
            checkpoint = run_dir / f"checkpoint-{step + 1}"
            checkpoint.mkdir(parents=True, exist_ok=True)
            for adapter in ("stage1", "stage2"):
                torch.save(get_peft_model_state_dict(model, adapter_name=adapter), checkpoint / f"{adapter}_adapter.pt")
                (run_dir / "adapters" / adapter).mkdir(parents=True, exist_ok=True)
                model.save_pretrained(run_dir / "adapters" / adapter, selected_adapters=[adapter])
            torch.save(optim.state_dict(), checkpoint / "optimizer.pt")
            state_tmp = checkpoint / "trainer_state.json.tmp"
            state_tmp.write_text(json.dumps({"global_step": step + 1, "seed": args.seed}, indent=2))
            state_tmp.replace(checkpoint / "trainer_state.json")
            model.save_pretrained(run_dir / "adapters", selected_adapters=["stage1", "stage2"])

    # Check that both adapters can be activated and composed as a sum of deltas.
    model.eval()
    probe = tokenizer("Answer briefly.", return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = {}
        for adapter in ("stage1", "stage2"):
            model.set_adapter(adapter)
            outputs[adapter] = model(**probe).logits[:, -1, :].float().cpu()
        model.add_weighted_adapter(
            adapters=["stage1", "stage2"], weights=[1.0, 1.0],
            adapter_name="combined_sum", combination_type="cat",
        )
        model.set_adapter("combined_sum")
        outputs["combined_sum"] = model(**probe).logits[:, -1, :].float().cpu()
    if not all(torch.isfinite(value).all() for value in outputs.values()):
        raise FloatingPointError("non-finite logits while testing adapter activation/combination")
    summary = {
        "run_id": args.run_id, "status": "smoke_passed", "steps": args.max_steps,
        "resumed_from_step": start_step, "cuda_device": torch.cuda.get_device_name(0),
        "cuda_memory_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2),
        "bf16": torch.cuda.is_bf16_supported(),
        "phase_order": ["stage1" if s % 2 == 0 else "stage2" for s in range(args.max_steps)],
        "adapter_outputs_finite": True,
        "combined_adapter": "cat sum of low-rank deltas via PEFT",
        "elapsed_seconds": round(time.time() - began, 3),
    }
    tmp = run_dir / "summary.json.tmp"
    tmp.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(run_dir / "summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
