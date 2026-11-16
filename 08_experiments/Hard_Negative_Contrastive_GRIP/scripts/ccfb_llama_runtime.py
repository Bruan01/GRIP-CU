#!/usr/bin/env python3
"""Llama 3.1 support for the CCF B pair, isolated from active Qwen workers."""
import hashlib
import re
from pathlib import Path
import subprocess
import sys

MODEL_SHORT = "llama3-8b"
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
REVISION = "0e9e39f249a16976918f6564b8830bc894c89659"
MIRROR = "https://modelscope.cn/models/LLM-Research/Meta-Llama-3.1-8B-Instruct/resolve/master"
MODEL_FOLDER = MODEL_ID.replace("/", "--")
MAX_LENGTH = 4096
# File identities from the pinned Meta snapshot. Small files use Git blob IDs.
FILES = {
    "config.json": (855, "0bb6fd75b3ad2fe988565929f329945262c2814e"),
    "generation_config.json": (184, "cc7276afd599de091142c6ed3005faf8a74aa257"),
    "model.safetensors.index.json": (23950, "0fd8120f1c6acddc268ebc2583058efaf699a771"),
    "special_tokens_map.json": (296, "02ee80b6196926a5ad790a004d9efd6ab1ba6542"),
    "tokenizer.json": (9085657, "5cc5f00a5b203e90a27a3bd60d1ec393b07971e8"),
    "tokenizer_config.json": (55351, "db88166e2bc4c799fd5d1ae643b75e84d03ee70e"),
    "model-00001-of-00004.safetensors": (4976698672, "2b1879f356aed350030bb40eb45ad362c89d9891096f79a3ab323d3ba5607668"),
    "model-00002-of-00004.safetensors": (4999802720, "09d433f650646834a83c580877bd60c6d1f88f7755305c12576b5c7058f9af15"),
    "model-00003-of-00004.safetensors": (4915916176, "fc1cdddd6bfa91128d6e94ee73d0ce62bfcdb7af29e978ddcab30c66ae9ea7fa"),
    "model-00004-of-00004.safetensors": (1168138808, "92ecfe1a2414458b4821ac8c13cf8cb70aed66b5eea8dc5ad9eeb4ff309d6d7b"),
}
METADATA = tuple(name for name in FILES if name.endswith(".json"))
WEIGHTS = tuple(name for name in FILES if name.endswith(".safetensors"))
MESSAGE = re.compile(r"<\|im_start\|>(system|user|assistant)\n(.*?)<\|im_end\|>\n?", re.S)
LLAMA_ASSISTANT = "<|start_header_id|>assistant<|end_header_id|>\n\n"


def verify_file(path, name):
    size, expected = FILES[name]
    if path.stat().st_size != size:
        raise ValueError(f"Incomplete model file: {path}")
    digest = hashlib.sha256() if len(expected) == 64 else hashlib.sha1(f"blob {size}\0".encode())
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024**2), b""):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise ValueError(f"Model file does not match {MODEL_ID} at {REVISION}: {path}")


def fetch_files(model_dir, names):
    """Resume within this model directory and verify against Meta before publishing."""
    model_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        dest = model_dir / name
        if dest.is_file():
            verify_file(dest, name)
            continue
        part = dest.with_suffix(dest.suffix + ".part")
        if not part.exists() or part.stat().st_size != FILES[name][0]:
            print(f"[llama] download {name}", flush=True)
            subprocess.run(["curl", "--fail", "--location", "--silent", "--show-error",
                            "--connect-timeout", "15", "--speed-time", "60", "--speed-limit", "1024",
                            "--retry", "2", "--max-filesize", str(FILES[name][0]),
                            "--continue-at", "-", "--output", str(part), f"{MIRROR}/{name}"], check=True)
        verify_file(part, name)
        part.replace(dest)
        print(f"[llama] verified {name}", flush=True)


def remaining_weight_bytes(model_dir):
    missing = 0
    for name in WEIGHTS:
        file = model_dir / name
        if file.is_file():
            continue
        part = file.with_suffix(file.suffix + ".part")
        size = part.stat().st_size if part.is_file() else 0
        missing += max(0, FILES[name][0] - size)
    return missing


def get_tokenizer(model_dir):
    from tokenizers.processors import TemplateProcessing
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, use_fast=True,
        padding_side="left", truncation_side="left", model_max_length=MAX_LENGTH)
    tok.pad_token = "<|finetune_right_pad_id|>"
    if tok.pad_token_id != 128004 or tok.eos_token != "<|eot_id|>" or not tok.is_fast:
        raise ValueError("Unexpected Llama 3.1 tokenizer special tokens")
    # The rendered native template already has BOS. Use the same single BOS
    # for the generation loss, candidate prefixes, and free-generation eval.
    tok.backend_tokenizer.post_processor = TemplateProcessing(single="$A", pair="$A $B")
    return tok


def parse_messages(text):
    matches = MESSAGE.findall(text)
    if tuple(role for role, _ in matches) != ("system", "user", "assistant") or MESSAGE.sub("", text).strip():
        raise ValueError("Expected a complete three-turn Qwen QA")
    return [{"role": role, "content": body} for role, body in matches]


def render_qa(tokenizer, text):
    messages = parse_messages(text)
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def render_context(tokenizer, text):
    body = text.rstrip()
    if not body.endswith("<|im_end|>"):
        raise ValueError("Expected a plain context task ending with the Qwen end token")
    return tokenizer.bos_token + body.removesuffix("<|im_end|>") + tokenizer.eos_token


def assistant_offset(text):
    try:
        return text.rindex(LLAMA_ASSISTANT) + len(LLAMA_ASSISTANT)
    except ValueError as exc:
        raise ValueError("Llama answer-only masking requires an assistant turn") from exc


def load_llama(model_name, model_cache_dir, *, dtype, device_map="auto", **_kwargs):
    from transformers import AutoModelForCausalLM

    if model_name != MODEL_ID:
        raise ValueError(f"Llama worker received a different base model: {model_name}")
    model_dir = Path(model_cache_dir) / MODEL_FOLDER
    tokenizer = get_tokenizer(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=dtype, device_map=device_map, attn_implementation="sdpa",
        local_files_only=True, use_safetensors=True)
    if len(tokenizer) != model.get_input_embeddings().num_embeddings:
        raise ValueError("Llama vocabulary mismatch; refusing to resize the embedding weights")
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in {"train", "score", "test"}:
        raise SystemExit("Usage: ccfb_llama_runtime.py {train,score,test} <existing runner arguments>")
    action = sys.argv.pop(1)
    import train_listed_contrastive as train
    from hard_negative_grip import listed_training
    from hard_negative_grip.task_file import assistant_answer_prefix, assistant_gold

    # Override only this worker's loading/rendering hooks. Qwen entry points
    # and the shared trainer, candidate score and evaluation remain in use.
    train.get_hf_llm_tokenizer = load_llama
    if action == "train":
        resolve = train.resolve_training_assets
        listed_training.answer_start_char = assistant_offset

        def resolve_llama_assets(args):
            record, contexts, texts, metas, evaluation = resolve(args)
            tok = get_tokenizer(Path(args.model_cache_dir) / MODEL_FOLDER)
            if args.stage == "s1":
                contexts = [render_context(tok, text) for text in contexts]
            if args.stage in {"listed", "b1"}:
                texts = [render_qa(tok, text) for text in texts]
                for text, meta in zip(texts, metas, strict=True):
                    if assistant_gold(text) != meta["positive_relation"].strip():
                        raise ValueError(f"Rendered gold changed for {meta['question_id']}")
                    meta["prefix_text"] = assistant_answer_prefix(text)
            return record, contexts, texts, metas, evaluation

        train.resolve_training_assets = resolve_llama_assets
        train.main()
    elif action == "score":
        import score_relation_candidates_offline as score
        from functools import lru_cache

        args = score.parse_args()
        tokenizer = get_tokenizer(Path(args.model_cache_dir) / MODEL_FOLDER)
        score.get_hf_llm_tokenizer = load_llama
        score.assistant_answer_prefix = lru_cache(maxsize=4096)(
            lambda text: assistant_answer_prefix(render_qa(tokenizer, text)))
        score.main()
    else:
        import eval_listed_decode
        eval_listed_decode.main()


if __name__ == "__main__":
    main()
