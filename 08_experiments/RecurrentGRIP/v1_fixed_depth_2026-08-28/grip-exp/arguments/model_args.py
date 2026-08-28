from dataclasses import dataclass, field


@dataclass
class ModelArguments:
    model_name: str = field(
        default="qwen-7b",
        metadata={"help": "The name of the model."}
    )

    load_dir: str = field(
        default=None,
        metadata={"help": "The directory to load the model from."}
    )

    model_source: str = field(
        default="auto",
        metadata={
            "help": "Base-model source: auto (local then ModelScope, falling back to Hugging Face), local, hf, or modelscope."
        }
    )

    model_cache_dir: str = field(
        default="model_cache",
        metadata={"help": "Directory used for downloaded base models."}
    )

    local_files_only: bool = field(
        default=False,
        metadata={"help": "Do not access a remote model hub; require a cached/local model."}
    )

    model_download_workers: int = field(
        default=1,
        metadata={"help": "Concurrent ModelScope download workers. Use 1 for reliable large-shard downloads on Windows."}
    )

    tokenize_max_length: int = field(
        default=4096,
        metadata={"help": "The maximum length of the tokenized input."}
    )

    padding_side: str = field(
        default="right",
        metadata={"help": "The padding side of the tokenizer."}
    )

    truncation_side: str = field(
        default="right",
        metadata={"help": "The truncation side of the tokenizer."}
    )

    dtype: str = field(
        default="bfloat16",
        metadata={"help": "The dtype of the model."}
    )

    quantization: bool = field(
        default=False,
        metadata={"help": "Whether to quantize the model."}
    )
