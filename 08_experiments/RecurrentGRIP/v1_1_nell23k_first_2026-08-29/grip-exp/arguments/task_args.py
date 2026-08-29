from dataclasses import dataclass, field


@dataclass
class TaskArguments:
    dataset_name: str = field(
        default="scene_graph",
        metadata={"help": "The name of the dataset to perform"}
    )

    task_generator_model_name: str = field(
        default="qwen-32b",
        metadata={"help": "The model name of the task generator."}
    )

    num_context_qa: int = field(
        default=20,
        metadata={"help": "The number of QA pairs to generate."}
    )

    num_reason_qa: int = field(
        default=20,
        metadata={"help": "The number of QA pairs to generate."}
    )

    k_shot: int = field(
        default=5,
        metadata={"help": "The number example questions to use."}
    )

    num_summarization: int = field(
        default=10,
        metadata={"help": "The number of subgraph summarization to generate."}
    )

    task_gen_max_length: int = field(
        default=1000,
        metadata={"help": "The maximum length for task generation."}
    )

    task_generator_use_vllm: bool = field(
        default=False,
        metadata={"help": "Use vLLM for task generation. Set False when vLLM is unavailable."}
    )

    task_generator_batch_size: int = field(
        default=1,
        metadata={"help": "HF task-generator batch size. Keep 1 on a 24 GB GPU for long prompts."}
    )

    task_cache_dir: str = field(
        default=None,
        metadata={"help": "Optional JSONL cache for generated training tasks; makes long task generation resumable."}
    )

    sample_node_attribute_task: bool = field(
        default=True,
        metadata={"help": "If true, sample node attribute task in context qa."}
    )

    repharse_context_qa: bool = field(
        default=True,
        metadata={"help": "If true, repharse the context qa task."}
    )

    context_upsampling: bool = field(
        default=True,
        metadata={"help": "If true, do upsampling in context memory task."}
    )

    format_as_instruction: bool = field(
        default=False,
        metadata={"help": "If true, do upsampling in context memory task."}
    )