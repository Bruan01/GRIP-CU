from dataclasses import dataclass, field


@dataclass
class InferenceArguments:
    api_key: str = field(
        default="${OPENAI_API_KEY}",
        metadata={"help": "The API key for the LLM provider."}
    )

    api_base: str = field(
        default="https://api.openai.com/v1/",
        metadata={"help": "The API base for the OPENAI."}
    )

    api_version: str = field(
        default=None,
        metadata={"help": "The API version for the OPENAI."}
    )

    use_vllm: bool = field(
        default=True,
        metadata={"help": "Whether to use vLLM for inference."}
    )

    gen_max_length: int = field(
        default=5000,
        metadata={"help": "The maximum length of the generated text."}
    )

    batch_size: int = field(
        default=-1,
        metadata={"help": "The batch size for inference."}
    )

    do_sample: bool = field(
        default=True,
        metadata={"help": "Whether to use sampling for inference."}
    )

    top_p: float = field(
        default=0.9,
        metadata={"help": "The top p for inference."}
    )

    temperature: float = field(
        default=0.6,
        metadata={"help": "The temperature for inference."}
    )

    no_graph_context: bool = field(
        default=False,
        metadata={"help": "Whether to add graph context to user query."}
    )

    use_subgraph: bool = field(
        default=False,
        metadata={"help": "Whether to use subgraph as graph context. If False, use full graph instead."}
    )

    index_format: bool = field(
        default=False,
        metadata={"help": "If true, use index + edge as graph to text method."}
    )
    report_input_token_count: bool = field(
        default=False,
        metadata={"help": "If true, report average input token."}

    )