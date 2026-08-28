from dataclasses import dataclass, field
from typing import List


@dataclass
class EvalArguments:
    do_evaluation: bool = field(
        default=True,
        metadata={"help": "Whether to run evaluation."}
    )

    metrics: List[str] = field(
        default_factory=lambda: ["em", "f1"],
        metadata={"help": "The metrics to evaluate."}
    )

    smooth: float = field(
        default=0.0,
        metadata={"help": "The smoothing factor for the belu score."}
    )

    llm_as_judge_model: str = field(
        default="qwen-32b",
        metadata={"help": "The model to use as judge model."}
    )
