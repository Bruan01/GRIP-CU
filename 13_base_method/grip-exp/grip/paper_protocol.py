"""Canonical GRIP paper protocols.

This module keeps the paper-facing settings in one place so that training and
evaluation entry points do not silently drift apart.  Hardware adaptations
(for example, a smaller micro-batch on a single RTX 3090) are represented
separately from the scientific protocol.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class PaperProtocol:
    dataset: str
    context_qa: int
    reasoning_qa: int
    summarization: int
    lora_r: int
    lora_alpha: int
    stage1_min_epochs: int
    stage1_max_epochs: int
    stage2_min_epochs: int
    stage2_max_epochs: int
    early_stop_threshold: float
    gradient_accumulation_steps: int
    gather_batches: bool
    adam_epsilon: float
    no_graph_context: bool
    use_subgraph: bool
    primary_metric: str
    judge_model: str | None
    num_test: int | None
    sample_node_attribute_task: bool
    repharse_context_qa: bool
    context_upsampling: bool
    format_as_instruction: bool
    task_generator_use_vllm: bool

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Values below follow the original GRIP paper shell/appendix protocol.
# The 3090-specific batch and checkpointing choices are kept in the runnable
# shell scripts rather than mixed into these scientific settings.
PAPER_PROTOCOLS = {
    "wn18rr": PaperProtocol(
        dataset="wn18rr",
        context_qa=8000,
        reasoning_qa=2000,
        summarization=6000,
        lora_r=24,
        lora_alpha=48,
        stage1_min_epochs=1,
        stage1_max_epochs=1,
        stage2_min_epochs=1,
        stage2_max_epochs=5,
        early_stop_threshold=0.15,
        gradient_accumulation_steps=512,
        gather_batches=False,
        adam_epsilon=1e-4,
        no_graph_context=True,
        use_subgraph=False,
        primary_metric="exact_match",
        judge_model=None,
        num_test=None,
        sample_node_attribute_task=False,
        repharse_context_qa=False,
        context_upsampling=False,
        # This is the value used by the original WN18RR shell.  The paper
        # describes context/summarization as plain text, while the released
        # implementation enables this shared flag for the full WN protocol.
        format_as_instruction=True,
        task_generator_use_vllm=True,
    ),
    "clegr_reasoning": PaperProtocol(
        dataset="clegr_reasoning",
        context_qa=20,
        reasoning_qa=160,
        summarization=20,
        lora_r=16,
        lora_alpha=32,
        stage1_min_epochs=5,
        stage1_max_epochs=50,
        stage2_min_epochs=5,
        stage2_max_epochs=50,
        early_stop_threshold=0.4,
        gradient_accumulation_steps=128,
        gather_batches=True,
        adam_epsilon=1e-4,
        no_graph_context=True,
        use_subgraph=False,
        primary_metric="llm_as_judge",
        judge_model="Qwen/Qwen2.5-32B-Instruct",
        num_test=400,
        sample_node_attribute_task=True,
        repharse_context_qa=True,
        context_upsampling=True,
        format_as_instruction=False,
        task_generator_use_vllm=True,
    ),
}


def get_paper_protocol(dataset: str) -> PaperProtocol:
    """Return a canonical protocol or raise a useful error."""

    key = dataset.lower()
    if key not in PAPER_PROTOCOLS:
        choices = ", ".join(sorted(PAPER_PROTOCOLS))
        raise ValueError(f"Unknown paper protocol dataset {dataset!r}; choose one of: {choices}")
    return PAPER_PROTOCOLS[key]


def validate_main_eval_flags(
    dataset: str,
    *,
    no_graph_context: bool,
    use_subgraph: bool,
) -> None:
    """Fail fast when a claimed paper-main evaluation has graph leakage."""

    protocol = get_paper_protocol(dataset)
    if no_graph_context != protocol.no_graph_context or use_subgraph != protocol.use_subgraph:
        raise ValueError(
            f"{dataset} paper-main evaluation requires "
            f"no_graph_context={protocol.no_graph_context} and "
            f"use_subgraph={protocol.use_subgraph}; "
            f"received no_graph_context={no_graph_context}, use_subgraph={use_subgraph}"
        )
