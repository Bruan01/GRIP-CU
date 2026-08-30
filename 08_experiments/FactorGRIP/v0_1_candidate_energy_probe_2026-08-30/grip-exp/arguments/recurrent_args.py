from dataclasses import dataclass, field
from typing import List


@dataclass
class RecurrentArguments:
    """Arguments exclusive to the Fixed-Depth RecurrentGRIP path."""

    recurrent_depth_train: int = field(
        default=2,
        metadata={"help": "Number of shared executor applications used during training."},
    )
    recurrent_depth_eval: int = field(
        default=2,
        metadata={"help": "Default number of executor applications used during evaluation."},
    )
    recurrent_depth_sweep: List[int] = field(
        default_factory=lambda: [1, 2, 3, 4, 5],
        metadata={"help": "Evaluation recurrence depths."},
    )
    executor_layer_index: int = field(
        default=-1,
        metadata={"help": "Decoder layer to repeat. -1 selects the middle decoder layer."},
    )
    recurrent_question_types: List[str] = field(
        default_factory=lambda: ["StationShortestCount"],
        metadata={"help": "CLEGR question types used by the recurrent pilot."},
    )
    train_hops: List[int] = field(
        default_factory=lambda: [1, 2],
        metadata={"help": "Shortest-path distances used for training."},
    )
    validation_hops: List[int] = field(
        default_factory=lambda: [1, 2],
        metadata={"help": "Shortest-path distances used for validation."},
    )
    test_hops: List[int] = field(
        default_factory=lambda: [3, 4],
        metadata={"help": "Shortest-path distances used for length-OOD testing."},
    )
    validation_fraction: float = field(
        default=0.2,
        metadata={"help": "Fraction of seen-hop questions held out for validation."},
    )
    save_step_hidden_states: bool = field(
        default=True,
        metadata={"help": "Save one pooled hidden vector per recurrence step."},
    )
    max_graphs: int = field(
        default=16,
        metadata={"help": "Maximum number of graphs used by the pilot. 0 keeps all graphs."},
    )
    max_questions_per_hop: int = field(
        default=32,
        metadata={"help": "Per-graph cap for each hop bucket. 0 keeps all questions."},
    )
    max_context_samples: int = field(
        default=0,
        metadata={"help": "Maximum graph-memory samples per graph; 0 keeps all samples."},
    )
    context_node_samples: int = field(
        default=0,
        metadata={"help": "Node declarations selected by the stratified context sampler; 0 disables this stratum."},
    )
    context_edge_samples: int = field(
        default=0,
        metadata={"help": "Edge facts selected relation-round-robin; 0 disables this stratum."},
    )
    context_sampling_seed: int = field(
        default=2026,
        metadata={"help": "Local seed for deterministic graph-context selection."},
    )
    adapter_control: str = field(
        default="all",
        metadata={"help": "Evaluation adapter condition: correct, shuffled, none, or all."},
    )
    evaluation_device: str = field(
        default="auto",
        metadata={"help": "Evaluation device: auto, cuda, cpu, or mps."},
    )
    require_cuda: bool = field(
        default=False,
        metadata={"help": "Fail immediately when CUDA is unavailable; enable this on the WSL 3090 runner."},
    )
    split_seed: int = field(
        default=2026,
        metadata={"help": "Seed used for deterministic question splitting."},
    )
    adapter_output_dir: str = field(
        default="outputs/recurrent_grip/adapters",
        metadata={"help": "Directory used to persist graph-specific recurrent adapters."},
    )
    wall_time_limit_minutes: int = field(
        default=120,
        metadata={"help": "Soft wall-time budget for a pilot run."},
    )
    hard_stop_minutes: int = field(
        default=180,
        metadata={"help": "Hard wall-time limit for a pilot run."},
    )

    def __post_init__(self) -> None:
        for name in ("recurrent_depth_train", "recurrent_depth_eval"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1")
        if not self.recurrent_depth_sweep or any(depth < 1 for depth in self.recurrent_depth_sweep):
            raise ValueError("recurrent_depth_sweep must contain positive integers")
        if not self.train_hops or not self.test_hops:
            raise ValueError("train_hops and test_hops must be non-empty")
        if set(self.train_hops) & set(self.test_hops):
            raise ValueError("train_hops and test_hops must be disjoint")
        if not 0.0 <= self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in [0, 1)")
        if self.adapter_control not in {"correct", "shuffled", "none", "all"}:
            raise ValueError("adapter_control must be one of: correct, shuffled, none, all")
        if self.evaluation_device not in {"auto", "cuda", "cpu", "mps"}:
            raise ValueError("evaluation_device must be one of: auto, cuda, cpu, mps")
        if self.max_graphs < 0 or self.max_questions_per_hop < 0 or self.max_context_samples < 0:
            raise ValueError("max_graphs, max_questions_per_hop, and max_context_samples cannot be negative")
        if self.context_node_samples < 0:
            raise ValueError("context_node_samples cannot be negative")
        if self.context_edge_samples < 0:
            raise ValueError("context_edge_samples cannot be negative")
        if self.max_context_samples > 0 and (self.context_node_samples > 0 or self.context_edge_samples > 0):
            raise ValueError(
                "max_context_samples cannot be combined with context_node_samples/context_edge_samples"
            )
        if self.wall_time_limit_minutes < 1:
            raise ValueError("wall_time_limit_minutes must be positive")
        if self.hard_stop_minutes < self.wall_time_limit_minutes:
            raise ValueError("hard_stop_minutes must be >= wall_time_limit_minutes")
