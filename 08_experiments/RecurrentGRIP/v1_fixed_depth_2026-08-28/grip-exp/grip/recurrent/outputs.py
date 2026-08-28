from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class RecurrentPrediction:
    graph_id: str
    question_id: str
    question: str
    target: list[str]
    true_hop: int
    recurrence_k: int
    adapter_id: str
    adapter_control: str
    raw_response: str
    response: str
    correct: bool
    latency_seconds: float
    peak_memory_bytes: int = 0
    step_pooled_hidden_states: list[list[float]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        if self.true_hop < 1 or self.recurrence_k < 1:
            raise ValueError("true_hop and recurrence_k must be positive")
        if self.adapter_control not in {"correct", "shuffled", "none"}:
            raise ValueError("invalid adapter_control")
        if not self.graph_id or not self.question_id:
            raise ValueError("graph_id and question_id are required")
        if self.latency_seconds < 0:
            raise ValueError("latency_seconds cannot be negative")
