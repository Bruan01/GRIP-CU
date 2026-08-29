from dataclasses import dataclass


@dataclass(frozen=True)
class RecurrentConfig:
    depth: int = 2
    layer_index: int = -1
    save_step_hidden_states: bool = True

    def __post_init__(self) -> None:
        if self.depth < 1:
            raise ValueError("depth must be at least 1")


def resolve_layer_index(num_layers: int, requested_index: int) -> int:
    if num_layers < 1:
        raise ValueError("the model must expose at least one decoder layer")
    if requested_index == -1:
        return num_layers // 2
    index = requested_index if requested_index >= 0 else num_layers + requested_index
    if index < 0 or index >= num_layers:
        raise IndexError(f"decoder layer index {requested_index} is invalid for {num_layers} layers")
    return index
