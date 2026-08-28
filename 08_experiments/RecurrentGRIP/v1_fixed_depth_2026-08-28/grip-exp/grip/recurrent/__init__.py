from .config import RecurrentConfig, resolve_layer_index
from .executor import (
    FixedDepthRecurrentBlock,
    capture_recurrent_trace,
    find_decoder_layers,
    get_recurrent_block,
    set_recurrent_depth,
    trace_recurrence,
    wrap_decoder_layer,
)
from .model import build_recurrent_peft_model, validate_adapter_scope
from .outputs import RecurrentPrediction

__all__ = [
    "RecurrentConfig",
    "resolve_layer_index",
    "FixedDepthRecurrentBlock",
    "capture_recurrent_trace",
    "find_decoder_layers",
    "get_recurrent_block",
    "set_recurrent_depth",
    "trace_recurrence",
    "wrap_decoder_layer",
    "build_recurrent_peft_model",
    "validate_adapter_scope",
    "RecurrentPrediction",
]
