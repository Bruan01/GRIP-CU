from peft import PeftModel

from grip.recurrent import set_recurrent_depth, validate_adapter_scope
from .train import train


def train_recurrent_graph(
    model: PeftModel,
    tokenizer,
    training_dataset,
    training_args,
    executor_layer_index: int,
    recurrent_depth_train: int,
    **kwargs,
):
    set_recurrent_depth(model, recurrent_depth_train)
    validate_adapter_scope(model, executor_layer_index)
    model.config.use_cache = False
    trained_model, tokenizer = train(
        model=model,
        tokenizer=tokenizer,
        training_dataset=training_dataset,
        training_args=training_args,
        **kwargs,
    )
    validate_adapter_scope(trained_model, executor_layer_index)
    return trained_model, tokenizer
