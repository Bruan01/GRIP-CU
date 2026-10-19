from dataclasses import dataclass, field
from typing import List


@dataclass
class CustomTrainingArguments:
    use_prefix: bool = field(
        default=False,
        metadata={"help": "Whether to use prefix tuning."}
    )

    num_virtual_tokens: int = field(
        default=20,
        metadata={"help": "The number of virtual tokens."}
    )

    encoder_hidden_size: int = field(
        default=128,
        metadata={"help": "The hidden size of the encoder."}
    )

    lora_r: int = field(
        default=4,
        metadata={"help": "The r of the LoRA."}
    )

    lora_alpha: int = field(
        default=32,
        metadata={"help": "The alpha of the LoRA."}
    )

    target_modules: List[str] = field(
        default_factory=lambda: ["k_proj", "v_proj", "q_proj"],
        metadata={"help": "The alpha of the LoRA."}
    )

    dropout: float = field(
        default=0.0,
        metadata={"help": "The dropout of the LoRA."}
    )

    gather_batches: bool = field(
        default=False,
        metadata={'help': "Force the trainer to update the model only once every epoch. It is implemented with "
                          "gradient accumulation and it may lead to more stable gradients."}
    )

    involve_qa_epochs: int = field(
        default=0,
        metadata={'help': "Number of training epochs that involve QA tasks."}
    )

    s1_stop_loss_threshold: float = field(
        default=0.2,
        metadata={'help': "The threshold of the loss to stop training for stage 1."}
    )

    s2_stop_loss_threshold: float = field(
        default=0.5,
        metadata={'help': "The threshold of the loss to stop training for stage 2."}
    )

    s1_min_epoch: float = field(
        default=1,
        metadata={'help': "The minimum training epoch for stage 1."}
    )

    s2_min_epoch: float = field(
        default=1,
        metadata={'help': "The minimum training epoch for stage 2."}
    )
    continue_training: bool = field(
        default=False,
        metadata={'help': "If ture, do continue training upon previou graphs"}
    )
