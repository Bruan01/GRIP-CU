from copy import deepcopy
from typing import Optional, Union

import torch
from peft import PeftModel
from torch.utils.data import Dataset
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

from grip.tasks import TaskDataset
from .utils import EarlyStoppingOnTrainLossCallback


def load_trainer(
        model: Union[PreTrainedModel, PeftModel],
        tokenizer: PreTrainedTokenizer,
        training_dataset: TaskDataset,
        training_args: TrainingArguments,
        eval_dataset: Optional[Union[Dataset, list]] = None,
        gather_batches: bool = False,
        optimizer: Optional[torch.optim.Optimizer] = None,
        stop_loss_threshold: float = 0.4,
        min_epoch: int = 1,
) -> tuple[Trainer, Union[PreTrainedModel, PeftModel]]:
    training_args = deepcopy(training_args)
    training_args.remove_unused_columns = False
    training_args.ddp_find_unused_parameters = False
    training_args.label_names = ["labels"]
    total_batch_size = training_args.gradient_accumulation_steps * training_args.per_device_train_batch_size
    if gather_batches or total_batch_size > len(training_dataset):
        training_args.gradient_accumulation_steps = len(training_dataset)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=training_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        optimizers=(optimizer, None),
        callbacks=[EarlyStoppingOnTrainLossCallback(threshold=stop_loss_threshold, min_epoch=min_epoch)],
    )

    return trainer, model


def train(
        model: Union[PreTrainedModel, PeftModel],
        tokenizer: PreTrainedTokenizer,
        training_args: TrainingArguments,
        training_dataset: TaskDataset,
        eval_dataset: Optional[Union[Dataset, list]] = None,
        involve_qa_epochs: int = 0,
        gather_batches: bool = True,
        s1_stop_loss_threshold: float = 0.1,
        s1_min_epoch: int = 1,
        s2_stop_loss_threshold: float = 0.5,
        s2_min_epoch: int = 1,
        **kwargs,
) -> tuple[Union[PreTrainedModel, PeftModel], PreTrainedTokenizer]:
    # load tokenizer
    torch.cuda.empty_cache()  # Manually release memory

    # Stage 1: Train on remember raw text.
    trainer, model = load_trainer(
        model=model,
        tokenizer=tokenizer,
        training_dataset=training_dataset,
        training_args=training_args,
        eval_dataset=eval_dataset,
        gather_batches=gather_batches,
        stop_loss_threshold=s1_stop_loss_threshold,
        min_epoch=s1_min_epoch,
    )
    if training_args.num_train_epochs > 0:
        trainer.train()

    # Stage 2: Train on raw text + reasoning tasks.
    if involve_qa_epochs > 0:
        training_dataset.enable_qa = True
        training_args_syn = deepcopy(training_args)
        training_args_syn.num_train_epochs = involve_qa_epochs
        trainer_syn, model = load_trainer(
            model=model,
            tokenizer=tokenizer,
            training_dataset=training_dataset,
            training_args=training_args_syn,
            eval_dataset=eval_dataset,
            gather_batches=gather_batches,
            optimizer=trainer.optimizer,
            stop_loss_threshold=s2_stop_loss_threshold,
            min_epoch=s2_min_epoch,
        )
        trainer_syn.train()

    # Clear cache
    for param in model.parameters():
        if param.requires_grad:
            param.grad = None
    torch.cuda.empty_cache()
    return model, tokenizer
