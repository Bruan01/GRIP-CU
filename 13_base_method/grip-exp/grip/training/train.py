from copy import deepcopy
from pathlib import Path
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
from .candidate_scorer import score_candidate_records
from .utils import EarlyStoppingOnTrainLossCallback


def _save_stage_checkpoint(trainer: Trainer, tokenizer: PreTrainedTokenizer, output_dir: Union[str, Path]) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(output_dir)

    # PEFT may nest adapter files under ``mylora``; keep each stage loadable directly.
    nested_dir = output_dir / "mylora"
    if nested_dir.is_dir():
        for item in nested_dir.iterdir():
            item.rename(output_dir / item.name)
        nested_dir.rmdir()


class CandidateInfoNCEDataCollator:
    """Keep candidate metadata beside the ordinary GRIP LM batch."""

    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.language_modeling = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    def __call__(self, features: list[dict]) -> dict:
        candidate_records = []
        generation_features = []
        for feature in features:
            feature = dict(feature)
            candidate_records.append(feature.pop("candidate_record", None))
            generation_features.append(feature)
        batch = self.language_modeling(generation_features)
        batch["candidate_records"] = candidate_records
        return batch


class CandidateInfoNCETrainer(Trainer):
    def __init__(self, *args, candidate_loss_weight: float, candidate_temperature: float,
                 candidate_max_length: int, candidate_score_clip: float,
                 candidate_loss_interval: int, **kwargs):
        super().__init__(*args, **kwargs)
        if candidate_loss_weight < 0:
            raise ValueError("candidate_loss_weight must be non-negative")
        if candidate_temperature <= 0:
            raise ValueError("candidate_temperature must be positive")
        if candidate_score_clip <= 0:
            raise ValueError("candidate_score_clip must be positive")
        if candidate_loss_interval < 1:
            raise ValueError("candidate_loss_interval must be at least 1")
        self.candidate_loss_weight = candidate_loss_weight
        self.candidate_temperature = candidate_temperature
        self.candidate_max_length = candidate_max_length
        self.candidate_score_clip = candidate_score_clip
        self.candidate_loss_interval = candidate_loss_interval
        self._candidate_microsteps = 0

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        candidate_records = inputs.pop("candidate_records", None)
        outputs = model(**inputs)
        generation_loss = outputs.loss
        self._candidate_microsteps += 1
        candidate_records = [record for record in candidate_records or [] if record is not None]
        if not candidate_records or self.candidate_loss_weight == 0:
            return (generation_loss, outputs) if return_outputs else generation_loss
        if self._candidate_microsteps % self.candidate_loss_interval != 0:
            return (generation_loss, outputs) if return_outputs else generation_loss

        score_result = score_candidate_records(
            model=model,
            tokenizer=self.processing_class,
            records=candidate_records,
            max_length=self.candidate_max_length,
            score_clip=self.candidate_score_clip,
        )
        candidate_scores = torch.stack(score_result.scores)
        labels = torch.tensor(
            [int(record["positive_index"]) for record in candidate_records],
            dtype=torch.long,
            device=candidate_scores.device,
        )
        candidate_loss = torch.nn.functional.cross_entropy(
            candidate_scores / self.candidate_temperature,
            labels,
        )
        loss = generation_loss + (
            self.candidate_loss_weight * self.candidate_loss_interval * candidate_loss
        )
        return (loss, outputs) if return_outputs else loss


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
         candidate_loss_weight: float = 0.0,
         candidate_temperature: float = 1.0,
         candidate_max_length: int = 4096,
         candidate_score_clip: float = 20.0,
         candidate_loss_interval: int = 1,
) -> tuple[Trainer, Union[PreTrainedModel, PeftModel]]:
    training_args = deepcopy(training_args)
    training_args.remove_unused_columns = False
    training_args.ddp_find_unused_parameters = False
    training_args.label_names = ["labels"]
    total_batch_size = training_args.gradient_accumulation_steps * training_args.per_device_train_batch_size
    if gather_batches or total_batch_size > len(training_dataset):
        training_args.gradient_accumulation_steps = len(training_dataset)
    trainer_class = CandidateInfoNCETrainer if candidate_loss_weight > 0 else Trainer
    data_collator = (
        CandidateInfoNCEDataCollator(tokenizer)
        if candidate_loss_weight > 0 else
        DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    )
    trainer_kwargs = {}
    if candidate_loss_weight > 0:
        trainer_kwargs.update(
            candidate_loss_weight=candidate_loss_weight,
            candidate_temperature=candidate_temperature,
            candidate_max_length=candidate_max_length,
            candidate_score_clip=candidate_score_clip,
            candidate_loss_interval=candidate_loss_interval,
        )
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=training_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        optimizers=(optimizer, None),
        callbacks=[EarlyStoppingOnTrainLossCallback(threshold=stop_loss_threshold, min_epoch=min_epoch)],
        **trainer_kwargs,
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
         stage1_output_dir: Optional[Union[str, Path]] = None,
         stage2_output_dir: Optional[Union[str, Path]] = None,
         candidate_loss_weight: float = 0.0,
         candidate_temperature: float = 1.0,
         candidate_max_length: int = 4096,
         candidate_score_clip: float = 20.0,
         candidate_loss_interval: int = 1,
         stage2_checkpoint_dir: Optional[Union[str, Path]] = None,
         stage2_checkpoint_interval: int = 0,
         resume_stage1_checkpoint: Optional[Union[str, Path]] = None,
         resume_stage2_checkpoint: Optional[Union[str, Path]] = None,
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
        candidate_loss_weight=0.0,
    )
    if training_args.num_train_epochs > 0:
        trainer.train(
            resume_from_checkpoint=(
                str(resume_stage1_checkpoint) if resume_stage1_checkpoint else None
            )
        )
        if stage1_output_dir is not None:
            _save_stage_checkpoint(trainer, tokenizer, stage1_output_dir)

    # Stage 2: Train on raw text + reasoning tasks.
    if involve_qa_epochs > 0:
        training_dataset.enable_qa = True
        training_args_syn = deepcopy(training_args)
        training_args_syn.num_train_epochs = involve_qa_epochs
        if stage2_checkpoint_dir is not None:
            training_args_syn.output_dir = str(stage2_checkpoint_dir)
        if stage2_checkpoint_interval > 0:
            if stage2_checkpoint_dir is None:
                raise ValueError('stage2_checkpoint_dir is required for periodic checkpoints.')
            training_args_syn.output_dir = str(stage2_checkpoint_dir)
            training_args_syn.save_strategy = 'steps'
            training_args_syn.save_steps = stage2_checkpoint_interval
            training_args_syn.save_total_limit = 1
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
            candidate_loss_weight=candidate_loss_weight,
            candidate_temperature=candidate_temperature,
            candidate_max_length=candidate_max_length,
            candidate_score_clip=candidate_score_clip,
            candidate_loss_interval=candidate_loss_interval,
        )
        trainer_syn.train(
            resume_from_checkpoint=(
                str(resume_stage2_checkpoint) if resume_stage2_checkpoint else None
            )
        )
        if stage2_output_dir is not None:
            _save_stage_checkpoint(trainer_syn, tokenizer, stage2_output_dir)

    # Clear cache
    for param in model.parameters():
        if param.requires_grad:
            param.grad = None
    torch.cuda.empty_cache()
    return model, tokenizer
