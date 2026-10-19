"""Helpers so long GPU runs can resume after a kill.

Trainer checkpoints live under ``trainer_s1/`` or ``trainer_<variant>/``.
The flattened PEFT adapter written at the end of a stage is the finished
artifact; a ``checkpoint-*`` directory is only an in-progress snapshot.
"""

from __future__ import annotations

from pathlib import Path


def adapter_is_complete(adapter_dir: Path) -> bool:
    """True when a PEFT adapter directory looks save-complete."""
    if not adapter_dir.is_dir():
        return False
    has_config = (adapter_dir / "adapter_config.json").is_file()
    has_weights = (adapter_dir / "adapter_model.safetensors").is_file() or (
        adapter_dir / "adapter_model.bin"
    ).is_file()
    return has_config and has_weights


def latest_trainer_checkpoint(trainer_dir: Path) -> Path | None:
    """Return the highest ``checkpoint-N`` that has ``trainer_state.json``."""
    if not trainer_dir.is_dir():
        return None
    best: Path | None = None
    best_step = -1
    for path in trainer_dir.glob("checkpoint-*"):
        if not path.is_dir():
            continue
        if not (path / "trainer_state.json").is_file():
            continue
        suffix = path.name.split("-", 1)[-1]
        try:
            step = int(suffix)
        except ValueError:
            continue
        if step > best_step:
            best_step = step
            best = path
    return best


def resolve_resume_checkpoint(
    trainer_dir: Path,
    *,
    resume_from_checkpoint: Path | None = None,
    no_resume: bool = False,
) -> str | bool:
    """Path to resume from, or ``False`` to start fresh."""
    if no_resume:
        return False
    if resume_from_checkpoint is not None:
        path = Path(resume_from_checkpoint)
        trainer_resolved = trainer_dir.resolve()
        path_resolved = path.resolve()
        belongs = path_resolved == trainer_resolved or trainer_resolved in path_resolved.parents
        if belongs:
            if not path.is_dir():
                raise FileNotFoundError(f"resume checkpoint not found: {path}")
            return str(path_resolved)
    last = latest_trainer_checkpoint(trainer_dir)
    return str(last) if last is not None else False


def checkpoint_training_kwargs(save_steps: int, save_total_limit: int) -> dict:
    """HuggingFace ``TrainingArguments`` fields for mid-run checkpoints."""
    if save_steps <= 0:
        return {
            "save_strategy": "no",
            "overwrite_output_dir": True,
        }
    return {
        "save_strategy": "steps",
        "save_steps": int(save_steps),
        "save_total_limit": max(1, int(save_total_limit)),
        "overwrite_output_dir": False,
        "load_best_model_at_end": False,
    }
