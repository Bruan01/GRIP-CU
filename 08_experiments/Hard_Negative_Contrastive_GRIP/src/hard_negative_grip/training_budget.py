"""Budget guards for the full original-GRIP candidate-learning run."""

from __future__ import annotations

DEFAULT_BATCH = 1
DEFAULT_ACCUM = 512
DEFAULT_EPOCHS = 10
DEFAULT_MIN_TRAIN_QA = 3253


def listed_update_budget(
    n_samples: int,
    *,
    batch: int = DEFAULT_BATCH,
    requested_accum: int = DEFAULT_ACCUM,
    epochs: float = DEFAULT_EPOCHS,
) -> dict:
    """Match the trainer's accumulation clamp and keep the planned steps explicit."""
    if n_samples < 0:
        raise ValueError("n_samples must be non-negative")
    if batch < 1 or requested_accum < 1:
        raise ValueError("batch and requested_accum must be positive")
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    effective_accum = requested_accum
    if batch * requested_accum > n_samples:
        effective_accum = max(1, n_samples // batch)
    updates_per_epoch = (
        (n_samples + batch * effective_accum - 1) // (batch * effective_accum)
        if n_samples
        else 0
    )
    return {
        "n_samples": n_samples,
        "batch": batch,
        "requested_accum": requested_accum,
        "effective_accum": effective_accum,
        "epochs": float(epochs),
        "steps_per_epoch": updates_per_epoch,
        "total_steps": int(epochs * updates_per_epoch),
        "accum_clamped": effective_accum != requested_accum,
        "underfit": bool(effective_accum != requested_accum or n_samples <= DEFAULT_MIN_TRAIN_QA),
        "tail_policy": "keep_incomplete_window",
    }


def refuse_underfit_listed_budget(
    n_samples: int,
    *,
    batch: int = DEFAULT_BATCH,
    requested_accum: int = DEFAULT_ACCUM,
    epochs: float = DEFAULT_EPOCHS,
) -> dict:
    budget = listed_update_budget(
        n_samples,
        batch=batch,
        requested_accum=requested_accum,
        epochs=epochs,
    )
    if budget["underfit"]:
        raise ValueError(
            "candidate-learning training requires the full task file: "
            f"n_samples={n_samples}, effective_accum={budget['effective_accum']}, "
            f"total_steps={budget['total_steps']}"
        )
    return budget


def assert_listed_training_budget(
    n_samples: int,
    *,
    listed_negative_source: str,
    skip_train: bool = False,
    allow_underfit: bool = False,
    batch: int = DEFAULT_BATCH,
    requested_accum: int = DEFAULT_ACCUM,
    epochs: float = DEFAULT_EPOCHS,
) -> dict | None:
    """Require a full task budget for the selected listed-negative method."""
    if skip_train or allow_underfit or listed_negative_source not in {
        "error_memory",
        "structure_counterfactual",
    }:
        return None
    return refuse_underfit_listed_budget(
        n_samples,
        batch=batch,
        requested_accum=requested_accum,
        epochs=epochs,
    )


def assert_error_memory_training_budget(
    n_samples: int,
    *,
    listed_negative_source: str,
    skip_train: bool = False,
    allow_underfit: bool = False,
    batch: int = DEFAULT_BATCH,
    requested_accum: int = DEFAULT_ACCUM,
    epochs: float = DEFAULT_EPOCHS,
) -> dict | None:
    """Backward-compatible alias for the pre-mainline budget guard."""
    return assert_listed_training_budget(
        n_samples,
        listed_negative_source=listed_negative_source,
        skip_train=skip_train,
        allow_underfit=allow_underfit,
        batch=batch,
        requested_accum=requested_accum,
        epochs=epochs,
    )
