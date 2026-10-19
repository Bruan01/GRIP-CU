from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from hard_negative_grip.run_protection import (  # noqa: E402
    adapter_is_complete,
    checkpoint_training_kwargs,
    latest_trainer_checkpoint,
    resolve_resume_checkpoint,
)


def test_latest_trainer_checkpoint_picks_highest_complete_step(tmp_path: Path) -> None:
    trainer_dir = tmp_path / "trainer_listed"
    stale = trainer_dir / "checkpoint-10"
    stale.mkdir(parents=True)
    (stale / "trainer_state.json").write_text("{}", encoding="utf-8")
    newer = trainer_dir / "checkpoint-70"
    newer.mkdir()
    (newer / "trainer_state.json").write_text("{}", encoding="utf-8")
    incomplete = trainer_dir / "checkpoint-80"
    incomplete.mkdir()
    (trainer_dir / "checkpoint-bad").mkdir()
    (trainer_dir / "checkpoint-bad" / "trainer_state.json").write_text("{}", encoding="utf-8")
    assert latest_trainer_checkpoint(trainer_dir) == newer
    assert latest_trainer_checkpoint(tmp_path / "missing") is None


def test_resolve_resume_checkpoint_respects_flags(tmp_path: Path) -> None:
    trainer_dir = tmp_path / "trainer_listed"
    ckpt = trainer_dir / "checkpoint-20"
    ckpt.mkdir(parents=True)
    (ckpt / "trainer_state.json").write_text(json.dumps({"global_step": 20}), encoding="utf-8")
    assert resolve_resume_checkpoint(trainer_dir) == str(ckpt)
    assert resolve_resume_checkpoint(trainer_dir, no_resume=True) is False
    assert resolve_resume_checkpoint(trainer_dir, resume_from_checkpoint=ckpt) == str(ckpt)
    other = tmp_path / "trainer_s1" / "checkpoint-3"
    other.mkdir(parents=True)
    (other / "trainer_state.json").write_text("{}", encoding="utf-8")
    assert resolve_resume_checkpoint(trainer_dir, resume_from_checkpoint=other) == str(ckpt)


def test_adapter_is_complete_requires_config_and_weights(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    assert adapter_is_complete(adapter) is False
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    assert adapter_is_complete(adapter) is False
    (adapter / "adapter_model.safetensors").write_bytes(b"x")
    assert adapter_is_complete(adapter) is True


def test_checkpoint_training_kwargs_default_enables_step_saves() -> None:
    enabled = checkpoint_training_kwargs(10, 2)
    assert enabled["save_strategy"] == "steps"
    assert enabled["save_steps"] == 10
    assert enabled["save_total_limit"] == 2
    assert enabled["overwrite_output_dir"] is False
    disabled = checkpoint_training_kwargs(0, 2)
    assert disabled["save_strategy"] == "no"
    assert disabled["overwrite_output_dir"] is True
