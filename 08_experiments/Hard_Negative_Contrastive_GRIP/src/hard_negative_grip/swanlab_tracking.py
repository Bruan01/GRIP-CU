"""Opt-in SwanLab recording for paired Stage-2 runs.

Importing this module has no network or GPU side effects. A run is created only
when ``start`` is called by the training entrypoint.
"""

from __future__ import annotations

import json
import traceback
from datetime import datetime, timezone
from pathlib import Path


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _numeric(values: dict) -> dict[str, float | int]:
    return {
        str(key): value for key, value in values.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }


class SwanLabTracker:
    def __init__(
        self,
        output_dir: Path,
        *,
        project: str,
        experiment_name: str,
        group: str,
        mode: str = "online",
    ) -> None:
        self.output_dir = output_dir
        self.project = project
        self.experiment_name = experiment_name
        self.group = group
        self.mode = mode
        self.run_id: str | None = None
        self.started = False
        self._swanlab = None

    def _write_status(self, status: str, **fields: object) -> None:
        payload = {"status": status, "at_utc": _now(), "swanlab_run_id": self.run_id, **fields}
        (self.output_dir / "run_status.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    def _event(self, event: str, **fields: object) -> None:
        with (self.output_dir / "events.log").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"at_utc": _now(), "event": event, **fields}, ensure_ascii=False) + "\n")

    def start(self, config: dict) -> None:
        if self.started:
            raise RuntimeError("SwanLab run already started")
        import swanlab

        run = swanlab.init(
            project=self.project,
            experiment_name=self.experiment_name,
            group=self.group,
            tags=["GRIP", "NELL23K", "Stage2", str(config.get("negative_sampling_mode", "random"))],
            mode=self.mode,
            config=config,
            logdir=str(self.output_dir / "swanlab"),
        )
        self._swanlab = swanlab
        self.run_id = str(getattr(run, "id", "")) or None
        self.started = True
        self._write_status("running", stage=config.get("stage"), experiment_name=self.experiment_name)
        self._event("started", swanlab_run_id=self.run_id, stage=config.get("stage"))
        (self.output_dir / "swanlab_run.json").write_text(
            json.dumps({"project": self.project, "group": self.group,
                        "experiment_name": self.experiment_name, "mode": self.mode,
                        "run_id": self.run_id}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def callback(self, phase: str):
        if not self.started:
            raise RuntimeError("SwanLab run not started")
        from transformers import TrainerCallback

        tracker = self

        class Recorder(TrainerCallback):
            def on_train_begin(self, args, state, control, **kwargs):
                tracker._event("train_begin", phase=phase, max_steps=state.max_steps)

            def on_log(self, args, state, control, logs=None, **kwargs):
                if not logs:
                    return
                values = {f"train/{phase}/{key}": value for key, value in _numeric(logs).items()}
                values[f"train/{phase}/global_step"] = state.global_step
                try:
                    import torch
                    if torch.cuda.is_available():
                        values["hardware/gpu_max_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
                except ImportError:
                    pass
                tracker._swanlab.log(values, step=state.global_step)
                tracker._event("train_log", phase=phase, global_step=state.global_step,
                               epoch=state.epoch, metrics=_numeric(logs),
                               gpu_max_allocated_gib=values.get("hardware/gpu_max_allocated_gib"))

        return Recorder()

    def log_evaluation(self, variant: str, summary: dict) -> None:
        if not self.started:
            return
        metrics = {}
        for section, values in summary.items():
            if isinstance(values, dict):
                metrics.update({f"eval/{variant}/{section}/{key}": value
                                for key, value in _numeric(values).items()})
            elif isinstance(values, (int, float)) and not isinstance(values, bool):
                metrics[f"eval/{variant}/{section}"] = values
        if metrics:
            self._swanlab.log(metrics)
        self._event("evaluation", variant=variant, summary=summary)

    def complete(self, **fields: object) -> None:
        if self.started:
            self._write_status("completed", **fields)
            self._event("completed", **fields)

    def fail(self, error: BaseException) -> None:
        if self.started:
            detail = "".join(traceback.format_exception(type(error), error, error.__traceback__))
            (self.output_dir / "failure.txt").write_text(detail, encoding="utf-8")
            self._write_status("failed", error=str(error))
            self._event("failed", error=str(error))

    def finish(self) -> None:
        if self.started:
            self._swanlab.finish()
            self.started = False
