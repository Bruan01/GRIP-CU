"""Strict-paper B1 options: answer-only L_QA, S2 early stop, Adam eps, test-only EM."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from hard_negative_grip.decode_io import eval_samples  # noqa: E402
from hard_negative_grip.listed_training import (  # noqa: E402
    ASSISTANT_MARKER,
    IGNORE_INDEX,
    ListedDataCollator,
    ListedQADataset,
    answer_only_labels,
)

SCRIPT = Path(__file__).parents[1] / "scripts/train_listed_contrastive.py"


class _CharTokenizer:
    """One token per character; supports offsets like a fast tokenizer."""

    pad_token_id = 0
    padding_side = "right"

    def __init__(self, offsets: bool = True) -> None:
        self.offsets = offsets

    def __call__(self, text, truncation=True, padding=False, return_offsets_mapping=False):
        ids = [ord(ch) for ch in text]
        out = {"input_ids": ids, "attention_mask": [1] * len(ids)}
        if return_offsets_mapping:
            if not self.offsets:
                raise NotImplementedError
            out["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return out


QA = "<|im_start|>user\nq?<|im_end|>\n" + ASSISTANT_MARKER + "<answer>a</answer><|im_end|>\n"


@pytest.mark.parametrize("offsets", [True, False])
def test_answer_only_labels_mask_prompt(offsets: bool) -> None:
    tok = _CharTokenizer(offsets)
    ids = tok(QA)["input_ids"]
    labels = answer_only_labels(QA, ids, tok)
    start = QA.rfind(ASSISTANT_MARKER) + len(ASSISTANT_MARKER)
    assert labels[:start] == [IGNORE_INDEX] * start
    assert labels[start:] == ids[start:]


def test_answer_only_requires_assistant_turn() -> None:
    with pytest.raises(ValueError):
        answer_only_labels("plain text", [1, 2], _CharTokenizer())


def test_dataset_only_adds_labels_when_requested() -> None:
    meta = {"positive_relation": "", "listed_relations": []}
    tok = _CharTokenizer()
    assert "labels" not in ListedQADataset([QA], [meta], tok)[0]
    item = ListedQADataset([QA], [meta], tok, answer_only=True)[0]
    assert item["labels"][0] == IGNORE_INDEX and item["labels"][-1] == ord("\n")


def test_collator_pads_own_labels_with_ignore_index() -> None:
    collator = ListedDataCollator.__new__(ListedDataCollator)
    collator.tokenizer = _CharTokenizer()

    def inner(features):
        width = max(len(f["input_ids"]) for f in features)
        ids = [f["input_ids"] + [0] * (width - len(f["input_ids"])) for f in features]
        return {"input_ids": torch.tensor(ids), "labels": torch.tensor(ids)}

    collator.inner = inner
    batch = collator([
        {"input_ids": [5, 6, 7], "labels": [IGNORE_INDEX, 6, 7]},
        {"input_ids": [8], "labels": [8]},
    ])
    assert batch["labels"].tolist() == [[IGNORE_INDEX, 6, 7], [8, IGNORE_INDEX, IGNORE_INDEX]]


def test_eval_samples_can_select_test_only() -> None:
    record = {"recurrent_questions": [
        {"split": "train"}, {"split": "validation"}, {"split": "test"}, {"split": "test"},
    ]}
    assert len(eval_samples(record)) == 3
    assert [s["split"] for s in eval_samples(record, ["test"])] == ["test", "test"]


@pytest.fixture(scope="module")
def script():
    try:
        spec = importlib.util.spec_from_file_location("train_paper_alignment_test", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except ImportError as exc:  # grip-exp not importable off-server
        pytest.skip(f"train script imports unavailable: {exc}")
    return module


def test_official_parser_is_strict(script) -> None:
    assert script.official_parse_answer("<answer>concept:a</answer>") == "concept:a"
    assert script.official_parse_answer("<answer>x</answer> <answer>y</answer>") == "x; y"
    assert script.official_parse_answer("[concept:a]") == ""
    assert script.official_exact_match("The concept:a.", ["concept:a"])  # articles/punctuation dropped


def test_em_summary_reports_official_em_per_split(script) -> None:
    rows = [
        {"split": "test", "correct": True, "official_correct": False, "in_list": True},
        {"split": "test", "correct": True, "official_correct": True, "in_list": True},
    ]
    summary = script.em_summary(rows)
    assert summary["test"]["em"] == 1.0
    assert summary["test"]["official_em"] == 0.5
    assert summary["all"]["official_em"] == 0.5


def test_training_arguments_use_paper_adam_epsilon(script, tmp_path) -> None:
    args = SimpleNamespace(
        per_device_train_batch_size=1, gradient_accumulation_steps=512, learning_rate=1e-3,
        weight_decay=0.0, adam_epsilon=1e-4, max_grad_norm=1.0, seed=2026,
        save_steps=0, save_total_limit=2,
    )
    try:
        targs = script.training_arguments(tmp_path, args, 10, 20000)
    except (ValueError, ImportError) as exc:  # bf16 needs a GPU
        pytest.skip(f"TrainingArguments unavailable here: {exc}")
    assert targs.adam_epsilon == 1e-4
    assert targs.weight_decay == 0.0
    assert (targs.adam_beta1, targs.adam_beta2) == (0.9, 0.98)
    assert targs.lr_scheduler_type == "linear"


def test_s2_early_stop_callback_matches_grip(script) -> None:
    callback = script.EarlyStoppingOnTrainLossCallback(threshold=0.15, min_epoch=1)
    control = SimpleNamespace(should_training_stop=False)
    callback.on_log(None, None, control, logs={"loss": 0.1, "epoch": 0.5})
    assert not control.should_training_stop
    callback.on_log(None, None, control, logs={"loss": 0.2, "epoch": 1.0})
    assert not control.should_training_stop
    callback.on_log(None, None, control, logs={"loss": 0.1, "epoch": 1.0})
    assert control.should_training_stop
