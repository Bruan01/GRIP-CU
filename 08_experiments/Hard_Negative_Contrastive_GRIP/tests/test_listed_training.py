from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from hard_negative_grip.listed_training import (  # noqa: E402
    EXTRA_KEYS,
    ListedContrastiveTrainer,
    ListedDataCollator,
    ListedQADataset,
    format_answer_prefix,
    listed_negatives,
    pack_decision_set_rows,
    pick_closed_set_answer,
    unwrap_for_scoring,
)


class _FakeChatTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert tokenize is False
        assert add_generation_prompt is True
        assert messages[0]["role"] == "system"
        assert "nell23k" in messages[1]["content"]
        return "CHAT|"


class _FakeEncodeTokenizer:
    unk_token_id = 0
    pad_token_id = 0

    def __call__(self, text, truncation=True, padding=False, add_special_tokens=True):
        tokens = [len(text), 7, 8]
        return {"input_ids": tokens, "attention_mask": [1] * len(tokens)}


def test_listed_negatives_drop_gold_and_keep_prompt_order() -> None:
    sample = {
        "answer": "rel_b",
        "candidate_relations": ["rel_b", "rel_a", "rel_c", "rel_b"],
    }
    assert listed_negatives(sample) == ["rel_a", "rel_c"]


def test_listed_negatives_parse_question_when_list_missing() -> None:
    sample = {
        "answer": "owns",
        "question": (
            "What is the relation between word node alice and word node bob? "
            "Selected from the following candidate answers: owns; visits; likes."
        ),
    }
    assert listed_negatives(sample) == ["visits", "likes"]


def test_format_answer_prefix_stops_at_answer_tag() -> None:
    prefix = format_answer_prefix(
        _FakeChatTokenizer(),
        title="nell23k",
        question="q",
        system_prompt="sys",
        question_template="Given {title}: {question}",
    )
    assert prefix == "CHAT|<answer>"


def test_listed_qa_dataset_keeps_the_decision_set() -> None:
    dataset = ListedQADataset(
        ["hello world"],
        [
            {
                "listed_relations": ["visits", "owns"],
                "positive_relation": "likes",
                "prefix_text": "PRE<answer>",
            }
        ],
        _FakeEncodeTokenizer(),
    )
    item = dataset[0]
    assert item["listed_relations"] == ["visits", "owns"]
    assert item["positive_relation"] == "likes"
    assert item["prefix_text"].endswith("<answer>")
    assert item["input_ids"] == [11, 7, 8]
    assert item["prefix_ids"] == [len("PRE<answer>"), 7, 8]
    assert len(item["relation_ids"]) == 3
    assert item["relation_ids"][0] == [len("likes"), 7, 8]
    assert item["lambda_candidate"] is None


def test_listed_qa_dataset_accepts_variable_negative_k() -> None:
    dataset = ListedQADataset(
        ["short", "longer negatives"],
        [
            {
                "listed_relations": ["a"],
                "positive_relation": "gold",
                "prefix_text": "PRE<answer>",
            },
            {
                "listed_relations": ["b", "c", "d"],
                "positive_relation": "gold",
                "prefix_text": "PRE<answer>",
            },
        ],
        _FakeEncodeTokenizer(),
    )
    assert len(dataset[0]["relation_ids"]) == 2
    assert len(dataset[1]["relation_ids"]) == 4
    assert dataset[0]["listed_relations"] == ["a"]
    assert dataset[1]["listed_relations"] == ["b", "c", "d"]


def test_listed_collator_does_not_tensorize_prompt_lists() -> None:
    collator = ListedDataCollator.__new__(ListedDataCollator)
    collator.inner = lambda features: {"input_ids": [feature["input_ids"] for feature in features]}
    batch = collator(
        [
            {
                "input_ids": [1, 2],
                "listed_relations": ["a", "b"],
                "positive_relation": "gold",
                "prefix_text": "PRE<answer>",
                "prefix_ids": [9, 7, 8],
                "relation_ids": [[4, 7, 8], [1, 7, 8], [1, 7, 8]],
                "lambda_candidate": 0.5,
            },
            {
                "input_ids": [3, 4],
                "listed_relations": ["c"],
                "positive_relation": "other",
                "prefix_text": "PRE2<answer>",
                "prefix_ids": [10, 7, 8],
                "relation_ids": [],
                "lambda_candidate": None,
            },
        ]
    )
    assert batch["input_ids"] == [[1, 2], [3, 4]]
    assert batch["listed_relations"] == [["a", "b"], ["c"]]
    assert batch["positive_relation"] == ["gold", "other"]
    assert batch["prefix_text"] == ["PRE<answer>", "PRE2<answer>"]
    assert batch["prefix_ids"] == [[9, 7, 8], [10, 7, 8]]
    assert batch["relation_ids"][0][0] == [4, 7, 8]
    assert batch["lambda_candidate"] == [0.5, None]
    for key in EXTRA_KEYS:
        assert key in batch


def test_unwrap_for_scoring_uses_accelerator_when_present() -> None:
    class _Accelerator:
        def unwrap_model(self, model):
            return f"unwrapped:{model}"

    assert unwrap_for_scoring("wrapped", _Accelerator()) == "unwrapped:wrapped"
    assert unwrap_for_scoring("plain") == "plain"


def test_listed_candidates_are_scored_in_one_forward() -> None:
    class _Tok:
        pad_token_id = 0

    class _Out:
        def __init__(self, logits):
            self.logits = logits

    class _Scorer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.calls = 0

        def forward(self, input_ids, attention_mask, use_cache=False):
            self.calls += 1
            batch, seq = input_ids.shape
            logits = torch.zeros(batch, seq, 16)
            for row in range(batch):
                for pos in range(seq - 1):
                    logits[row, pos, int(input_ids[row, pos + 1])] = 8.0
            return _Out(logits)

    trainer = ListedContrastiveTrainer.__new__(ListedContrastiveTrainer)
    trainer.candidate_forwards = 0
    trainer.accelerator = None
    scorer = _Scorer()
    scores = trainer._score_candidate_rows(
        scorer,
        _Tok(),
        [[1, 2, 3, 4], [1, 2, 5]],
        [2, 2],
        torch.device("cpu"),
    )
    assert scorer.calls == 1
    assert trainer.candidate_forwards == 2
    assert scores.shape == (2,)
    assert scores[0] > scores[1] - 1e-5


def test_listed_log_fields_average_the_logging_window() -> None:
    trainer = ListedContrastiveTrainer.__new__(ListedContrastiveTrainer)
    trainer._reset_loss_window()
    trainer.last_generation_loss = 9.0
    trainer.last_candidate_loss = 0.0
    trainer.last_lambda_candidate = 1.0
    trainer.candidate_forwards = 10
    trainer._window_samples = 2
    trainer._window_generation_sum = 0.2 + 0.4
    trainer._window_weighted_candidate_sum = 0.5
    trainer._window_contrastive_samples = 1
    trainer._window_contrastive_loss_sum = 1.0
    trainer._window_lambda_sum = 0.5
    fields = trainer.listed_log_fields()
    assert fields["generation_loss"] == pytest.approx(0.3)
    assert fields["candidate_loss"] == pytest.approx(0.25)
    assert fields["contrastive_candidate_loss"] == 1.0
    assert fields["lambda_candidate"] == 0.5
    assert fields["last_microbatch_candidate_loss"] == 0.0


def test_batch_lambda_uses_per_qa_weights_and_global_fallback() -> None:
    trainer = ListedContrastiveTrainer.__new__(ListedContrastiveTrainer)
    trainer.lambda_candidate = 1.0
    weights = trainer._batch_lambda([0.25, None], 2, torch.device("cpu"))
    assert weights.tolist() == [0.25, 1.0]
    fallback = trainer._batch_lambda(None, 2, torch.device("cpu"))
    assert fallback.tolist() == [1.0, 1.0]


def test_listed_qa_dataset_keeps_per_qa_lambda() -> None:
    dataset = ListedQADataset(
        ["hello world"],
        [
            {
                "listed_relations": ["visits", "owns"],
                "positive_relation": "likes",
                "prefix_text": "PRE<answer>",
                "lambda_candidate": 0.55,
            }
        ],
        _FakeEncodeTokenizer(),
    )
    assert dataset[0]["lambda_candidate"] == 0.55


def test_pick_closed_set_answer_uses_argmax_and_prompt_order_ties() -> None:
    relations = ["owns", "visits", "likes"]
    assert pick_closed_set_answer(relations, [1.0, 3.0, 2.0]) == "visits"
    assert pick_closed_set_answer(relations, [2.0, 2.0, 1.0]) == "owns"


class _TinyLM(torch.nn.Module):
    """One scalar parameter. Generation loss is that scalar times a token count."""

    def __init__(self, value: float = 0.4) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(value))

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        count = float(input_ids.shape[0])
        loss = self.weight * count
        return type("Out", (), {"loss": loss, "logits": self.weight.view(1, 1, 1)})()


class _ScriptedCandidate(ListedContrastiveTrainer):
    """Replace 10-way scoring with fixed per-question losses."""

    def __init__(self, *args, scripted: list[float | None], **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._scripted = list(scripted)
        self._cursor = 0

    def _listed_candidate_loss(self, model, prefix_ids_batch, relation_ids_batch, **kwargs):
        weights = kwargs["weights"]
        losses = []
        sample_losses: list[torch.Tensor | None] = []
        for index, _prefix in enumerate(prefix_ids_batch):
            value = self._scripted[self._cursor]
            self._cursor += 1
            if value is None:
                sample_losses.append(None)
                continue
            sample = model.weight.new_tensor(value)
            sample_losses.append(sample)
            losses.append(weights[index] * sample)
        if not losses:
            return model.weight.new_zeros(()), sample_losses
        return torch.stack(losses).mean(), sample_losses


def _microbatch(token_count: int, *, prefix: bool, weight: float | None) -> dict:
    ids = torch.arange(token_count).view(1, token_count)
    return {
        "input_ids": ids,
        "attention_mask": torch.ones_like(ids),
        "labels": ids.clone(),
        "prefix_ids": [[1, 2]] if prefix else [[]],
        "relation_ids": [[[3], [4]]] if prefix else [[]],
        "prefix_text": ["prefix"] if prefix else [""],
        "positive_relation": ["gold"] if prefix else [""],
        "listed_relations": [["other"]] if prefix else [[]],
        "lambda_candidate": [weight],
    }


def _manual_window_gradient(model: _TinyLM, batches: list[dict], scripted: list[float | None]) -> torch.Tensor:
    """Mean of microbatch losses, then one backward. This is the target scale."""
    pieces = []
    cursor = 0
    for batch in batches:
        count = float(batch["input_ids"].shape[0])
        generation = model.weight * count
        weights = batch["lambda_candidate"]
        terms = []
        for weight in weights:
            value = scripted[cursor]
            cursor += 1
            if value is None or weight is None or float(weight) <= 0:
                continue
            terms.append(model.weight.new_tensor(float(weight) * value))
        candidate = torch.stack(terms).mean() if terms else generation.new_zeros(())
        pieces.append(generation + candidate)
    loss = torch.stack(pieces).mean()
    loss.backward()
    assert model.weight.grad is not None
    return model.weight.grad.detach().clone()


def _trainer_window_gradient(
    batches: list[dict],
    scripted: list[float | None],
    *,
    configured_accum: int,
) -> tuple[torch.Tensor, ListedContrastiveTrainer]:
    model = _TinyLM()
    trainer = _ScriptedCandidate(
        model=model,
        args=_cpu_training_args(configured_accum),
        train_dataset=[{"input_ids": [1]}],
        processing_class=object(),
        scripted=scripted,
        lambda_candidate=1.0,
    )
    trainer.current_gradient_accumulation_steps = len(batches)
    trainer.accelerator.gradient_state.sync_gradients = False
    for batch in batches:
        trainer.training_step(model, batch)
    assert model.weight.grad is not None
    return model.weight.grad.detach().clone(), trainer


def _cpu_training_args(accum: int):
    from transformers import TrainingArguments

    return TrainingArguments(
        output_dir="/tmp/listed-accum-normalization",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=accum,
        learning_rate=1e-4,
        report_to="none",
        use_cpu=True,
        disable_tqdm=True,
    )


def test_training_step_matches_mean_of_microbatch_losses() -> None:
    batches = [
        _microbatch(2, prefix=True, weight=0.5),
        _microbatch(4, prefix=False, weight=None),
        _microbatch(1, prefix=True, weight=1.0),
        _microbatch(3, prefix=True, weight=0.25),
    ]
    scripted = [0.8, None, 1.2, 0.4]
    manual = _manual_window_gradient(_TinyLM(), batches, scripted)
    trained, trainer = _trainer_window_gradient(batches, scripted, configured_accum=4)
    assert torch.allclose(trained, manual)
    assert not torch.allclose(trained * 4, manual)
    assert trainer.model_accepts_loss_kwargs is False


def test_short_final_window_divides_by_its_actual_size() -> None:
    batches = [
        _microbatch(2, prefix=True, weight=1.0),
        _microbatch(3, prefix=False, weight=None),
    ]
    scripted = [0.6, None]
    manual = _manual_window_gradient(_TinyLM(), batches, scripted)
    trained, _trainer = _trainer_window_gradient(batches, scripted, configured_accum=4)
    assert torch.allclose(trained, manual)
    over_divided = _manual_window_gradient(_TinyLM(), batches, scripted) * (2 / 4)
    assert not torch.allclose(trained, over_divided)


def test_logged_candidate_loss_keeps_earlier_microbatches() -> None:
    batches = [
        _microbatch(2, prefix=True, weight=0.5),
        _microbatch(2, prefix=False, weight=None),
    ]
    scripted = [0.8, None]
    _gradient, trainer = _trainer_window_gradient(batches, scripted, configured_accum=2)
    fields = trainer.listed_log_fields()
    assert fields["candidate_loss"] == pytest.approx(0.2)
    assert fields["contrastive_candidate_loss"] == pytest.approx(0.8)
    assert fields["lambda_candidate"] == pytest.approx(0.5)
    assert fields["last_microbatch_candidate_loss"] == 0.0
    # Both microbatches have one sample; token counts are 2 and 2, so the
    # mean generation term is weight * 1, not the sum of the two counts.
    assert fields["generation_loss"] == pytest.approx(0.4)


def test_pack_decision_set_rows_share_one_prefix() -> None:
    class _Tok:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [len(text)]}

    rows, prefix_lens = pack_decision_set_rows(_Tok(), "PRE<answer>", ["owns", "visits"])
    assert prefix_lens == [1, 1]
    assert rows[0][0] == rows[1][0] == len("PRE<answer>")
    assert rows[0][1] == len("owns")
    assert rows[1][1] == len("visits")
