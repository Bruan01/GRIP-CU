from __future__ import annotations

import sys
from pathlib import Path

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
            },
            {
                "input_ids": [3, 4],
                "listed_relations": ["c"],
                "positive_relation": "other",
                "prefix_text": "PRE2<answer>",
                "prefix_ids": [10, 7, 8],
                "relation_ids": [],
            },
        ]
    )
    assert batch["input_ids"] == [[1, 2], [3, 4]]
    assert batch["listed_relations"] == [["a", "b"], ["c"]]
    assert batch["positive_relation"] == ["gold", "other"]
    assert batch["prefix_text"] == ["PRE<answer>", "PRE2<answer>"]
    assert batch["prefix_ids"] == [[9, 7, 8], [10, 7, 8]]
    assert batch["relation_ids"][0][0] == [4, 7, 8]
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


def test_pick_closed_set_answer_uses_argmax_and_prompt_order_ties() -> None:
    relations = ["owns", "visits", "likes"]
    assert pick_closed_set_answer(relations, [1.0, 3.0, 2.0]) == "visits"
    assert pick_closed_set_answer(relations, [2.0, 2.0, 1.0]) == "owns"


def test_pack_decision_set_rows_share_one_prefix() -> None:
    class _Tok:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [len(text)]}

    rows, prefix_lens = pack_decision_set_rows(_Tok(), "PRE<answer>", ["owns", "visits"])
    assert prefix_lens == [1, 1]
    assert rows[0][0] == rows[1][0] == len("PRE<answer>")
    assert rows[0][1] == len("owns")
    assert rows[1][1] == len("visits")
