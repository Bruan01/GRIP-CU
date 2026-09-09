from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from hard_negative_grip.listed_training import (  # noqa: E402
    EXTRA_KEYS,
    ListedDataCollator,
    ListedQADataset,
    format_answer_prefix,
    listed_negatives,
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
    def __call__(self, text, truncation=True, padding=False):
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
            },
            {
                "input_ids": [3, 4],
                "listed_relations": ["c"],
                "positive_relation": "other",
                "prefix_text": "PRE2<answer>",
            },
        ]
    )
    assert batch["input_ids"] == [[1, 2], [3, 4]]
    assert batch["listed_relations"] == [["a", "b"], ["c"]]
    assert batch["positive_relation"] == ["gold", "other"]
    assert batch["prefix_text"] == ["PRE<answer>", "PRE2<answer>"]
    for key in EXTRA_KEYS:
        assert key in batch


def test_unwrap_for_scoring_uses_accelerator_when_present() -> None:
    class _Accelerator:
        def unwrap_model(self, model):
            return f"unwrapped:{model}"

    assert unwrap_for_scoring("wrapped", _Accelerator()) == "unwrapped:wrapped"
    assert unwrap_for_scoring("plain") == "plain"
