from __future__ import annotations

import random
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from hard_negative_grip.listed_training import ListedContrastiveTrainer  # noqa: E402
from hard_negative_grip.scoring import (  # noqa: E402
    encode_without_specials,
    normalized_continuation_log_likelihood,
    pack_decision_set_rows,
    pack_token_rows,
    score_candidate_rows,
    score_candidates,
)
from hard_negative_grip.task_file import assistant_answer_prefix  # noqa: E402

VOCAB = 32
PREFIX = "Q<answer>"
# Short, long, and mixed-length relation-like continuations.
RELATIONS = [
    "r",
    "ab",
    "owns",
    "visits",
    "concept:atdate",
    "concept:worksfor",
    "concept:sportsgameteam",
    "concept:mutualproxyfor",
]


class CharTokenizer:
    """Deterministic char tokenizer; no BOS/EOS unless the text contains them."""

    pad_token_id = 0
    unk_token_id = 1

    def __call__(self, text, add_special_tokens=False, truncation=True, padding=False):
        ids = [(ord(char) % (VOCAB - 2)) + 2 for char in text]
        if not ids:
            ids = [self.unk_token_id]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


class ForwardOnlyLM(torch.nn.Module):
    """Slow-path scorer: full-sequence logits, no ``model`` / ``lm_head`` split."""

    def __init__(self, vocab: int = VOCAB, hidden: int = 8, seed: int = 2026) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.embed = torch.nn.Embedding(vocab, hidden)
        self.lm_head = torch.nn.Linear(hidden, vocab, bias=False)

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        logits = self.lm_head(self.embed(input_ids))
        return SimpleNamespace(logits=logits)


class TinyCausalLM(torch.nn.Module):
    """Fast-path scorer matching listed training: ``model`` + ``lm_head``."""

    def __init__(self, vocab: int = VOCAB, hidden: int = 8, seed: int = 2026) -> None:
        super().__init__()
        torch.manual_seed(seed)
        embed = torch.nn.Embedding(vocab, hidden)
        proj = torch.nn.Linear(hidden, hidden)
        self.lm_head = torch.nn.Linear(hidden, vocab, bias=False)

        class _Body(torch.nn.Module):
            def __init__(self, embed_layer, proj_layer):
                super().__init__()
                self.embed = embed_layer
                self.proj = proj_layer

            def forward(self, input_ids, attention_mask=None, use_cache=False):
                hidden_states = self.proj(torch.tanh(self.embed(input_ids)))
                return (hidden_states,)

        self.model = _Body(embed, proj)

    def get_base_model(self):
        return self

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        hidden = self.model(input_ids, attention_mask=attention_mask, use_cache=use_cache)[0]
        return SimpleNamespace(logits=self.lm_head(hidden))


def _qa(question: str, answer: str) -> str:
    return (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\nGiven the context graph titled nell23k, please answer "
        f"the following question: {question} Response in the following format:"
        "<answer>[answer]</answer><|im_end|>\n"
        f"<|im_start|>assistant\n<answer>{answer}</answer><|im_end|>\n"
    )


def _relation_qa_pool() -> list[tuple[str, str, list[str]]]:
    pairs = [
        ("concept_sportsgame_championship", "concept_sportsteam_red_wings", "concept:sportsgameteam"),
        ("concept_ceo_dieter_zetsche", "concept_company_chrysler", "concept:worksfor"),
        ("concept_city_detroit", "concept_stateorprovince_michigan", "concept:citylocatedinstate"),
        ("concept_person_alice", "concept_person_bob", "concept:hasfriend"),
        ("concept_book_the_deerslayer", "concept_writer_cooper", "concept:bookwriter"),
        ("concept_team_red_wings", "concept_sport_hockey", "concept:teamplaysinleague"),
        ("concept_airport_dtw", "concept_city_detroit", "concept:atdate"),
        ("concept_person_x", "concept_person_y", "r"),
    ]
    extras = ["concept:mutualproxyfor", "owns", "visits", "concept:atdate", "ab"]
    pool = []
    for src, tgt, gold in pairs:
        text = _qa(f"what is the relation between {src} and {tgt}?", gold)
        prefix = assistant_answer_prefix(text)
        candidates = [gold, *[rel for rel in extras if rel != gold]]
        pool.append((prefix, gold, candidates))
    return pool


def legacy_score_candidate_rows(scorer, tokenizer, rows, prefix_lens, device):
    """Frozen copy of the original listed-training slow-path scorer.

    This is the math that used to live inside ``score_candidate_rows`` before
    the unified wrapper: pack with right padding, then mean log-probability of
    continuation tokens via ``normalized_continuation_log_likelihood``.
    """
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = 0
    input_ids, attention_mask, seq_lens = pack_token_rows(
        rows, pad_id=int(pad_id), device=device
    )
    prefix_lengths = torch.tensor(prefix_lens, dtype=torch.long, device=device)
    outputs = scorer(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    )
    return normalized_continuation_log_likelihood(
        outputs.logits,
        input_ids,
        prefix_lengths,
        seq_lens,
    )


def test_score_candidates_match_legacy_and_training_rows() -> None:
    tokenizer = CharTokenizer()
    model = ForwardOnlyLM()
    device = torch.device("cpu")
    rng = random.Random(2026)
    pool = _relation_qa_pool()
    sample = rng.sample(pool, k=8)
    max_abs = 0.0
    saw_single = False
    saw_multi = False
    for prefix, _gold, candidates in sample:
        lengths = [len(encode_without_specials(tokenizer, rel)) for rel in candidates]
        saw_single = saw_single or any(length == 1 for length in lengths)
        saw_multi = saw_multi or any(length > 1 for length in lengths)
        rows, prefix_lens = pack_decision_set_rows(tokenizer, prefix, candidates)
        assert min(prefix_lens) != max(len(row) for row in rows)
        old = legacy_score_candidate_rows(model, tokenizer, rows, prefix_lens, device)
        training = score_candidate_rows(model, tokenizer, rows, prefix_lens, device)
        unified = score_candidates(model, tokenizer, prefix, candidates, device=device)
        new = torch.tensor(unified["candidate_score"][0])
        trainer = ListedContrastiveTrainer.__new__(ListedContrastiveTrainer)
        trainer.candidate_forwards = 0
        trainer.accelerator = None
        trainer_scores = trainer._score_candidate_rows(
            model, tokenizer, rows, prefix_lens, device
        )
        assert torch.allclose(old, new, atol=1e-6, rtol=0)
        assert torch.allclose(training, new, atol=1e-6, rtol=0)
        assert torch.allclose(trainer_scores, new, atol=1e-6, rtol=0)
        assert unified["candidate_token_length"][0] == lengths
        max_abs = max(max_abs, float((old - new).abs().max()))
    assert saw_single and saw_multi
    assert max_abs < 1e-6


def test_prompt_tokens_do_not_enter_candidate_score() -> None:
    tokenizer = CharTokenizer()
    prefix_len = len(encode_without_specials(tokenizer, PREFIX))
    answer = "owns"
    answer_len = len(encode_without_specials(tokenizer, answer))
    assert prefix_len >= 1
    assert answer_len >= 1

    class _Scorer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

        def forward(self, input_ids, attention_mask=None, use_cache=False):
            batch, seq = input_ids.shape
            logits = torch.zeros(batch, seq, VOCAB)
            for row in range(batch):
                for pos in range(seq - 1):
                    target = int(input_ids[row, pos + 1])
                    # Prefix next-token positions get a strongly negative peak;
                    # answer tokens get a strongly positive peak.
                    logits[row, pos, target] = -20.0 if pos + 1 < prefix_len else 20.0
            return SimpleNamespace(logits=logits)

    result = score_candidates(_Scorer(), tokenizer, PREFIX, [answer], device=torch.device("cpu"))
    score = result["candidate_score"][0][0]
    assert result["candidate_token_length"][0][0] == answer_len
    # Answer-only mean log-probability of a +20 peak is ~0, never positive.
    assert score > -1e-4
    rows, _prefix_lens = pack_decision_set_rows(tokenizer, PREFIX, [answer])
    packed = pack_token_rows(rows, pad_id=0, device=torch.device("cpu"))
    input_ids, _, seq_lens = packed
    logits = _Scorer()(input_ids).logits
    all_positions = normalized_continuation_log_likelihood(
        logits,
        input_ids,
        torch.ones(1, dtype=torch.long),
        seq_lens,
    )
    # Averaging prefix tokens would pull the mean well below the answer-only score.
    assert float(all_positions) < -1.0
    assert float(all_positions) < score - 1.0


def test_padding_does_not_enter_candidate_score() -> None:
    tokenizer = CharTokenizer()
    model = ForwardOnlyLM()
    device = torch.device("cpu")
    short, long = "r", "concept:sportsgameteam"
    separate_short = score_candidates(model, tokenizer, PREFIX, [short], device=device)
    separate_long = score_candidates(model, tokenizer, PREFIX, [long], device=device)
    packed = score_candidates(model, tokenizer, PREFIX, [short, long], device=device)
    assert packed["candidate_token_length"][0] == [
        separate_short["candidate_token_length"][0][0],
        separate_long["candidate_token_length"][0][0],
    ]
    assert packed["candidate_token_length"][0][0] < packed["candidate_token_length"][0][1]
    assert abs(packed["candidate_score"][0][0] - separate_short["candidate_score"][0][0]) < 1e-6
    assert abs(packed["candidate_score"][0][1] - separate_long["candidate_score"][0][0]) < 1e-6


def test_encode_many_matches_per_string_ids() -> None:
    tokenizer = CharTokenizer()
    answers = ["r", "owns", "concept:worksfor"]
    from hard_negative_grip.scoring import encode_many_without_specials

    batched = encode_many_without_specials(tokenizer, answers)
    singles = [encode_without_specials(tokenizer, answer) for answer in answers]
    assert batched == singles


def test_candidate_token_length_is_answer_only() -> None:
    tokenizer = CharTokenizer()
    model = ForwardOnlyLM()
    answers = ["r", "owns", "concept:worksfor"]
    result = score_candidates(model, tokenizer, PREFIX, answers, device=torch.device("cpu"))
    prefix_len = len(encode_without_specials(tokenizer, PREFIX))
    expected = [len(encode_without_specials(tokenizer, answer)) for answer in answers]
    assert result["candidate_token_length"][0] == expected
    assert prefix_len not in expected
    assert expected[0] == 1
    assert expected[-1] > expected[0]


def test_multi_token_candidate_uses_mean_not_sum() -> None:
    tokenizer = CharTokenizer()

    class _Constant(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

        def forward(self, input_ids, attention_mask=None, use_cache=False):
            batch, seq = input_ids.shape
            logits = torch.full((batch, seq, VOCAB), -20.0)
            for row in range(batch):
                for pos in range(seq - 1):
                    logits[row, pos, int(input_ids[row, pos + 1])] = 4.0
            return SimpleNamespace(logits=logits)

    result = score_candidates(
        _Constant(),
        tokenizer,
        PREFIX,
        ["r", "concept:worksfor"],
        device=torch.device("cpu"),
    )
    lengths = result["candidate_token_length"][0]
    scores = result["candidate_score"][0]
    assert lengths[0] == 1
    assert lengths[1] > 1
    assert abs(scores[0] - scores[1]) < 1e-6
    # A sum score would grow with length; the mean stays flat.


def test_batch_scoring_matches_per_item_scoring() -> None:
    tokenizer = CharTokenizer()
    model = ForwardOnlyLM()
    device = torch.device("cpu")
    prefixes = [prefix for prefix, _, _ in _relation_qa_pool()[:3]]
    groups = [RELATIONS[:4], RELATIONS[2:6], RELATIONS[4:]]
    batched = score_candidates(model, tokenizer, prefixes, groups, device=device)
    for index, (prefix, answers) in enumerate(zip(prefixes, groups)):
        single = score_candidates(model, tokenizer, prefix, answers, device=device)
        assert single["candidate_token_length"][0] == batched["candidate_token_length"][index]
        for left, right in zip(single["candidate_score"][0], batched["candidate_score"][index]):
            assert abs(left - right) < 1e-6


def test_positive_and_negative_use_the_same_scorer() -> None:
    tokenizer = CharTokenizer()
    model = TinyCausalLM()
    device = torch.device("cpu")
    positive = "concept:worksfor"
    negatives = ["owns", "r", "concept:atdate"]
    together = score_candidates(
        model, tokenizer, PREFIX, [positive, *negatives], device=device
    )
    pos_only = score_candidates(model, tokenizer, PREFIX, [positive], device=device)
    neg_only = score_candidates(model, tokenizer, PREFIX, negatives, device=device)
    assert abs(together["candidate_score"][0][0] - pos_only["candidate_score"][0][0]) < 1e-6
    for index, score in enumerate(together["candidate_score"][0][1:]):
        assert abs(score - neg_only["candidate_score"][0][index]) < 1e-6
    assert together["candidate_token_length"][0][0] == pos_only["candidate_token_length"][0][0]


def test_causal_fast_path_matches_legacy_full_logits() -> None:
    tokenizer = CharTokenizer()
    model = TinyCausalLM()
    device = torch.device("cpu")
    candidates = ["r", "owns", "concept:worksfor"]
    rows, prefix_lens = pack_decision_set_rows(tokenizer, PREFIX, candidates)
    old = legacy_score_candidate_rows(model, tokenizer, rows, prefix_lens, device)
    new = torch.tensor(
        score_candidates(model, tokenizer, PREFIX, candidates, device=device)["candidate_score"][0]
    )
    assert torch.allclose(old, new, atol=1e-5, rtol=0)
