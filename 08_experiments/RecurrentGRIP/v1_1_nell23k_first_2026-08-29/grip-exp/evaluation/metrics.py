from collections import Counter
from functools import partial
from typing import Callable

import numpy as np
from torch import Tensor
# from torchmetrics.text import BLEUScore, Perplexity

from constants import SYSTEM_PROMPT
from models import get_inf_model, BaseInferenceModel
from .utils import *


def get_eval_function(metric: str) -> Callable:
    if metric == "bleu1":
        return partial(compute_bleu, n_gram=1)
    elif metric == "bleu2":
        return partial(compute_bleu, n_gram=2)
    elif metric == "bleu3":
        return partial(compute_bleu, n_gram=3)
    elif metric == "bleu4":
        return partial(compute_bleu, n_gram=4)
    elif metric == "ppl":
        return compute_perplexity
    elif metric == "f1":
        return compute_f1_score
    elif metric == "em":
        return compute_em_score
    elif metric == "hit":
        return compute_hit_score
    elif metric == "retrieval_score":
        return compute_retrieval_score
    elif metric == "llm":
        return compute_llm_score
    else:
        raise ValueError(f"Unknown metric name: {metric}")


def compute_bleu(
        preds: list[str],
        targets: list[str],
        n_gram: int,
        smooth=True,
        **kwargs,
) -> float:
    bleu = BLEUScore(n_gram=n_gram, smooth=smooth)
    preds = [normalize_answer(pred) for pred in preds]
    targets = [normalize_answer(tgt) for tgt in targets]
    return bleu(preds, [[tgt] for tgt in targets]).item()


def compute_perplexity(
        preds: Tensor,
        targets: Tensor,
        ignore_index: int = -100,
        **kwargs,
) -> float:
    perp = Perplexity(ignore_index=ignore_index)
    return perp(preds, targets).item()


def compute_f1_score(
        preds: list[str],
        targets: list[list[str]],
        aggregation_fn: Callable = np.max,
        **kwargs,
) -> float:
    assert len(preds) == len(targets)

    def compute_f1(pred, target) -> float:
        pred_tokens = normalize_answer(pred).split()
        target_tokens = normalize_answer(target).split()
        common = Counter(pred_tokens) & Counter(target_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0.0
        precision = 1.0 * num_same / len(pred_tokens)
        recall = 1.0 * num_same / len(target_tokens)
        return 2 * (precision * recall) / (precision + recall)

    total_f1 = 0.0
    for target_list, pred in zip(targets, preds):
        f1_scores = [compute_f1(pred, target) for target in target_list]
        aggregated_f1 = aggregation_fn(f1_scores)
        total_f1 += aggregated_f1

    avg_f1 = total_f1 / len(preds) if preds else 0.0
    return avg_f1


def compute_em_score(
        preds: list[str],
        targets: list[list[str]],
        aggregate_fn: Callable = np.max,
        **kwargs
) -> float:
    assert len(preds) == len(targets)
    total_em = 0
    for target_list, pred in zip(targets, preds):
        em_scores = [1.0 if normalize_answer(target) == normalize_answer(pred) else 0.0 for target in target_list]
        aggregated_em = aggregate_fn(em_scores)
        total_em += aggregated_em

    return total_em / len(preds) if preds else 0.0


def compute_hit_score(
        preds: list[str],
        targets: list[list[str]],
        aggregate_fn: Callable = np.max,
        **kwargs
) -> float:
    assert len(preds) == len(targets)
    total_hit = 0
    for target_list, pred in zip(targets, preds):
        hit_scores = [1.0 if normalize_answer(target) in normalize_answer(pred) else 0.0 for target in
                      target_list]
        aggregated_hit = aggregate_fn(hit_scores)
        total_hit += aggregated_hit
    return total_hit / len(preds) if preds else 0.0


def compute_retrieval_score(
        source_embeddings: np.ndarray,
        target_embeddings: np.ndarray,
        threshold: float = 0.8,
        **kwargs,
) -> float:
    similarity_matrix = source_embeddings @ target_embeddings.T
    retrieval_scores = 0.0
    for i in range(similarity_matrix.shape[0]):
        if np.sum(similarity_matrix[i] > threshold) >= 1:
            retrieval_scores += 1.0
        else:
            retrieval_scores += 0.0
    return retrieval_scores / similarity_matrix.shape[0]


def compute_llm_score(
        preds: list[str],
        targets: list[list[str]],
        eval_model: BaseInferenceModel,
        **kwargs,
) -> float:
    questions = kwargs["questions"]
    assert len(preds) == len(targets) == len(questions)
    if len(preds) == 0:
        return 0.0

    user_prompt = """
    You will be given one question, one predicted answer from an llm candidate model, and the ground truth answers. 
    Be as best as you can to evaluate whether the answer generated from the model match one of the ground truth. If 
    be included, response yes, otherwise no. A match is not required to be exactly the same, 
    semantically included can be regarded as a match.
    Please DON'T output quotes when outputting your evaluation. 
    Here is some examples:
    --Example 1--
    Question: What is in the box?
    Candidate answer: a pizza slice.
    Ground truth: [pizza, pizza slice].
    Evaluation: yes

    --Example 2--
    Question: On which side of the image are the chairs?
    Candidate answer: The chairs are on the left side of the image.
    Ground truth: [right]
    Evaluation: no


    --Example 3--
    Question: Who is wearing the hat?
    Candidate answer: the guy at coordinate (410, 163)
    Ground truth: [guy]
    Evaluation: yes

    --Real task--
    Question: {question}
    Candidate answer: {pred}
    Ground truth: {target}
    Evaluation: 
    """
    
    user_content = [user_prompt.format(question=q, pred=p, target=t) for q, p, t in zip(questions, preds, targets)]
    results = eval_model.inference(user_contents=user_content, system_prompt=SYSTEM_PROMPT)
    scores = [normalize_answer(r["response"]) == "yes" for r in results]
    return sum(scores) / len(scores)
