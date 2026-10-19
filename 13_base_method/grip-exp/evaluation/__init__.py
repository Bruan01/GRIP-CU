from typing import Any, Union

from numpy import ndarray
from torch import Tensor

from .metrics import get_eval_function


def auto_eval(
        preds: Union[list[str], ndarray, Tensor],
        targets: Union[list[str], ndarray, Tensor],
        metric: str,
        **kwargs) -> Any:
    return get_eval_function(metric)(preds, targets, **kwargs)


def auto_eval_batch(
        preds: Union[list[str], ndarray, Tensor],
        targets: Union[list[str], ndarray, Tensor],
        metrics: list[str],
        **kwargs) -> dict:
    metric_dict = {}
    for metric in metrics:
        metric_dict[metric] = auto_eval(preds, targets, metric, **kwargs)
    return metric_dict
