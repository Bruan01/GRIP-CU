from typing import Any

from transformers import HfArgumentParser

from .base_args import BaseArguments
from .eval_args import EvalArguments
from .inf_args import InferenceArguments
from .model_args import ModelArguments
from .task_args import TaskArguments
from .train_args import CustomTrainingArguments


def parse_args(
        class_clusters: tuple[Any | tuple[Any]],
        no_dict: tuple[Any],
        return_config: bool = False) -> tuple:
    class_set = set()
    for cluster in class_clusters:
        if isinstance(cluster, tuple):
            class_set.update(set(cluster))
        else:
            class_set.add(cluster)
    class_tuple = tuple(class_set)
    parser = HfArgumentParser(class_tuple)
    arg_list = parser.parse_args_into_dataclasses()
    arg_dict = {c: a for c, a in zip(class_tuple, arg_list)}
    returns = ()
    for cluster in class_clusters:
        if isinstance(cluster, tuple):
            temp = {}
            for item in cluster:
                temp.update(dict(vars(arg_dict[item])))
            returns += (temp,)
        else:
            if cluster in no_dict:
                returns += (arg_dict[cluster],)
            else:
                returns += (dict(vars(arg_dict[cluster])),)
    if return_config:
        config = {}
        for arg in arg_list:
            config.update({k: v for k, v in dict(vars(arg)).items() if isinstance(v, int | float | bool | str)})
        return returns, config
    return returns
