from typing import Optional

import numpy as np
from transformers import PreTrainedTokenizer

from constants import ANSWER_TAG, SYSTEM_PROMPT
from grip.tasks.utils import shuffle_graph, clean_graph, compute_node_edge_weight
from models import BaseInferenceModel
from .gen_base import GenGraphTaskBase


def renormalize_weights(weights, clip=100):
    """
    Normalize weights to [0, 1], then scale so min is 1 and max is 'clip'.
    Args:
        weights (list of float or int): The input weights.
        clip (int): The maximum allowed value for the scaled weights.
    Returns:
        list of int: The scaled, integer, and clipped weights.
    """
    min_weight = min(weights)
    max_weight = max(weights)
    if max_weight == min_weight:
        # All weights are the same, set all to clip
        return [clip for _ in weights]
    # Normalize to [0, 1]
    normalized = [(w - min_weight) / (max_weight - min_weight) for w in weights]
    # Scale to [1, clip]
    scaled = [n * (clip - 1) + 1 for n in normalized]
    # Convert to integer and clip
    int_scaled = [min(max(int(round(s)), 1), clip) for s in scaled]
    return int_scaled

class GenGraphContextTask(GenGraphTaskBase):

    instruction_template = "Recite the information in the context graph {title} accurately."

    node_context_format = "In context graph {title}, there is a node {node}."

    edge_context_format = "In context graph {title}, the node {src} is {rel} the node {tgt}."


    def __init__(
            self,
            graph_list: list,
            title_list: list[str],
            tokenizer: PreTrainedTokenizer,
            task_generator: Optional[BaseInferenceModel] = None,
            task_generator_model_name: str = "qwen-32b",
            task_gen_max_length: int = 1000,
            context_upsampling: bool = True,
            format_as_instruction: bool = False,
            **kwargs,
    ):
        super().__init__(
            graph_list=graph_list,
            title_list=title_list,
            tokenizer=tokenizer,
            task_generator=task_generator,
            task_generator_model_name=task_generator_model_name,
            task_gen_max_length=task_gen_max_length,
            **kwargs,
        )
        self.context_upsampling = context_upsampling
        self.format_as_instruction = format_as_instruction

    def gen_task(self, gen_empty_task=False) -> list:
        if gen_empty_task:
            return [[] for _ in range(len(self.graph_list))]

        train_context_result = []
        for index, (graph, title) in enumerate(zip(self.graph_list, self.title_list)):
            sample_train_context = []
            node_list = graph["node_list"]
            edge_list = graph["edge_list"]
            edge_index = np.array(graph["edge_index"])

            # clean graph, remove repeated edges and self-loops
            edge_list, edge_index = clean_graph(edge_list, edge_index)

            # random shuffle the graph order
            node_list, edge_list, edge_index = shuffle_graph(node_list, edge_list, edge_index)

            num_node = len(node_list)
            num_edge = len(edge_list)

            if self.context_upsampling:
                node_weight, edge_weight = compute_node_edge_weight(edge_index, num_node)
                node_up_sampling = renormalize_weights(node_weight, 3)
                edge_up_sampling = renormalize_weights(edge_weight, 3)
            else:
                node_up_sampling = [1 for _ in range(num_node)]
                edge_up_sampling = [1 for _ in range(num_edge)]

            for node, up_sample_num in zip(node_list, node_up_sampling):
                node_text = self.node_context_format.format(node=node, title=title)
                for _ in range(up_sample_num):
                    if self.format_as_instruction:
                        answer = node_text
                        sample = self.instruction_template.format(title=title)
                        sample = self.create_chat_message(sample, answer)
                    else:
                        sample = node_text
                    sample_train_context.append(sample)

            # plain edge information
            for edge, up_sample_num in zip(edge_list, edge_up_sampling):
                edge_text = self.edge_context_format.format(src=edge[0], tgt=edge[2], rel=edge[1], title=title)
                for _ in range(up_sample_num):
                    if self.format_as_instruction:
                        answer = edge_text
                        sample = self.instruction_template.format(title=title)
                        sample = self.create_chat_message(sample, answer)
                    else:
                        sample = edge_text
                    sample_train_context.append(sample)

            train_context_result.append(sample_train_context)

        return self.sample_post_process(train_context_result)
