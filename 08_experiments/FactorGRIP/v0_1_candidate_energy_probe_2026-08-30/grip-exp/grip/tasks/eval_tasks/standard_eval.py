import warnings

import numpy as np
from scipy.sparse import csr_array
from transformers import PreTrainedTokenizer

from grip.tasks.utils import sample_fixed_hop_size_neighbor, edge_with_index_to_sequence, edge_list_to_sequence
from .base import BaseEvalDataset


class StandardEvalDataset(BaseEvalDataset):
    template = (
        "Given the context graph titled {title}, please answer the following question: {question} Please enclose the answer in <answer></answer> and response in "
        "the following format: <answer>[answer]</answer>. Please DON'T output quotes and strictly follow the format.")
        
    context_format = "This is the graph contains relevant context that can answer the question: {context_graph}"

    def __init__(
            self,
            questions: list,
            answers: list,
            tokenizer: PreTrainedTokenizer,
            graph: dict,
            no_graph_context: bool = False,
            use_subgraph: bool = False,
            index_format: bool = False,
            **kwargs,
    ):
        super().__init__(questions=questions, answers=answers, tokenizer=tokenizer, graph=graph, **kwargs)
        self.no_graph_context = no_graph_context
        self.use_subgraph = use_subgraph
        self.index_format = index_format
        if not self.no_graph_context and not self.use_subgraph:
            if self.index_format:
                self.context_graph = edge_with_index_to_sequence(graph["node_list"], graph["edge_list"], graph["edge_index"], shuffle=True)
            else:
                self.context_graph = edge_list_to_sequence(graph["edge_list"], shuffle=True)
        else:
            self.context_graph = ""

        if self.use_subgraph:
            edge_index = np.array(graph["edge_index"]).T
            num_nodes = edge_index.max() + 1
            self.edge_index = csr_array((np.arange(len(graph["edge_index"])), (edge_index[0], edge_index[1]),),
                                        shape=(num_nodes, num_nodes), )
        else:
            self.edge_index = None

    def __getitem__(self, idx):
        question = self.questions[idx]
        if isinstance(question, list):
            question, roots = question
        else:
            roots = None
        answer = self.answers[idx]
        template = self.template.format(question=question, title=self.title)
        # template = self.template.format(question=question) + self.answer_format

        if self.no_graph_context:
            user_content = template
        else:
            # for question that use subgraph as context, question must contains target node index.
            if self.use_subgraph and not roots:
                warnings.warn("Subgraph sampling is enabled, but no root nodes are provided in evaluation data. "
                              "Automatically use blank graph context. Please double check your evaluation data.")
            if self.use_subgraph and roots:
                nodes, subgraph = sample_fixed_hop_size_neighbor(
                    self.edge_index,
                    np.arange(len(self.graph["edge_index"])),
                    roots,
                    hop=3,
                    max_nodes_per_hop=5,
                )
                if self.index_format:
                    subgraph_edge_list = [self.graph["edge_list"][i] for i in subgraph.data if
                                        i < len(self.graph["edge_list"])]
                    subgraph_node_list = [self.graph["node_list"][i] for i in nodes]   
                    subgraph_coo = subgraph.tocoo()
                    subgraph_edge_index = (np.array([subgraph_coo.row, subgraph_coo.col]).T).tolist()
                    subgraph_edge_index = [index for i, index in enumerate(subgraph_edge_index) if subgraph.data[i] < len(self.graph["edge_list"])]
                    context_graph = edge_with_index_to_sequence(
                        subgraph_node_list,
                        subgraph_edge_list,
                        subgraph_edge_index,
                        shuffle=True
                    )                 
                else:
                    subgraph_edge_list = [self.graph["edge_list"][i] for i in subgraph.data if
                                        i < len(self.graph["edge_list"])]
                    context_graph = edge_list_to_sequence(subgraph_edge_list, shuffle=True)
            else:
                context_graph = self.context_graph
            context = self.context_format.format(context_graph=context_graph)
            user_content = context + "\n" + template

        return user_content, question, answer
