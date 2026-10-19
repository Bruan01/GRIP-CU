import random
from typing import Optional

import numpy as np
from transformers import PreTrainedTokenizer

from constants import ANSWER_TAG
from models import BaseInferenceModel
from .gen_base import GenGraphTaskBase


class GenBridgeTask(GenGraphTaskBase):
    """Compositional bridging task generator.

    Enumerates length-2 train paths ``A -(r1)- B -(r2)- C`` whose composed target
    ``(A, r3, C)`` is ALSO a train edge, then emits a 10-way relation-prediction
    question that *forces* the model to derive ``r3`` from the two hops. This supplies
    the structured two-hop supervision that GRIP otherwise lacks (cf. Two-Hop Curse,
    arXiv 2411.16353; Coverage Principle, arXiv 2505.20278).

    Two modes (controlled by ``bridge_mode``):
      * ``explicit`` : the prompt states the intermediate path before asking for the
                       relation between A and C (composition supervision).
      * ``plain``    : only asks for the relation between A and C (ablation).
    """

    answer_format = f"<{ANSWER_TAG}>" + "{answer}" + f"</{ANSWER_TAG}>"

    template = """Given the context graph titled {title}, please answer the following question: {question} Response in the following format:""" + answer_format.format(
        answer="[answer]")

    q_format = ("What is the relation between word node {src} and word node {tgt}? "
                "Selected from the following candidate answers: {candidates}.")

    explicit_prefix = ("In the context graph, the node {a} is {r1} the node {b}, "
                      "and the node {b} is {r2} the node {c}. ")

    def __init__(
            self,
            graph_list: list,
            title_list: list[str],
            tokenizer: PreTrainedTokenizer,
            task_generator: Optional[BaseInferenceModel] = None,
            task_generator_model_name: str = "qwen-32b",
            task_gen_max_length: int = 1000,
            num_bridge_qa: int = 0,
            bridge_mode: str = "explicit",
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
        self.num_bridge_qa = num_bridge_qa
        self.bridge_mode = bridge_mode

    # ------------------------------------------------------------------ #
    def _collect_chains(self, graph: dict):
        """Return list of (A, r1, B, r2, C, r3) where (A,r3,C) is a train edge."""
        edge_list = graph.get("edge_list", [])
        adj = {}
        edge_set = set()
        rels = set()
        for e in edge_list:
            src, rel, tgt = e[0], e[1], e[2]
            adj.setdefault(src, []).append((rel, tgt))
            edge_set.add((src, tgt, rel))
            rels.add(rel)

        chains = []
        for a, out_a in adj.items():
            for r1, b in out_a:
                for r2, c in adj.get(b, []):
                    if c == a:
                        continue
                    # composed target relation between A and C
                    r3 = None
                    for (rr, tt) in adj.get(a, []):
                        if tt == c:
                            r3 = rr
                            break
                    if r3 is None:
                        continue
                    chains.append((a, r1, b, r2, c, r3))
        return chains, sorted(rels)

    def _make_question(self, chain, rels, rng):
        a, r1, b, r2, c, r3 = chain
        non_target = [x for x in rels if x != r3]
        if len(non_target) >= 9:
            neg = rng.permutation(non_target)[:9].tolist()
        else:
            # not enough distinct relations; sample with replacement from non-target only
            extra = rng.choice(non_target, 9 - len(non_target), replace=True).tolist() if non_target else []
            neg = non_target + extra
        candidates = "; ".join(rng.permutation(neg + [r3]).tolist())
        q = self.q_format.format(src=a, tgt=c, candidates=candidates)
        if self.bridge_mode == "explicit":
            q = self.explicit_prefix.format(a=a, r1=r1, b=b, r2=r2, c=c) + q
        return q, r3

    # ------------------------------------------------------------------ #
    def gen_task(self, gen_empty_task=False) -> list:
        if gen_empty_task or self.num_bridge_qa <= 0:
            return [[] for _ in range(len(self.graph_list))]

        rng = np.random.default_rng(2026)
        results = [[] for _ in range(len(self.graph_list))]
        for gi, (graph, title) in enumerate(zip(self.graph_list, self.title_list)):
            chains, rels = self._collect_chains(graph)
            if not chains:
                continue
            # sample (with replacement if requested volume exceeds pool size)
            idx = rng.choice(len(chains), size=self.num_bridge_qa, replace=len(chains) < self.num_bridge_qa)
            for i in idx:
                q, r3 = self._make_question(chains[i], rels, rng)
                user_prompt = self.template.format(title=title, question=q)
                answer = self.answer_format.format(answer=r3)
                results[gi].append(self.create_chat_message(user_prompt, answer))
        return self.sample_post_process(results)
