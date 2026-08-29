import random
from typing import Optional

import numpy as np
from transformers import PreTrainedTokenizer

from constants import SYSTEM_PROMPT, ANSWER_TAG, TUPLE_DELIMITER
from grip.tasks.utils import sample_fixed_hop_size_neighbor, clean_graph, shuffle_graph, compute_node_edge_weight, weighted_sampling
from models import BaseInferenceModel
from .gen_base import GenGraphTaskBase


class GenReasoningQATask(GenGraphTaskBase):
    gen_system_prompt = SYSTEM_PROMPT

    gen_multihop_user_prompt = """
    You are given several text snippets from a graph as context. Generate two diverse, reasoning-focused questions and 
    answers based on the context. 
     - The new questions should require multi-hop/indirect reasoning and use ≥2 distinct facts from the text snippets.
     - Avoid trivial lookups (e.g., complete an edge) and avoid quantity questions (“how many”).
     - Use natural phrasing; do not explicitly states “graph”/“node(s)” in question/answer.
     - Keep answers concise (single words or short phrases). 
     - The two new questions must target different facts and differ from each other.
     - Answers must be correct, concise, complete (if multiple elements can be answer, include all of them) and supported by the text snippets.
     - For each question and answer, provide one brief evidence sentence from the context.
     
    text snippets are provided below:
    {context}

    Please first provide the evidence and then provide question and answer in the following format: 
    \nEvidence: [evidence] \nQuestion: [question] \nAnswer: [answer] {tuple_delimiter}
    \nEvidence: [evidence] \nQuestion: [question] \nAnswer: [answer]
    Please DON'T output quotes and separate each questions by {tuple_delimiter}, strictly follow the format. 
    """
    #
    # gen_local_user_prompt = """
    # You are given several text snippets from a graph as context. Generate two diverse, reasoning-focused questions and
    # answers based on the context.
    #  - Each question should focus on retrieving a single fact using partial information (e.g., infer the entity
    #    from an attribute like color, appearance, or infer an attribute from the entity).
    #  - Avoid trivial lookups (e.g., complete an edge) and avoid quantity questions (“how many”).
    #  - Use natural phrasing; do not explicitly states “graph”/“node(s)” in question/answer.
    #  - Keep answers concise (single words or short phrases).
    #  - The two new questions must target different facts and differ from each other.
    #  - Answers must be correct, concise, complete (if multiple elements can be answer, include all of them) and supported by the subgraph.
    #  - For each question and answer, provide one brief evidence sentence from the context.
    #
    # text snippets are provided below:
    # {context}
    #
    # Please first provide the evidence and then provide question and answer in the following format:
    # \nEvidence: [evidence] \nQuestion: [question] \nAnswer: [answer] {tuple_delimiter}
    # \nEvidence: [evidence] \nQuestion: [question] \nAnswer: [answer]
    # Please DON'T output quotes and separate each questions by {tuple_delimiter}, strictly follow the format.
    # """

    gen_global_user_prompt = """
    You are given several text snippets from a graph as context. Generate two diverse, reasoning-focused questions and 
    answers based on the context. 

    - The new questions should focus on reasoning over the full subgraph and use >2 distinct facts from the subgraph.
    - Avoid trivial lookups (e.g., complete an edge) and avoid quantity questions (“how many”).
    - The answer must include >= 2 element and separate them with comma or "and".
    - Use natural phrasing; do not explicitly states “graph”/“node(s)” in question/answer.
    - Keep answers concise (single words or short phrases). 
    - The two new questions must target different facts and differ from each other.
    - Answers must be correct, concise, complete (if multiple elements can be answer, include all of them) and supported by the subgraph.

    text snippets are provided below:
    {context}

    Please first provide the evidence and then provide question and answer in the following format: 
    \nEvidence: [evidence] \nQuestion: [question] \nAnswer: [answer] {tuple_delimiter}
    \nEvidence: [evidence] \nQuestion: [question] \nAnswer: [answer]
    Please DON'T output quotes and separate each questions by {tuple_delimiter}, strictly follow the format. 
    """

    gen_binary_user_prompt = """
    You are given several text snippets from a graph as context. Generate two diverse, reasoning-focused questions and 
    answers based on the context. 

    - One question should be answered yes and another should be answered no.
    - The question can be ask in the following manner: is there, are there, does, can, has is it, et al. 
    - Use natural phrasing; do not explicitly states “graph”/“node(s)” in question/answer.
    - The two new questions must target different facts and differ from each other.
    - Answers must be correct, concise, complete (if multiple elements can be answer, include all of them) and supported by the subgraph.

    text snippets are provided below:
    {context}

    Please first provide the evidence and then provide question and answer in the following format: 
    \nEvidence: [evidence] \nQuestion: [question] \nAnswer: [answer] {tuple_delimiter}
    \nEvidence: [evidence] \nQuestion: [question] \nAnswer: [answer]
    Please DON'T output quotes and separate each questions by {tuple_delimiter}, strictly follow the format. 
    """


    gen_kshot_user_prompt = """
    You are given several text snippets from a graph as context, along with some sampled questions from ANOTHER graph. Create two new 
    questions to test understanding of model on the provided text snippets.
    1. Select two sample questions that best fit the provided text snippets. Two sample questions must differ in at least one dimension:
    Interrogative (who/what/how/is-are/when/where), Format (descriptive/comparative/reasoning/yes-no), 
    Focus (attributes/relationships/numerical details/multi-entity connections)

    2. For each selected sample question, write a new question with the same type and style, but the answer is fully grounded in the text snippets. Specifcially,
    - The new questions should focus on reasoning over the provided text snippets.
    - Avoid trivial lookups (e.g., complete an edge) and avoid quantity questions (“how many”).
    - Use natural phrasing; do not explicitly states “graph”/“node(s)” in question/answer.
    - The two new questions must target different facts and differ from each other.
    - Answers must be correct, concise, complete (if multiple elements can be answer, include all of them).
    - DO NOT leverage any semantic information from sample questions. The new question and answer should grounded by provided text snippets.
    
    text snippets are provided below:
    {context}

    sampled questions are provided below:
    {sample_questions}

    Please first provide the referred sample question and then provide question and answer in the following format: 
    \nReferred question: [referred question] \nQuestion: [question] \nAnswer: [answer] {tuple_delimiter}
    \nReferred question: [referred question] \nQuestion: [question] \nAnswer: [answer]
    Please DON'T output quotes and separate each questions by {tuple_delimiter}, strictly follow the format. 

    """

    gen_s2_user_prompt = """
    You are given several text snippets representing real-world facts as context and an QA pair generated from it. Your task 
    is to evaluate the correctness of question and answer pair based on the text snippets. Here is the instruction:
    1. For each question and answer pair, determine if the question and answer is reasonable and fully grounded by the provided context (answer cannot be I don't know). 
    2. If the question/answer is wrong or not related to the provided context, generate a CORRECTED question/answer based on the context.
    3. If the answer and question are correct but can be further improved, provide an IMPROVED answer that is more concise and precise.
    4. If the answer and question already correct and optimal, simply restate the original question and answer.
        
    The context and QA pairs are provided below:
    --Context--
    {context}
    --QA pair--
    Question : {question}
    Answer : {answer}
    Please first provide whether is QA pair is changed and then provide the final question and answer in the following format: 
    \nChanged: [yes or no] \nQuestion: [question] \nAnswer: [answer] 
    Please DON'T output quotes and strictly follow the format. 
    
    """

    answer_format = f"<{ANSWER_TAG}>" + "{answer}" + f"</{ANSWER_TAG}>"

    template = """Given the context graph titled {title}, please answer the following question: {question} Response in the following format:""" + answer_format.format(
        answer="[answer]")

    def __init__(
            self,
            graph_list: list,
            title_list: list[str],
            tokenizer: PreTrainedTokenizer,
            task_generator: Optional[BaseInferenceModel] = None,
            refer_data: Optional[list[dict]] = None,
            task_generator_model_name: str = "qwen-32b",
            task_gen_max_length: int = 1000,
            num_reason_qa: int = 10,
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
        self.num_qa = num_reason_qa
        self.add_kshot_qa = True
        train_question_list = []
        if refer_data is not None:
            for d in refer_data:
                questions = d["questions"]
                answers = d["answers"]
                # Some datasets may contains root node in the questions.
                for q, a in zip(questions, answers):
                    if isinstance(q, list):
                        q = q[0]
                    qa_pair = "Question: {question}\nAnswer: {answer}".format(question=q.strip(), answer=a.strip())
                    train_question_list.append(qa_pair)
        else:
            # train_question_list = [
            #     "What are the main contributions of Ada Lovelace to the early history of computing?",
            #     "Which differences distinguish the philosophies of Confucianism and Daoism?",
            #     "How did the invention of the printing press influence literacy rates in Europe?",
            #     "How many plays are attributed to William Shakespeare?",
            #     "Who were the key allies of the United States during World War II?",
            #     "When was the Great Wall of China first constructed, and during which dynasty?",
            #     "In what ways did the Industrial Revolution reshape urban life in Britain?",
            #     "What is the relationship between photosynthesis, carbon dioxide, and the global climate system?",
            #     "Could the Apollo 13 mission have succeeded without the improvised solutions developed by NASA engineers?",
            # ]
            self.add_kshot_qa = False
        self.train_question_list = train_question_list

    def gen_task(self, gen_empty_task=False) -> list:
        if gen_empty_task or self.num_qa <= 0:
            return [[] for _ in range(len(self.graph_list))]
        qa_s1_tasks = []
        qa_graph_index = []
        qa_title_list = []
        context_list = []
        for index, (graph, title) in enumerate(zip(self.graph_list, self.title_list)):
            edge_list = graph["edge_list"]
            node_list = graph["node_list"]
            edge_index = np.array(graph["edge_index"])

            # clean graph, remove repeated edges and self-loops
            edge_list, edge_index = clean_graph(edge_list, edge_index)

            # random shuffle the graph order
            node_list, edge_list, edge_index = shuffle_graph(node_list, edge_list, edge_index)
            num_nodes = len(node_list)

            node_weights, edge_weights = compute_node_edge_weight(edge_index, num_nodes)

            question_type = 0
            for _ in range(self.num_qa):

                sample_edge_index = weighted_sampling(list(range(len(edge_list))), edge_weights, 6)
                subgraph_edges = [edge_list[e] for e in sample_edge_index]

                context = ""
                entry_format = "the {src} is {rel} the {tgt}; "
                for entry in subgraph_edges:
                    context += entry_format.format(src=entry[0], rel=entry[1], tgt=entry[2])

                if not self.add_kshot_qa and question_type == 3:
                    question_type = 0

                gen_prompt = [# self.gen_local_user_prompt,
                            self.gen_global_user_prompt,
                            self.gen_multihop_user_prompt,
                            self.gen_binary_user_prompt,
                            self.gen_kshot_user_prompt][question_type]

                if question_type == 3:
                    k_shot_examples = random.sample(self.train_question_list, k=8)
                    k_shot_examples = "; ".join(k_shot_examples)
                    qa_task = gen_prompt.format(context=context, tuple_delimiter=TUPLE_DELIMITER,
                            sample_questions=k_shot_examples)
                else:
                    qa_task = gen_prompt.format(context=context, tuple_delimiter=TUPLE_DELIMITER)

                qa_s1_tasks.append(qa_task.strip())
                context_list.append(context)
                qa_graph_index.append(index)
                qa_title_list.append(title)
                question_type += 1
                question_type = question_type % 4
        qa_s1_result = self.infer_cached("reasoning_qa_s1", qa_s1_tasks, self.gen_system_prompt)
        qa_s2_tasks = []
        qa_graph_s2_index = []
        qa_graph_s2_title = []
        for qa_pairs, index, title, context in zip(qa_s1_result, qa_graph_index, qa_title_list, context_list):
            qa_pairs = qa_pairs["response"].strip()
            qa_pairs = qa_pairs.split(TUPLE_DELIMITER)
            if len(qa_pairs) == 1:
                qa_pairs = qa_pairs[0].split("\n\n")
            for qa_pair in qa_pairs:
                qa_pair = qa_pair.strip().split("\n")
                if len(qa_pair) != 3:
                    continue
                question_type, question, answer = qa_pair
                question = question.split(":")[-1].strip("\n").strip()
                answer = answer.split(":")[-1].strip("\n").strip()
                qa_task = self.gen_s2_user_prompt.format(context=context, question=question, answer=answer)
                qa_s2_tasks.append(qa_task.strip())
                qa_graph_s2_index.append(index)
                qa_graph_s2_title.append(title)

        qa_s2_result = self.infer_cached("reasoning_qa_s2", qa_s2_tasks, self.gen_system_prompt)
        task_gen_results = [[] for _ in range(len(self.graph_list))]
        for qa_pair, index, title in zip(qa_s2_result, qa_graph_s2_index, qa_graph_s2_title):
            qa_pair = qa_pair["response"].strip()
            qa_pair = qa_pair.strip().split("\n")
            if len(qa_pair) != 3:
                continue
            question_type, question, answer = qa_pair
            question = question.split(":")[-1].strip("\n").strip()
            answer = answer.split(":")[-1].strip("\n").strip()
            answer = self.answer_format.format(answer=answer)
            user_prompt = self.template.format(
                title=title,
                question=question,
            )
            qa_train_text = self.create_chat_message(user_prompt, answer)
            task_gen_results[index].append(qa_train_text)
        return self.sample_post_process(task_gen_results)
