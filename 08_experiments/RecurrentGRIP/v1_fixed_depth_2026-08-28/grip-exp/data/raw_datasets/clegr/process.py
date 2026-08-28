import random
from os import path as osp
import tarfile

import torch

from utils import download_hf_file, make_dir, save_list_json

HF_REPO_ID = "tenseisoham/CLEGR"


def _slice_optional_field(data, field_name: str, start: int, count: int):
    values = getattr(data, field_name, None)
    if values is None:
        return None
    return [values[index] for index in range(start, start + count)]


def process_clegr(save_dir: str):
    make_dir(save_dir)
    if not osp.exists(osp.join(save_dir, "data_for_hf.tar.gz")):
        download_hf_file(HF_REPO_ID, subfolder="data", filename="data_for_hf.tar.gz", local_dir=save_dir)
        with tarfile.open(osp.join(save_dir, "data_for_hf.tar.gz"), "r:gz") as tar:
            tar.extractall(path=save_dir)

    for dataset in ["clegr-facts", "clegr-reasoning", "clegr-facts-large", "clegr-reasoning-large"]:
        if "large" in dataset:
            dataset_name = "-".join(dataset.split("-")[:-1])
            data_tuple = torch.load(
                osp.join(
                    save_dir,
                    "datasets_for_hf",
                    "clegr-large",
                    dataset_name,
                    "processed",
                    "data_list.pt",
                ),
                weights_only=False,
            )
        else:
            data_tuple = torch.load(
                osp.join(save_dir, "datasets_for_hf", dataset, "processed", "data_list.pt"),
                weights_only=False,
            )
        data, slices, data_cls = data_tuple
        data = data_cls.from_dict(data)

        graph_id_dict = {}
        for graph_id in data.graph_id:
            if graph_id not in graph_id_dict:
                graph_id_dict[graph_id] = 0
            graph_id_dict[graph_id] += 1

        edge_slices = 0
        question_slices = 0
        graph_list = []

        for graph_id, num_question in graph_id_dict.items():
            node_list = data.node_texts[question_slices]
            edge_texts = data.edge_texts[question_slices]
            num_edges = len(edge_texts)

            edge_index = data.edge_index[:, edge_slices:edge_slices + num_edges]
            assert edge_index.max() == len(node_list) - 1
            assert edge_index.min() == 0
            edge_index = edge_index.transpose(0, 1).tolist()
            edge_list = [
                [node_list[edge_index[i][0]], edge_texts[i], node_list[edge_index[i][1]]]
                for i in range(num_edges)
            ]
            questions = _slice_optional_field(data, "question", question_slices, num_question)
            answers = _slice_optional_field(data, "label", question_slices, num_question)
            question_types = _slice_optional_field(data, "question_type", question_slices, num_question)
            question_groups = _slice_optional_field(data, "question_group", question_slices, num_question)
            question_subgroups = _slice_optional_field(data, "question_subgroup", question_slices, num_question)
            record = {
                "title": f"clegr graph {graph_id}",
                "id": graph_id,
                "graph": {
                    "node_list": node_list,
                    "edge_list": edge_list,
                    "edge_index": edge_index,
                },
                "questions": questions,
                "answers": answers,
                "full_answers": answers,
            }
            if question_types is not None:
                record["question_types"] = question_types
            if question_groups is not None:
                record["question_groups"] = question_groups
            if question_subgroups is not None:
                record["question_subgroups"] = question_subgroups
            graph_list.append(record)
            edge_slices += num_edges * num_question
            question_slices += num_question

        index_list = list(range(len(graph_list)))
        random.shuffle(index_list)
        num_train = int(0.2 * len(graph_list))
        train_data_list = [graph_list[i] for i in index_list[:num_train]]
        test_data_list = [graph_list[i] for i in index_list[num_train:]]

        output_dataset = "_".join(dataset.split("-"))
        make_dir(osp.join(osp.dirname(save_dir), output_dataset))
        save_list_json(osp.join(osp.dirname(save_dir), output_dataset, "processed_train.json"), train_data_list)
        save_list_json(osp.join(osp.dirname(save_dir), output_dataset, "processed_test.json"), test_data_list)


if __name__ == "__main__":
    process_clegr("outputs/data/clegr")
