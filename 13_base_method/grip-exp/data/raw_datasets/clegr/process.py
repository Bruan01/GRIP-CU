import random
from os import path as osp
import tarfile
import torch

from utils import make_dir, download_hf_file, extract_zip, save_list_json

HF_REPO_ID = "tenseisoham/CLEGR"


def process_clegr(save_dir: str):
    make_dir(save_dir)
    if not osp.exists(osp.join(save_dir, "data_for_hf.tar.gz")):
        download_hf_file(HF_REPO_ID, subfolder="data", filename="data_for_hf.tar.gz", local_dir=save_dir)
        with tarfile.open(osp.join(save_dir, "data_for_hf.tar.gz"), "r:gz") as tar:
            tar.extractall(path=save_dir)

    for dataset in ["clegr-facts", "clegr-reasoning", "clegr-facts-large", "clegr-reasoning-large"]:
        if "large" in dataset:
            dataset_name = "-".join(dataset.split("-")[:-1])
            data_tuple = torch.load(osp.join(save_dir, "datasets_for_hf", "clegr-large",
                                             dataset_name, "processed", "data_list.pt"), weights_only=False)
        else:
            data_tuple = torch.load(osp.join(save_dir, "datasets_for_hf", dataset, "processed", "data_list.pt"), weights_only=False)
        data, slices, data_cls = data_tuple
        data = data_cls.from_dict(data)

        graph_id_dict = {}
        for id in data.graph_id:
            if id not in graph_id_dict:
                graph_id_dict[id] = 0
            graph_id_dict[id] += 1

        edge_slices = 0
        question_slices = 0
        graph_list = []

        for id, num_question in graph_id_dict.items():
            node_list = data.node_texts[question_slices]
            edge_texts = data.edge_texts[question_slices]
            num_edges = len(edge_texts)

            edge_index = data.edge_index[:, edge_slices:edge_slices + num_edges]
            assert edge_index.max() == len(node_list) - 1
            assert edge_index.min() == 0
            edge_index = edge_index.transpose(0, 1).tolist()
            edge_list = [[node_list[edge_index[i][0]], edge_texts[i], node_list[edge_index[i][1]]] for i in
                         range(num_edges)]
            questions = [data.question[j] for j in range(question_slices, question_slices + num_question)]
            answers = [data.label[j] for j in range(question_slices, question_slices + num_question)]
            graph_list.append({
                "title": f"clegr graph {id}",
                "id": id,
                "graph": {
                    "node_list": node_list,
                    "edge_list": edge_list,
                    "edge_index": edge_index,
                },
                "questions": questions,
                "answers": answers,
                "full_answers": answers,
            })
            edge_slices += num_edges * num_question
            question_slices += num_question

        # split
        index_list = list(range(len(graph_list)))
        random.shuffle(index_list)
        num_train = int(0.2 * len(graph_list))
        train_data_list = [graph_list[i] for i in index_list[:num_train]]
        test_data_list = [graph_list[i] for i in index_list[num_train:]]

        dataset = "_".join(dataset.split("-"))
        make_dir(osp.join(osp.dirname(save_dir), dataset))
        save_list_json(osp.join(osp.dirname(save_dir), dataset, "processed_train.json"), train_data_list)
        save_list_json(osp.join(osp.dirname(save_dir), dataset, "processed_test.json"), test_data_list)


if __name__ == "__main__":
    process_clegr("outputs/data/clegr")
