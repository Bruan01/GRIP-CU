from os import path as osp

from datasets import load_dataset

from utils import make_dir, save_list_json


def process_webqsp(save_dir: str):
    make_dir(save_dir)
    train_dataset = load_dataset("rmanluo/RoG-webqsp", split="train")
    val_dataset = load_dataset("rmanluo/RoG-webqsp", split="validation")
    test_dataset = load_dataset("rmanluo/RoG-webqsp", split="test")

    for split in ["test", "train", "val"]:
        dataset = train_dataset if split == "train" else val_dataset if split == "val" else test_dataset
        data_list = []
        for data in dataset:
            question = data["question"]
            answer = data["answer"]
            graph = data["graph"]

            question_list = []
            answer_list = []
            q_entity_list = []
            a_entity_list = []
            node_set = set()
            edge_set = set()
            for edge in graph:
                src, rel, tgt = edge
                node_set.add(src)
                node_set.add(tgt)
                edge_set.add((src, rel, tgt))
            question_list.append(question)
            answer_list.append(answer)
            q_entity_list.append(data["q_entity"])
            a_entity_list.append(data["a_entity"])
            node_list = list(node_set)
            edge_list = list(edge_set)
            edge_index = [[node_list.index(edge[0]), node_list.index(edge[2])] for edge in edge_list]
            graph = {
                "node_list": node_list,
                "edge_list": edge_list,
                "edge_index": edge_index,
            }
            id = data["id"]
            data_list.append(
                {
                    "title": f"webqsp graph {id}",
                    "graph": graph,
                    "questions": question_list,
                    "answers": answer_list,
                    "q_entities": q_entity_list,
                    "a_entities": a_entity_list,
                    "id": str(id),
                }
            )
        save_list_json(osp.join(save_dir, f"processed_{split}.json"), data_list)


if __name__ == "__main__":
    process_webqsp("outputs/data/webqsp")
