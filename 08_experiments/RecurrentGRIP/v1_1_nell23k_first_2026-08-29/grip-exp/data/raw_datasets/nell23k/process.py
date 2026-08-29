import json
from os import path as osp

import numpy as np

from utils import make_dir, download_hf_file, save_list_json
import shutil

def process_nell23k(save_dir: str):
    make_dir(save_dir)
    shutil.copy2("data/raw_datasets/nell23k/train.txt", osp.join(save_dir, "train.txt"))
    shutil.copy2("data/raw_datasets/nell23k/valid.txt", osp.join(save_dir, "valid.txt"))
    shutil.copy2("data/raw_datasets/nell23k/test.txt", osp.join(save_dir, "test.txt"))
    shutil.copy2("data/raw_datasets/nell23k/entity2text.json", osp.join(save_dir, "entity2text.json"))

    entity2text = {}
    with open(osp.join(save_dir, "entity2text.json"), "r") as f:
        data = json.load(f)

    for k in data:
        entity = k
        # if data[k]["label"] is not None:
        #     entity += data[k]["label"] + ", "
        # if data[k]["description"] is not None:
        #     entity += data[k]["description"]
        if not entity:
            entity = "missing"
        entity2text[k] = entity.strip().strip(".")
    entity2id = {entity: i for i, entity in enumerate(entity2text.keys())}

    file_path = osp.join(save_dir, "train.txt")
    edge_list = []
    edge_index = []
    unique_rel = set()

    with open(file_path) as f:
        file_data = [line.split() for line in f.read().split("\n")[:-1]]
    unknown_entity = 0
    for triplet in file_data:
        if triplet[0] not in entity2id:
            entity2text[triplet[0]] = "Unknown"
            entity2id[triplet[0]] = len(entity2id)
            unknown_entity += 1
        if triplet[2] not in entity2id:
            entity2text[triplet[2]] = "Unknown"
            entity2id[triplet[2]] = len(entity2id)
            unknown_entity += 1
        rel = triplet[1]
        unique_rel.add(rel)
        edge_list.append([entity2text[triplet[0]], rel, entity2text[triplet[2]]])
        edge_index.append([entity2id[triplet[0]], entity2id[triplet[2]]])

    graph = {
        "edge_index": edge_index,
        "edge_list": edge_list,
        "node_list": list(entity2text.values()),
    }
    unique_rel = list(unique_rel)
    WAY = 10
    Q_format = ("What is the relation between word node {src} and word node {tgt}? "
                "Selected from the following candidate answers: {candidates}.")
    for name in ["valid", "test"]:
        file_path = osp.join(save_dir, name + ".txt")
        with open(file_path) as f:
            file_data = [line.split() for line in f.read().split("\n")[:-1]]
        unknown_entity = 0
        questions = []
        answers = []
        for triplet in file_data:
            if triplet[0] not in entity2id:
                entity2text[triplet[0]] = "Unknown"
                entity2id[triplet[0]] = len(entity2id)
                unknown_entity += 1
            if triplet[2] not in entity2id:
                entity2text[triplet[2]] = "Unknown"
                entity2id[triplet[2]] = len(entity2id)
                unknown_entity += 1
            rel = triplet[1]
            unique_rel_copy = unique_rel.copy()
            unique_rel_copy.remove(rel)
            negative = np.random.permutation(unique_rel_copy)[:WAY - 1]
            candidates = "; ".join(np.random.permutation(negative.tolist() + [rel]))
            q = Q_format.format(src=entity2text[triplet[0]], tgt=entity2text[triplet[2]], candidates=candidates)
            a = rel
            questions.append([q, [entity2id[triplet[0]], entity2id[triplet[2]]]])
            answers.append(a)
        # random select 3000
        # selected_index = np.random.choice(len(questions), 3000, replace=False)
        # questions = [questions[i] for i in selected_index]
        # answers = [answers[i] for i in selected_index]
        # breakpoint()
        data_list = [{"title": "nell23k", "graph": graph, "questions": questions, "answers": answers, "id": "0"}]
        if name == "valid":
            name = "val"
        save_list_json(osp.join(save_dir, f"processed_{name}.json"), data_list)


if __name__ == "__main__":
    process_nell23k("outputs/data/nell23k")
