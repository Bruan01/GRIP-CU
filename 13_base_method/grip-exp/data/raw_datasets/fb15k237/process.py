import json
from os import path as osp

import numpy as np

from utils import make_dir, download_hf_file, save_list_json

HF_REPO_ID = "WFRaain/TAG_datasets"


def process_fb15k237(save_dir: str):
    make_dir(save_dir)
    if not osp.exists(osp.join(save_dir, "entity2wikidata.json")):
        download_hf_file(HF_REPO_ID, subfolder="FB15K237", filename="entity2wikidata.json", local_dir=save_dir)
    if not osp.exists(osp.join(save_dir, "train.txt")):
        download_hf_file(HF_REPO_ID, subfolder="FB15K237", filename="train.txt", local_dir=save_dir)
    if not osp.exists(osp.join(save_dir, "test.txt")):
        download_hf_file(HF_REPO_ID, subfolder="FB15K237", filename="test.txt", local_dir=save_dir)
    if not osp.exists(osp.join(save_dir, "valid.txt")):
        download_hf_file(HF_REPO_ID, subfolder="FB15K237", filename="valid.txt", local_dir=save_dir)
    if not osp.exists(osp.join(save_dir, "fb15k237.json")):
        download_hf_file(HF_REPO_ID, subfolder="FB15K237", filename="fb15k237.json", local_dir=save_dir)

    entity2text = {}
    with open(osp.join(save_dir, "entity2wikidata.json"), "r") as f:
        data = json.load(f)

    for k in data:
        entity = ""
        if data[k]["label"] is not None:
            entity += data[k]["label"] + ", "
        if data[k]["description"] is not None:
            entity += data[k]["description"]
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
    Q_format = ("What is the relation between node {src} and node {tgt}? "
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
        # random select 10000
        selected_index = np.random.choice(len(questions), 10000, replace=False)
        questions = [questions[i] for i in selected_index]
        answers = [answers[i] for i in selected_index]
        data_list = [{"title": "fb15k237", "graph": graph, "questions": questions, "answers": answers, "id": "0"}]
        if name == "valid":
            name = "val"
        save_list_json(osp.join(save_dir, f"processed_{name}.json"), data_list)


if __name__ == "__main__":
    process_fb15k237("outputs/data/fb15k237")
