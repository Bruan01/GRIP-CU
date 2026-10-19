from os import path as osp

import numpy as np

from utils import make_dir, download_hf_file, save_list_json

HF_REPO_ID = "WFRaain/TAG_datasets"


def process_wn18rr(save_dir: str):
    make_dir(save_dir)
    if not osp.exists(osp.join(save_dir, "entity2text.txt")):
        download_hf_file(HF_REPO_ID, subfolder="WN18RR", filename="entity2text.txt", local_dir=save_dir)
    if not osp.exists(osp.join(save_dir, "train.txt")):
        download_hf_file(HF_REPO_ID, subfolder="WN18RR", filename="train.txt", local_dir=save_dir)
    if not osp.exists(osp.join(save_dir, "test.txt")):
        download_hf_file(HF_REPO_ID, subfolder="WN18RR", filename="test.txt", local_dir=save_dir)
    if not osp.exists(osp.join(save_dir, "valid.txt")):
        download_hf_file(HF_REPO_ID, subfolder="WN18RR", filename="valid.txt", local_dir=save_dir)
    if not osp.exists(osp.join(save_dir, "wn18rr.json")):
        download_hf_file(HF_REPO_ID, subfolder="WN18RR", filename="wn18rr.json", local_dir=save_dir)

    entity2text = {}
    with open(osp.join(save_dir, "entity2text.txt"), "r") as f:
        lines = f.readlines()
        for line in lines:
            tmp = line.strip().split("\t")
            entity2text[tmp[0]] = tmp[1]

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
        rel = " ".join(rel.split("_")[1:])
        unique_rel.add(rel)
        edge_list.append([entity2text[triplet[0]], rel, entity2text[triplet[2]]])
        edge_index.append([entity2id[triplet[0]], entity2id[triplet[2]]])

    graph = {
        "edge_index": edge_index,
        "edge_list": edge_list,
        "node_list": list(entity2text.values()),
    }
    unique_rel = list(unique_rel)
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
            rel = " ".join(rel.split("_")[1:])
            candidates = "; ".join(np.random.permutation(unique_rel))
            q = Q_format.format(src=entity2text[triplet[0]], tgt=entity2text[triplet[2]], candidates=candidates)
            a = rel
            questions.append([q, [entity2id[triplet[0]], entity2id[triplet[2]]]])
            answers.append(a)

        data_list = [{"title": "wn18rr", "graph": graph, "questions": questions, "answers": answers, "id": "0"}]
        if name == "valid":
            name = "val"
        save_list_json(osp.join(save_dir, f"processed_{name}.json"), data_list)


if __name__ == "__main__":
    process_wn18rr("outputs/data/wn18rr")
