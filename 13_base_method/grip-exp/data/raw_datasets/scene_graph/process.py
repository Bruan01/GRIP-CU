import json
from os import path as osp

import pandas as pd
import torch
from tqdm import tqdm

from utils import make_dir, download_hf_file, extract_zip, save_list_json

HF_REPO_ID = "WFRaain/TAG_datasets"


def textualize_graph(data: dict) -> tuple[list[str], list[list[str]], list[list[int]]]:
    objectid2name = {}
    objectid2nodeid = {object_id: idx for idx, object_id in enumerate(data['objects'].keys())}
    for object_id in data['objects'].keys():
        object_name = data["objects"][object_id]["name"]
        x, y, w, h = (data["objects"][object_id]["x"], data["objects"][object_id]["y"],
                      data["objects"][object_id]["w"], data["objects"][object_id]["h"])
        object_attribute = ", ".join(data["objects"][object_id]["attributes"])
        object_description = f"{object_attribute} {object_name} at coordinate ({x}, {y}) with width {w} and height {h}"
        objectid2name[object_id] = object_description.strip()

    node_list = ["" for _ in range(len(objectid2name))]
    for object_id, node_id in objectid2nodeid.items():
        node_list[node_id] = objectid2name[object_id]

    edge_list = []
    edge_index = []
    for object_id in data["objects"].keys():
        src = objectid2name[object_id]
        for rel_data in data["objects"][object_id]["relations"]:
            rel = rel_data["name"]
            tgt = objectid2name[rel_data["object"]]
            edge = [src, rel, tgt]
            edge_list.append(edge)
            edge_index.append([objectid2nodeid[object_id], objectid2nodeid[rel_data["object"]]])
    return node_list, edge_list, edge_index


def process_scene_graph(save_dir: str):
    make_dir(save_dir)
    if (not osp.exists(osp.join(save_dir, "train_sceneGraphs.json"))
            or not osp.exists(osp.join(save_dir, "val_sceneGraphs.json"))):
        download_hf_file(HF_REPO_ID, subfolder="scenegraph", filename="sceneGraphs.zip", local_dir=save_dir)
        extract_zip(osp.join(save_dir, "sceneGraphs.zip"), save_dir)
    if not osp.exists(osp.join(save_dir, "scene_graph_split.pt")):
        download_hf_file(HF_REPO_ID, subfolder="scenegraph", filename="scene_graph_split.pt", local_dir=save_dir)
    if not osp.exists(osp.join(save_dir, "questions.csv")):
        download_hf_file(HF_REPO_ID, subfolder="scenegraph", filename="questions.csv", local_dir=save_dir)

    dataset = json.load(open(osp.join(save_dir, "train_sceneGraphs.json"), "r"))
    question_df = pd.read_csv(osp.join(save_dir, "questions.csv"))
    graph_split = torch.load(osp.join(save_dir, "scene_graph_split.pt"), weights_only=False)
    image_ids = question_df.image_id.unique()
    train_data_list = []
    val_data_list = []
    test_data_list = []
    for image_id in tqdm(image_ids):
        if image_id in graph_split["train"]:
            data_list = train_data_list
        elif image_id in graph_split["val"]:
            data_list = val_data_list
        else:
            data_list = test_data_list

        object_ = dataset[str(image_id)]
        node_list, edge_list, edge_index = textualize_graph(object_)
        questions = question_df[question_df.image_id == image_id]["question"].tolist()
        answers = question_df[question_df.image_id == image_id]["answer"].tolist()
        full_answers = question_df[question_df.image_id == image_id]["full_answer"].tolist()
        data_list.append({
            "title": f"scene graph {image_id}",
            "id": str(image_id),
            "graph": {
                "node_list": node_list,
                "edge_list": edge_list,
                "edge_index": edge_index,
            },
            "questions": questions,
            "answers": answers,
            "full_answers": full_answers,
        })

    save_list_json(osp.join(save_dir, "processed_train.json"), train_data_list)
    save_list_json(osp.join(save_dir, "processed_val.json"), val_data_list)
    save_list_json(osp.join(save_dir, "processed_test.json"), test_data_list)


if __name__ == "__main__":
    process_scene_graph("outputs/data/scene_graph")
