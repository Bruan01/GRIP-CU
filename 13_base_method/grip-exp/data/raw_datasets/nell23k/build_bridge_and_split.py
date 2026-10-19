"""Build NELL23K novel-composition eval splits and (optionally) bridge-training data.

This script is PURE PYTHON (no LLM, no torch) and mirrors the graph construction
logic in ``data/raw_datasets/nell23k/process.py`` so that the produced
``processed_test_*.json`` files are byte-for-byte compatible with the existing
``infer_eval_nell23k.py`` loader.

Splits are defined purely by TRAIN-GRAPH TOPOLOGY, so no test triple is ever used
as a training label (zero leakage):

* ``direct``      : test triple (s, r, t) where s and t share a DIRECT train edge.
* ``novel_2hop``  : test triple (s, r, t) where s and t have NO direct train edge
                    but are connected by >=1 length-2 train path s-r1-B-r2-t.
* ``pattern_novel`` (subset of novel_2hop):
                    the composition pattern (r1, r2) -> r never appears among
                    train length-2 chains, i.e. true generalization.

Outputs (process.py format):
    outputs/data/nell23k/processed_test_direct.json
    outputs/data/nell23k/processed_test_novel2hop.json
    outputs/data/nell23k/processed_test_patternnovel.json
    outputs/data/nell23k/split_stats.json
"""

import argparse
import json
import os
from os import path as osp

import numpy as np


def save_list_json(file_path, data):
    """Write JSONL (one object per line) to match the repo's loader."""
    with open(file_path, "w", encoding="utf-8") as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")


def load_triples(path):
    triples = []
    with open(path) as f:
        for line in f.read().split("\n")[:-1]:
            parts = line.split()
            if len(parts) == 3:
                triples.append((parts[0], parts[1], parts[2]))
    return triples


def build_entity_maps(train_triples, test_triples):
    """Replicate process.py entity->text / entity->id construction."""
    entity2text = {}
    with open("data/raw_datasets/nell23k/entity2text.json") as f:
        data = json.load(f)
    for k in data:
        entity = k
        if not entity:
            entity = "missing"
        entity2text[k] = entity.strip().strip(".")

    def ensure(e):
        if e not in entity2text:
            entity2text[e] = "Unknown"

    for h, _, t in train_triples:
        ensure(h)
        ensure(t)
    for h, _, t in test_triples:
        ensure(h)
        ensure(t)

    entity2id = {e: i for i, e in enumerate(entity2text.keys())}
    return entity2text, entity2id


def build_train_graph(train_triples, entity2text, entity2id):
    edge_list, edge_index = [], []
    unique_rel = set()
    for h, r, t in train_triples:
        unique_rel.add(r)
        edge_list.append([entity2text[h], r, entity2text[t]])
        edge_index.append([entity2id[h], entity2id[t]])
    graph = {
        "edge_index": edge_index,
        "edge_list": edge_list,
        "node_list": list(entity2text.values()),
    }
    return graph, sorted(unique_rel)


def build_adjacency(edge_index, num_nodes):
    out_adj = [[] for _ in range(num_nodes)]
    in_adj = [[] for _ in range(num_nodes)]
    rel_out = {}
    for (s, t) in edge_index:
        out_adj[s].append(t)
        in_adj[t].append(s)
    # relation lookup by (s,t) pair -- NELL23K edges are unique per pair
    pair_rel = {}
    return out_adj, in_adj, pair_rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/raw_datasets/nell23k")
    ap.add_argument("--out-dir", default="outputs/data/nell23k")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--way", type=int, default=10)
    ap.add_argument("--max-per-split", type=int, default=0,
                    help="0 = use all; otherwise cap each split (for fast 0.5B ablations)")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    train_triples = load_triples(osp.join(args.data_dir, "train.txt"))
    valid_triples = load_triples(osp.join(args.data_dir, "valid.txt"))
    test_triples = load_triples(osp.join(args.data_dir, "test.txt"))

    entity2text, entity2id = build_entity_maps(train_triples, test_triples)
    graph, unique_rel = build_train_graph(train_triples, entity2text, entity2id)

    num_nodes = len(entity2id)
    edge_index = graph["edge_index"]
    pair_rel = {}  # (s,t) -> relation (train)
    for (s, t), (h, r, tt) in zip(edge_index, graph["edge_list"]):
        pair_rel[(s, t)] = r

    # adjacency
    out_adj = [[] for _ in range(num_nodes)]
    for s, t in edge_index:
        out_adj[s].append(t)

    # direct train edge set
    direct_train_pairs = set(pair_rel.keys())

    # --- 2-hop train paths: mid_paths[(s,t)] = list of (r1, r2, mid) ---
    mid_paths = {}
    for (s, t), (h, r1, _) in zip(edge_index, graph["edge_list"]):
        for mid in out_adj[t]:
            if mid == s:
                continue
            r2 = pair_rel.get((t, mid))
            if r2 is None:
                continue
            mid_paths.setdefault((s, mid), []).append((r1, r2, t))

    # --- 3-hop reachability (exactly 3 steps, no shorter path) ---
    two_hop_neighbors = {s: set(ts) for s, ts in mid_paths.items()}
    three_hop_neighbors = {}
    for s in range(num_nodes):
        seen = set([s]) | set(out_adj[s]) | two_hop_neighbors.get(s, set())
        for u in two_hop_neighbors.get(s, set()):
            for v in out_adj[u]:
                if v not in seen:
                    three_hop_neighbors.setdefault(s, set()).add(v)

    # --- bridge-training chain count: 2-hop chains (A-r1-B-r2-C) where (A,r3,C) is ALSO a train edge ---
    bridge_chain_count = 0
    chain_to_target = {}  # (r1,r2) -> set(r3) for pattern_novel labeling
    for (s, t), chains in mid_paths.items():
        r3 = pair_rel.get((s, t))
        if r3 is None:
            continue
        bridge_chain_count += len(chains)
        for (r1, r2, _) in chains:
            chain_to_target.setdefault((r1, r2), set()).add(r3)

    Q_FORMAT = ("What is the relation between word node {src} and word node {tgt}? "
                "Selected from the following candidate answers: {candidates}.")

    def make_record(triples, split_name):
        questions, answers = [], []
        for h, r, t in triples:
            if h not in entity2id or t not in entity2id:
                continue
            sid, tid = entity2id[h], entity2id[t]
            rels = unique_rel.copy()
            if r not in rels:
                rels.append(r)
            rels.remove(r)
            neg = rng.permutation(rels)[: args.way - 1]
            candidates = "; ".join(rng.permutation(list(neg) + [r]).tolist())
            q = Q_FORMAT.format(src=entity2text[h], tgt=entity2text[t], candidates=candidates)
            questions.append([q, [sid, tid]])
            answers.append(r)
        rec = [{"title": "nell23k", "graph": graph, "questions": questions,
                "answers": answers, "id": "0"}]
        out_path = osp.join(args.out_dir, f"processed_test_{split_name}.json")
        # JSONL (one object per line) to match the repo's load_list_json / save_list_json.
        save_list_json(out_path, rec)
        return len(questions), out_path

    # partition test triples
    direct_ts, novel2_ts, novel3_ts, pnovel_ts = [], [], [], []
    for trip in test_triples:
        h, r, t = trip
        sid = entity2id.get(h)
        tid = entity2id.get(t)
        if sid is None or tid is None:
            continue
        if (sid, tid) in direct_train_pairs:
            direct_ts.append(trip)
            continue
        chains = mid_paths.get((sid, tid))
        if chains:
            novel2_ts.append(trip)
            # pattern_novel: none of the chains' (r1,r2) maps to r in chain_to_target
            is_pnovel = True
            for (r1, r2, _) in chains:
                if r in chain_to_target.get((r1, r2), set()):
                    is_pnovel = False
                    break
            if is_pnovel:
                pnovel_ts.append(trip)
        elif tid in three_hop_neighbors.get(sid, set()):
            novel3_ts.append(trip)

    # novel_composition = union of 2-hop and 3-hop indirect compositions
    novel_comp_ts = novel2_ts + novel3_ts

    def cap(ts):
        if args.max_per_split and len(ts) > args.max_per_split:
            idx = rng.choice(len(ts), args.max_per_split, replace=False)
            return [ts[i] for i in idx]
        return ts

    n_direct, p_direct = make_record(cap(direct_ts), "direct")
    n_novel, p_novel = make_record(cap(novel2_ts), "novel2hop")
    n_novel3, p_novel3 = make_record(cap(novel3_ts), "novel3hop")
    n_comp, p_comp = make_record(cap(novel_comp_ts), "novel_composition")
    n_pnovel, p_pnovel = make_record(cap(pnovel_ts), "patternnovel")

    stats = {
        "seed": args.seed,
        "way": args.way,
        "train_edges": len(train_triples),
        "valid_edges": len(valid_triples),
        "test_edges_total": len(test_triples),
        "bridge_training_chains (2-hop w/ direct target)": bridge_chain_count,
        "split_direct": n_direct,
        "split_novel2hop": n_novel,
        "split_novel3hop": n_novel3,
        "split_novel_composition": n_comp,
        "split_patternnovel": n_pnovel,
        "paths_direct": p_direct,
        "paths_novel2hop": p_novel,
        "paths_novel3hop": p_novel3,
        "paths_novel_composition": p_comp,
        "paths_patternnovel": p_pnovel,
    }
    with open(osp.join(args.out_dir, "split_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
