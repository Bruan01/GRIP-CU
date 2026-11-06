"""Train-frequency-bin matched random negatives; no held-out statistics.

The Full bank supplies only a per-question sequence of log2-frequency bins.
Uniform draws without replacement inside each legal bin keep those quotas and
slot bins exact. Counts inside a bin need not be identical (factor <2 range).
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import math
import random

from .frozen_random import uniform_bank_row

SOURCE = "train_only_frequency_bin_matched_random_without_replacement"
RULE = "match_full_slot_floor_log2_train_edge_count_then_uniform_within_legal_bin"


def frequency_bin(count: int) -> int:
    if type(count) is not int or count < 1:
        raise ValueError("frequency must be a positive integer from train edges")
    return count.bit_length()-1


def matched_bank_row(rows, *, relation_order, known_relations, frequencies,
                     parent_row, seed=2026, k=9):
    # Reuse the strict vocabulary, train split, gold and legal-pool audit.
    row, pool = uniform_bank_row(rows, relation_order=relation_order,
                                known_relations=known_relations, seed=seed, k=k)
    qid = row["question_id"]
    parent = parent_row["negative_relations"]
    if (parent_row["question_id"] != qid or parent_row.get("split") != "train"
            or parent_row["positive_relation"] != row["positive_relation"]
            or parent_row["matched_train_relation"] != row["matched_train_relation"]
            or len(parent) != k or len(set(parent)) != k
            or not set(parent).issubset(pool["legal_relations"])):
        raise ValueError("parent Full identity/K/legality mismatch")
    if set(frequencies) != set(relation_order):
        raise ValueError("train-frequency vocabulary mismatch")
    bins = {name: frequency_bin(count) for name,count in frequencies.items()}
    buckets = defaultdict(list)
    for name in pool["legal_relations"]:
        buckets[bins[name]].append(name)
    slot_bins = [bins[name] for name in parent]
    quota = Counter(slot_bins)
    for bucket, needed in quota.items():
        if len(buckets[bucket]) < needed:
            raise ValueError(f"{qid}: frequency bin {bucket} shortfall; no fallback allowed")
    digest = hashlib.sha256(f"freqmatch-v1::{seed}::{qid}".encode()).digest()
    row_seed = int.from_bytes(digest[:8], "big")
    rng = random.Random(row_seed)
    drawn = {b: iter(rng.sample(buckets[b], quota[b])) for b in sorted(quota)}
    selected = [next(drawn[b]) for b in slot_bins]
    seen = Counter()
    details = []
    for name,b in zip(selected,slot_bins):
        size = len(buckets[b])
        details.append({"relation":name, "train_edge_count":frequencies[name], "frequency_bin":b,
                        "bin_pool_size":size, "bin_quota":quota[b],
                        "slot_initial_probability":1/size, "conditional_probability":1/(size-seen[b]),
                        "inclusion_probability":quota[b]/size, "train_known_absent":True})
        seen[b] += 1
    row.update(negative_source=SOURCE, selection_rule=RULE, negative_relations=selected,
               uniform_negative_relations=selected, selected_negatives=details,
               parent_full_negative_relations=parent, parent_slot_bins=slot_bins,
               parent_train_counts=[frequencies[n] for n in parent],
               frequency_bin_rule="floor_log2_train_edge_count", frequency_matching="exact_slot_bins_not_exact_counts",
               row_seed=row_seed)
    audit = dict(pool, negative_relations=selected, row_seed=row_seed,
                 full_slot_bins=slot_bins, control_slot_bins=[bins[n] for n in selected],
                 bin_quotas={str(b):q for b,q in sorted(quota.items())},
                 legal_bin_sizes={str(b):len(names) for b,names in sorted(buckets.items())},
                 exact_bin_match=True, fallback_count=0,
                 forced_bin_count=sum(len(buckets[b])==q for b,q in quota.items()),
                 forced_slot_count=sum(q for b,q in quota.items() if len(buckets[b])==q))
    validate_frequency_manifest({qid:row}, k=k)
    return row,audit


def validate_frequency_manifest(manifest, *, k=9):
    if not manifest or k < 1:
        raise ValueError("frequency_matched requires a nonempty bank and positive K")
    for qid,row in manifest.items():
        negatives=row.get("negative_relations",[]); selected=row.get("selected_negatives",[])
        parent=row.get("parent_full_negative_relations",[]); bins=row.get("parent_slot_bins",[])
        counts=row.get("parent_train_counts",[])
        if (row.get("question_id") != qid or row.get("split") != "train"
                or row.get("negative_source") != SOURCE or row.get("selection_rule") != RULE
                or row.get("filter_splits") != ["train"] or type(row.get("seed")) is not int
                or any(row.get(f)!=k for f in ("k","actual_k","requested_k"))
                or len(negatives)!=k or len(set(negatives))!=k or len(selected)!=k
                or len(parent)!=k or len(set(parent))!=k or len(bins)!=k or len(counts)!=k
                or row.get("pool_size",0)<k or not row.get("pool_sha256")
                or row.get("hard_negative_relations")!=[] or row.get("uniform_negative_relations")!=negatives
                or row.get("lambda_candidate") is not None
                or [d.get("relation") for d in selected]!=negatives
                or set(negatives)&set(row.get("excluded_train_relations",[]))
                or row.get("positive_relation") in negatives or row.get("matched_train_relation") in negatives
                or row.get("frequency_bin_rule")!="floor_log2_train_edge_count"
                or row.get("frequency_matching")!="exact_slot_bins_not_exact_counts"):
            raise ValueError(f"{qid}: frequency bank identity/K/provenance mismatch")
        quota=Counter(bins); seen=Counter(); sizes={}
        for d,b,parent_count in zip(selected,bins,counts):
            size=d.get("bin_pool_size",0)
            if (frequency_bin(parent_count)!=b or frequency_bin(d["train_edge_count"])!=b
                    or d.get("frequency_bin")!=b or size<quota[b] or size>row["pool_size"]
                    or d.get("bin_quota")!=quota[b] or d.get("train_known_absent") is not True
                    or (b in sizes and sizes[b]!=size)
                    or not math.isclose(d.get("slot_initial_probability",-1),1/size)
                    or not math.isclose(d.get("conditional_probability",-1),1/(size-seen[b]))
                    or not math.isclose(d.get("inclusion_probability",-1),quota[b]/size)):
                raise ValueError(f"{qid}: frequency quota/probability mismatch")
            sizes[b]=size; seen[b]+=1
        if sum(sizes.values())>row["pool_size"]:
            raise ValueError(f"{qid}: impossible legal bin sizes")
