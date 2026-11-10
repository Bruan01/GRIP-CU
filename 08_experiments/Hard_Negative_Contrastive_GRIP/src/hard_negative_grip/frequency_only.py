"""Independent train-edge-frequency negatives, with sequential weighted draws.

The selection API accepts no teacher scores, structural profiles, or Full bank.
At each draw a remaining legal relation has probability f(r)/sum_remaining f.
"""
from __future__ import annotations

import hashlib
import math
import random

from .frozen_random import pool_digest

SOURCE = "train_only_edge_frequency_proportional_without_replacement"
RULE = "sequential_pps_without_replacement_weight_equals_train_edge_count"


def frequency_only_bank_row(*, question_id, positive_relation, matched_train_relation,
                            relation_order, known_relations, frequencies, seed=2026, k=9):
    vocabulary = set(relation_order)
    if (not question_id or type(seed) is not int or type(k) is not int or k < 1
            or len(vocabulary) != len(relation_order) or set(frequencies) != vocabulary
            or matched_train_relation not in vocabulary
            or any(type(v) is not int or v < 1 for v in frequencies.values())):
        raise ValueError("invalid frequency-only train vocabulary/counts/identity")
    excluded = set(known_relations) | {matched_train_relation}
    if not excluded <= vocabulary:
        raise ValueError("known train relations outside vocabulary")
    pool = sorted(vocabulary - excluded)
    if len(pool) < k:
        raise ValueError("frequency-only legal pool shortfall; no fallback")
    row_seed = int.from_bytes(hashlib.sha256(f"freqonly-v1::{seed}::{question_id}".encode()).digest()[:8], "big")
    rng = random.Random(row_seed)
    remaining = pool.copy()
    total = sum(frequencies[n] for n in pool)
    weight_left = total
    selected = []
    for _ in range(k):
        ticket = rng.randrange(weight_left)
        cumulative = 0
        for name in remaining:
            cumulative += frequencies[name]
            if ticket < cumulative:
                break
        selected.append(dict(relation=name, train_edge_count=frequencies[name],
                             initial_probability=frequencies[name]/total,
                             conditional_probability=frequencies[name]/weight_left,
                             remaining_total_frequency=weight_left, train_known_absent=True))
        remaining.remove(name)
        weight_left -= frequencies[name]
    names = [d["relation"] for d in selected]
    row = dict(question_id=question_id, split="train", positive_relation=positive_relation,
               matched_train_relation=matched_train_relation, gold_relation=matched_train_relation,
               negative_source=SOURCE, selection_rule=RULE, filter_splits=["train"],
               negative_relations=names, hard_negative_relations=[], uniform_negative_relations=names,
               k=k, actual_k=k, requested_k=k, seed=seed, row_seed=row_seed,
               pool_size=len(pool), pool_sha256=pool_digest(pool), pool_total_frequency=total,
               excluded_train_relations=sorted(excluded), selected_negatives=selected,
               frequency_exponent=1, uses_full_bank=False, uses_teacher_scores=False,
               uses_structural_profiles=False)
    validate_frequency_only_manifest({question_id: row}, k=k)
    return row, dict(question_id=question_id, legal_relations=pool,
                     pool_sha256=pool_digest(pool), pool_total_frequency=total,
                     excluded_train_relations=sorted(excluded), row_seed=row_seed)


def validate_frequency_only_manifest(manifest, *, k=9):
    if not manifest or type(k) is not int or k < 1:
        raise ValueError("frequency_only needs a nonempty bank and positive K")
    for qid, row in manifest.items():
        names = row.get("negative_relations", [])
        details = row.get("selected_negatives", [])
        total = row.get("pool_total_frequency", 0)
        if (row.get("question_id") != qid or row.get("split") != "train"
                or row.get("negative_source") != SOURCE or row.get("selection_rule") != RULE
                or row.get("filter_splits") != ["train"] or type(row.get("seed")) is not int
                or any(row.get(f) != k for f in ("k", "actual_k", "requested_k"))
                or len(names) != k or len(set(names)) != k or len(details) != k
                or [d.get("relation") for d in details] != names
                or row.get("hard_negative_relations") != []
                or row.get("uniform_negative_relations") != names
                or row.get("pool_size", 0) < k or not row.get("pool_sha256")
                or type(total) is not int or total < row["pool_size"]
                or row.get("frequency_exponent") != 1 or row.get("lambda_candidate") is not None
                or any(row.get(f) is not False for f in ("uses_full_bank", "uses_teacher_scores", "uses_structural_profiles"))
                or "parent_full_negative_relations" in row or "parent_slot_bins" in row
                or set(names) & set(row.get("excluded_train_relations", []))
                or row.get("positive_relation") in names or row.get("matched_train_relation") in names):
            raise ValueError(f"{qid}: frequency-only identity/K/provenance mismatch")
        remaining = total
        for d in details:
            count = d.get("train_edge_count")
            if (type(count) is not int or not 0 < count <= remaining
                    or d.get("remaining_total_frequency") != remaining
                    or d.get("train_known_absent") is not True
                    or not math.isclose(d.get("initial_probability", -1), count/total)
                    or not math.isclose(d.get("conditional_probability", -1), count/remaining)):
                raise ValueError(f"{qid}: frequency-only draw probability mismatch")
            remaining -= count
        if remaining < row["pool_size"] - k:
            raise ValueError(f"{qid}: impossible remaining train frequencies")
