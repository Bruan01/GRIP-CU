"""Remove explicitly audited discarded weights, preserving paper evidence.

This is a one-off cleanup with fixed paths, not an automatic retention daemon.
Use --apply only after reviewing plan.json. Active FB and incomplete CoDEx runs
are excluded. Checkpoint JSON, training_args.bin, configs and tokenizers remain.
"""
import argparse
import hashlib
import json
import os
import shutil
import struct
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
HNG = REPO / "08_experiments/Hard_Negative_Contrastive_GRIP"
NELL = HNG / "results/hpo/fb_nell_direct_em_20261027/nell23k"
RUNS = HNG / "results/runs"
BASE = REPO / "13_base_method/grip-exp"
ARCHIVE = Path("/mnt/e/GRIP-CU/hpo_checkpoint_archive/fb_nell_direct_em_20261027/nell23k")
REMOVE_NAMES = {"adapter_model.safetensors", "adapter_model.bin", "optimizer.pt", "scheduler.pt", "rng_state.pth"}


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sample_sha(path):
    """Identify discarded binaries without rereading tens of GB of retired weights."""
    h = hashlib.sha256()
    size = path.stat().st_size
    h.update(str(size).encode())
    with path.open("rb") as stream:
        h.update(stream.read(1024 * 1024))
        if size > 1024 * 1024:
            stream.seek(max(1024 * 1024, size - 1024 * 1024))
            h.update(stream.read(1024 * 1024))
    return h.hexdigest()


def validate_adapter(path):
    if not (path / "adapter_config.json").is_file():
        raise RuntimeError(f"Retained adapter config missing: {path}")
    weight = path / "adapter_model.safetensors"
    with weight.open("rb") as stream:
        length = struct.unpack("<Q", stream.read(8))[0]
        if not 0 < length < 32 * 1024 * 1024:
            raise RuntimeError(f"Invalid retained safetensors header: {weight}")
        header = json.loads(stream.read(length))
    ends = [value["data_offsets"][1] for key, value in header.items() if key != "__metadata__"]
    if not ends or 8 + length + max(ends) != weight.stat().st_size:
        raise RuntimeError(f"Truncated retained adapter: {weight}")
    return weight


def active_output_roots():
    roots = []
    for proc in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            args = proc.read_bytes().decode(errors="replace").split("\0")
        except (OSError, PermissionError):
            continue
        for flag in ("--output_dir", "--resume_from_checkpoint", "--resume_stage1_checkpoint", "--resume_stage2_checkpoint"):
            if flag in args:
                index = args.index(flag)
                if index + 1 < len(args) and args[index + 1].startswith("/"):
                    roots.append(Path(args[index + 1]).resolve())
    return roots


def build_plan():
    groups = []
    retained = []
    states = {row["number"]: row["state"] for row in read(NELL / "trials.json")}
    for item in read(NELL / "checkpoint_archive_manifest.json")["moved"]:
        folder = Path(item["archive"]).resolve()
        if not folder.is_relative_to(ARCHIVE) or states.get(item["trial"]) != "PRUNED":
            raise RuntimeError(f"Archive is not a certified pruned trial: {item}")
        groups.append({"path": str(folder), "reason": "PRUNED NELL trial; metrics, predictions, trial DB and configs retained", "trial": item["trial"]})

    completed = [
        (RUNS / "nell23k_structure_counterfactual_3090_20261021_06_train", [230, 240]),
        (NELL / "trials/trial_0000/train_run", [239, 240]),
        (RUNS / "wn18rr_sccl_hpo_s2026_lc0p25", [100, 125]),
    ]
    for root, checkpoints in completed:
        metadata = read(root / "listed/adapter/run_metadata.json")
        if (read(root / "stage_status.json").get("status") != "completed"
                or metadata["trainer_global_step"] != metadata["trainer_max_steps"]):
            raise RuntimeError(f"Run is not complete: {root}")
        retained.append(validate_adapter(root / "listed/adapter"))
        for step in checkpoints:
            groups.append({"path": str(root / f"trainer_listed/checkpoint-{step}"),
                           "reason": "Completed full-budget run; complete final adapter retained"})

    baseline = BASE / "outputs/paper_runs/wn18rr_taskfile_3090"
    if read(baseline / "summary.json").get("status") != "complete":
        raise RuntimeError("WN baseline is not complete")
    for stage, steps in [(1, [300, 320]), (2, [125])]:
        retained.append(validate_adapter(baseline / f"stage{stage}_adapter"))
        for step in steps:
            groups.append({"path": str(baseline / f"stage{stage}_checkpoints/checkpoint-{step}"),
                           "reason": "Completed WN B1 training; both final stage adapters retained"})

    # Keep the newest resumable checkpoint of a partially trained candidate.
    partial = RUNS / "wn18rr_sccl_hpo_s2026_lc0p5"
    retained.append(validate_adapter(partial / "listed/adapter"))
    newest = partial / "trainer_listed/checkpoint-50"
    if read(newest / "trainer_state.json")["global_step"] != 50 or not (newest / "optimizer.pt").is_file():
        raise RuntimeError("WN partial candidate has no newer resumable checkpoint")
    groups.append({"path": str(partial / "trainer_listed/checkpoint-25"),
                   "reason": "Superseded checkpoint; checkpoint-50 and candidate adapter retained"})

    for path in [BASE / "artifacts/nell23k_paper_7b/lora_adapter",
                 NELL / "followups/trial0_step8_replay/train_run/listed/adapter",
                 NELL / "followups/trial0_step12_extension/train_run/listed/adapter",
                 NELL / "followups/trial7_step8_replay/train_run/listed/adapter",
                 RUNS / "wn18rr_sccl_hpo_s2026_lc2p0/listed/adapter"]:
        retained.append(validate_adapter(path))

    # These paths are needed for later continuation, including the current job.
    protected_trees = [
        HNG / "results/hpo/nell_fb_guard_20261029",
        NELL / "followups/trial0_step8_replay",
        NELL / "followups/trial0_step12_extension",
        NELL / "followups/trial7_step8_replay",
        partial / "trainer_listed/checkpoint-50",
        RUNS / "wn18rr_sccl_hpo_s2026_lc2p0",
        BASE / "outputs/paper_runs/codexm_taskfile_3090",
        Path("/mnt/e/GRIP-CU/model_cache"),
        REPO / "04_benchmark/data",
    ] + active_output_roots()
    files = []
    nonweight = {}
    for group in groups:
        folder = Path(group["path"]).resolve()
        if any(folder.is_relative_to(p.resolve()) or p.resolve().is_relative_to(folder) for p in protected_trees):
            raise RuntimeError(f"Cleanup overlaps an active/protected tree: {folder}")
        for file in folder.rglob("*"):
            if not file.is_file():
                continue
            if file.is_symlink():
                raise RuntimeError(f"Unexpected symlink inside cleanup target: {file}")
            if file.name in REMOVE_NAMES:
                stat = file.stat()
                files.append({"path": str(file), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                              "nlink": stat.st_nlink, "reason": group["reason"]})
            else:
                nonweight[str(file)] = sha(file)
    # Historical predictions/score dumps are the evidence for paper tables and
    # failure analysis. Their hashes are checked after removal as well.
    for folder in [NELL, RUNS / "nell23k_structure_counterfactual_3090_20261021_06_train",
                   HNG / "results/hpo/wn18rr_sccl_seed2026", REPO / "11_paper"]:
        for file in folder.rglob("*"):
            if file.is_file() and not file.is_symlink() and file.suffix in {".json", ".jsonl", ".csv", ".md", ".tex", ".bib", ".sqlite3"}:
                nonweight[str(file)] = sha(file)
    return {"created_at": now(), "authorized_by": "User asked to delete unused weights while preserving paper work",
            "groups": groups, "files": files, "logical_bytes": sum(row["bytes"] for row in files),
            "retained_adapter_files": sorted(set(map(str, retained))),
            "protected_trees": list(map(str, protected_trees)), "evidence_sha256": nonweight,
            "policy": "Delete only named weight/optimizer/RNG binaries; preserve all other files and current/incomplete runs"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    plan = build_plan()
    if not args.apply:
        write(HERE / "plan.json", plan)
        print(json.dumps({"files": len(plan["files"]), "gib": plan["logical_bytes"] / 2**30,
                          "groups": len(plan["groups"]), "retained_adapters": len(plan["retained_adapter_files"])}), flush=True)
        return
    reviewed = read(HERE / "plan.json")
    ledger = HERE / "deleted_files.jsonl"
    previous = {}
    if ledger.exists():
        for line in ledger.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if not Path(row["path"]).exists():
                previous[row["path"]] = row
    remaining = [row for row in reviewed["files"] if row["path"] not in previous]
    if plan["files"] != remaining:
        raise RuntimeError("Weight inventory changed since planning; refusing deletion")
    plan = reviewed
    if (HERE / "result.json").exists():
        raise RuntimeError("This one-off cleanup already has a result")
    before = {drive: shutil.disk_usage(drive)._asdict() for drive in ["/mnt/c", "/mnt/e"]}
    preserved = read(HERE / "retained_adapters.json") if (HERE / "retained_adapters.json").exists() else {}
    if preserved and set(preserved) != set(plan["retained_adapter_files"]):
        raise RuntimeError("Retained adapter inventory changed")
    cache_mtime = (HERE / "retained_adapters.json").stat().st_mtime_ns if preserved else 0
    for file in plan["retained_adapter_files"]:
        if file not in preserved or Path(file).stat().st_mtime_ns > cache_mtime:
            print(f"Hashing retained adapter {file}", flush=True)
            preserved[file] = sha(Path(file))
    write(HERE / "retained_adapters.json", preserved)
    removed = list(previous.values())
    for index, row in enumerate(remaining):
        path = Path(row["path"])
        stat = path.stat()
        if stat.st_size != row["bytes"] or stat.st_mtime_ns != row["mtime_ns"]:
            raise RuntimeError(f"Weight changed after review: {path}")
        # Final paper models receive full SHA256 before and after cleanup.
        # Retired binaries use an explicitly labelled content sample hash.
        record = {**row, "content_sample_sha256": sample_sha(path),
                  "content_sample_rule": "SHA256(decimal byte size + first MiB + last MiB); not a full-file checksum",
                  "deleted_at": now()}
        with (HERE / "deleted_files.jsonl").open("a", encoding="utf-8") as journal:
            journal.write(json.dumps(record, ensure_ascii=False) + "\n")
            journal.flush()
            os.fsync(journal.fileno())
        path.unlink()
        removed.append(record)
        print(f"Deleted {index+1}/{len(plan['files'])}: {path}", flush=True)
    for file, digest in preserved.items():
        if sha(Path(file)) != digest:
            raise RuntimeError(f"Retained adapter changed: {file}")
    for file, digest in plan["evidence_sha256"].items():
        if sha(Path(file)) != digest:
            raise RuntimeError(f"Paper evidence changed: {file}")
    for group in plan["groups"]:
        write(Path(group["path"]) / "WEIGHTS_REMOVED_20261029.json", {
            "removed_at": now(), "reason": group["reason"], "ledger": str(HERE / "deleted_files.jsonl"),
            "note": "Weight binaries were deleted. Remaining checkpoint state/config files are provenance, not a resumable checkpoint."})
    after = {drive: shutil.disk_usage(drive)._asdict() for drive in before}
    result = {"completed_at": now(), "deleted_files": len(removed),
              "deleted_logical_bytes": sum(row["bytes"] for row in removed),
              "disk_before": before, "disk_after": after,
              "retained_adapters_verified": len(preserved), "evidence_files_verified": len(plan["evidence_sha256"]),
              "active_fb_state": read(HNG / "results/hpo/nell_fb_guard_20261029/status.json")["action"],
              "deleted_files_ledger": str(HERE / "deleted_files.jsonl"), "validation": "passed"}
    write(HERE / "result.json", result)
    write(REPO / "11_paper/artifacts/weight_retention_20261029.json", {
        "recorded_at": now(), "cleanup_result": str(HERE / "result.json"),
        "retained_adapter_sha256": preserved,
        "retention_boundary": "Final paper adapters, selected NELL candidates and resumable/incomplete runs retained; discarded-trial binaries and completed-run optimizer checkpoints removed. Historical metrics/predictions/configs verified unchanged."})
    archive_manifest = read(NELL / "checkpoint_archive_manifest.json")
    archive_manifest["permanent_cleanup_20261029"] = {
        "result": str(HERE / "result.json"), "ledger": str(HERE / "deleted_files.jsonl"),
        "note": "Moved entries above describe historical archival; their weight binaries have now been permanently deleted under user instruction. Other provenance files remain."}
    write(NELL / "checkpoint_archive_manifest.json", archive_manifest)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
