#!/usr/bin/env python3
"""Audit existing NELL HPO evidence on CPU; never train or change selection."""
from __future__ import annotations

import collections
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import csv
import statistics

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
HPO = ROOT / "08_experiments/Hard_Negative_Contrastive_GRIP/results/hpo"
OLD = HPO / "fb_nell_direct_em_20261027/nell23k"
NEW = HPO / "stage2_three_dataset_20261030"
SOURCES: dict[str, str] = {}


def read(path: Path, *, lines: bool = False):
    payload = path.read_bytes()
    SOURCES[str(path)] = hashlib.sha256(payload).hexdigest()
    if lines:
        return [json.loads(line) for line in payload.decode().splitlines() if line.strip()]
    return json.loads(payload)


def errors(rows):
    return dict(collections.Counter(
        "correct" if row["correct"] else
        "wrong_in_list" if row["in_list"] else "wrong_out_of_list"
        for row in rows
    ))


def paired(before, after):
    assert len(before) == len(after)
    assert all(a["question_id"] == b["question_id"] and
               a["question"] == b["question"] and a["target"] == b["target"]
               for a, b in zip(before, after))
    counts = collections.Counter(
        "both_correct" if a["correct"] and b["correct"] else
        "rescue" if b["correct"] else
        "regression" if a["correct"] else "both_wrong"
        for a, b in zip(before, after)
    )
    assert sum(counts.values()) == len(before)
    return {**counts, "count": len(before),
            "net_correct_gain": counts["rescue"] - counts["regression"],
            "comparison": "identical question, target, and candidate-list prompt"}


def main():
    contract = read(OLD / "run_contract.json")
    old_run = read(OLD / "followups/trial0_step8_replay/train_run/run_config.json")
    adapter = Path(old_run["s1_adapter"])
    metadata = read(adapter / "grip_run_metadata.json")
    new_contract = read(NEW / "nell23k/dataset_contract.json")
    true_s1 = read(Path(new_contract["s1_adapter"]).parent / "checkpoint_manifest.json")
    assert old_run["s1_adapter"] == old_run["paths"]["b1_adapter"]
    assert metadata["stage2_epochs"] == 10
    assert true_s1["stage"] == "stage1"
    assert true_s1["task_sha256"] == new_contract["task_sha256"]

    trial_path = OLD / "trials.csv"
    SOURCES[str(trial_path)] = hashlib.sha256(trial_path.read_bytes()).hexdigest()
    trials = list(csv.DictReader(trial_path.open()))
    states = dict(collections.Counter(row["state"] for row in trials))
    pruning = dict(collections.Counter(
        row["user_attrs_pruned_at_step"] for row in trials if row["state"] == "PRUNED"
    ))
    assert len(trials) == 12 and sum(states.values()) == 12
    fail_log = OLD / "trials/trial_0008/train_rung_1.log"
    log = fail_log.read_bytes()
    SOURCES[str(fail_log)] = hashlib.sha256(log).hexdigest()
    assert b"No space left on device" in log

    folders = {
        "b1": "baseline_validation",
        "step8": "followups/trial0_step8_replay/validation_screen",
        "step12": "followups/trial0_step12_extension/validation_screen",
        "step240": "winner_trial_0000/validation_screen",
    }
    predictions = {key: read(OLD / folder / "predictions.jsonl", lines=True)
                   for key, folder in folders.items()}
    validation = {}
    for key, folder in folders.items():
        summary = read(OLD / folder / "summary.json")
        rows = predictions[key]
        assert len(rows) == 300
        assert abs(sum(row["correct"] for row in rows) / len(rows)
                   - summary["all"]["em"]) < 1e-12
        validation[key] = {"em": summary["all"]["em"], "errors": errors(rows)}
    pairs = {"b1_to_" + key: paired(predictions["b1"], predictions[key])
             for key in ("step8", "step12", "step240")}
    pairs["step8_to_step12"] = paired(predictions["step8"], predictions["step12"])

    test8 = read(OLD / "followups/trial0_step8_replay/test/predictions.jsonl", lines=True)
    test240 = read(OLD / "winner_trial_0000/test/predictions.jsonl", lines=True)
    assert all(a["question"] == b["question"] and a["target"] == b["target"]
               for a, b in zip(test8, test240)) and len(test8) == len(test240) == 4943
    old_b1 = read(ROOT / "13_base_method/grip-exp/outputs/grip_inf/nell23k/nell23k_qwen7b_full_seed2026.jsonl", lines=True)
    assert len(old_b1) == len(test8)
    same_target = sum(a["target"] == b["target"] for a, b in zip(old_b1, test8))
    same_stem = sum(a["question"].split("Selected from")[0] == b["question"].split("Selected from")[0]
                    for a, b in zip(old_b1, test8))
    same_prompt = sum(a["question"] == b["question"] for a, b in zip(old_b1, test8))
    def candidate_set(row):
        return {value.strip().rstrip(".") for value in
                row["question"].split("Selected from the following candidate answers: ", 1)[1].split(";")}
    same_candidate_sets = sum(candidate_set(a) == candidate_set(b) for a, b in zip(old_b1, test8))
    assert same_target == same_stem == 4943 and same_prompt == 0
    test = {
        "count": len(test8), "step8_em": sum(r["correct"] for r in test8) / len(test8),
        "step240_em": sum(r["correct"] for r in test240) / len(test240),
        "historical_b1_em": sum(r["response"] in r["target"] for r in old_b1) / len(old_b1),
        "paper_reference": contract["paper_em_reference"],
        "same_targets": same_target, "same_question_stems": same_stem,
        "identical_full_prompts": same_prompt,
        "identical_candidate_sets_ignoring_order": same_candidate_sets,
        "historical_b1_direct_comparability": False,
        "step8_errors": errors(test8),
        "test_used_for_new_model_selection_by_this_audit": False,
    }

    histories = {}
    for label, folder, step in [
        ("step8", "followups/trial0_step8_replay/train_run", 8),
        ("step240", "trials/trial_0000/train_run", 240),
    ]:
        state = read(OLD / folder / f"trainer_listed/checkpoint-{step}/trainer_state.json")
        assert state["max_steps"] == 240
        histories[label] = [{key: item.get(key) for key in
                             ("step", "generation_loss", "candidate_loss", "learning_rate", "epoch")}
                            for item in state["log_history"] if item.get("step") in (1, 2, 8, 240)]

    bank = read(OLD / "k9_from_k15/counterfactual_bank.jsonl", lines=True)
    negatives = [n for row in bank for n in row["selected_negatives"]]
    assert len(bank) == 3253 and len(negatives) == 3253 * 9
    eligibility = read(NEW / "nell23k/assets/relation_eligibility.json")
    gate = {
        "old_relation_rows": len(bank), "new_verified_rows": len(eligibility["eligible_question_ids"]),
        "unverified_rows": len(eligibility["excluded_relation_qa"]),
        "generation_qa_count": eligibility["generation_qa_count"],
        "unverified_is_not_proof_of_wrong_labels": True,
        "median_selected_negative_score_gap": statistics.median(n["score_gap"] for n in negatives),
        "negatives_scoring_at_least_gold": sum(n["candidate_score"] >= n["gold_score"] for n in negatives),
        "negative_count": len(negatives),
        "rows_with_negative_scoring_at_least_gold": sum(any(n["candidate_score"] >= n["gold_score"]
                                                             for n in row["selected_negatives"]) for row in bank),
        "interpretation": "frozen B1 teacher scores only; mixture intentionally includes easy negatives; no isolated causal test",
    }

    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "parent": str(OLD), "question": "Why did prior NELL Stage-2 HPO fail to beat the paper reference?",
        "scope": "CPU audit of existing artifacts; no new GPU job, training, test selection, or controller change",
        "execution_envelope": "Existing RTX3090 is occupied by adopted FB prerequisite; this audit uses Python standard library on CPU",
        "lineage": {"old_initialization": str(adapter), "old_metadata": metadata,
                    "new_stage1": new_contract["s1_adapter"], "new_stage1_manifest": true_s1},
        "search": {"trial_count": len(trials), "states": states, "pruned_at_step": pruning,
                   "rungs": contract["multi_fidelity_optimizer_steps"], "disk_full_failed_trial": 8,
                   "lambda_zero_control_present": any(float(row["params_lambda_candidate"]) == 0 for row in trials)},
        "validation": validation, "paired_validation": pairs, "test": test,
        "training_loss_windows": histories, "bank_audit": gate,
        "inference_boundary": {
            "observed": ["Initialization was already Stage2-trained B1", "300-item validation gain was net 3 questions",
                         "Out-of-list errors rose from 9 to 24 with full continuation", "Most configurations were pruned by step2",
                         "Historical B1 full-test candidate lists differ in all4943 prompts", "One trial failed on disk capacity"],
            "hypotheses_not_proven": ["Continued high-LR training disrupts a strong B1", "Aggressive pruning misses late-improving trials",
                                      "Fixed-bank saturation and train-pair uncertainty limit generalization"],
            "cannot_conclude": ["Method has no benefit", "Only a larger lambda is needed", "Extra epochs guarantee gains",
                                "87.639% minus87.457% is an isolated method effect"],
        },
        "next_route": "experiment: fixed true Stage1; matched Stage2 B1 and same exact evaluation prompts; larger validation and checkpoint selection before test",
        "existing_queued_corrections": {key: new_contract[key] for key in
                                        ("seed", "search_stage", "screen_size", "prune_after", "s1_adapter", "b1_adapter", "expected_listed")},
        "deferred_validation": "lambda=0 matched-budget control; low-LR bounded refinement if still needed. No automatic new runs added by this audit.",
        "paper_role": "reference_only: adaptive exploratory failure diagnosis, not a new positive main result",
        "validation_status": "passed: artifact counts, prompt identity, EM recomputation, lineage and gate invariants",
        "sources_sha256": SOURCES,
    }
    (OUT / "findings.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    report = f"""# NELL Stage2 搜索失败分析

这是对既有实验的 CPU 审计，没有启动训练、改变后台队列或据测试表现选择新参数。
父实验：`{OLD.relative_to(ROOT)}`。结论仅用于决定下一轮实验，不能写成新的正向效果。

1. **训练起点混淆已证实，影响程度尚未隔离。** 旧配置的 s1_adapter 和 b1_adapter 都指向同一个已经完成 Stage2 10 epochs 的 B1。旧实验测量的是 B1 上追加训练，不能当成从固定真实 Stage1 开始的同预算 Stage2 比较。近 1e-3 的学习率伴随第二步生成损失从约0.152升到0.819；与扰动已训练模型相符，但不能仅凭日志证明遗忘机制。

2. **选模证据偏弱。** 300题验证集中 B1 正确254题，8步正确257题；配对修正14题、回退11题，净增3题。单题就是0.333个百分点，不适合分辨约0.1个百分点的目标收益。这里不是统计显著性结论，而是实际观测分辨率。

| 模型 | 相同300题验证EM | 候选内答错 | 候选外答错 |
|---|---:|---:|---:|
| B1 |84.667%|37|9|
| 追加8步 |85.667%|34|9|
| 追加12步 |83.667%|31|18|
| 追加240步 |83.333%|26|24|

3. **继续优化训练目标，没有转化成自由生成收益。** 8步到12步，验证候选内错误减少3题，但候选外错误增加9题。全预算240步测试是{100*test['step240_em']:.3f}%，低于8步的{100*test['step8_em']:.3f}%。训练末期加权候选损失约0.000102、生成损失约0.127908；训练损失很低并不代表测试正确率提高。候选排序监督与自由生成目标存在差异，是有观测支持的机制解释，但仍缺控制实验来隔离因果。

4. **12组搜索并不充分。** {states}；剪枝步数分布为{pruning}。最早1步就淘汰模型，而最终最好的trial0本身也是从第1步70.333%爬到第8步85.667%。这证明早期分数会明显变化，不证明被剪枝的任何一组一定能赢。trial8因磁盘满失败。只有1组完成8步搜索上限并继续到240步；没有lambda=0的匹配对照。

5. **历史完整测试比较口径不同。** 历史B1为{100*test['historical_b1_em']:.3f}%，8步为{100*test['step8_em']:.3f}%；4943条的实体对和gold相同，但完整候选列表提示一致数为0，忽略顺序后的候选集合一致数为{same_candidate_sets}。不能把二者差值当作严格同提示的模型收益。8步仍低于记录中的论文参考87.740%；本审计没有重新核验论文原始提示协议。

6. **训练候选的潜在限制。** 3253条旧对比QA中279条未被新审计确定性验证为train中的实体对/gold；这不等于证明标签错误。原冻结bank的29277个负候选中只有473个教师分数不低于gold，只有224条QA含这类负例；其中位gold-score gap为{gate['median_selected_negative_score_gap']:.5f}。混合采样有意包含容易负例，因此这些数字只能支持“固定bank可能较快饱和”的假设，不能单独判定采样错误。

## 下一步优先级

先用真实固定Stage1，统一候选提示和解码，比较匹配预算的普通Stage2与SC-CL；用验证集选定实际checkpoint后冻结再测试。不要把旧B1追加训练中8步较好，直接解释为真实Stage1启动时也应只训练8步。

已排队配置中已有真实Stage1/独立配套B1、1000题初筛、完整验证终选、32步后剪枝以及2974条可验证train关系QA；12014条生成QA保持不变。这些修正尚未产生新的GPU结果，也没有保证方法会超过论文。

lambda=0同预算对照与更低学习率的局部搜索是后续有价值的鉴别实验；本次没有增加GPU任务。当前单张3090由FB恢复任务占用，新的NELL搜索仍排队，这解释最近没有新结果，不解释旧实验本身为何失败。

复核：运行同目录`analyze.py`，细节、逐项原始路径与SHA256在`findings.json`。本报告不更新论文主表或正向结论；采用的是自适应探索性旧实验，不能当作独立确认实验。
"""
    (OUT / "report.md").write_text(report)
    print(json.dumps({"validation": result["validation_status"], "search": result["search"],
                      "paired_validation": pairs, "test": test, "bank_audit": gate,
                      "report": str(OUT / "report.md")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
