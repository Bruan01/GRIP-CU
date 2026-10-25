"""CPU-only regression for interrupted training-time generation evaluation."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

SCRIPT = Path(__file__).parents[1] / "scripts/train_listed_contrastive.py"


def test_evaluate_variant_resumes_without_redecoding_finished_question(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("train_fb_resume_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    samples = [dict(question_id=f"test:{i}", split="test", question=f"q{i}", answer="a") for i in range(2)]
    first = dict(question_id="test:0", split="test", question="q0", target=["a"], correct=True)
    path = tmp_path / "predictions_correct.jsonl"
    path.write_text(json.dumps(first) + '\n{"question_id":')
    calls = []

    def evaluate(model, tokenizer, record, max_length, *, skip_question_ids, pred_path, splits=None):
        calls.append(skip_question_ids)
        assert skip_question_ids == {"test:0"}
        row = dict(first, question_id="test:1", question="q1")
        module.append_jsonl(pred_path, row)
        return [row]

    monkeypatch.setattr(module, "evaluate_adapter", evaluate)
    monkeypatch.setattr(module, "em_summary", lambda rows, **_: {"all": {"em": 1.0, "count": len(rows)}})
    args = SimpleNamespace(no_resume=False, gen_max_length=32)
    summary = module.evaluate_variant(args, {"recurrent_questions": samples}, tmp_path / "adapter", tmp_path,
                                      model=object(), tokenizer=object())
    assert summary["all"]["count"] == 2
    assert len(module.load_jsonl(path)) == 2
    module.evaluate_variant(args, {"recurrent_questions": samples}, tmp_path / "adapter", tmp_path,
                            model=object(), tokenizer=object())
    assert len(calls) == 1
