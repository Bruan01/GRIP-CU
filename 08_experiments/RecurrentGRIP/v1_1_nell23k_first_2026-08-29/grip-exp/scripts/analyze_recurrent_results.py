import argparse
import csv
import json
from pathlib import Path

from evaluation.recurrent_metrics import summarize_recurrent_predictions


def load_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _write_bucket_csv(path: Path, buckets: dict[str, dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["bucket", "count", "accuracy"])
        for bucket, values in buckets.items():
            writer.writerow([bucket, values["count"], values["accuracy"]])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    rows = load_jsonl(args.input_file)
    summary = summarize_recurrent_predictions(rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_bucket_csv(output_dir / "hop_k_accuracy.csv", summary["by_hop_and_k"])
    _write_bucket_csv(output_dir / "k_adapter_accuracy.csv", summary["by_k_and_adapter"])
    _write_bucket_csv(
        output_dir / "split_k_adapter_accuracy.csv",
        summary["by_split_k_and_adapter"],
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
