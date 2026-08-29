import argparse
import json
from pathlib import Path

from grip.tasks.recurrent_tasks import build_dataset_hop_split


def load_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def save_jsonl(path: str, rows: list[dict]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--question_types", nargs="+", default=["StationShortestCount"])
    parser.add_argument("--train_hops", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--validation_hops", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--test_hops", nargs="+", type=int, default=[3, 4])
    parser.add_argument("--validation_fraction", type=float, default=0.2)
    parser.add_argument("--max_graphs", type=int, default=16)
    parser.add_argument("--max_questions_per_hop", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    records = load_jsonl(args.input_file)
    output, stats = build_dataset_hop_split(
        records,
        max_graphs=args.max_graphs,
        train_hops=args.train_hops,
        validation_hops=args.validation_hops,
        test_hops=args.test_hops,
        question_types=args.question_types,
        validation_fraction=args.validation_fraction,
        max_questions_per_hop=args.max_questions_per_hop,
        seed=args.seed,
    )
    save_jsonl(args.output_file, output)
    stats_path = str(Path(args.output_file).with_suffix(Path(args.output_file).suffix + ".stats.json"))
    Path(stats_path).write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"graphs": len(output), "output_file": args.output_file, "stats": stats}, ensure_ascii=False))


if __name__ == "__main__":
    main()
