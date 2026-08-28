"""Small-budget RecurrentGRIP launcher.

This launcher trains graph adapters once and evaluates correct, cyclically shuffled,
and disabled-adapter controls. Original GRIP and GRIP+More-QA remain separate runners
so their original execution path is not modified.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", default="outputs/recurrent_grip/clegr_reasoning/pilot.jsonl")
    parser.add_argument("--training_output_dir", default="outputs/recurrent_grip/trainer")
    parser.add_argument("--model_name", default="qwen-0.5b")
    parser.add_argument("--max_graphs", type=int, default=16)
    parser.add_argument("--involve_qa_epochs", type=int, default=1)
    parser.add_argument("--gen_max_length", type=int, default=32)
    parser.add_argument("--wall_time_limit_minutes", type=int, default=120)
    parser.add_argument("--hard_stop_minutes", type=int, default=180)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("extra_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    extra_args = args.extra_args[1:] if args.extra_args[:1] == ["--"] else args.extra_args
    command = [
        sys.executable,
        str(Path(__file__).with_name("run_recurrent_grip.py")),
        "--input_file", args.input_file,
        "--output_file", args.output_file,
        "--output_dir", args.training_output_dir,
        "--model_name", args.model_name,
        "--max_graphs", str(args.max_graphs),
        "--involve_qa_epochs", str(args.involve_qa_epochs),
        "--gen_max_length", str(args.gen_max_length),
        "--adapter_control", "all",
        "--wall_time_limit_minutes", str(args.wall_time_limit_minutes),
        "--hard_stop_minutes", str(args.hard_stop_minutes),
        "--seed", str(args.seed),
        *extra_args,
    ]
    print("command:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
