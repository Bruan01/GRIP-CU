import argparse
import os

from data import *
from utils import set_random_seed

OUTPUT_DIR = "outputs/data"
DATASET_DICT = {
    "scene_graph": process_scene_graph,
    "fb15k237_2": process_fb15k237_2,
    "wn18rr": process_wn18rr,
    "clegr": process_clegr,
    "nell23k": process_nell23k,
    "codexm": process_codexm,
}


def main():
    parser = argparse.ArgumentParser(description="Create deterministic GRIP processed datasets.")
    parser.add_argument(
        "--datasets", nargs="+", choices=sorted(DATASET_DICT), default=sorted(DATASET_DICT),
        help="Datasets to process (defaults to all supported datasets).",
    )
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--overwrite", action="store_true", help="Regenerate a selected dataset even if test data exists.")
    args = parser.parse_args()

    set_random_seed(args.seed)
    for dataset in args.datasets:
        output_path = os.path.join(args.output_dir, dataset)
        test_path = os.path.join(output_path, "processed_test.json")
        if os.path.exists(test_path) and not args.overwrite:
            print(f"Skipping {dataset}: {test_path} already exists. Pass --overwrite to regenerate it.")
            continue
        print(f"Processing {dataset} dataset into {output_path}...")
        DATASET_DICT[dataset](output_path)


if __name__ == "__main__":
    main()
