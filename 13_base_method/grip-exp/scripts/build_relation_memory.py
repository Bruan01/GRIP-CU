"""Build an auditable CECR relation memory from a TSV training graph."""

import argparse
from pathlib import Path
import sys

# Allow ``python scripts/build_relation_memory.py`` from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grip.relation_cecr import build_relation_memory_from_triples, save_relation_memory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-triples-file', required=True)
    parser.add_argument('--output-file', required=True)
    parser.add_argument('--max-support-examples', type=int, default=8)
    args = parser.parse_args()

    memory = build_relation_memory_from_triples(
        args.train_triples_file,
        max_support_examples=args.max_support_examples,
    )
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    save_relation_memory(memory, args.output_file)
    print(f'Wrote {len(memory.cards)} relation cards to {args.output_file}')


if __name__ == '__main__':
    main()
