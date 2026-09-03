"""Download s1K and build block-wise conditional SEDD training records."""

import argparse
import json

from sft.data import prepare_s1_dataset


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="simplescaling/s1K-1.1")
    parser.add_argument("--output_dir", default="data/s1k_1_1_sedd")
    parser.add_argument("--tokenizer", default="gpt2")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--validation_ratio", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--target_length", type=int, default=256)
    parser.add_argument("--max_question_tokens", type=int, default=384)
    parser.add_argument(
        "--max_chunks_per_example",
        type=int,
        default=0,
        help="0 keeps all reasoning chunks; positive values truncate long traces.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    metadata = prepare_s1_dataset(
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        tokenizer_name=args.tokenizer,
        cache_dir=args.cache_dir,
        seed=args.seed,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        max_length=args.max_length,
        target_length=args.target_length,
        max_question_tokens=args.max_question_tokens,
        max_chunks_per_example=args.max_chunks_per_example,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
