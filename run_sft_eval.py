"""Evaluate conditional denoising and optionally generate a reasoning answer."""

import argparse

import torch
from datasets import load_from_disk
from omegaconf import OmegaConf
from transformers import GPT2TokenizerFast

from load_model import load_model
from model.ema import ExponentialMovingAverage
from run_sft_train import evaluate
from sft.data import load_sft_metadata, make_dataloader
from sft.sampling import generate_reasoning


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=("validation", "test"))
    parser.add_argument("--question", default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--max_blocks", type=int, default=None)
    parser.add_argument("--raw_weights", action="store_true", help="Do not use EMA weights")
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("SEDD evaluation requires CUDA and Flash Attention")
    device = torch.device("cuda")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    cfg = OmegaConf.create(checkpoint["config"])
    model, graph, noise = load_model(checkpoint["pretrained_model"], device)
    model.load_state_dict(checkpoint["model"])
    if not args.raw_weights and "ema" in checkpoint:
        ema = ExponentialMovingAverage(model.parameters(), decay=cfg.training.ema)
        ema.load_state_dict(checkpoint["ema"])
        ema.copy_to(model.parameters())

    datasets = load_from_disk(cfg.data_dir)
    loader = make_dataloader(
        datasets[args.split],
        cfg.training.eval_batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
    )
    metrics = evaluate(model, noise, graph, loader, cfg, device)
    print(
        f"{args.split} loss={metrics['loss']:.5f} "
        f"denoise_accuracy={metrics['denoise_accuracy']:.4f}"
    )

    if args.question:
        metadata = load_sft_metadata(cfg.data_dir)
        tokenizer = GPT2TokenizerFast.from_pretrained(metadata["tokenizer_name"])
        result = generate_reasoning(
            model=model,
            graph=graph,
            noise=noise,
            tokenizer=tokenizer,
            question=args.question,
            max_length=metadata["max_length"],
            target_length=metadata["target_length"],
            max_question_tokens=metadata["max_question_tokens"],
            max_blocks=args.max_blocks or cfg.sampling.max_blocks,
            steps=args.steps or cfg.sampling.steps,
            predictor=cfg.sampling.predictor,
        )
        print("\nGenerated reasoning:\n")
        print(result["text"])


if __name__ == "__main__":
    main()
