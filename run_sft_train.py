"""Single-GPU supervised fine-tuning entry point for conditional SEDD."""

import argparse
import math
import os
import random
import time

import numpy as np
import torch
from datasets import load_from_disk
from omegaconf import OmegaConf
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from load_model import load_model
from model.ema import ExponentialMovingAverage
from sft.data import load_sft_metadata, make_dataloader
from sft.losses import conditional_score_entropy


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/sft_s1.yaml")
    parser.add_argument(
        "overrides",
        nargs="*",
        help="OmegaConf dot-list overrides, e.g. training.epochs=1",
    )
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def autocast_context(precision):
    if precision == "bf16":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    if precision == "fp16":
        return torch.autocast("cuda", dtype=torch.float16)
    return torch.autocast("cuda", enabled=False)


def move_batch(batch, device):
    return {
        key: batch[key].to(device, non_blocking=True)
        for key in ("input_ids", "condition_mask", "loss_mask")
    }


@torch.no_grad()
def evaluate(model, noise, graph, loader, cfg, device):
    model.eval()
    losses = []
    accuracies = []
    for batch_index, batch in enumerate(loader):
        if batch_index >= cfg.training.max_eval_batches:
            break
        batch = move_batch(batch, device)
        with autocast_context(cfg.training.precision):
            loss, metrics = conditional_score_entropy(
                model,
                noise,
                graph,
                **batch,
                sampling_eps=cfg.training.sampling_eps,
            )
        losses.append(loss.float())
        accuracies.append(metrics["denoise_accuracy"].float())
    model.train()
    return {
        "loss": torch.stack(losses).mean().item(),
        "denoise_accuracy": torch.stack(accuracies).mean().item(),
    }


def save_checkpoint(path, model, optimizer, scheduler, ema, cfg, epoch, step):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "ema": ema.state_dict(),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "pretrained_model": cfg.pretrained_model,
        "epoch": epoch,
        "step": step,
    }
    temporary_path = path + ".tmp"
    torch.save(payload, temporary_path)
    os.replace(temporary_path, path)


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))
    if not torch.cuda.is_available():
        raise RuntimeError("SEDD uses Flash Attention; CUDA is required for SFT training")
    seed_everything(cfg.seed)
    device = torch.device("cuda")

    metadata = load_sft_metadata(cfg.data_dir)
    datasets = load_from_disk(cfg.data_dir)
    train_loader = make_dataloader(
        datasets["train"],
        cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
    )
    validation_loader = make_dataloader(
        datasets["validation"],
        cfg.training.eval_batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
    )

    model, graph, noise = load_model(cfg.pretrained_model, device)
    if not graph.absorb:
        raise ValueError("The conditional SFT implementation currently requires absorb graph")
    if model.config.model.length != metadata["max_length"]:
        raise ValueError(
            f"Processed length {metadata['max_length']} does not match model length "
            f"{model.config.model.length}"
        )
    model.train()
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )
    updates_per_epoch = math.ceil(
        len(train_loader) / cfg.training.gradient_accumulation_steps
    )
    total_updates = max(updates_per_epoch * cfg.training.epochs, 1)
    warmup_updates = int(total_updates * cfg.training.warmup_ratio)

    def lr_scale(step):
        if warmup_updates and step < warmup_updates:
            return float(step + 1) / warmup_updates
        progress = (step - warmup_updates) / max(total_updates - warmup_updates, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    scheduler = LambdaLR(optimizer, lr_scale)
    ema = ExponentialMovingAverage(model.parameters(), decay=cfg.training.ema)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.training.precision == "fp16")
    start_epoch = 0
    global_step = 0

    if cfg.training.resume_from:
        checkpoint = torch.load(cfg.training.resume_from, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        ema.load_state_dict(checkpoint["ema"])
        start_epoch = checkpoint["epoch"]
        global_step = checkpoint["step"]

    os.makedirs(cfg.output_dir, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(cfg.output_dir, "config.yaml"))
    optimizer.zero_grad(set_to_none=True)
    accumulation = cfg.training.gradient_accumulation_steps
    started = time.time()

    for epoch in range(start_epoch, cfg.training.epochs):
        for batch_index, batch in enumerate(train_loader):
            batch = move_batch(batch, device)
            group_start = (batch_index // accumulation) * accumulation
            group_size = min(accumulation, len(train_loader) - group_start)
            with autocast_context(cfg.training.precision):
                loss, metrics = conditional_score_entropy(
                    model,
                    noise,
                    graph,
                    **batch,
                    sampling_eps=cfg.training.sampling_eps,
                )
                scaled_loss = loss / group_size
            scaler.scale(scaled_loss).backward()

            is_update = (batch_index + 1) % accumulation == 0 or (
                batch_index + 1 == len(train_loader)
            )
            if not is_update:
                continue
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            ema.update(model.parameters())
            global_step += 1

            if global_step % cfg.training.log_every == 0:
                elapsed = time.time() - started
                print(
                    f"step={global_step} epoch={epoch + 1} loss={loss.item():.5f} "
                    f"accuracy={metrics['denoise_accuracy'].item():.4f} "
                    f"lr={scheduler.get_last_lr()[0]:.3e} elapsed={elapsed:.1f}s"
                )

            if global_step % cfg.training.eval_every == 0:
                ema.store(model.parameters())
                ema.copy_to(model.parameters())
                values = evaluate(model, noise, graph, validation_loader, cfg, device)
                ema.restore(model.parameters())
                print(
                    f"validation step={global_step} loss={values['loss']:.5f} "
                    f"accuracy={values['denoise_accuracy']:.4f}"
                )

            if global_step % cfg.training.save_every == 0:
                save_checkpoint(
                    os.path.join(cfg.output_dir, f"checkpoint_{global_step}.pt"),
                    model,
                    optimizer,
                    scheduler,
                    ema,
                    cfg,
                    epoch,
                    global_step,
                )

        save_checkpoint(
            os.path.join(cfg.output_dir, "checkpoint_last.pt"),
            model,
            optimizer,
            scheduler,
            ema,
            cfg,
            epoch + 1,
            global_step,
        )


if __name__ == "__main__":
    main()
