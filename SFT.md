# Conditional SEDD fine-tuning on s1K-1.1

This repository now contains a minimal end-to-end framework for supervised
fine-tuning SEDD as a conditional, block-wise diffusion model.

## Data representation

Each teacher response is split into fixed-size target blocks. A training row is
laid out as:

```text
[Question + recent reasoning history] [current teacher target] [EOS padding]
             condition_mask=True         loss_mask=True      condition=True
```

Only the target block is corrupted by the absorbing diffusion process. Score
entropy is computed only for teacher tokens selected by `loss_mask`. Splitting
is performed by question before responses are expanded into blocks, preventing
chunks of the same question from leaking across train, validation, and test.

The default layout is 768 context tokens plus a 256-token target block, keeping
the 1024-token context length used by the released SEDD checkpoints.

## 1. Prepare s1K-1.1

```powershell
python prepare_s1.py `
  --dataset simplescaling/s1K-1.1 `
  --output_dir data/s1k_1_1_sedd
```

The command downloads the raw dataset, uses the GPT-2 tokenizer, makes a
deterministic 80/10/10 question-level split, and saves a Hugging Face
`DatasetDict` plus `sft_metadata.json`.

For a quick pipeline smoke test, limit the number of blocks per question:

```powershell
python prepare_s1.py --max_chunks_per_example 2
```

When a response exceeds the cap, preprocessing keeps the first `N - 1` blocks
and the final answer/EOS block. This is useful for a medium-scale experiment,
but omitted middle reasoning should still be restored for the final experiment.

## 2. Train

Edit `configs/sft_s1.yaml`, particularly the pretrained model, batch size,
gradient accumulation, and output directory. Then run:

```powershell
python run_sft_train.py --config configs/sft_s1.yaml
```

OmegaConf dot-list overrides are accepted:

```powershell
python run_sft_train.py --config configs/sft_s1.yaml `
  training.epochs=1 training.eval_every=50
```

The current entry point intentionally targets one CUDA GPU. It retains EMA,
gradient accumulation, mixed precision, validation, periodic checkpoints, and
resume support. The original SEDD model requires a CUDA-compatible Flash
Attention installation.

Periodic checkpoints record the next dataloader batch, so an interrupted epoch
can continue without replaying its completed prefix. Only the two newest
numbered checkpoints are retained by default to limit disk usage; configure
this with `training.keep_step_checkpoints`.

By default, each training record independently samples 10 diffusion times and
10 corresponding corruptions. Their losses are averaged before the optimizer
update. They are evaluated sequentially, so activation memory stays close to a
single time sample while training compute is approximately 10 times larger.
Configure this with:

```yaml
training:
  time_samples_per_example: 10
  eval_time_samples_per_example: 10
```

Setting either value to `1` recovers the original single-time Monte Carlo
estimator. Evaluation accuracy is aggregated over the actual number of masked
target tokens rather than averaging per-batch percentages.

## 3. Evaluate and generate

Evaluate conditional score entropy and masked-token denoising accuracy:

```powershell
python run_sft_eval.py `
  --checkpoint exp_local/sedd_s1_sft/checkpoint_last.pt
```

Run block-wise conditional generation:

```powershell
python run_sft_eval.py `
  --checkpoint exp_local/sedd_s1_sft/checkpoint_last.pt `
  --question "Find all integers n such that ..." `
  --steps 128 `
  --max_blocks 8
```

Generation holds the question and recent reasoning history fixed, initializes
the next target block as all MASK, applies the analytic reverse-diffusion
sampler, and repeats until GPT-2 EOS or `max_blocks`.

## Files

- `prepare_s1.py`: preprocessing CLI.
- `sft/data.py`: s1 field extraction, question-level split, chunk construction.
- `sft/losses.py`: conditional score-entropy objective.
- `sft/sampling.py`: projected conditional sampling and multi-block generation.
- `run_sft_train.py`: training and checkpointing.
- `run_sft_eval.py`: held-out evaluation and interactive generation.
- `tests/test_sft_data.py`: layout and masking tests.

This is a correctness-oriented baseline. It does not yet include multi-GPU
training, LoRA, mathematical answer extraction, or benchmark decontamination.
