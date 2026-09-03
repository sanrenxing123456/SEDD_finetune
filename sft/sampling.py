"""Block-wise conditional sampling for SEDD supervised models."""

import torch

import sampling
from .data import build_inference_record


@torch.no_grad()
def sample_target_block(
    model,
    graph,
    noise,
    condition_ids,
    condition_mask,
    steps=128,
    predictor="analytic",
    eps=1e-5,
):
    device = next(model.parameters()).device
    condition_ids = condition_ids.to(device)
    condition_mask = condition_mask.to(device).bool()
    if condition_ids.ndim == 1:
        condition_ids = condition_ids.unsqueeze(0)
        condition_mask = condition_mask.unsqueeze(0)

    def projector(x):
        return torch.where(condition_mask, condition_ids, x)

    sampler = sampling.get_pc_sampler(
        graph=graph,
        noise=noise,
        batch_dims=tuple(condition_ids.shape),
        predictor=predictor,
        steps=steps,
        denoise=True,
        eps=eps,
        device=device,
        proj_fun=projector,
    )
    return projector(sampler(model))


@torch.no_grad()
def generate_reasoning(
    model,
    graph,
    noise,
    tokenizer,
    question,
    max_length=1024,
    target_length=256,
    max_question_tokens=384,
    max_blocks=8,
    steps=128,
    predictor="analytic",
):
    """Generate fixed-size diffusion blocks until EOS or max_blocks."""
    history_ids = []
    blocks = []
    device = next(model.parameters()).device
    model.eval()

    for _ in range(max_blocks):
        record = build_inference_record(
            tokenizer=tokenizer,
            question=question,
            history_ids=history_ids,
            max_length=max_length,
            target_length=target_length,
            max_question_tokens=max_question_tokens,
        )
        condition_ids = torch.tensor(record["input_ids"], dtype=torch.long, device=device)
        condition_mask = torch.tensor(record["condition_mask"], dtype=torch.bool, device=device)
        sampled = sample_target_block(
            model,
            graph,
            noise,
            condition_ids,
            condition_mask,
            steps=steps,
            predictor=predictor,
        )[0]
        generated = sampled[~condition_mask].tolist()
        if tokenizer.eos_token_id in generated:
            eos_index = generated.index(tokenizer.eos_token_id)
            generated = generated[:eos_index]
            blocks.append(generated)
            history_ids.extend(generated)
            break
        blocks.append(generated)
        history_ids.extend(generated)

    text = tokenizer.decode(history_ids, skip_special_tokens=True)
    return {"text": text, "token_ids": history_ids, "blocks": blocks}
