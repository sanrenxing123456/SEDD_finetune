"""Conditional score-entropy objective used by SEDD supervised tuning."""

import torch

from model import utils as model_utils


def conditional_score_entropy(
    model,
    noise,
    graph,
    input_ids,
    condition_mask,
    loss_mask,
    sampling_eps=1e-3,
    t=None,
):
    """Corrupt only target positions and compute loss only on supervised tokens."""
    condition_mask = condition_mask.bool()
    loss_mask = loss_mask.bool()
    if torch.any(condition_mask & loss_mask):
        raise ValueError("condition_mask and loss_mask must not overlap")
    if not torch.all(loss_mask.any(dim=-1)):
        raise ValueError("every sample must contain at least one supervised token")

    batch_size = input_ids.shape[0]
    if t is None:
        t = (
            (1.0 - sampling_eps)
            * torch.rand(batch_size, device=input_ids.device)
            + sampling_eps
        )
    sigma, dsigma = noise(t)
    perturbed = graph.sample_transition(input_ids, sigma[:, None])
    perturbed = torch.where(condition_mask, input_ids, perturbed)

    log_score = model_utils.get_score_fn(model, train=model.training)(perturbed, sigma)
    token_loss = graph.score_entropy(
        log_score, sigma[:, None], perturbed, input_ids
    )
    weights = loss_mask.to(token_loss.dtype)
    per_example = (token_loss * weights).sum(dim=-1)
    per_example = per_example / weights.sum(dim=-1).clamp_min(1)
    loss = (dsigma * per_example).mean()

    with torch.no_grad():
        masked_targets = loss_mask & (perturbed == graph.dim - 1)
        predictions = log_score[..., :-1].argmax(dim=-1)
        correct = ((predictions == input_ids) & masked_targets).sum()
        count = masked_targets.sum()
        accuracy = correct.float() / count.clamp_min(1)

    metrics = {
        "denoise_accuracy": accuracy.detach(),
        "correct_tokens": correct.detach(),
        "masked_tokens": count.detach(),
        "mean_t": t.mean().detach(),
    }
    return loss, metrics
