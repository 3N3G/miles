"""Read-only before/after scores on the exact batch used for an optimizer step."""

import torch


def compute_update_direction(before, after, *, advantages, behavior, loss_masks, eps_clip, eps_clip_high):
    """Measure improvement of the detached-weight CISPO surrogate on fixed data.

    Positive gain means the update favors positive advantages and suppresses
    negative ones, using the pre-update importance weights. This is a software
    diagnostic, not a measurement of newly generated answer quality.
    """
    gains, positive, negative = [], [], []
    for previous, current, advantage, old, mask in zip(before, after, advantages, behavior, loss_masks, strict=True):
        if not (previous.shape == current.shape == advantage.shape == old.shape == mask.shape):
            raise ValueError("Update-direction scores, advantages and masks must align")
        active = mask.bool()
        if not active.any():
            continue
        previous, current, advantage, old = [x.detach()[active].float() for x in (previous, current, advantage, old)]
        if not all(torch.isfinite(x).all() for x in (previous, current, advantage, old)):
            raise ValueError("Nonfinite active token in update-direction observation")
        weights = torch.exp((previous - old).clamp(-20, 20)).clamp(1 - eps_clip, 1 + eps_clip_high)
        change = current - previous
        gains.append(float((weights * advantage * change).mean()))
        mean_advantage = float(advantage.mean())
        if mean_advantage > 0:
            positive.append(float(change.mean()))
        elif mean_advantage < 0:
            negative.append(float(change.mean()))
    if not gains:
        raise ValueError("Update-direction observation has no active samples")
    result = {
        "surrogate_gain": sum(gains) / len(gains),
        "samples": len(gains),
        "positive_advantage_samples": len(positive),
        "negative_advantage_samples": len(negative),
    }
    if positive:
        result["positive_advantage_mean_logprob_change"] = sum(positive) / len(positive)
    if negative:
        result["negative_advantage_mean_logprob_change"] = sum(negative) / len(negative)
    return result
