import pytest
import torch
import torch.distributed as dist
from tests.fast.backends.training_utils.loss.loss_test_utils import make_args, make_parallel_state
from tests.fast.dist_utils import init_gloo, run_multiprocess

from miles.backends.training_utils.data.context_parallel import get_sum_of_sample_mean
from miles.backends.training_utils.loss.hub import losses as losses_module
from miles.backends.training_utils.loss.hub.logit_processors import _observed_sampling_entropy
from miles.backends.training_utils.loss.hub.math_utils import compute_cispo_loss
from miles.backends.training_utils.metrics.update_direction import compute_update_direction
from miles.utils.sampling_mask import RolloutSamplingMask


@pytest.mark.parametrize("clip_low,clip_high", [(10.0, 4.0), (0.2, 0.28)])
def test_cispo_matches_reference_gradient_with_padding_and_unequal_lengths(clip_low, clip_high):
    # Two unequal response lengths; include positive/negative advantages,
    # clipped/unclipped ratios, and extreme but finite log ratios.
    current = torch.tensor([-1.0, -2.0, -3.0, -4.0, -1.0, -20.0, -0.1], requires_grad=True)
    delta = torch.tensor([0.0, 2.0, -2.0, 0.5, 30.0, -30.0, 1.0])
    old = (current.detach() - delta).requires_grad_(True)
    advantages = torch.tensor([1.0, -2.0, 3.0, -4.0, 2.0, 1.0, 0.0])
    mask = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0])
    losses, clipfrac = compute_cispo_loss(current, old - current, advantages, clip_low, clip_high)
    loss = (losses * mask).sum() / mask.sum()
    loss.backward()
    # Independent analytical derivative of the reference detached-weight objective.
    weights = delta.clamp(-20, 20).exp().clamp(1 - clip_low, 1 + clip_high)
    expected = -weights * advantages * mask / mask.sum()
    torch.testing.assert_close(current.grad, expected)
    assert old.grad is None
    torch.testing.assert_close(loss.detach(), (expected * current.detach()).sum())
    assert clipfrac[1] == 1


def test_cispo_gradient_is_not_ppo_ratio_gradient():
    logp = torch.tensor([-2.0], requires_grad=True)
    loss, _ = compute_cispo_loss(logp, torch.tensor([-2.0]) - logp, torch.ones(1), 10.0, 4.0)
    loss.sum().backward()
    torch.testing.assert_close(logp.grad, torch.tensor([-1.0]))


def test_policy_dispatch_preserves_masked_token_sum(monkeypatch):
    make_parallel_state()
    args = make_args(
        policy_loss_type="cispo",
        eps_clip=10.0,
        eps_clip_high=4.0,
        use_rollout_logprobs=True,
        skip_actor_forward_only=True,
        entropy_coef=0.0,
        observe_training_entropy=False,
        calculate_per_token_loss=True,
    )
    current = torch.tensor([-1.0, -2.0, -3.0, float("-inf")], requires_grad=True)
    delta = torch.tensor([0.0, 2.0, -2.0, 0.0])
    old = current.detach() - delta
    advantages = torch.tensor([1.0, -2.0, 3.0, float("nan")])
    masks = [torch.ones(1), torch.tensor([1.0, 1.0, 0.0])]
    batch = dict(
        advantages=[advantages[:1], advantages[1:]],
        rollout_log_probs=[old[:1], old[1:]],
        total_lengths=[3, 5],
        response_lengths=[1, 3],
        loss_masks=masks,
        unconcat_tokens=[],
    )
    monkeypatch.setattr(
        losses_module,
        "get_log_probs_and_entropy",
        lambda *a, **kw: {
            "log_probs": [current[:1], current[1:]],
        },
    )
    reducer = get_sum_of_sample_mean([3, 5], [1, 3], masks, True, "thd", None)
    # Miles token mode returns a sum; the training schedule divides by the
    # distributed active-token count. Include that normalization explicitly.
    loss, metrics = losses_module.policy_loss_function(args, batch, torch.zeros(1), reducer)
    (loss / 3).backward()
    expected = -delta[:3].exp().clamp(max=5) * advantages[:3] / 3
    torch.testing.assert_close(current.grad[:3], expected)
    assert current.grad[3] == 0
    assert torch.isfinite(metrics["loss"])


def _filtered_batch(logits, *, singleton=False):
    choices = torch.tensor([1, 2, 3])
    support = [[int(token)] if singleton else [int(token), 0] for token in choices]
    mask = RolloutSamplingMask.from_mask_list(support)
    ids, offsets = mask._as_tensors()
    old = torch.stack([logits[0, i, support[i]].log_softmax(-1)[0] for i in range(3)]).detach()
    return dict(
        advantages=[torch.tensor([1.0, -0.5, 0.3])],
        rollout_log_probs=[old],
        ref_log_probs=[logits[0, :3].detach().log_softmax(-1).gather(1, choices[:, None]).squeeze(-1)],
        unconcat_tokens=[torch.tensor([0, 1, 2, 3])],
        total_lengths=[4],
        response_lengths=[3],
        loss_masks=[torch.ones(3)],
        rollout_sampling_mask_ids=[ids],
        rollout_sampling_mask_offsets=[offsets],
    )


def _filtered_args(**overrides):
    return make_args(
        policy_loss_type="cispo",
        use_rollout_logprobs=True,
        skip_actor_forward_only=True,
        use_sampling_support_replay=True,
        eps_clip=10.0,
        eps_clip_high=4.0,
        entropy_coef=0.0,
        observe_training_entropy=True,
        use_kl_loss=True,
        kl_loss_coef=0.0,
        kl_loss_type="low_var_kl",
        vocab_size=4,
        **overrides,
    )


def test_observed_entropy_is_full_policy_and_reference_kl_is_zero_at_identical_weights():
    make_parallel_state()
    logits = torch.tensor([[[1.0, 0.2, -0.5, 0.4]] * 4], requires_grad=True)
    batch = _filtered_batch(logits, singleton=True)
    args = _filtered_args()
    loss, metrics = losses_module.policy_loss_function(args, batch, logits, torch.mean)
    loss.backward()
    full = logits.detach()[0, :3].log_softmax(-1)
    expected_entropy = -(full.exp() * full).sum(-1).mean()
    torch.testing.assert_close(metrics["full_policy_entropy"], expected_entropy)
    assert metrics["full_policy_entropy"] > 0
    torch.testing.assert_close(metrics["entropy_loss"], expected_entropy)
    assert metrics["sampling_entropy"] == 0  # The actual sampler permits one token.
    torch.testing.assert_close(metrics["kl_loss"], torch.tensor(0.0), atol=1e-7, rtol=0)
    torch.testing.assert_close(metrics["reference_sampled_log_ratio"], torch.tensor(0.0), atol=1e-7, rtol=0)
    assert not any(value.requires_grad for value in metrics.values())
    assert torch.count_nonzero(logits.grad) == 0


def test_entropy_and_zero_coefficient_reference_diagnostics_do_not_change_cispo_gradients():
    make_parallel_state()
    source = torch.tensor([[[0.2, 0.6, -0.3, 0.8]] * 4])
    outputs = []
    for observe in (False, True):
        logits = source.clone().requires_grad_(True)
        batch = _filtered_batch(logits)
        args = _filtered_args()
        args.observe_training_entropy = observe
        args.use_kl_loss = observe
        batch["ref_log_probs"] = [batch["ref_log_probs"][0] - 0.2]
        loss, metrics = losses_module.policy_loss_function(args, batch, logits, torch.mean)
        loss.backward()
        outputs.append((loss.detach(), logits.grad))
        if observe:
            expected = torch.tensor(0.2).neg().expm1() + 0.2
            torch.testing.assert_close(metrics["kl_loss"], expected)
            torch.testing.assert_close(metrics["reference_sampled_log_ratio"], torch.tensor(0.2))
    torch.testing.assert_close(outputs[0][0], outputs[1][0])
    torch.testing.assert_close(outputs[0][1], outputs[1][1])


@pytest.mark.parametrize("change,expected", [(0.1, 0.1), (-0.1, -0.1), (0, 0)])
def test_update_direction_detects_favorable_reversed_and_zero_updates(change, expected):
    result = compute_update_direction(
        [torch.zeros(2)],
        [torch.tensor([change, -change])],
        advantages=[torch.tensor([1.0, -1.0])],
        behavior=[torch.zeros(2)],
        loss_masks=[torch.ones(2)],
        eps_clip=10,
        eps_clip_high=4,
    )
    assert result["surrogate_gain"] == pytest.approx(expected)


def _check_sampling_entropy_tp2(rank, world_size, port):
    init_gloo(rank, world_size, port=port)
    try:
        scores = torch.tensor([[1.0, 0.5, -0.5, 0.2], [0.1, 0.4, -0.2, 0.8]])
        mask = torch.tensor([[False, False, True, False], [True, False, False, True]])
        local = slice(rank * 2, (rank + 1) * 2)
        observed = _observed_sampling_entropy(
            scores[:, local], mask[:, local], group=dist.group.WORLD, chunk_size=1, temperature=1.0
        )
        full = scores.masked_fill(~mask, torch.finfo(scores.dtype).min).log_softmax(-1)
        torch.testing.assert_close(observed, -(full.exp() * full).sum(-1), atol=1e-6, rtol=1e-6)
        assert observed[0] == 0
    finally:
        dist.destroy_process_group()


def test_sampling_entropy_handles_a_tp_rank_with_no_allowed_tokens():
    run_multiprocess(_check_sampling_entropy_tp2)
