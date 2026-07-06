import torch
from typing import Callable, Literal
from transformers import PreTrainedModel, PreTrainedTokenizer
from cs336_alignment.algs.utils import get_response_log_probs, tokenize_prompt_and_output



def compute_rollout_rewards(reward_fn:Callable[[str, str], dict[str, float]],
                            rollout_responses: list[str],
                            repeated_ground_truths: list[str]) -> tuple[torch.Tensor, dict[str, float]]:
    rewards = []
    metadata = {"mean_reward": 0.0,
                "mean_format_reward": 0.0,
                "mean_answer_reward": 0.0}
    for response, gt_answer in zip(rollout_responses, repeated_ground_truths):
        reward_dict = reward_fn(response, gt_answer)
        reward = reward_dict["reward"]
        format_reward = reward_dict["format_reward"]
        answer_reward = reward_dict["answer_reward"]
        rewards.append(reward)
        metadata["mean_reward"] += reward
        metadata["mean_format_reward"] += format_reward
        metadata["mean_answer_reward"] += answer_reward
    length = len(rollout_responses)
    metadata["mean_reward"] /= length
    metadata["mean_format_reward"] /= length
    metadata["mean_answer_reward"] /= length

    rewards = torch.tensor(rewards, dtype=torch.float32)

    return (rewards, metadata)


def compute_group_normalized_rewards(raw_rewards: torch.Tensor,
                                     group_size: int,
                                     baseline: Literal["mean", "none"] = "mean",
                                     advantage_eps: float = 1e-6,
                                     advantage_normalizer: Literal["std", "none", "mean"] = "std",
):
    advantages = []
    metadata = {"mean_reward": 0.0,
                "std_reward": 0.0,
                "max_reward": 0.0,
                "min_reward": 0.0}

    for i in range(0, len(raw_rewards), group_size):
        group_rewards = raw_rewards[i: i + group_size]
        if baseline == "mean":
            group_subtraction = torch.mean(group_rewards)
        else:
            group_subtraction = 0
        if advantage_normalizer == "std":
            group_norm = torch.std(group_rewards) + advantage_eps
        elif advantage_normalizer == "mean":
            group_norm = torch.mean(group_rewards) + advantage_eps
        else:
            group_norm = 1

        normalized_rewards = (group_rewards - group_subtraction) / group_norm

        advantages.extend(normalized_rewards.tolist())
    advantages = torch.tensor(advantages, dtype=torch.float32)

    return (advantages, metadata)


def compute_policy_gradient_loss(raw_rewards_or_advantages: torch.Tensor,
                                 policy_log_probs: torch.Tensor,
                                 importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
                                 old_log_probs: torch.Tensor | None = None,
                                 cliprange: float | None = None,
                                 response_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if raw_rewards_or_advantages.dim() == 1:
        raw_rewards_or_advantages = raw_rewards_or_advantages.unsqueeze(-1)

    metadata = {}
    if importance_reweighting_method == "none":
        per_token_policy_gradient_loss = -raw_rewards_or_advantages * policy_log_probs

    else:
        raise NotImplementedError

    return (per_token_policy_gradient_loss, metadata)


def aggregate_loss_across_microbatch(per_token_policy_gradient_loss: torch.Tensor,
                                     mask: torch.Tensor,
                                     loss_normalization: Literal["sequence", "constant"] = "sequence",
                                     normalization_constant: int | None = None) -> torch.Tensor:
    mask_ = mask.to(per_token_policy_gradient_loss.dtype)
    valid_per_token_policy_gradient_loss = mask_ * per_token_policy_gradient_loss
    loss = valid_per_token_policy_gradient_loss.sum(dim=-1)
    if loss_normalization == "sequence":
        valid_len = torch.sum(mask, dim=-1)
        loss /= valid_len
        normed_loss = torch.mean(loss)

    elif loss_normalization == "constant":
        normed_loss = torch.sum(loss, dim=-1) / normalization_constant

    return normed_loss


def grpo_train_step(model: PreTrainedModel,
                    tokenizer: PreTrainedTokenizer,
                    optimizer: torch.optim.Optimizer,
                    gradient_accumulation_steps: int,
                    max_grad_norm: float | None,
                    reward_fn: Callable[[str, str], dict[str, float]],
                    repeated_prompts: list[str],
                    rollout_responses: list[str],
                    repeated_ground_truths: list[str],
                    group_size: int,
                    # Reward normalization
                    baseline: Literal["mean", "none"] = "mean",
                    advantage_eps: float = 1e-6,
                    advantage_normalizer: Literal["std", "none", "mean"] = "std",
                    # Importance reweighting and clipping
                    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
                    old_log_probs: torch.Tensor | None = None,
                    cliprange: float | None = None,
                    # Loss normalization
                    loss_normalization: Literal["sequence", "constant"] = "sequence",
                    normalization_constant: int | None = None) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    raw_rewards, metadata = compute_rollout_rewards(
        reward_fn, rollout_responses, repeated_ground_truths
    )
    advantages, _ = compute_group_normalized_rewards(
        raw_rewards, group_size, baseline, advantage_eps, advantage_normalizer
    )

    batch_size = len(repeated_prompts)
    microbatch_size = batch_size // gradient_accumulation_steps
    device = next(model.parameters()).device
    total_loss = torch.zeros((), device=device)
    total_entropy = 0.0

    optimizer.zero_grad()
    for start in range(0, batch_size, microbatch_size):
        end = start + microbatch_size
        batch = tokenize_prompt_and_output(
            repeated_prompts[start:end], rollout_responses[start:end], tokenizer
        )
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        response_mask = batch["response_mask"].to(device)

        output = get_response_log_probs(
            model, input_ids, labels, return_token_entropy=True
        )
        per_token_loss, _ = compute_policy_gradient_loss(
            advantages[start:end].to(device),
            output["log_probs"],
            importance_reweighting_method,
            None if old_log_probs is None else old_log_probs[start:end, :labels.shape[1]].to(device),
            cliprange,
            response_mask,
        )
        loss = aggregate_loss_across_microbatch(
            per_token_loss, response_mask, loss_normalization, normalization_constant
        ) / gradient_accumulation_steps
        loss.backward()
        total_loss += loss.detach()
        total_entropy += (output["token_entropy"] * response_mask).sum().item()

    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    optimizer.zero_grad()

    metadata["loss"] = total_loss.item()
    metadata["grad_norm"] = grad_norm
    metadata["token_entropy"] = total_entropy / sum(
        len(tokenizer.encode(response, add_special_tokens=False))
        for response in rollout_responses
    )
    return total_loss, metadata
