import argparse
import json
import random
import tomllib
from pathlib import Path

import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer

from cs336_alignment.algs.grpo import compute_rollout_rewards, grpo_train_step
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.vllm_utils import VLLMServer


def load_jsonl(path, n_examples):
    with open(path) as f:
        return [json.loads(line) for line in f][:n_examples]


def get_ground_truth(example):
    return example["answer"].split("####")[-1].strip()


def main(config_path):
    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    train_config = config["training"]
    seed = train_config["seed"]
    random.seed(seed)
    torch.manual_seed(seed)

    train_data = load_jsonl(config["data"]["train_path"], config["data"]["n_train_examples"])
    val_data = load_jsonl(config["data"]["val_path"], config["data"]["n_val_examples"])
    random.shuffle(train_data)
    prompt_template = Path(config["data"]["prompt_path"]).read_text()

    model_id = config["model"]["model_id"]
    device = config["model"]["device"]
    policy = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=train_config["learning_rate"],
        betas=(0.9, 0.95),
        weight_decay=0.0,
    )
    run = wandb.init(project=config["logging"]["wandb_project"], config=config)

    server = VLLMServer(
        model_id=model_id,
        gpu=config["model"]["vllm_gpu"],
        seed=seed,
    )
    server.start()
    server.init_weight_sync(device)

    group_size = train_config["group_size"]
    prompts_per_step = train_config["rollout_batch_size"] // group_size
    sampling_params = {
        "temperature": config["sampling"]["temperature"],
        "max_tokens": config["sampling"]["max_tokens"],
        "n": group_size,
        "seed": seed,
        "stop": ["</answer>"],
        "include_stop_str_in_output": True,
    }

    try:
        for step in range(train_config["num_rollout_steps"]):
            batch = train_data[step * prompts_per_step:(step + 1) * prompts_per_step]
            prompts = [prompt_template.replace("{question}", x["question"]) for x in batch]
            ground_truths = [get_ground_truth(x) for x in batch]

            server.sync_policy_weights(policy)
            completions = server.generate_completions(prompts, sampling_params)
            responses = [x.text for x in completions]
            repeated_prompts = [prompt for prompt in prompts for _ in range(group_size)]
            repeated_ground_truths = [answer for answer in ground_truths for _ in range(group_size)]

            _, metrics = grpo_train_step(
                model=policy,
                tokenizer=tokenizer,
                optimizer=optimizer,
                gradient_accumulation_steps=train_config["gradient_accumulation_steps"],
                max_grad_norm=train_config["max_grad_norm"],
                reward_fn=r1_zero_reward_fn,
                repeated_prompts=repeated_prompts,
                rollout_responses=responses,
                repeated_ground_truths=repeated_ground_truths,
                group_size=group_size,
            )
            run.log({f"train/{key}": value for key, value in metrics.items()}, step=step)

            if (step + 1) % train_config["eval_interval"] == 0:
                server.sync_policy_weights(policy)
                val_prompts = [prompt_template.replace("{question}", x["question"]) for x in val_data]
                val_answers = [get_ground_truth(x) for x in val_data]
                val_params = sampling_params | {"n": 1, "seed": seed + step + 1}
                val_completions = server.generate_completions(val_prompts, val_params)
                val_responses = [x.text for x in val_completions]
                _, val_metrics = compute_rollout_rewards(
                    r1_zero_reward_fn, val_responses, val_answers
                )
                val_metrics["mean_response_length"] = sum(
                    len(x.token_ids) for x in val_completions
                ) / len(val_completions)
                run.log({f"val/{key}": value for key, value in val_metrics.items()}, step=step)

        output_dir = config["logging"]["output_dir"]
        policy.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
    finally:
        server.stop()
        run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/grpo.toml")
    args = parser.parse_args()
    main(args.config)
