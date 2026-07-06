import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_id_or_path: str, device: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_id_or_path,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=device,
    )
    model.eval()
    return model


def generate(model, tokenizer, prompt, max_new_tokens, seed):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    torch.manual_seed(seed)

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            do_sample=True,
            temperature=1.0,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(
        output[0, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    if "</answer>" in response:
        response = response.split("</answer>", 1)[0] + "</answer>"

    return response


def main(args):
    if not torch.cuda.is_available():
        raise RuntimeError("没有检测到 CUDA GPU")

    if torch.cuda.device_count() < 2:
        raise RuntimeError("该模式需要两张 CUDA GPU")

    print(f"正在加载训练前模型到 {args.base_device}...")
    base_model = load_model(args.base_model, args.base_device)
    print(f"正在加载训练后模型到 {args.trained_device}...")
    trained_model = load_model(args.trained_model, args.trained_device)

    tokenizer = AutoTokenizer.from_pretrained(args.trained_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompt_template = Path(args.prompt).read_text(encoding="utf-8")

    print("\n输入数学问题；输入 quit 退出。")
    while True:
        question = input("\n问题> ").strip()
        if question.lower() in {"quit", "exit", "q"}:
            break
        if not question:
            continue

        prompt = prompt_template.replace("{question}", question)

        print("\n正在生成训练前回答...")
        before = generate(
            base_model,
            tokenizer,
            prompt,
            args.max_new_tokens,
            args.seed,
        )
        print(f"\n[训练前]\n<think>{before}")

        print("\n正在生成训练后回答...")
        after = generate(
            trained_model,
            tokenizer,
            prompt,
            args.max_new_tokens,
            args.seed,
        )
        print(f"\n[训练后]\n<think>{after}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trained-model", required=True, help="训练后模型目录")
    parser.add_argument("--base-model", default="allenai/OLMo-2-0425-1B")
    parser.add_argument("--prompt", default="cs336_alignment/prompts/r1_zero.prompt")
    parser.add_argument("--base-device", default="cuda:0")
    parser.add_argument("--trained-device", default="cuda:1")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    main(parser.parse_args())
