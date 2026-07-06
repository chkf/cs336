import torch
import torch.nn.functional as F


def tokenize_prompt_and_output(prompt_strs: list[str],
                               output_strs: list[str],
                               tokenizer) -> dict:
    prompt_tokens = tokenizer(prompt_strs,
                              add_special_tokens=False,
                              padding=False,
                              truncation=False,
                              return_attention_mask=False)

    output_tokens = tokenizer(output_strs,
                              add_special_tokens=False,
                              padding=False,
                              truncation=False,
                              return_attention_mask=False)

    input_ids = []
    response_mask = []

    for p_ids, o_ids in zip(prompt_tokens["input_ids"], output_tokens["input_ids"]):
        combined_ids = p_ids + o_ids
        input_ids.append(combined_ids)
        mask = ([False] * len(p_ids)) + ([True] * len(o_ids))
        response_mask.append(mask)

    MAX_LEN = max(len(ids) for ids in input_ids)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def pad(x, value):
        return x + [value] * (MAX_LEN - len(x))

    full = torch.tensor([pad(x, pad_id) for x in input_ids], dtype=torch.long)
    response_mask = torch.tensor([pad(x, False) for x in response_mask], dtype=torch.bool)

    assert full.shape == response_mask.shape

    input_ids = full[:, :-1].contiguous()
    labels = full[:, 1:].contiguous()
    response_mask = response_mask[:, 1:].contiguous()

    assert input_ids.shape == labels.shape == response_mask.shape

    return {"input_ids": input_ids,
            "labels": labels,
            "response_mask": response_mask}


def get_response_log_probs(model,
                           input_ids: torch.Tensor,
                           labels: torch.Tensor,
                           return_token_entropy: bool = False) -> dict[str, torch.Tensor]:
    logits = model(input_ids=input_ids).logits

    logp = F.log_softmax(logits, dim=-1)
    log_probs = logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)

    res = {"log_probs": log_probs}

    if return_token_entropy:
        probs = torch.exp(logp)
        entropy = -torch.sum(probs * logp, dim=-1)
        res["token_entropy"] = entropy
    return res
