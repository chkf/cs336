import typer
import os
from typing import Annotated
from src.tokenization.tokenizer import Tokenizer

from src.tokenization.tokenizer_trainer import TokenizerTrainer
from loguru import logger
import torch
from src.nn.network import TransformerLM
from src.nn.optimizer import AdamW
from pathlib import Path
import src.nn.functional as F
import numpy as np
import regex as re
import yaml
import shutil
# import time
from easydict import EasyDict

app = typer.Typer(help="Tokenizer training and evaluation tools")


@app.command()
def train_tokenizer(
    corpus_path: Annotated[str, typer.Argument(help="corpus file path")], 
    vocab_size: Annotated[int, typer.Argument(help="target vocab size")], 
    special_tokens: Annotated[list[str] | None, typer.Option(help="special tokens list")] = None, 
    save_path: Annotated[str, typer.Option(help="save directory")] = "save",  
):
    if special_tokens is None:
        special_tokens = ["<|endoftext|>"]

    corpus = os.path.splitext(os.path.basename(corpus_path))[0]
    save_dir = os.path.join(save_path, corpus)
    os.makedirs(save_dir, exist_ok=True) 

    logger.info("start training tokenizer")
    tokenizer = TokenizerTrainer(corpus_path, vocab_size, special_tokens)
    vocab, merges = tokenizer.train()
    tokenizer.save(save_dir)

    longest_token_length = max(len(token) for token in vocab.values())
    longest_tokens = [token for token in vocab.values() if len(token) == longest_token_length]

    logger.success(f"Tokenizer saved to: {save_dir}")
    logger.info(f"Vocab size: {len(vocab)}")
    logger.info(f"Merges count: {len(merges)}")
    logger.info(f"Longest token length: {longest_token_length} (Tokens: {longest_tokens!r})")


@app.command()
def train(cfg_file: Annotated[str, typer.Argument(help="config file path")]):
    with open(cfg_file, 'r') as f:
        cfg = EasyDict(yaml.safe_load(f))

    output_dir = Path(cfg.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cfg_file, output_dir / Path(cfg_file).name)

    model = TransformerLM(cfg.model.vocab_size,
                          cfg.model.max_seq_len,
                          cfg.model.d_model,
                          cfg.model.num_layers,
                          cfg.model.num_heads,
                          cfg.model.d_ff,
                          cfg.model.theta)

    optimizer = AdamW(model.parameters(),
                      cfg.optimizer.lr,
                      cfg.optimizer.weight_decay,
                      (cfg.optimizer.beta1, cfg.optimizer.beta2),
                      cfg.optimizer.eps)

    checkpoint_regex = re.compile(r"checkpoint_(\d+)\.pt")
    max_iteration = -1
    for file in output_dir.iterdir():
        match = checkpoint_regex.match(file.name)
        if match:
            iteration = int(match.group(1))
            if iteration > max_iteration:
                max_iteration = iteration
    current_iteration = 0

    if max_iteration >= 0:
        logger.info(f"Resuming from checkpoint at iteration {max_iteration}")
        checkpoint_path = output_dir / f"checkpoint_{max_iteration}.pt"
        current_iteration = F.load_checkpoint(checkpoint_path, model, optimizer)
        current_iteration += 1
    else:
        model._reset_params()

    model = model.to(torch.device(cfg.model.device))
    model.train()

    logger.info(f"Loading training dataset from {cfg.training.train_data}")
    train_dataset = np.memmap(cfg.training.train_data, dtype=np.uint16)

    logger.info(f"Starting training from iteration {current_iteration} to {cfg.training.total_iterations - 1}")
    # total_tokens = 0
    # t1 = time.time()
    for iter in range(current_iteration + 1, cfg.training.total_iterations + 1):
        optimizer.zero_grad()
        lr = F.get_lr_cosine_schedule(iter,
                                      cfg.optimizer.lr,
                                      0.0,
                                      cfg.training.warmup_iter,
                                      cfg.training.cosine_cycle_iter)
        for param_group in optimizer.param_groups:
            param_group["alpha"] = lr

        x_batch, y_batch = F.load_batch(train_dataset,
                                        cfg.training.batch_size,
                                        cfg.model.max_seq_len,
                                        cfg.model.device)
        logits = model(x_batch)
        loss = F.cross_entropy_loss(logits, y_batch)
        loss.backward()
        F.gradient_clipping(model.parameters(), cfg.training.max_norm)
        optimizer.step()
        if iter % cfg.training.log_step == 0:
            # t2 = time.time()
            # elapsed = t2 - t1
            # batch_per_sec = cfg.training.batch_size / elapsed
            # tokens_per_sec = batch_per_sec * cfg.model.max_seq_len
            # total_tokens += cfg.training.batch_size * cfg.model.max_seq_len
            # t1 = time.time()
            logger.info(f"Iteration {iter}: train loss = {loss.item():.4f}")

        if iter != 0 and iter % cfg.training.save_step == 0:
            save_path = output_dir / f"checkpoint_{iter}.pt"
            F.save_checkpoint(model, optimizer, iter, save_path)


@app.command()
def encode_file(tokenizer_path: Annotated[str, typer.Argument(help="Path to the tokenizer directory")],
                corpus_path: Annotated[str, typer.Argument(help="Path to the corpus file")],
                output_path: Annotated[str, typer.Argument(help="Path to save the encoded output")],
                split_special_token: str = "<|endoftext|>"):

    tokenizer_path = Path(tokenizer_path)
    vocab_path = tokenizer_path / "vocab.json"
    merges_path = tokenizer_path / "merges.txt"

    tokenizer = Tokenizer.from_files(vocab_path,
                                     merges_path,
                                     [split_special_token])

    tokenizer.encode_file(corpus_path,
                          split_special_token,
                          output_path)


@app.command()
def decode(prompt: str,
           model_path: str,
           tokenizer_path: str,
           checkpoint_number: int | None = 20000,
           max_length: int = 50,
           temperature: float = 1.0,
           p: float = 0.9) -> str:
    save_path = Path(model_path)
    checkpoint_path = save_path / f"checkpoint_{checkpoint_number}.pt"
    config_path = save_path / "basic.yaml"
    tokenizer_path = Path(tokenizer_path)
    vocab_path = tokenizer_path / "vocab.json"
    merges_path = tokenizer_path / "merges.txt"

    tokenizer = Tokenizer.from_files(vocab_path,
                                     merges_path)
    with open(config_path, 'r') as f:
        cfg = EasyDict(yaml.safe_load(f))
    model = TransformerLM(cfg.model.vocab_size,
                          cfg.model.max_seq_len,
                          cfg.model.d_model,
                          cfg.model.num_layers,
                          cfg.model.num_heads,
                          cfg.model.d_ff,
                          cfg.model.theta)
    
    F.load_checkpoint(checkpoint_path, model)
    model.eval()
    device = torch.device(cfg.model.device)
    model.to(device)

    input_token = tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_token], device=device)

    while len(input_token) < max_length:
        logits = model(input_tensor)
        logits = logits[:, -1, :].flatten()
        logits = F.softmax(logits / temperature)

        logits, index = logits.sort(dim=-1, descending=True)

        count = 0
        sum_p = 0.0
        while sum_p < p and count < logits.size(0):
            sum_p += logits[count].item()
            count += 1

        filtered_logits = logits[:count]
        filtered_logits = filtered_logits / filtered_logits.sum()

        next_token_id = index[torch.multinomial(filtered_logits, num_samples=1).item()].item()

        input_token.append(next_token_id)
        input_tensor = torch.tensor([input_token], device=device)

        next_token = tokenizer.decode([next_token_id])
        if next_token == "<|endoftext|>":
            break
    print(tokenizer.decode(input_token))





if __name__ == "__main__":
    app()
