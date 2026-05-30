
import torch
from src.nn.network import TransformerLM
from src.nn.optimizer import AdamW
from pathlib import Path
import src.nn.functional as F
import numpy as np
import regex as re
import yaml
import shutil
import loguru
import time


logger = loguru.logger


def train(cfg_file: str):
    with open(cfg_file, 'r') as f:
        cfg = yaml.safe_load(f, Loader=yaml.FullLoader)

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
    total_tokens = 0
    t1 = time.time()
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
                                        model.device)
        logits = model(x_batch)
        loss = F.cross_entropy_loss(logits, y_batch)
        loss.backward()
        F.gradient_clipping(model.parameters(), cfg.training.max_norm)
        optimizer.step()
        if iteration % cfg.training.log_step == 0:
            t2 = time.time()
            elapsed = t2 - t1
            batch_per_sec = cfg.training.batch_size / elapsed
            tokens_per_sec = batch_per_sec * cfg.model.context_length
            total_tokens += cfg.training.batch_size * cfg.model.context_length
            t1 = time.time()
            logger.info(f"Iteration {iteration}: train loss = {loss.item():.4f}")

        if iteration != 0 and iteration % cfg.training.save_step == 0:
            save_path = output_dir / f"checkpoint_{iteration}.pt"
            F.save_checkpoint(model, optimizer, iteration, save_path)


if __name__ == '__main__':
    train("assignment1-basics-main/basic.yaml")
