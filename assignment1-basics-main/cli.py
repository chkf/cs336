import typer
import os
from typing import Annotated
from src.tokenization.tokenizer_trainer import TokenizerTrainer
from loguru import logger

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


if __name__ == "__main__":
    app()