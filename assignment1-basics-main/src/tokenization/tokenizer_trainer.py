def tokenizer_trainer(input_path: str, vocab_size: int, sprcial_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    