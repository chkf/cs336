from abc import ABC, abstractmethod
import json
import os
from loguru import logger

class TokenizerTrainerBase(ABC):
    vocab: dict[int, bytes]
    merges: list[tuple[bytes, bytes]]

    @abstractmethod
    def train(self) -> tuple[dict, list[tuple[bytes, bytes]]]:
        pass

    @staticmethod
    def _bytes_to_unicode():
        """
        chr()
        ord()
        """
        bs = list(range(ord("!"), ord("~")+1)) + list(range(ord("?"), ord("?") + 1)) + list(range(ord("?"), ord("?") + 1))
        cs = bs[:]
        n = 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256+n)
                n += 1
        cs = [chr(n) for n in cs]
        return dict(zip(bs, cs))

    def save(self, out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)

        vocab_file = os.path.join(out_dir, "vocab.json")
        merges_file = os.path.join(out_dir, "merges.txt")

        byte_encoder = self._bytes_to_unicode()

        vocab_to_save = {k:"".join(byte_encoder[b] for b in v) for k, v in self.vocab.items()}
        
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(vocab_to_save, f, indent=4)

        with open(merges_file, "w", encoding="utf-8") as f:
            for b1, b2 in self.merges:
                s1 = "".join(byte_encoder[b] for b in b1)
                s2 = "".join(byte_encoder[b] for b in b2)
                f.write(f"{s1} {s2}\n")
        logger.info(f"tokenier saved to {out_dir}")


class TokenizerTrainer(TokenizerTrainerBase):
    def __init__(
            self, 
            corpus_path: str,
            vocab_size: int,
            special_tokens: list[str],
            pre_tokenizer_cls) -> None:
        super().__init__()

        self.corpus_path = corpus_path
        self.target_vocab_size = vocab_size
        self.special_tokens = [token.encode() for token in special_tokens]
        self.pre_tokenizer = pre_tokenizer_cls()

        self.vocab = {}
        self.merges: list[tuple[bytes, bytes]] = []
        self.current_vocab_size = 0

        