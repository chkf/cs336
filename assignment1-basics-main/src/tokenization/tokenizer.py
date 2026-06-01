from collections.abc import Iterator, Iterable
from .pre_tokenizer import NativePreTokenizer, FileChunkIterator
from cachetools import LRUCache
from collections import defaultdict
from multiprocessing import Pool
from .utils import bytes_to_unicode

import json
import os
import numpy as np
import tqdm


class Tokenizer:
    def __init__(self,
                 vocab: dict[int, bytes],
                 merges: list[tuple[bytes, bytes]],
                 special_tokens: list[str] | None = None,
                 pre_tokenizer_cls=NativePreTokenizer):
        self.vocab = vocab
        self.merges = merges
        if special_tokens:
            self.special_tokens = [token.encode("utf-8") for token in special_tokens]
        else:
            self.special_tokens = []

        byte_encoder = bytes_to_unicode()
        byte_encoder_reverse = {v: k for k, v in byte_encoder.items()}

        self.vocab_to_id = {}

        for idx, token in self.vocab.items():
            token_str = token.decode("utf-8")
            token_bytes = bytes(byte_encoder_reverse[ch] for ch in token_str)
            self.vocab_to_id[token_bytes] = idx

        self.ranks = {pair: i for i, pair in enumerate(self.merges)}

        self.pre_tokenizer = pre_tokenizer_cls()

        self.cache = LRUCache(maxsize=1000)

    @classmethod
    def from_files(cls,
                   vocab_filepath: str,
                   merges_filepath: str,
                   special_tokens: list[str] | None = None):
        vocab = defaultdict(bytes)
        merges: list[tuple[bytes, bytes]] = []

        with open(vocab_filepath, "rb") as f:
            vocab_str = json.load(f)

        for token_id, token_bytes in vocab_str.items():
            vocab[token_id] = token_bytes.encode("utf-8")

        with open(merges_filepath, "rb") as f:
            for line in f:
                line = line.rstrip()
                if not line:
                    continue
                parts = line.split(b" ")
                merges.append((parts[0], parts[1]))
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str | bytes) -> list[int]:
        byte_text = text.encode("utf-8") if isinstance(text, str) else text
        id_list = []
        for pre_token in self.pre_tokenizer.pre_tokenize(byte_text, self.special_tokens):
            id_list.extend(self._encode_one_token(pre_token))

        return id_list

    def _encode_one_token(self, pre_token: bytes) -> list[int]:
        if pre_token in self.cache:
            return self.cache[pre_token]

        if pre_token in self.vocab_to_id:
            return [self.vocab_to_id[pre_token]]

        word = [bytes([b]) for b in pre_token]

        while len(word) > 1:
            min_rank = float("inf")
            min_pair = None
            for i in range(len(word) - 1):
                pair = (word[i], word[i+1])
                rank = self.ranks.get(pair, float("inf"))
                if rank < min_rank:
                    min_rank = rank
                    min_pair = pair

            if min_pair is None or min_rank == float("inf"):
                break

            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == min_pair[0] and word[i+1] == min_pair[1]:
                    new_word.append(min_pair[0]+min_pair[1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word

        token_list = [self.vocab_to_id[token] for token in word]

        self.cache[pre_token] = token_list
        return token_list

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            byte_text = text.encode("utf-8")
            for pre_token in self.pre_tokenizer.pre_tokenize(byte_text, self.special_tokens):
                yield from self._encode_one_token(pre_token)

    def encode_file(self,
                    file_path: str,
                    split_token: str = "<|endoftext|>",
                    save_file: str | None = None) -> None:
        file_iter = FileChunkIterator(file_path,
                                      os.cpu_count()*100,
                                      split_token.encode("utf-8"),
                                      return_bytes=True,
                                      desired_bytes=1024*1024)
        self.encode_batch(file_iter, save_file=save_file)

    # TODO: to learn
    def _worker(self, text):
        return self.encode(text)

    def encode_batch(self,
                     texts: Iterable[str | bytes],
                     save_file: str):
        cpu_count = os.cpu_count()
        num_workers = cpu_count - 1

        with open(save_file, "wb") as f_out, Pool(processes=num_workers) as pool:
            for chunk_ids in tqdm.tqdm(
                pool.imap(self._worker, texts, chunksize=32),
                desc="Encoding",
                total=len(texts)
            ):
                if chunk_ids:
                    res = np.array(chunk_ids, dtype=np.uint16)
                    res.tofile(f_out)

    def decode(self, ids: list[int]) -> str:
        bytes_list = [self.vocab[id] for id in ids]

        return b"".join(bytes_list).decode("utf-8", errors="replace")
