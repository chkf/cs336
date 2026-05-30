from collections.abc import Iterator, Iterable
from .pre_tokenizer import NativePreTokenizer
from cachetools import LRUCache
from collections import defaultdict
import regex as re


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

        self.vocab_to_id = {token: idx for idx, token in self.vocab.items()}

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

        vocab_pattern = rb'"(.*)"'
        with open(vocab_filepath, "rb") as f:
            for idx, match in re.finditer(vocab_pattern, f.read()):
                token = match.group(1)
                # TODO:TEST
                vocab[idx] = token

        # merges_pattern = rb"^(.*) (.*)$"
        with open(merges_filepath, "rb") as f:
            for line in f:
                # TODO:match = re.match(merges_pattern)
                line = line.rstrip()
                if not line:
                    continue
                parts = line.split(b" ")
                merges.append((parts[0], parts[1]))
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        byte_text = text.encode("utf-8")
        id_list = []

        for pre_token in self.pre_tokenizer.pre_tokenize(byte_text, self.special_tokens):
            id_list.extend(self._encode_one_pre_token(pre_token))

        return id_list

    def _encode_one_pre_token(self, pre_token: bytes) -> list[int]:
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
                yield from self._encode_one_pre_token(pre_token)

    def decode(self, ids: list[int]) -> str:
        bytes_list = [self.vocab[id] for id in ids]

        return b"".join(bytes_list).decode("utf-8", errors="replace")
