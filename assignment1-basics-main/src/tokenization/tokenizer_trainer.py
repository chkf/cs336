from abc import ABC, abstractmethod
import json
import os
from loguru import logger
import heapq
from tqdm import tqdm
from line_profiler import profile
from .pre_tokenizer import NativePreTokenizer
from collections import defaultdict


class ComparablePair:
    __slots__ = ("pair")

    def __init__(self, pair):
        self.pair = pair

    def __lt__(self, other):
        return self.pair > other.pair

    def __eq__(self, other):
        return self.pair == other.pair

    def __repr__(self):
        return str(self.pair)


class TokenizerTrainerBase(ABC):
    vocab: dict[int, bytes]
    merges: list[tuple[bytes, bytes]]

    def __init__(
            self,
            corpus_path: str,
            vocab_size: int,
            special_tokens: list[str],
            split_special_token: bytes,
            pre_tokenizer_cls) -> None:
        super().__init__()

        self.corpus_path = corpus_path
        self.target_vocab_size = vocab_size
        self.special_tokens = [token.encode("utf-8") for token in special_tokens]
        self.split_special_token = split_special_token
        self.pre_tokenizer = pre_tokenizer_cls()

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

        vocab_to_save = {k: "".join(byte_encoder[b] for b in v) for k, v in self.vocab.items()}

        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(vocab_to_save, f, indent=4)

        with open(merges_file, "w", encoding="utf-8") as f:
            for b1, b2 in self.merges:
                s1 = "".join(byte_encoder[b] for b in b1)
                s2 = "".join(byte_encoder[b] for b in b2)
                f.write(f"{s1} {s2}\n")
        logger.info(f"tokenier saved to {out_dir}")


@profile
class TokenizerTrainer(TokenizerTrainerBase):
    def __init__(self,
                 corpus_path,
                 vocab_size,
                 special_tokens,
                 split_special_token=b"<|endoftext|>",
                 pre_tokenizer_cls=NativePreTokenizer):
        super().__init__(corpus_path, vocab_size, special_tokens, split_special_token, pre_tokenizer_cls)

        self.vocab: dict[int, bytes] = defaultdict(bytes)
        self.merges: list[tuple[bytes, bytes]] = []

        self.current_vocab_size = 0
        self.pre_token_count: dict[bytes, int] = defaultdict(int)

        self.words_list: list[bytes] = []
        self.words_count: list[int] = []

        self.pair_counts: dict[tuple[bytes, bytes], int] = defaultdict(int)
        self.pair_heap: list[tuple] = []

        self.relevant_words: dict[tuple[bytes, bytes], set] = defaultdict(set)

        for ascii_code in range(256):
            self._add_token(bytes([ascii_code]))
        for token in self.special_tokens:
            self._add_token(token)

        self.pre_token_count = self.pre_tokenizer(
            corpus_path=self.corpus_path,
            special_tokens=self.special_tokens,
            split_special_token=self.split_special_token
        )
        for word, count in self.pre_token_count.items():
            self.words_list.append(list(tuple(bytes([b]) for b in word)))
            self.words_count.append(count)

        for idx, word in enumerate(self.words_list):
            for i in range(len(word) - 1):
                pair = (word[i], word[i+1])
                self.pair_counts[pair] += self.words_count[idx]
                self.relevant_words[pair].add(idx)

        for pair, count in self.pair_counts.items():
            heapq.heappush(self.pair_heap, (-count, ComparablePair(pair)))

    def _add_token(self, token: bytes) -> None:
        self.vocab[self.current_vocab_size] = token
        self.current_vocab_size += 1

    def _merge_pair(self, pair: tuple[bytes, bytes]) -> None:
        merged_token = pair[0] + pair[1]
        self._add_token(merged_token)
        self.merges.append(pair)

        relevant_words_list = list(self.relevant_words[pair])

        for idx in relevant_words_list:
            word = self.words_list[idx]
            count = self.words_count[idx]

            i = 0
            while i < len(word) - 1:
                if word[i] == pair[0] and word[i+1] == pair[1]:
                    if i > 0:
                        prev_pair = (word[i-1], word[i])
                        self.pair_counts[prev_pair] -= count
                        heapq.heappush(self.pair_heap, (-self.pair_counts[prev_pair], ComparablePair(prev_pair)))

                    if i < len(word) - 2:
                        next_pair = (word[i+1], word[i+2])
                        self.pair_counts[next_pair] -= count
                        heapq.heappush(self.pair_heap, (-self.pair_counts[next_pair], ComparablePair(next_pair)))

                    word[i] = pair[0] + pair[1]
                    del word[i+1]

                    if i > 0:
                        new_prev = (word[i-1], word[i])
                        self.relevant_words[new_prev].add(idx)
                        self.pair_counts[new_prev] += count
                        heapq.heappush(self.pair_heap, (-self.pair_counts[new_prev], ComparablePair(new_prev)))

                    if i < len(word) - 1:
                        new_next = (word[i], word[i+1])
                        self.relevant_words[new_next].add(idx)
                        self.pair_counts[new_next] += count
                        heapq.heappush(self.pair_heap, (-self.pair_counts[new_next], ComparablePair(new_next)))
                else:
                    i += 1

        del self.pair_counts[pair]
        del self.relevant_words[pair]

    def train(self) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        for _ in tqdm(range(self.current_vocab_size, self.target_vocab_size), desc="training tokenizer"):
            while self.pair_heap:
                neg_count, wrapper = heapq.heappop(self.pair_heap)
                count = -neg_count
                pair = wrapper.pair

                if count == self.pair_counts[pair]:
                    break

            if pair is None:
                raise ValueError("heap is empty")

            self._merge_pair(pair)

        logger.info("train tokenizer done")

        return self.vocab, self.merges
