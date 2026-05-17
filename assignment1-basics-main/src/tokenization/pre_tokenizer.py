import os
import mmap
from loguru import logger
from abc import ABC, abstractmethod
from line_profiler import profile
from collections.abc import Iterator
from collections import Counter, defaultdict
import regex as re
from tqdm import tqdm


class FileChunkIterator:
    def __init__(self,
                 corpus_path: str,
                 desired_num_chunks: int,
                 split_special_token: bytes,
                 return_bytes: bool = True,
                 desired_bytes: int | None = None):
        self.corpus_path = corpus_path
        self.desired_num_chunks = desired_num_chunks
        self.split_special_token = split_special_token
        self.return_bytes = return_bytes
        self.desired_bytes = desired_bytes

        self.boundaries = []
        self._find_chunk_boundaries()

    def _find_chunk_boundaries(self):
        file_size = os.path.getsize(self.corpus_path)

        if self.desired_bytes is not None:
            desired_num_chunks = file_size // self.desired_bytes

        with open(self.corpus_path, "+rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            chunk_size = file_size // desired_num_chunks

            self.boundaries.append(0)

            for i in range(1, desired_num_chunks):
                pos = mm.find(self.split_special_token, i*chunk_size)

                if pos != -1:
                    self.boundaries.append(pos)
                else:
                    break

            self.boundaries.append(file_size)
        
        self.boundaries = sorted(list(set(self.boundaries)))
        logger.info(f"chunk boundary finding done")
    
    def __len__(self) -> int:
        return max(0, len(self.boundaries) - 1)
        
    def __iter__(self) -> Iterator[bytes] | Iterator[str]:
        with open(self.corpus_path, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            for i in range(len(self.boundaries)-1):
                start = self.boundaries[i]
                end = self.boundaries[i+1]
                chunk = mm[start:end]
                
                if self.return_bytes:
                    yield chunk
                else:
                    yield chunk.decode("utf-8")


class PreTokenizer(ABC):
    @abstractmethod
    def __call__(
        self, 
        corpus_path: str,
        special_tokens: list[bytes],
        split_special_token: bytes
        ) -> dict[bytes, int]:
        pass

    def _process_chunk(self, chunk: bytes, special_tokens: list[bytes]) -> dict[bytes, int]:
        pattern = re.compile(rb"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

        pre_token_count = defaultdict(int)

        for mini_chunk in re.split(b"|".join([re.escape(token) for token in special_tokens]), chunk):
            pre_tokens = Counter(re.findall(pattern, mini_chunk))
            for pre_token, count in pre_tokens.items():
                pre_token_count[pre_token] += count
        
        return pre_token_count
    
    def pre_tokenize(self, input_bytes: bytes, special_tokens: list[bytes]) -> Iterator[bytes]:
        special_tokens = sorted(special_tokens, key =lambda token: len(token), reverse=True)

        pattern = re.compile(rb"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

        if special_tokens:
            special_tokens_pattern = (b"|".join([re.escape(token) for token in special_tokens]))
        else:
            special_tokens_pattern = b""

        special_tokens_pattern = b"(" + special_tokens_pattern + b")"

        if special_tokens:
            for mini_chunk in re.splititer(special_tokens_pattern, input_bytes):
                if mini_chunk in special_tokens:
                    yield mini_chunk
                    continue
                for pre_token in re.finditer(pattern, mini_chunk):
                    yield pre_token.group()
        else:
            for pre_token in re.finditer(pattern, input_bytes):
                yield pre_token.group()
                    

@profile
class NativePreTokenizer(PreTokenizer):
    def __call__(
        self, 
        corpus_path: str,
        special_tokens: list[bytes],
        split_special_token: bytes,
        num_chunks: int = 8,

        ) -> dict[bytes, int]:

        total_pre_token_count = defaultdict(int)
        file_size = os.path.getsize(corpus_path)

        chunk_iter = FileChunkIterator(corpus_path, num_chunks, split_special_token, desired_bytes=4096)

        with open(corpus_path, "br") as f:
            for chunk in tqdm(chunk_iter, desc="Processing chunks"):
                chunk_pre_token_count = self._process_chunk(chunk, special_tokens)
                for pre_token, count in chunk_pre_token_count.items():
                    total_pre_token_count[pre_token] += count
        
        logger.info(f"pre-tokenize done")
        return total_pre_token_count

            
    