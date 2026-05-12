"""
Tokenizer 模块 —— BPE (Byte-Pair Encoding) 分词器的核心实现。

BPE 是一种子词（subword）分词算法，核心思想是：
1. 从字符级别开始，把每个字节当作一个独立的 token
2. 统计相邻 token 对的出现频率，找到最常出现的 pair
3. 将最频繁的 pair 合并成一个新的 token
4. 重复步骤 2-3，直到达到目标词表大小

这个文件中定义的 Tokenizer 类就是用来执行 BPE 编码（文本 → token id 列表）
和解码（token id 列表 → 文本）的。
"""

import json
import os
from collections.abc import Iterable, Iterator  # 用于类型标注，表示可迭代对象和迭代器
from multiprocessing import Pool  # 多进程并行处理，加速批量编码
from pathlib import Path  # 更方便地面向对象的文件路径操作
from typing import Self  # Python 3.11+ 的类型，表示"当前类自身"

import numpy as np  # 数值计算库
import regex as re  # 比标准库 re 更强大的正则表达式库（支持 Unicode 更好）
from cachetools import LRUCache  # LRU (Least Recently Used) 缓存，最近最少使用淘汰策略
from tqdm import tqdm  # 进度条显示

# 从同一包中导入其他模块
from ._types import PreToken, Vocab  # PreToken: 预分词后的 token 类型; Vocab: 词表类型（dict[int, bytes]）
from .pre_tokenization import MultiProcessPreTokenizer  # 多进程预分词器
from .utils import find_chunk_boundaries, make_file_str_iter  # 文件分块工具函数


class Tokenizer:
    """
    BPE 分词器。

    包含两个核心操作：
    - encode: 将文本转换为 token id 序列
    - decode: 将 token id 序列还原为文本

    属性:
        vocab (Vocab): 词表，从 token_id → bytes 的映射
        merges (list[tuple[bytes, bytes]]): 合并规则列表，按优先级排序
        special_tokens (list[bytes]): 特殊 token 列表（如 <|endoftext|>）
        token_to_id (dict[bytes, int]): vocab 的反向映射，从 bytes → token_id
        ranks (dict[tuple[bytes, bytes], int]): 合并规则优先级字典，(a,b) → 优先级编号
        cache (LRUCache): 缓存已编码的 PreToken，避免重复计算
        pre_tokenizer: 预分词器实例
    """

    def __init__(
        self,
        vocab: Vocab,  # 词表：{0: b'a', 1: b'b', 2: b'ab', ...}
        merges: list[tuple[bytes, bytes]],  # 合并规则：[ (b'a', b'b'), (b'ab', b'c'), ... ]
        special_tokens: list[str] | None = None,  # 特殊 token，如 ["<|endoftext|>"]
    ):
        # ---- 保存基本属性 ----
        self.vocab = vocab
        self.merges = merges
        # 将特殊 token 的字符串编码为 bytes 存储
        self.special_tokens = [token.encode() for token in special_tokens] if special_tokens else []

        # ---- 确保所有特殊 token 都在词表中（如果不在则自动添加） ----
        for token in self.special_tokens:
            if token not in self.vocab.values():
                # 给新 token 分配一个新的 id（等于当前词表大小）
                self.vocab[len(self.vocab)] = token

        # ---- 构建反向映射和优先级字典（方便快速查找） ----
        # token_to_id: bytes → id，例如 {b'a': 0, b'b': 1, b'ab': 2}
        self.token_to_id = {token: idx for idx, token in self.vocab.items()}
        # ranks: (a, b) → 合并优先级编号，编号越小越优先合并
        self.ranks = {pair: i for i, pair in enumerate(merges)}

        # ---- 初始化缓存和预分词器 ----
        # LRU 缓存：最多缓存 10000 个 token 的编码结果，超出时淘汰最久未使用的
        self.cache = LRUCache(maxsize=10000)
        self.pre_tokenizer = MultiProcessPreTokenizer()

        # ---- 一致性校验：词表和反向映射的大小必须相同（不能有重复 token） ----
        assert len(self.vocab) == len(self.token_to_id), "Vocab contains duplicate tokens."

    # ==================== 编码 (encode) ====================

    def encode(self, text: str | bytes) -> list[int]:
        """
        将一段文本编码为 token id 列表。

        流程：
        1. 如果输入是字符串，先编码为 UTF-8 bytes
        2. 用预分词器把 bytes 切分成多个 PreToken（按空格、标点等边界切分）
        3. 对每个 PreToken 调用 _encode_one_token 进行 BPE 编码
        4. 把所有结果拼接成一个列表

        参数:
            text: 要编码的文本（字符串或 bytes）
        返回:
            token id 列表，如 [12, 345, 67, 8]
        """
        # 字符串 → bytes（UTF-8 编码）
        byte_text = text.encode("utf-8") if isinstance(text, str) else text
        id_list = []
        # 预分词：将 bytes 切分成 PreToken 列表
        for pre_token in self.pre_tokenizer.pre_tokenize(byte_text, self.special_tokens):
            # 对每个 PreToken 单独做 BPE 编码，结果追加到总列表
            id_list.extend(self._encode_one_token(pre_token))
        return id_list

    def encode_batch(
        self,
        texts: Iterable[str | bytes],  # 一批文本
        num_workers: int | None = None,  # 并行进程数，None 则自动选择
        save_file: str | None = None,  # 如果指定，编码结果直接存到二进制文件
    ) -> list[list[int]]:
        """
        批量编码多段文本，使用多进程并行加速。

        参数:
            texts: 多段文本的可迭代对象
            num_workers: 并行进程数（默认 CPU 核心数 - 1）
            save_file: 如果给定，结果以 np.uint16 格式写入文件，不返回列表
        返回:
            token id 列表的列表，如 [[1,2,3], [4,5,6], ...]
        """
        if not texts:
            return []

        # 自动确定进程数：CPU 核心数 - 1（至少 1 个）
        if num_workers is None:
            cpu_count = os.cpu_count() or 10  # 如果获取失败，默认 10
            num_workers = max(1, cpu_count - 1)

        results = []

        if save_file is None:
            # 模式 1：编码结果存入内存的 list
            with Pool(processes=num_workers) as pool:
                # imap 是惰性 map，逐个返回结果；tqdm 显示进度条
                for chunk_ids in tqdm(
                    pool.imap(self.encode, texts, chunksize=1),
                    desc="Encoding",
                    total=len(list(texts)),  # 注意：list(texts) 会消费可迭代对象
                ):
                    results.append(chunk_ids)
        else:
            # 模式 2：编码结果直接写入二进制文件（节省内存）
            with open(save_file, "wb") as f_out, Pool(processes=num_workers) as pool:
                for chunk_ids in tqdm(
                    pool.imap(self.encode, texts, chunksize=1),
                    desc="Encoding",
                    total=len(list(texts)),
                ):
                    # 转换为 16 位无符号整数数组（词表大小通常 < 65536）
                    res = np.array(chunk_ids, dtype=np.uint16)
                    res.tofile(f_out)  # 直接写入二进制文件

        return results

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        以流式方式编码文本，返回一个迭代器。

        与 encode 的区别：
        - encode 一次性返回整个列表
        - encode_iterable 逐个产出 token id，适合处理超长文本（节省内存）

        参数:
            iterable: 文本的可迭代对象（如文件中逐行读取）
        产出:
            逐个 token id
        """
        for text in iterable:
            byte_text = text.encode("utf-8")
            for pre_token in self.pre_tokenizer.pre_tokenize(byte_text, self.special_tokens):
                # yield from：把 _encode_one_token 的结果逐个产出
                yield from self._encode_one_token(pre_token)

    def _encode_one_token(self, token: PreToken) -> list[int]:
        """
        对一个 PreToken（bytes）执行 BPE 编码。
        这是 BPE 编码的核心算法。

        BPE 编码步骤：
        1. 检查缓存，命中则直接返回
        2. 如果整个 token 刚好在词表里，直接返回其 id
        3. 把 token 拆成单个字节的列表：[b'a', b'b', b'c', ...]
        4. 重复以下过程直到无法合并：
           a. 遍历所有相邻 pair，找到在 ranks 中优先级最高的 pair
           b. 从头到尾扫描 word，将匹配的 pair 合并成一个新的 token
        5. 将最终的 bytes 列表映射为 id 列表

        参数:
            token: 一个 PreToken（bytes 类型）
        返回:
            token id 列表
        """
        # ---- 步骤 1：检查缓存 ----
        # LRU 缓存会自动处理淘汰，最常访问的 token 会保留在缓存中
        if token in self.cache:
            return self.cache[token]

        # ---- 步骤 2：直接匹配整个 token ----
        if token in self.token_to_id:
            return [self.token_to_id[token]]

        # ---- 步骤 3：初始化为单字节列表 ----
        # 例如 b"abc" → [b'a', b'b', b'c']
        word = [token[i : i + 1] for i in range(len(token))]

        # ---- 步骤 4：迭代合并 ----
        while len(word) > 1:
            # 4a. 找到当前 word 中优先级最高的相邻 pair
            min_rank = float("inf")  # 初始化为无穷大
            min_pair = None

            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])  # 相邻两个 token 组成的 pair
                rank = self.ranks.get(pair, float("inf"))  # 查找该 pair 的优先级
                if rank < min_rank:  # 找到优先级更高的 pair
                    min_rank = rank
                    min_pair = pair

            # 4b. 如果没有可合并的 pair，结束迭代
            if min_pair is None or min_rank == float("inf"):
                break

            # 4c. 执行合并：从左到右扫描，将匹配的 pair 合并
            new_word = []
            i = 0
            while i < len(word):
                # 检查当前位置 i 和 i+1 是否构成要合并的 pair
                if i < len(word) - 1 and word[i] == min_pair[0] and word[i + 1] == min_pair[1]:
                    # 匹配到了 → 合并为一个 token
                    new_word.append(word[i] + word[i + 1])
                    i += 2  # 跳过两个位置
                else:
                    # 没匹配到 → 保留原 token
                    new_word.append(word[i])
                    i += 1
            word = new_word  # 用合并后的新列表替换旧列表，进入下一轮迭代

        # ---- 步骤 5：将最终的 bytes 列表映射为 id 列表 ----
        ids_list = [self.token_to_id[token_bytes] for token_bytes in word]
        # 存入缓存，下次遇到相同的 token 可直接返回
        self.cache[token] = ids_list
        return ids_list

    # ==================== 解码 (decode) ====================

    def decode(self, token_ids: list[int]) -> str:
        """
        将 token id 列表解码回文本字符串。

        流程：
        1. 将每个 id 通过 vocab 映射回 bytes
        2. 把所有 bytes 拼接起来
        3. 用 UTF-8 解码为字符串（用 errors="replace" 处理无法解码的字节）

        参数:
            token_ids: token id 列表
        返回:
            解码后的字符串
        """
        # id → bytes 映射
        byte_list = [self.vocab[token_id] for token_id in token_ids]
        # 拼接所有 bytes 并解码为 UTF-8 字符串
        return b"".join(byte_list).decode("utf-8", errors="replace")

    # ==================== 从文件加载 (类方法) ====================

    @classmethod
    def from_file(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None) -> Self:
        """
        从 vocab 文件和 merges 文件构建 Tokenizer 实例。

        这是 OpenAI GPT-2 风格的词表和合并文件格式：
        - vocab 文件：每行是 `"token_bytes"` 的 base64 表示
        - merges 文件：每行是 `token1 token2`，表示合并规则

        参数:
            vocab_filepath: 词表文件路径
            merges_filepath: 合并规则文件路径
            special_tokens: 特殊 token 列表，默认 None
        返回:
            Tokenizer 实例
        """
        vocab: Vocab = {}

        # ---- 解析 vocab 文件 ----
        # 词表文件中每行格式：用引号括起来的 token（字符串形式存储的 bytes）
        # 正则匹配: " 开头，捕获内容，" 结尾
        vocal_pattern = rb'"(.*)"'
        idx = 0
        with open(vocab_filepath, "rb") as vf:
            # finditer 返回所有匹配的迭代器
            for match in re.finditer(vocal_pattern, vf.read()):
                token = match.group(1)  # 提取引号内的内容（bytes）
                vocab[idx] = token
                idx += 1

        # ---- 解析 merges 文件 ----
        # 每行格式：token1 token2（用空格分隔的 bytes）
        merge_pattern = rb"^(.*) (.*)$"
        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath, "rb") as mf:
            for line in mf:
                match = re.match(merge_pattern, line.rstrip())
                if match:
                    token1 = match.group(1)
                    token2 = match.group(2)
                    merges.append((token1, token2))

        # 用解析出的 vocab 和 merges 构造 Tokenizer
        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def encode_file(
        self,
        file_path: str,  # 要编码的文件路径
        split_token: str = "<|endoftext|>",  # 分块分割 token
        save_file: str | None = None,  # 保存路径
    ) -> list[int]:
        """
        编码整个文件。

        处理流程：
        1. 找文件的分块边界（在特殊 token 处分割，避免多进程时切断 token）
        2. 创建一个文件迭代器，按分块读取
        3. 调用 encode_batch 进行批量编码

        参数:
            file_path: 要编码的文件路径
            split_token: 用于分割文档的特殊 token
            save_file: 可选，编码结果保存路径
        返回:
            token id 列表
        """
        # 找到分块边界：在 split_token 处分割，每块约 1MB
        boundries = find_chunk_boundaries(
            file_path=file_path,
            desired_num_chunks=(os.cpu_count() or 8) * 100,  # 目标分块数 = CPU 核心数 × 100
            split_special_token=split_token.encode(),
            desize_bytes=1024 * 1024,  # 每块约 1MB
        )
        # 创建文件迭代器，按边界分块读取文件内容
        file_iter = make_file_str_iter(file_path=file_path, boundaries=boundries, return_bytes=True)
        # 批量编码
        ret = self.encode_batch(file_iter, save_file=save_file)
        # 展平嵌套列表：[[1,2], [3,4]] → [1,2,3,4]
        return [list_id for sublist in ret for list_id in sublist]

    @classmethod
    def from_my_save(cls, save_dir: str | Path, special_tokens: list[str] | None = None) -> Self:
        """
        从自定义格式的保存目录加载 Tokenizer。

        这是作业中自己训练的 BPE 分词器的保存格式：
        - vocab.json: JSON 格式，{token_string: token_id}
        - merges.txt: 每行 token1 token2

        参数:
            save_dir: 保存目录路径
            special_tokens: 特殊 token 列表，默认使用 ["<|endoftext|>"]
        返回:
            Tokenizer 实例
        """
        # 默认特殊 token
        if special_tokens is None:
            special_tokens = ["<|endoftext|>"]

        directory = Path(save_dir)

        # ---- 加载 vocab.json ----
        with open(directory / "vocab.json", encoding="utf-8") as f:
            vocab_str = json.load(f)  # 类型：{str: int}，如 {"a": 0, "b": 1}

        vocab = {}
        vocab_size = 0  # 只是占位，没有实际使用（原代码的残留变量）
        for token_str, token_id in vocab_str.items():
            # 注意：这里用 latin-1（ISO 8859-1）编码，而不是 UTF-8
            # 这是因为 BPE token 可能是任意字节序列，用 latin-1 可以无损映射
            # 每个 0-255 的字节恰好对应一个 latin-1 字符
            token_bytes = token_str.encode("latin-1")
            vocab[token_id] = token_bytes
        vocab_size = vocab_size  # 无实际作用的赋值

        # ---- 加载 merges.txt ----
        merges = []
        merges_file = directory / "merges.txt"
        if merges_file.exists():
            with open(merges_file, encoding="utf-8") as f:
                for line in f:
                    regex = r"^(.*) (.*)$"  # 匹配 "token1 token2" 格式
                    match = re.match(regex, line.rstrip())
                    if match:
                        # 同样用 latin-1 编码恢复原始 bytes
                        token1 = match.group(1).encode("latin-1")
                        token2 = match.group(2).encode("latin-1")
                        merges.append((token1, token2))

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)