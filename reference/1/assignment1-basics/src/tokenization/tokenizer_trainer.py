"""
BPE (Byte Pair Encoding) 分词器训练器
======================================
本文件实现了 BPE 算法的训练过程。BPE 是 GPT 系列等大模型常用的分词算法，
其核心思想是：从字节级别开始，不断把语料中"出现频次最高的相邻字节对"合并成一个新 token，
直到词表大小达到目标值为止。

文件中包含三个类：
1. ComparablePair      ：用于把"字节对"放进堆里时自定义比较规则的小工具类。
2. TokenizerTrainerBase：训练器基类，提供 save() 等公共方法。
3. TokenizerTrainer    ：纯 Python 实现的 BPE 训练器（带优化：用堆 + 倒排索引加速）。
4. TokenizerTrainerC   ：调用 C++ 扩展 (tokenizer_cpp) 的训练器，速度更快。
"""

import heapq            # Python 自带的最小堆 (heap) 实现，用于快速找出"频次最高的字节对"
import timeit           # 计时工具，用于测量 C++ 训练器耗时
from collections import defaultdict  # 带默认值的字典，访问不存在的 key 时自动创建
from pathlib import Path             # 面向对象的路径处理库

import tokenizer_cpp                 # C++ 编写、用 pybind11 暴露给 Python 的扩展模块
import tqdm                          # 进度条库
from line_profiler import profile    # 逐行性能分析装饰器（仅做性能调试用）
from loguru import logger            # 第三方日志库，比标准库 logging 更易用

from ._types import Token, Vocab     # 类型别名：Token=bytes，Vocab=dict[int, bytes]
from .pre_tokenization import MultiProcessPreTokenizer  # 预分词器（多进程版本）


class ComparablePair:
    """
    一个对"字节对 (bytes, bytes)"做自定义比较的小包装器。

    为什么需要它？
    -------------
    Python 的 `heapq` 是「最小堆」——堆顶始终是最小的元素。
    我们要的却是「频次最高的字节对」，所以一般做法是把 (-count, pair) 推进堆里，
    让 -count 最小相当于 count 最大。

    但是当两个字节对 count 相同时，BPE 算法要求按"字典序较大的字节对优先合并"
    （这是 cs336 作业里的规定，也是 GPT-2 的实际行为）。
    因此我们需要让 pair 之间的比较反过来：pair 越大反而越"小"，从而堆顶优先取它。
    这正是 __lt__ 中 `self.pair > other.pair` 的含义。
    """

    __slots__ = ("pair",)  # __slots__ 限定该类只能有一个属性 pair，节省内存（创建几百万个对象时很有用）

    def __init__(self, pair):
        self.pair = pair

    def __lt__(self, other):
        # 注意是 ">"，把比较反向：让"字典序大的"被 heap 当成"小的"
        return self.pair > other.pair

    def __eq__(self, other):
        return self.pair == other.pair

    def __repr__(self):
        return str(self.pair)


class TokenizerTrainerBase:
    """训练器基类：定义统一的接口，并提供保存 vocab/merges 文件的功能。"""

    vocab: Vocab                              # 词表：{token_id: token_bytes}
    merges: list[tuple[bytes, bytes]]         # 合并规则列表：按合并的先后顺序记录每一次合并的 (a, b)

    def train(self) -> tuple[Vocab, list[tuple[bytes, bytes]]]:
        """子类必须实现的训练方法。"""
        raise NotImplementedError

    def save(self, directory: str | Path) -> None:
        """
        把训练得到的词表和合并规则保存到磁盘，方便后续加载使用。
        - vocab.json ：词表，key 是 token 的 latin-1 字符串表示，value 是 token id
        - merges.txt ：每行一条合并规则： "a b"，表示把 a 和 b 合并成 ab
        """
        import json

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)  # 不存在则递归创建目录

        vocab_file = directory / "vocab.json"
        merges_file = directory / "merges.txt"

        # 把 {id: bytes} 转换为 {str: id} 的形式，方便人查看 / 与 HuggingFace 格式兼容
        vocab_to_save = {}
        for token_id, token_bytes in self.vocab.items():
            # 用 latin-1 编码可以保证任意字节序列都能唯一映射回 str（每个 byte 对应一个字符）
            token_str = token_bytes.decode("latin-1")
            vocab_to_save[token_str] = token_id

        # 按 token_id 排序，让 vocab.json 看起来更整齐
        vocab_to_save = {k: v for k, v in sorted(vocab_to_save.items(), key=lambda item: item[1])}

        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(vocab_to_save, f, indent=2, ensure_ascii=True)

        with open(merges_file, "w", encoding="utf-8") as f:
            for p1, p2 in self.merges:
                s1 = p1.decode("latin-1")
                s2 = p2.decode("latin-1")
                f.write(f"{s1} {s2}\n")

        logger.info(f"Model saved to {directory}")


class TokenizerTrainer(TokenizerTrainerBase):
    """
    纯 Python 实现的 BPE 训练器。

    核心数据结构（关键！理解这些就理解了算法）：
    ------------------------------------------------
    - pre_token_count       : {pre_token_bytes: 出现次数}
                              预分词后每个"词"出现的次数。BPE 在每个 pre_token 内部做合并，
                              不会跨 pre_token 合并（这是 GPT-2 风格的预分词的作用）。
    - pre_token_states      : {pre_token_bytes: 当前 token 列表}
                              每个 pre_token 当前被切分成哪些 token（一开始是逐字节切分，随着合并越来越粗）。
    - pair_counts           : {(token_a, token_b): 该 pair 在所有语料中的总出现次数}
                              即 BPE 训练时关心的"字节对频次表"。
    - pair_to_pretokens     : {pair: 包含该 pair 的 pre_token 集合}
                              倒排索引：合并 pair 时，只需更新这些 pre_token，不必扫所有词。
    - pair_heap             : 最大堆（用 -count + ComparablePair 模拟），方便 O(log N) 取出频次最高的 pair。
    """

    def __init__(
        self,
        corpos_path: str,                                # 语料文件路径（注意：这里拼写是 "corpos"）
        vocab_size: int,                                 # 目标词表大小
        special_tokens: list[str],                       # 特殊 token，如 "<|endoftext|>"
        split_special_token: Token = b"<|endoftext|>",   # 切分文档用的特殊 token（在它处切开，避免跨文档合并）
        pre_tokenizer_cls=MultiProcessPreTokenizer,      # 预分词器类（默认是多进程版本）
    ) -> None:
        self.pre_tokenizer = pre_tokenizer_cls()
        self.target_vocab_size = vocab_size
        # 把 special_tokens 从 str 转成 bytes（BPE 内部全部以 bytes 运作）
        self.special_tokens = [token.encode() for token in special_tokens]
        self.corpos_path = corpos_path
        self.split_special_token = split_special_token

        # ---- 训练过程中的状态变量 ----
        self.vocab_size = 0                              # 当前词表大小
        self.vocab = {}                                  # 词表 {id: bytes}
        self.pair_counts = defaultdict(int)              # 字节对 -> 频次
        self.merges: list[tuple[bytes, bytes]] = []      # 合并规则（按顺序）
        self.pre_token_states: dict[bytes, list[bytes]] = {}  # 每个 pre_token 当前的 token 序列

        # 倒排索引：pair -> 包含此 pair 的 pre_token 集合（合并时只更新这些）
        self.pair_to_pretokens: dict[tuple[bytes, bytes], set[bytes]] = defaultdict(set)

        self.pair_heap = []                              # 最大堆，存 (-count, ComparablePair(pair))

    def _init(self) -> None:
        """
        初始化阶段：
        1) 把特殊 token 加入词表；
        2) 把 0~255 这 256 个字节都加入词表（保证任何字节序列都能被分词）；
        3) 调用预分词器，获得每个 pre_token 的频次；
        4) 把每个 pre_token 切成「单字节列表」作为初始状态；
        5) 统计所有相邻字节对的频次，构建堆和倒排索引。
        """
        # 1) 添加特殊 token
        for token in self.special_tokens:
            self._add_token(token)
        # 2) 添加 256 个基础字节（这样任何 UTF-8 字节都能表示）
        for ascii_code in range(256):
            self._add_token(bytes([ascii_code]))

        # 3) 预分词 -> {pre_token: count}
        self.pre_token_count = self.pre_tokenizer(
            corpos_path=self.corpos_path,
            split_special_token=self.split_special_token,
            special_tokens=self.special_tokens,
        )

        # 4) & 5) 初始化每个 pre_token 的"当前切分状态"和 pair_counts
        for pre_token, count in self.pre_token_count.items():
            # 初始状态：把一个 bytes 词拆成单字节 [b'h', b'e', b'l', b'l', b'o']
            self.pre_token_states[pre_token] = [bytes([b]) for b in pre_token]
            # 统计相邻字节对
            for idx in range(len(self.pre_token_states[pre_token]) - 1):
                pair = (self.pre_token_states[pre_token][idx], self.pre_token_states[pre_token][idx + 1])
                self.pair_counts[pair] += count
                self.pair_to_pretokens[pair].add(pre_token)

        # 把所有 pair 推入堆，准备开始合并
        for pair, count in self.pair_counts.items():
            heapq.heappush(self.pair_heap, (-count, ComparablePair(pair)))

    def _determine_merge_pair(self) -> tuple[Token, Token] | None:
        """
        从堆顶取出"当前频次最高的 pair"。

        为什么不能直接 heap.pop() 一次就返回？
        --------------------------------------
        因为 _merge_pair() 修改 pair_counts 时只是"再 push 一次"，并没有把
        旧的 (count, pair) 条目从堆里删掉（heapq 不支持高效删除任意元素）。
        所以堆里可能存在「过期条目」——它的 count 跟 pair_counts 中真实值对不上。
        我们用「懒惰删除」策略：弹出来后检查 count 是否仍然等于 pair_counts[pair]，
        相等就用，不等就丢掉，继续弹下一个。
        """
        while self.pair_heap:
            neg_count, wrapper = heapq.heappop(self.pair_heap)
            count = -neg_count
            pair = wrapper.pair

            if count == self.pair_counts[pair]:  # 不是过期数据，可以使用
                return pair
            # 否则丢弃这个过期条目，继续循环

        return None  # 堆空了，没有可合并的 pair

    def _merge_pair(self, pair: tuple[Token, Token]) -> None:
        """
        执行一次合并：把所有 pre_token 中出现的 pair=(A,B) 合并成 AB，
        同时更新 pair_counts、pair_to_pretokens、pair_heap。

        关键：这里只遍历"包含 pair 的 pre_token"（来自倒排索引），不是全部词，效率高。
        """
        merged_token = pair[0] + pair[1]    # 合并后的新 token，例如 b'h' + b'e' -> b'he'
        self._add_token(merged_token)       # 加入词表

        affected_pre_tokens = self.pair_to_pretokens[pair]

        for pre_token in affected_pre_tokens:
            state = self.pre_token_states[pre_token]   # 这个词当前的 token 序列
            count = self.pre_token_count[pre_token]    # 这个词出现了多少次

            idx = 0
            # 在 state 里扫描所有 (state[idx], state[idx+1]) == pair 的位置并就地合并
            while idx < len(state) - 1:
                if (state[idx], state[idx + 1]) == pair:
                    # 把两个 token 合并成一个：state[idx] 改成新 token，pop 掉 state[idx+1]
                    state[idx] = merged_token
                    state.pop(idx + 1)

                    # ---- 更新「左邻 pair」的统计 ----
                    # 原本是 (last, A)，现在变成 (last, AB)
                    if idx > 0:
                        last_token = state[idx - 1]
                        old_prev_pair = (last_token, pair[0])
                        self.pair_counts[old_prev_pair] -= count
                        heapq.heappush(
                            self.pair_heap, (-self.pair_counts[old_prev_pair], ComparablePair(old_prev_pair))
                        )

                        new_prev_pair = (last_token, merged_token)
                        self.pair_counts[new_prev_pair] += count
                        self.pair_to_pretokens[new_prev_pair].add(pre_token)
                        heapq.heappush(
                            self.pair_heap, (-self.pair_counts[new_prev_pair], ComparablePair(new_prev_pair))
                        )

                    # ---- 更新「右邻 pair」的统计 ----
                    # 原本是 (B, next)，现在变成 (AB, next)
                    if idx < len(state) - 1:
                        next_token = state[idx + 1]
                        old_next_pair = (pair[1], next_token)
                        self.pair_counts[old_next_pair] -= count
                        heapq.heappush(
                            self.pair_heap, (-self.pair_counts[old_next_pair], ComparablePair(old_next_pair))
                        )

                        new_next_pair = (merged_token, next_token)
                        self.pair_counts[new_next_pair] += count
                        self.pair_to_pretokens[new_next_pair].add(pre_token)
                        heapq.heappush(
                            self.pair_heap, (-self.pair_counts[new_next_pair], ComparablePair(new_next_pair))
                        )

                    # 注意：合并后 idx 不递增——因为合并后位置 idx 的 token 变了，
                    # 但下一次循环检查的是 (state[idx], state[idx+1])，正好可以处理"AAA"这种连续合并的情况（idx 现在指向 merged_token，下一对是 (merged_token, 下一个)）
                else:
                    idx += 1

        # pair 已合并完毕，从两个表里删除（再也不会出现 pair=(A,B) 这个组合了）
        del self.pair_counts[pair]
        del self.pair_to_pretokens[pair]

    def _add_token(self, token: Token) -> None:
        """把一个新 token 加入词表，token id 自增。"""
        self.vocab[self.vocab_size] = token
        self.vocab_size += 1

    def train(self) -> tuple[Vocab, list[tuple[bytes, bytes]]]:
        """
        BPE 训练主循环：
        反复挑出频次最高的 pair 合并，直到达到目标词表大小或没有可合并的 pair 为止。
        """
        self._init()

        tqdm.tqdm.write(f"Initial vocabulary size: {self.vocab_size}")
        # 还需要做多少次合并 = 目标 - 当前(已经包含了 256 个字节 + 特殊 token)
        num_merges_needed = self.target_vocab_size - self.vocab_size
        if num_merges_needed <= 0:
            return {}, self.merges  # 目标小于初始大小，无需训练

        with tqdm.tqdm(total=num_merges_needed, desc="Training BPE") as pbar:
            while self.vocab_size < self.target_vocab_size:
                merge_pair = self._determine_merge_pair()
                if merge_pair is None:
                    break  # 没有 pair 可以再合并了，提前结束

                self.merges.append(merge_pair)   # 记录合并历史（顺序很重要！推理时要按相同顺序应用）
                self._merge_pair(merge_pair)     # 执行合并

                pbar.update(1)
                pbar.set_description(f"Vocab size: {self.vocab_size}")

        tqdm.tqdm.write(f"Training complete. Final vocab size: {self.vocab_size}")
        return self.vocab, self.merges


@profile  # line_profiler 的装饰器：会逐行统计这个类里方法的运行时间，便于性能分析
class TokenizerTrainerC(TokenizerTrainerBase):
    """
    使用 C++ 扩展实现的 BPE 训练器。

    Python 版的逻辑虽然清楚，但循环开销大；C++ 版速度快得多（一般 10x ~ 100x）。
    这个类只是一个"壳子"：把语料预分词放到 Python 侧（方便利用现有多进程逻辑），
    然后把结果传给 C++ 对象，由 C++ 完成核心 BPE 训练。
    """

    def __init__(
        self,
        corpos_path: str,
        vocab_size: int,
        special_tokens: list[str],
        split_special_token: Token = b"<|endoftext|>",
        pre_tokenizer_cls=MultiProcessPreTokenizer,
    ):
        self.pre_tokenizer = pre_tokenizer_cls()
        self.target_vocab_size = vocab_size
        self.special_tokens = [token.encode() for token in special_tokens]
        self.corpos_path = corpos_path
        self.split_special_token = split_special_token

        # 这些字段在 C++ 版里其实只起占位作用（与基类接口兼容），实际训练由 self.trainer 完成
        self.vocab_size = 0
        self.vocab = {}
        self.pair_counts = defaultdict(int)
        self.merges: list[tuple[bytes, bytes]] = []
        self.pre_token_states: dict[bytes, list[bytes]] = {}

        # 创建 C++ 训练器对象（在 csrc/ 中实现，通过 pybind11 暴露给 Python）
        self.trainer = tokenizer_cpp.TokenizerTrainerC(vocab_size)

    def train(self) -> tuple[dict, list[tuple[bytes, bytes]]]:
        # 1) Python 端做预分词（多进程读取语料并切分成 pre_token 计数）
        self.pre_token_count = self.pre_tokenizer(
            corpos_path=self.corpos_path,
            split_special_token=self.split_special_token,
            special_tokens=self.special_tokens,
        )
        # 2) 把数据喂给 C++ 训练器
        self.trainer.LoadData(self.special_tokens, self.pre_token_count)

        # 3) 调用 C++ 端的 train()，并计时
        t1 = timeit.default_timer()
        self.vocab, self.merges = self.trainer.train()
        t1 = timeit.default_timer() - t1

        logger.info(
            f"TokenizerTrainerC training time: {t1:.2f} seconds, {self.target_vocab_size / t1:.2f} tokens/second"
        )
        return self.vocab, self.merges
