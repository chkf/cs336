# =============================================================================
# 预分词（Pre-tokenization）模块
# =============================================================================
# 本模块实现了将原始文本（字节流）切分成"预分词单元"（pre-tokens）的功能。
#
# 什么是预分词？
# ─────────────
# 在 BPE（Byte Pair Encoding）等分词算法中，第一步通常不是直接在字符级别操作，
# 而是先把文本切分成一些"粗略的片段"，这些片段就是 pre-tokens。
#
# 举例：
#   原始文本: "Hello, world!"
#   预分词结果: ["Hello", ",", " world", "!"]
#
# 之后 BPE 会在每个 pre-token 内部进行子词合并，不会跨越 pre-token 边界。
# 这样做的好处是：
#   1. 常见的标点、空格被提前分开了，BPE 不需要学习跨越它们的合并规则
#   2. 特殊 token（如 <|endoftext|>）可以提前识别，不被错误拆分
#
# 本文件包含三个类：
#   - PreTokenizer (抽象基类): 定义了预分词的接口和通用方法
#   - NativePreTokenizer: 单进程版本，按顺序逐块处理
#   - MultiProcessPreTokenizer: 多进程版本，利用多核 CPU 并行处理
# =============================================================================

# ---- 标准库导入 ----
import os       # 文件系统操作，如获取文件大小
import time     # 计时，统计预处理耗时
from abc import ABC, abstractmethod               # 定义抽象基类
from collections import Counter, defaultdict      # Counter: 计数; defaultdict: 带默认值的字典
from collections.abc import Iterator              # 类型注解：迭代器
from multiprocessing import Pool, cpu_count       # Pool: 多进程池; cpu_count: 获取 CPU 核心数

# ---- 第三方库导入 ----
import regex as re        # 增强版正则表达式库，支持 \p{L} 等 Unicode 属性
import tqdm               # 进度条显示
from line_profiler import profile    # 性能分析装饰器（标记需要逐行分析性能的函数/类）
from loguru import logger            # 结构化日志输出

# ---- 本地模块导入 ----
from ._types import Chunk, PreTokenCount, Token
# Chunk = bytes（一个文本块，字节类型）
# PreTokenCount = defaultdict[str, int]（预分词到出现次数的映射）
# Token = bytes（一个 token，字节类型）

from .utils import find_chunk_boundaries
# find_chunk_boundaries: 将大文件切分成多个块的工具函数。
# 它会根据特殊分割 token（如 <|endoftext|>）找到合适的切分边界，
# 确保不会在 token 中间切断。


# @profile 装饰器：标记这个类的所有方法，让 line_profiler 可以
# 逐行分析性能（在开发/调试时使用，生产环境可去掉）
@profile
class PreTokenizer(ABC):
    """
    预分词器抽象基类
    ================
    定义了预分词需要的方法：
    - _merge_pre_token_counts: 合并多个计数结果（静态方法）
    - _process_chunk: 处理单个文本块
    - pre_tokenize: 对字节串进行预分词，逐个返回 token
    - __call__: 抽象方法，子类必须实现，用于处理整个语料库文件

    设计模式：模板方法模式（Template Method Pattern）
    - 基类提供了 _process_chunk、pre_tokenize 等通用实现
    - 子类只需实现 __call__ 来定义"如何遍历文件块并汇总结果"
    """

    # ------------------------------------------------------------------
    # 静态方法：合并多个预分词计数字典
    # ------------------------------------------------------------------
    @staticmethod
    def _merge_pre_token_counts(*pre_token_counts: PreTokenCount) -> PreTokenCount:
        """合并多个 PreTokenCount 字典为一个。

        为什么需要合并？
        每个文本块（chunk）处理后会得到一个 {token: 出现次数} 的字典，
        最后需要把所有这些字典合并成一个，才能得到整个语料库的统计结果。

        Args:
            *pre_token_counts: 可变数量的 PreTokenCount 字典

        Returns:
            PreTokenCount: 合并后的统计字典
        """
        merged_pre_token_count: PreTokenCount = defaultdict(int)
        # defaultdict(int): 访问不存在的 key 时自动返回 0，不需要做 if key in dict 检查
        for pre_token_count in pre_token_counts:
            for pre_token, count in pre_token_count.items():
                merged_pre_token_count[pre_token] += count  # 累加相同 token 的出现次数
        return merged_pre_token_count

    # ------------------------------------------------------------------
    # 处理单个文本块（chunk）
    # ------------------------------------------------------------------
    def _process_chunk(self, chunk: Chunk, special_tokens: list[Token]) -> PreTokenCount:
        """
        处理一个文本块，返回该块中每个 pre-token 的出现次数。

        处理流程：
          1. 先用特殊 token 把文本块切分成 mini_chunk
             （这样特殊 token 不会和普通文本混在一起）
          2. 在每个 mini_chunk 中用正则表达式匹配 pre-tokens
          3. 统计每个 pre-token 的出现次数

        Args:
            chunk (bytes): 要处理的文本块（字节串）
            special_tokens (list[Token]): 特殊 token 列表，如 [b"<|endoftext|>"]

        Returns:
            PreTokenCount: {pre_token (bytes): 出现次数 (int)}
        """
        pre_token_count: PreTokenCount = defaultdict(int)

        # ---- 正则表达式解析 ----
        # 这是 GPT-2 风格的预分词正则，用 raw bytes + verbose 模式编写：
        # rb"""..."""  = raw bytes + verbose 模式（空格和注释不参与匹配）
        #
        # 模式分解（用 | 分隔，表示"或"）：
        #   '(?:[sdmt]|ll|ve|re)   → 匹配英文缩写：'s, 'd, 'm, 't, 'll, 've, 're
        #                               如: don't → ["don", "'t"]
        #   ?\p{L}+                 → 匹配字母序列（前面可选一个空格）
        #                               \p{L} = 任意 Unicode 字母（包括中文）
        #   ?\p{N}+                 → 匹配数字序列（前面可选一个空格）
        #   ?[^\s\p{L}\p{N}]+      → 匹配标点/符号（前面可选一个空格）
        #                               ^\\s\\p{L}\\p{N} = 不是空白、不是字母、不是数字
        #   \s+(?!\S)              → 匹配末尾空白（后面没有非空白字符）
        #   \s+                     → 匹配其余空白
        #
        # 关键设计：字母/数字/符号前面都有可选空格 " ?"，这样空格被"粘"在前面的词上。
        # 例如 "hello world" → ["hello", " world"]
        # 这样在 BPE 合并时可以保持词的完整性，同时知道前面是否有空格。
        pattern = re.compile(rb"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

        # 第一步：用所有特殊 token 作为分隔符切分文本
        # re.escape(token): 转义特殊 token 中的正则特殊字符（如 |, (, ) 等）
        # b"|".join(...): 构造 "token1|token2|token3" 的正则模式
        for mini_chunk in re.split(b"|".join([re.escape(token) for token in special_tokens]), chunk):
            # 第二步：在每个 mini_chunk 中用 pattern 匹配 pre-tokens
            # Counter: 自动统计每个匹配结果的次数
            pre_tokens = Counter(re.findall(pattern, mini_chunk))
            # 第三步：累加到总计数中
            for pre_token, count in pre_tokens.items():
                pre_token_count[pre_token] += count
        return pre_token_count

    # ------------------------------------------------------------------
    # 对字节串进行预分词（逐个产出 token）
    # ------------------------------------------------------------------
    def pre_tokenize(self, str_bytes: bytes, special_token_list: list[Token]) -> Iterator[Token]:
        """
        对输入的字节串进行预分词，逐个产出 token。

        与 _process_chunk 的区别：
          _process_chunk 返回的是 {token: 计数}（统计用）
          pre_tokenize 返回的是 Iterator[Token]，逐个产出（用于实际分词/推理时）

        处理逻辑：
          1. 将特殊 token 从长到短排序（优先匹配更长的 token）
          2. 构建特殊 token 的正则模式
          3. 用特殊 token 作为分隔符切分文本
          4. 遇到特殊 token 直接产出，遇到普通文本用正则切分后产出

        Args:
            str_bytes (bytes): 输入字节串
            special_token_list (list[Token]): 特殊 token 列表

        Yields:
            Token (bytes): 逐个产出的预分词 token
        """
        # TODO: 提示：以下步骤待完善
        # 0. 特殊 token 要通过 split 先分割出来
        # 1. 特殊 token 按照从长到短排序

        # 按长度降序排列：长的优先匹配
        # 例如 [b"<|endoftext|>", b"<|a|>"] → 先匹配长的 <|endoftext|>，
        # 避免 <|a|> 先匹配导致 <|endoftext|> 被错误拆分
        special_token_list = sorted(special_token_list, key=len, reverse=True)  # match longer tokens first

        # 构建正则: (token1|token2|token3)
        special_token_pattern = (
            b"|".join([re.escape(token) for token in special_token_list]) if special_token_list else b""
        )
        special_token_pattern = b"(" + special_token_pattern + b")"  # 括号用于捕获分组

        # 预分词正则（同 _process_chunk 中的解释）
        pattern = re.compile(rb"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

        if special_token_list:
            # 有特殊 token：先用它们切分
            # re.splititer: 类似 re.split 但返回迭代器（节省内存），同时保留分隔符
            for mini_chunk in re.splititer(special_token_pattern, str_bytes):
                if mini_chunk in special_token_list:
                    # 如果当前片段是特殊 token，直接产出（不再进一步切分）
                    yield mini_chunk
                    continue
                # 普通文本片段：用正则进一步切分
                for token_match in re.finditer(pattern, mini_chunk):
                    yield token_match.group()  # group() 返回匹配的字节串
        else:
            # 没有特殊 token：直接用正则切分全文
            for token_match in re.finditer(pattern, str_bytes):
                yield token_match.group()

    # ------------------------------------------------------------------
    # 抽象方法：处理整个语料库文件
    # ------------------------------------------------------------------
    @abstractmethod
    def __call__(self, corpos_path: str, split_special_token: Token, special_tokens: list[Token]) -> PreTokenCount:
        """
        预分词整个语料库文件（抽象方法，由子类实现）。

        为什么对象可以像函数一样被调用？
        实现了 __call__ 方法后，实例就可以像函数一样使用：
          tokenizer = NativePreTokenizer()
          result = tokenizer("data.txt", split_token, special_tokens)  # 直接调用实例

        Args:
            corpos_path (str): 语料库文件的路径
            split_special_token (Token): 用于分割语料库的特殊 token
                （例如 b"<|endoftext|>"，用于在文档边界处切分）
            special_tokens (list[Token]): 所有特殊 token 列表
                （这些 token 不会被正则切分，保持完整）

        Returns:
            PreTokenCount: {pre_token: 出现次数}，整个语料库的统计结果
        """
        # 这是一个抽象方法，没有函数体。
        # 子类（NativePreTokenizer, MultiProcessPreTokenizer）必须实现它。


# =============================================================================
# 单进程预分词器
# =============================================================================
class NativePreTokenizer(PreTokenizer):
    """
    单进程版本的预分词器。

    工作流程：
      1. 打开语料库文件
      2. 用 find_chunk_boundaries 计算块的边界
         （以 split_special_token 为边界，确保不切断 token）
      3. 逐块读取数据 → 调用 _process_chunk 处理 → 合并结果
      4. 输出处理耗时和速度

    适用场景：小文件、调试、没有多核 CPU 的环境。
    """

    def __call__(
        self, corpos_path: str, split_special_token: Token, special_tokens: list[Token], num_chunks: int = 8
    ) -> PreTokenCount:
        """
        执行预分词。

        Args:
            corpos_path: 语料库文件路径
            split_special_token: 分割用的特殊 token
            special_tokens: 特殊 token 列表
            num_chunks: 期望的块数量（默认 8），用于在 tqdm 进度条中显示

        Returns:
            PreTokenCount: 整个语料库的预分词统计
        """
        pre_token_count: PreTokenCount = defaultdict(int)

        start_time = time.time()  # 记录开始时间
        with open(corpos_path, mode="br") as f:  # "br" = binary read，以二进制模式只读
            file_size = os.path.getsize(corpos_path)  # 获取文件大小（字节数）

            # 计算块的边界位置（每个边界是文件中的字节偏移量）
            # find_chunk_boundaries 保证每个块的边界都落在 split_special_token 上，
            # 这样不会在 token 中间切断文件
            chunk_boundaries = find_chunk_boundaries(
                file_path=corpos_path,
                desired_num_chunks=num_chunks,
                split_special_token=split_special_token,
            )
            # chunk_boundaries 形如 [0, 1000, 2000, 3000, ...]，相邻元素构成一个块

            # 逐块处理，tqdm 显示进度条
            for i in tqdm.tqdm(range(len(chunk_boundaries) - 1), desc="Pre-tokenizing corpus"):
                start = chunk_boundaries[i]        # 块起始位置
                end = chunk_boundaries[i + 1]      # 块结束位置
                f.seek(start)                      # 文件指针移动到起始位置
                chunk = f.read(end - start)        # 读取该块的全部字节
                chunk_pre_token_count = self._process_chunk(chunk, special_tokens)  # 处理该块
                # 将本块结果合并到总计数
                for pre_token, count in chunk_pre_token_count.items():
                    pre_token_count[pre_token] += count

        end_time = time.time()  # 记录结束时间

        # 输出处理统计：耗时、速度
        logger.info(
            "Takes {:.2f} seconds to pre-tokenize the corpus file {:.2f}, speed: {:.2f} bytes/second",
            end_time - start_time,                        # 耗时（秒）
            corpos_path,                                   # 文件路径
            file_size / (end_time - start_time),           # 处理速度（字节/秒）
        )
        return pre_token_count


# =============================================================================
# 多进程预分词器
# =============================================================================
@profile
class MultiProcessPreTokenizer(PreTokenizer):
    """
    多进程版本的预分词器。

    与 NativePreTokenizer 的核心区别：
      利用 multiprocessing.Pool 并行处理多个文本块，充分利用多核 CPU。

    工作流程：
      1. 计算 CPU 核心数，确定块数量（核心数 × 100）
      2. 用 find_chunk_boundaries 切分文件
      3. 将所有块的信息打包成参数列表 chunks_args
      4. 用进程池 pool.imap_unordered 并行处理所有块
         （imap_unordered = 无序的迭代器映射，哪个先完成就先返回）
      5. 合并所有并行处理的结果

    适用场景：大文件、多核服务器、需要高性能的场景。
    """

    # ------------------------------------------------------------------
    # 根据文件偏移量读取并处理一个块
    # ------------------------------------------------------------------
    def _process_chunk_with_boundry(
        self, corpos_path: str, start: int, end: int, special_tokens: list[Token]
    ) -> PreTokenCount:
        """
        根据字节偏移量读取文件的一个片段并进行预分词。

        这个方法被设计为可以在子进程中独立调用：
        - 每个子进程自己打开文件（文件句柄不能跨进程共享）
        - seek 到指定位置读取指定长度
        - 调用父类的 _process_chunk 处理

        Args:
            corpos_path: 语料库文件路径
            start: 块的起始字节偏移
            end: 块的结束字节偏移
            special_tokens: 特殊 token 列表

        Returns:
            PreTokenCount: 该块的预分词统计
        """
        with open(corpos_path, mode="br") as f:  # 每个子进程独立打开文件
            f.seek(start)                        # 跳转到块起始位置
            chunk = f.read(end - start)          # 读取 end - start 个字节
            pre_token_count = self._process_chunk(chunk, special_tokens)  # 处理
        return pre_token_count

    # ------------------------------------------------------------------
    # 主入口：并行预分词整个语料库
    # ------------------------------------------------------------------
    def __call__(self, corpos_path: str, split_special_token: Token, special_tokens: list[Token]) -> PreTokenCount:
        """
        并行执行预分词。

        Args:
            corpos_path: 语料库文件路径
            split_special_token: 分割用的特殊 token
            special_tokens: 特殊 token 列表

        Returns:
            PreTokenCount: 整个语料库的预分词统计
        """
        final_pre_token_count: PreTokenCount = defaultdict(int)

        start_time = time.time()
        file_size = os.path.getsize(corpos_path)
        num_cpus = cpu_count()  # 获取系统 CPU 逻辑核心数

        # 块数量 = CPU 核心数 × 100
        # 为什么是 100 倍？为了让每个块不太大，方便负载均衡。
        # 如果只有 CPU 核心数个块，有的块可能特别大处理很慢，其他进程闲置等待。
        # 更多的块 → 更细粒度的负载均衡
        desired_chunks = num_cpus * 100

        # 计算所有块边界
        chunk_boundaries = find_chunk_boundaries(
            file_path=corpos_path,
            desired_num_chunks=desired_chunks,
            split_special_token=split_special_token,
        )

        # 构建工作参数列表：每个元素是 (文件路径, 起始位置, 结束位置, 特殊token列表)
        chunks_args = []
        for i in range(len(chunk_boundaries) - 1):
            start = chunk_boundaries[i]
            end = chunk_boundaries[i + 1]
            chunks_args.append((corpos_path, start, end, special_tokens))

        logger.info(f"Splitting task into {len(chunks_args)} chunks.")

        # ---- 多进程并行处理 ----
        # Pool(processes=num_cpus * 2): 创建进程池，进程数 = CPU 核心数 × 2
        # 为什么 × 2？因为文件 I/O 操作会阻塞，用更多进程可以让 CPU 在等待 I/O 时
        # 切换到其他进程，提高整体吞吐量。
        with Pool(processes=num_cpus * 2) as pool:
            # imap_unordered: 类似 map，但是：
            #   - i 表示 iterator（迭代器），惰性求值，节省内存
            #   - unordered 表示不保证返回顺序（哪个先完成就先返回）
            # 相比 map，imap_unordered 可以更早开始处理已完成的结果
            chunk_iter = pool.imap_unordered(self._worker_wrapper, chunks_args)

            # 收集并合并所有结果
            for chunk_result in tqdm.tqdm(chunk_iter, total=len(chunks_args), desc="Pre-tokenizing"):
                for token, count in chunk_result.items():
                    final_pre_token_count[token] += count

        end_time = time.time()
        logger.info(
            "Takes {:.2f} seconds to pre-tokenize, speed: {:.2f} bytes/second",
            end_time - start_time,
            file_size / (end_time - start_time),
        )
        return final_pre_token_count

    # ------------------------------------------------------------------
    # 静态方法：子进程工作函数的包装器
    # ------------------------------------------------------------------
    @staticmethod
    def _worker_wrapper(args):
        """
        多进程池的工作函数包装器。

        为什么需要这个包装器？
        multiprocessing.Pool 的工作函数必须是：
          1. 模块级别的函数（不能在类内部直接使用实例方法）
          2. 可被 pickle 序列化（用于跨进程传输）

        而 _process_chunk_with_boundry 是实例方法，不能直接传给 pool。
        解决方案：创建一个静态方法包装器，在子进程中创建新的 MultiProcessPreTokenizer
        实例，然后调用其 _process_chunk_with_boundry 方法。

        Args:
            args (tuple): (corpos_path, start, end, special_tokens)

        Returns:
            PreTokenCount: 该块的预分词统计
        """
        # 在子进程中创建新的 tokenizer 实例（不共享状态，独立运行）
        tokenizer_instance = MultiProcessPreTokenizer()
        return tokenizer_instance._process_chunk_with_boundry(*args)  # *args 解包元组


# =============================================================================
# 模块独立运行时的演示代码
# =============================================================================
if __name__ == "__main__":
    """
    演示两种预分词器的用法：
      1. MultiProcessPreTokenizer（多进程版本）
      2. NativePreTokenizer（单进程版本）

    要运行此演示：
      python -m src.tokenization.pre_tokenization
    （需要存在 data/owt_valid.txt 文件）
    """
    file = "data/owt_valid.txt"                              # 语料库文件路径
    split_token = b"<|endoftext|>"                           # 文档分割 token
    special_tokens = [b"<|endoftext|>", b"<|startoftext|>"]  # 特殊 token 列表

    # 测试多进程版本
    tokenizer_mp = MultiProcessPreTokenizer()
    tokenizer_mp(file, split_token, special_tokens)

    # 测试单进程版本
    tokenizer = NativePreTokenizer()
    tokenizer(file, split_token, special_tokens)