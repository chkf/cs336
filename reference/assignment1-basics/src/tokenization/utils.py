# ============================================================================
# mmap（内存映射文件）简介
# ============================================================================
# mmap 是 Linux/Unix 提供的一种高效文件读取方式。
# 普通读取：从磁盘 → 内核缓冲区 → 用户内存（复制一次）
# mmap 读取：直接把文件"映射"到内存地址空间，访问内存就是读文件（零拷贝）
# 优点：快、省内存，操作系统会自动缓存，适合大文件随机访问。
import mmap
import os
import time
from collections.abc import Iterator

from loguru import logger

from ._types import Token


def find_chunk_boundaries(
    file_path: str,
    desired_num_chunks: int,
    split_special_token: Token,
    desize_bytes: int | None = None,  # desize_bytes: 期望的每个 chunk 大小（字节）。如果指定了，会覆盖 desired_num_chunks
) -> list[int]:
    """
    找到文件的分块边界位置（字节偏移量列表）。

    为什么要分块？
    - 语料文件可能很大（几 GB），无法一次性读入内存
    - 需要切成多个小块，逐块处理（或多进程并行处理）
    - 但不能在任意位置切——可能在单词中间、特殊 token 中间切断
    - 所以要根据特殊 token（如 <|endoftext|>）的位置来切，保证切分点"安全"

    工作原理：
    1. 把文件等分成 desired_num_chunks 份（得到每个 chunk 的"理想起始位置"）
    2. 从每个理想位置开始，向后搜索特殊 token
    3. 找到特殊 token 的位置就是"切分点"（chunk boundary）

    例如：文件 100MB，要 4 个 chunk，特殊 token 是 b"<|endoftext|>"
    理想切割点: 0MB,  25MB,     50MB,     75MB,   100MB
                     ↓          ↓          ↓
                   在附近找    在附近找    在附近找
                 <|endoftext|> <|endoftext|> <|endoftext|>
    实际切割点: 0,  25.3MB,    50.1MB,    75.5MB, 100MB

    参数：
        file_path: 语料文件的路径
        desired_num_chunks: 期望的分块数量
        split_special_token: 用于分割的特殊 token（bytes 类型）
        desize_bytes: 如果指定，按此大小计算块数（覆盖 desired_num_chunks）

    返回：
        list[int]: 分块边界的字节偏移量列表，包含 0 和文件大小
                   例：[0, 26542080, 53084160, 79626240, 106168320]
    """
    start = time.time()  # 计时开始
    chunk_boundaries = []  # 存放所有分块边界位置
    file_size = os.path.getsize(file_path)  # 获取文件总大小（字节）

    # 如果指定了每个 chunk 的大小，重新计算 chunk 数量
    if desize_bytes is not None:
        desired_num_chunks = file_size // desize_bytes  # // 是整除，例: 100 // 25 = 4

    # ------------------------------------------------------------------
    # 使用 mmap 打开文件（"r+b" 模式：可读写打开，但只读映射）
    # mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ):
    #   - f.fileno(): 获取文件的"文件描述符"（一个整数 ID）
    #   - 0: 映射整个文件
    #   - ACCESS_READ: 只读映射
    # with 语句会自动关闭文件和 mmap（即使出错也会关闭）
    # ------------------------------------------------------------------
    with open(file_path, "r+b") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        # 计算每个 chunk 的理想大小
        chunk_size = file_size // desired_num_chunks

        # 文件开头总是第一个边界
        chunk_boundaries.append(0)

        # 找到第 1, 2, ..., desired_num_chunks-1 个切分点
        for i in range(1, desired_num_chunks):
            # 理想切分位置 = 第 i 个 chunk 的开始位置
            target_pos = i * chunk_size

            # mm.find(token, start)：
            #   从 start 位置开始，在映射的内存中搜索 token
            #   返回找到的位置（字节偏移），如果没找到返回 -1
            found_at = mm.find(split_special_token, target_pos)

            if found_at != -1:  # 找到了特殊 token
                chunk_boundaries.append(found_at)
            else:              # 没找到（文件尾部可能没有特殊 token 了）
                break           # 停止搜索，剩下的部分归到最后一个 chunk

        # 文件末尾总是最后一个边界
        chunk_boundaries.append(file_size)

    end_time = time.time()
    # logger.info 输出日志信息（用 loguru 库，比 print 更好）
    # {end_time - start:.2f} 是格式化输出，保留两位小数
    logger.info(f"Chunk boundary finding took {end_time - start:.2f} seconds.")

    # sorted(list(set(...)))：去重 + 排序，确保边界是严格递增且不重复的
    return sorted(list(set(chunk_boundaries)))


# ============================================================================
# FileChunkIterator 类
# ============================================================================
# 这是一个"迭代器"（Iterator）类。
# 迭代器是 Python 中用来逐个产生数据的对象，好处是：
#   - 不需要把所有数据一次性装进内存
#   - 用 for 循环遍历时，每次只取一个元素
#   - 符合 Python 的"惰性求值"哲学
#
# 这个类的作用：
#   给定文件路径和分块边界列表，依次产生每个 chunk 的内容
#   可以选择返回 bytes（原始字节）或 str（解码后的字符串）
# ============================================================================
class FileChunkIterator:
    def __init__(self, file_path: str, boundaries: list[int], return_bytes: bool = True):
        """
        初始化迭代器。

        参数：
            file_path: 文件路径
            boundaries: 分块边界列表，例如 [0, 100, 200, 300]
            return_bytes: True → 返回 bytes，False → 返回 str（UTF-8 解码）
        """
        self.file_path = file_path
        self.boundaries = boundaries
        self.return_bytes = return_bytes

    def __len__(self) -> int:
        """
        返回 chunk 的数量。实现了 __len__ 后可以用 len(obj) 获取长度。

        例如：boundaries = [0, 100, 200, 300]
             有 3 个区间：[0, 100), [100, 200), [200, 300)
             所以 len = 3
        """
        # max(0, ...) 防止边界数小于 2 时出现负数
        return max(0, len(self.boundaries) - 1)

    def __iter__(self) -> Iterator[bytes] | Iterator[str]:
        """
        实现 __iter__ 让这个对象可以用 for 循环遍历。
        这是一个"生成器方法"（包含 yield 的函数就是生成器）。

        使用 mmap 再次映射文件，然后按 boundaries 定义的区间逐个产出 chunk：
          boundaries = [0, 100, 200, 300]
          第 1 次 yield: mm[0:100]   （第1个chunk）
          第 2 次 yield: mm[100:200] （第2个chunk）
          第 3 次 yield: mm[200:300] （第3个chunk）

        用法：
          iterator = FileChunkIterator("data.txt", [0, 100, 200])
          for chunk in iterator:
              process(chunk)
        """
        # 再次用 with + mmap 打开文件（每次迭代都重新映射，保证线程/进程安全）
        with open(self.file_path, "rb") as f, mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ) as mm:
            # 遍历每一对相邻的边界点
            for i in range(len(self.boundaries) - 1):
                start = self.boundaries[i]      # 当前 chunk 起始位置
                end = self.boundaries[i + 1]    # 当前 chunk 结束位置
                chunk = mm[start:end]            # 切片读取（不会复制数据，只是"视图"）

                if self.return_bytes:
                    yield chunk                 # 返回原始字节
                else:
                    yield chunk.decode("utf-8", errors="replace")  # 解码为字符串


def make_file_str_iter(file_path: str, boundaries: list[int], return_bytes: bool = True) -> FileChunkIterator:
    """
    工厂函数（Factory Function）：创建并返回 FileChunkIterator 实例。

    为什么需要一个工厂函数？
    - 给创建过程起一个更具描述性的名字
    - 如果以后需要换一种迭代器实现，只需修改这里
    - 让调用方代码更简洁

    参数和 FileChunkIterator.__init__ 完全一致。
    """
    return FileChunkIterator(file_path, boundaries, return_bytes)
