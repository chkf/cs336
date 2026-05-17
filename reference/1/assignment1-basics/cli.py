"""
Tokenizer 训练与评估命令行工具 (CLI)
==============================================
本文件提供了一个基于 Typer 框架的命令行工具，用于：
1. 训练 BPE (Byte Pair Encoding) 分词器
2. 比较两个已训练分词器的词汇表重叠情况
3. 检查分词器的压缩率
4. 使用分词器对语料进行编码
5. 学习率调优实验
6. 训练神经网络模型
7. 使用训练好的模型进行文本生成（解码）
"""

# ============================================================================
# 导入模块 (Imports)
# ============================================================================

import sys  # 系统相关功能，用于访问命令行参数、标准错误输出等
import timeit  # 性能计时工具，用于测量代码执行时间
from pathlib import Path  # 面向对象的文件路径处理库，比 os.path 更现代化
from pprint import pprint  # 美化打印 (pretty print)，用于打印复杂数据结构
from typing import Annotated  # 用于为类型添加额外元数据（如 Typer 的帮助信息）

import regex as re  # 更强大的正则表达式库，支持 Unicode 属性等高级特性
import torch  # PyTorch 深度学习框架
import typer  # 用于构建命令行应用的库，基于类型注解自动生成 CLI
from loguru import logger  # 现代化日志库，比标准 logging 更简洁易用

# 导入项目内部模块
from src.tokenization.tokenizer_trainer import TokenizerTrainer, TokenizerTrainerC
# TokenizerTrainer: BPE 分词器的纯 Python 训练器
# TokenizerTrainerC: 使用 C++ 加速的 BPE 训练器，性能更快

# ============================================================================
# 创建 Typer 应用实例
# ============================================================================

# Typer 是 FastAPI 作者开发的 CLI 框架，通过装饰器将函数转换为命令行子命令
app = typer.Typer(help="Tokenizer 训练与评估工具")

# ============================================================================
# 配置日志 (Logger Configuration)
# ============================================================================

# 移除默认的日志处理器，以便自定义配置
logger.remove()

# 添加一个新的日志处理器，输出到标准错误流 (stderr)
# 这样日志信息不会与程序的正常输出 (stdout) 混在一起
logger.add(
    sys.stderr,  # 输出目标：标准错误流
    # 自定义日志格式：
    # {time} - 时间戳，格式为 YYYY-MM-DD HH:mm:ss，绿色显示
    # {level} - 日志级别，左对齐8个字符宽度
    # {message} - 日志消息内容
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    level="INFO",  # 只显示 INFO 级别及以上的日志（DEBUG < INFO < WARNING < ERROR）
)


# ============================================================================
# 子命令 1: 训练分词器 (train_tokenizer)
# ============================================================================

@app.command()  # @app.command() 装饰器将该函数注册为 CLI 子命令
def train_tokenizer(
    # Annotated 用于同时指定类型和 Typer 参数信息
    # typer.Argument 表示这是位置参数（必须按顺序提供）
    corpos_path: Annotated[str, typer.Argument(help="corpos file path")],
    vocab_size: Annotated[int, typer.Argument(help="target vocabulary size")],
    # typer.Option 表示这是可选参数（通过 --xxx 提供）
    # list[str] | None 表示该参数可以是字符串列表或 None（可选）
    special_tokens: Annotated[list[str] | None, typer.Option(help="special tokens list")] = None,
    save_path: Annotated[str, typer.Option(help="save directory")] = "save",
    # --cpp/--py 表示可以通过 --cpp 或 --py 来设置该布尔值
    use_cpp: Annotated[bool, typer.Option("--cpp/--py", help="use C++ accelerated implementation")] = True,
):
    """
    训练 BPE 分词器并保存结果

    工作流程：
    1. 读取语料文件
    2. 使用 BPE 算法训练分词器，学习最优的词汇合并规则
    3. 将训练好的词汇表和合并规则保存到磁盘

    BPE (Byte Pair Encoding) 原理简介：
    - 从字符级别开始，统计所有相邻 token 对的出现频率
    - 每次选择出现频率最高的 token 对进行合并
    - 重复上述过程直到词汇表达到目标大小
    """
    # ---- 参数初始化 ----
    # 如果没有指定特殊 token，则使用默认的 <|endoftext|>
    # <|endoftext|> 是 GPT 系列模型常用的文档分隔 token
    if special_tokens is None:
        special_tokens = ["<|endoftext|>"]

    # ---- 语料文件检查 ----
    corpus_path_obj = Path(corpos_path)
    corpus_name = corpus_path_obj.stem  # .stem 获取文件名（不含扩展名），如 "corpus" from "corpus.txt"
    if not corpus_path_obj.exists():
        logger.error(f"Corpus file not found: {corpos_path}")
        raise typer.Exit(code=1)  # 以错误码 1 退出程序

    # ---- 准备保存目录 ----
    # 保存路径结构：save_path/corpus_name/
    save_dir = Path(save_path) / corpus_name
    save_dir.mkdir(parents=True, exist_ok=True)  # 递归创建目录，如果已存在不报错

    # ---- 配置文件日志 ----
    # 同时将日志写入文件，便于后续查看训练记录
    log_file = save_dir / "training.log"
    # rotation="10 MB": 当日志文件达到 10MB 时自动轮转（创建新文件）
    # retention="5 days": 保留最近 5 天的日志文件
    log_sink_id = logger.add(log_file, rotation="10 MB", retention="5 days", level="DEBUG")

    # ---- 选择训练器后端 ----
    # 根据 use_cpp 参数选择纯 Python 或 C++ 加速版本
    # C++ 版本使用 pybind11 绑定，可以显著加速训练过程
    TrainerClass = TokenizerTrainerC if use_cpp else TokenizerTrainer
    backend_name = TrainerClass.__name__  # __name__ 获取类的字符串名称

    # 记录训练开始信息
    logger.info(f"Starting training for [{corpus_name}] using [{backend_name}]")
    logger.info(f"Config: vocab_size={vocab_size}, special_tokens={special_tokens}")

    # ---- 执行训练 ----
    try:
        # 创建训练器实例并执行训练
        tokenizer = TrainerClass(corpos_path=corpos_path, vocab_size=vocab_size, special_tokens=special_tokens)
        # train() 返回两个结果：
        #   vocab: 词汇表 (dict)，从 token ID 到 token 字符串的映射
        #   merges: 合并规则 (dict)，记录 BPE 的合并顺序
        vocab, merges = tokenizer.train()
    except Exception:
        # logger.exception 会自动记录完整的异常堆栈信息
        logger.exception("Training failed due to an error")
        raise typer.Exit(code=1) from None  # from None 抑制异常链，只显示干净的错误

    # ---- 保存分词器 ----
    # 将词汇表和合并规则保存到指定目录
    tokenizer.save(save_dir)

    # ---- 输出训练统计信息 ----
    # 找出词汇表中最长的 token（通常可能是中文词语、URL 等）
    longest_token_length = max(len(token) for token in tokenizer.vocab.values())
    longest_tokens = [token for token in tokenizer.vocab.values() if len(token) == longest_token_length]

    logger.success(f"Tokenizer saved to: {save_dir}")
    logger.info(f"Vocab size: {len(vocab)}")  # 词汇表大小
    logger.info(f"Merges count: {len(merges)}")  # 合并规则数量
    logger.info(f"Longest token length: {longest_token_length} (Tokens: {longest_tokens!r})")
    # !r 表示使用 repr() 显示，可以看到特殊字符的转义形式

    # 关闭文件日志处理器
    logger.remove(log_sink_id)

    return vocab, merges


# ============================================================================
# 子命令 2: 比较分词器 (compare_tokenizer)
# ============================================================================

@app.command()
def compare_tokenizer(
    dir1: Annotated[str, typer.Argument(help="First Tokenizer directory")],
    dir2: Annotated[str, typer.Argument(help="Second Tokenizer directory")],
    show_details: Annotated[bool, typer.Option(help="Whether to show detailed overlap parts")] = False,
):
    """
    比较两个已训练分词器的词汇表重叠情况

    用途：评估不同训练配置（词汇量大小、语料来源等）对分词器的影响
    重叠率越高，说明两个分词器学习的 token 划分越相似
    """
    d1 = Path(dir1)
    d2 = Path(dir2)

    # ---- 查找词汇表文件 ----
    # 词汇表可能以 vocab.json 或 vocab.txt 格式保存
    # .json 格式包含更多元数据，.txt 格式更简单
    v1_path = d1 / "vocab.json" if (d1 / "vocab.json").exists() else d1 / "vocab.txt"
    v2_path = d2 / "vocab.json" if (d2 / "vocab.json").exists() else d2 / "vocab.txt"

    # ---- 检查所有必需文件是否存在 ----
    # 需要：两个词汇表文件 + 两个合并规则文件
    files_to_check = [v1_path, d1 / "merges.txt", v2_path, d2 / "merges.txt"]

    if not all(p.exists() for p in files_to_check):
        # 如果有任何文件缺失，记录错误并列出所有检查过的路径
        logger.error(f"Missing files in directories. Checked: {[str(p) for p in files_to_check]}")
        raise typer.Exit(code=1)

    # ---- 加载词汇表 ----
    # 定义一个内部辅助函数来加载和解析词汇表
    def load_vocab(path: Path):
        """从文件读取词汇表，使用 eval() 解析 Python 字典格式"""
        with open(path, encoding="utf-8") as f:
            return eval(f.read())  # eval 将字符串解析为 Python 字典，注意：仅用于受信任的文件

    vocab1_dict = load_vocab(v1_path)
    vocab2_dict = load_vocab(v2_path)

    # ---- 加载合并规则 ----
    # 合并规则文件每行一个规则，读取为集合以便快速求交集
    with open(d1 / "merges.txt", encoding="utf-8") as f:
        merges1_set = set(line.strip() for line in f)  # strip() 移除行首尾空白字符
    with open(d2 / "merges.txt", encoding="utf-8") as f:
        merges2_set = set(line.strip() for line in f)

    # ---- 提取词汇表的 token 集合（只看 keys，即 token ID）----
    vocab1_set = set(vocab1_dict.keys())
    vocab2_set = set(vocab2_dict.keys())

    # ---- 计算交集 ----
    # set.intersection() 求两个集合的共同元素
    common_vocab = vocab1_set.intersection(vocab2_set)
    common_merges = merges1_set.intersection(merges2_set)

    # ---- 输出比较报告 ----
    logger.info("--- Comparison Report ---")
    logger.info(f"Dir 1: {d1} | Dir 2: {d2}")

    # 计算重叠百分比
    # max(len(...), 1) 防止除零错误
    v_overlap = len(common_vocab) / max(len(vocab1_set), 1) * 100
    m_overlap = len(common_merges) / max(len(merges1_set), 1) * 100

    logger.info(f"Vocab overlap: {len(common_vocab)} tokens ({v_overlap:.2f}%)")
    # :.2f 表示保留两位小数的浮点数格式
    logger.info(f"Merges overlap: {len(common_merges)} merges ({m_overlap:.2f}%)")

    # 如果用户要求详细输出，显示共同 token 和合并规则的片段
    if show_details:
        logger.info(f"Common tokens snippet: {list(common_vocab)[:10]}...")  # 只显示前 10 个
        logger.info(f"Common merges snippet: {list(common_merges)[:10]}...")


# ============================================================================
# 子命令 3: 检查压缩率 (check_compression_ratio)
# ============================================================================

@app.command()
def check_compression_ratio(
    tokenizer_path: Annotated[str, typer.Argument(help="Path to the tokenizer directory")],
    corpos_path: Annotated[str, typer.Argument(help="Path to the corpus file")],
    num_samples: Annotated[int, typer.Argument(help="Number of samples to use for compression ratio check")] = 10,
    split_special_token: str = "<|endoftext|>",
):
    """
    检查分词器的压缩率是否符合预期

    压缩率 = 原始文本字节数 / 编码后的 token 数量

    压缩率越高，说明分词器对文本的"理解"越好：
    - 常见词/子词会被合并为更少的 token
    - 例如：英文文本的典型压缩率约为 3-5（即每 token 约等于 0.6-1 个单词）

    工作方式：
    1. 用 split_special_token 将语料切分为多个样本
    2. 对每个样本进行编码
    3. 比较原始字节数和编码 token 数
    """
    # 延迟导入 (lazy import)：只在需要时才导入，加快 CLI 启动速度
    from src.tokenization.tokenizer import Tokenizer

    tokenizer_dir = Path(tokenizer_path)

    # 查找词汇表文件（优先 .json 格式）
    vocab_file = (
        tokenizer_dir / "vocab.json" if (tokenizer_dir / "vocab.json").exists() else tokenizer_dir / "vocab.txt"
    )
    merges_file = tokenizer_dir / "merges.txt"

    # 验证分词器文件完整性
    if not vocab_file.exists() or not merges_file.exists():
        logger.error(f"Tokenizer files not found in: {tokenizer_path}")
        raise typer.Exit(code=1)

    # 从保存的文件中加载分词器
    tokenizer = Tokenizer.from_my_save(tokenizer_path, [split_special_token])

    total_original_size = 0  # 原始文本的总字节数
    total_encoded_size = 0   # 编码后的总 token 数

    # 编译正则表达式，用于按特殊 token 切分文本
    # re.escape() 确保特殊 token 中的特殊正则字符（如 |）被正确转义
    pattern = re.compile(re.escape(split_special_token))

    # ---- 读取并切分语料 ----
    with open(corpos_path, encoding="utf-8") as f:
        data = f.read()  # 读取整个文件内容
        # 用特殊 token 切分文本，取前 num_samples 个样本
        # 例如：如果 split_token 是 <|endoftext|>，则按文档边界切分
        samples = pattern.split(data)[:num_samples]
        del data  # 释放大文件占用的内存

    # ---- 执行编码并计时 ----
    t1 = timeit.default_timer()  # 记录开始时间
    all = []
    for sample in samples:
        # 将文本编码为 UTF-8 字节序列
        sample_bytes = sample.encode("utf-8")
        total_original_size += len(sample_bytes)  # 累加原始字节数
        # 使用分词器编码：将文本转换为 token ID 序列
        encoded = tokenizer.encode(sample)
        total_encoded_size += len(encoded)  # 累加 token 数量
        all.append((sample, encoded))

    # ---- 批量编码验证 ----
    # 对比单样本编码和批量编码的结果是否一致（用于验证批量编码的正确性）
    ret = tokenizer.encode_batch(samples)
    t2 = timeit.default_timer()  # 记录结束时间

    # 逐对比较单样本编码与批量编码的结果
    # strict=False 允许两个列表长度不同（不会报错）
    for sample, encoded in zip(samples, ret, strict=False):
        assert tokenizer.encode(sample) == encoded  # 断言单个编码和批量编码结果一致
        sample_bytes = sample.encode("utf-8")
        total_original_size += len(sample_bytes)
        total_encoded_size += len(encoded)

    # 输出编码性能信息
    logger.info(
        f"Encoding time for {num_samples} samples: {t2 - t1:.2f} seconds. "
        f"Throughput: {total_original_size / (t2 - t1):.2f} bytes/second"
    )

    # 边界检查：没有有效数据时退出
    if total_original_size == 0:
        logger.error("No valid data found in the corpus for compression ratio check.")
        raise typer.Exit(code=1)

    # ---- 计算压缩率 ----
    # 压缩率 = 原始字节数 / token 数量
    # 例如：1000 字节的文本被编码为 250 个 token → 压缩率 = 4.0
    compression_ratio = total_original_size / total_encoded_size
    logger.info(f"Compression ratio over {num_samples} samples: {compression_ratio:.2f}")


# ============================================================================
# 子命令 4: 编码文件 (encode_file)
# ============================================================================

@app.command()
def encode_file(
    tokenizer_path: Annotated[str, typer.Argument(help="Path to the tokenizer directory")],
    corpos_path: Annotated[str, typer.Argument(help="Path to the corpus file")],
    output_path: Annotated[str, typer.Argument(help="Path to save the encoded output")],
    split_special_token: str = "<|endoftext|>",
):
    """
    使用指定分词器对语料文件进行编码并保存结果

    编码：将原始文本转换为 token ID 序列（整数列表）
    这是训练神经网络之前的关键预处理步骤

    例如：
        输入文本: "Hello world"
        编码结果: [15496, 995]  (具体 ID 取决于分词器)

    输出文件可以直接被模型训练代码读取使用
    """
    # 延迟导入，只在需要时才加载 Tokenizer 类
    from src.tokenization.tokenizer import Tokenizer

    output = Path(output_path)
    # 确保输出目录存在（如果不存在则创建）
    if not output.parent.exists():
        output.parent.mkdir(parents=True, exist_ok=True)

    # 从保存的目录加载训练好的分词器
    tokenizer = Tokenizer.from_my_save(tokenizer_path, [split_special_token])

    # ---- 执行文件编码并计时 ----
    t1 = timeit.default_timer()
    # encode_file 会按 split_token 切分文件，逐块编码并保存
    tokenizer.encode_file(corpos_path, split_token=split_special_token, save_file=output_path)
    t2 = timeit.default_timer()

    # 获取输入文件大小，用于计算吞吐量
    size = Path(corpos_path).stat().st_size  # st_size 返回文件字节数
    logger.info(
        f"Encoding time for file {corpos_path}: {t2 - t1:.2f} seconds. "
        f"Throughput: {size / (t2 - t1):.2f} bytes/second"
    )
    # 吞吐量 (Throughput)：每秒处理的字节数，衡量编码性能

    # 注释掉的代码：原来可能是将结果保存为 numpy 二进制格式
    # out = np.array(encoded_ids, dtype=np.uint16)
    # out.tofile(output_path)

    logger.info(f"Encoded output saved to: {output}")


# ============================================================================
# 子命令 5: 学习率调优实验 (learning_rate_tuning)
# ============================================================================

@app.command()
def learning_rate_tuning():
    """
    学习率调优实验

    目的：演示不同学习率对 SGD (随机梯度下降) 优化过程的影响

    实验设置：
    - 创建一个随机初始化的 10x10 权重矩阵
    - 用三个不同的学习率 (10, 100, 1000) 分别进行优化
    - 观察损失函数的变化趋势

    学习率是深度学习中最关键的超参数之一：
    - 太小：收敛速度慢，需要更多训练步骤
    - 太大：可能导致不收敛，损失值震荡甚至发散
    - 适中：快速且稳定地收敛到最小值
    """
    from src.nn.optimizer import SGD  # SGD: Stochastic Gradient Descent 随机梯度下降

    # 创建一个 10x10 的随机权重矩阵（标准正态分布，乘以 5 放大）
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    # torch.nn.Parameter 告诉 PyTorch 这是一个可训练参数

    lr_list = [10, 100, 1000]  # 三个不同的学习率进行对比
    res = {}  # 存储每个学习率的损失历史

    for lr in lr_list:
        # 克隆原始权重，确保每个学习率从相同的初始状态开始
        # .clone().detach().requires_grad_(True) 的每一步：
        #   clone(): 创建张量的完整副本
        #   detach(): 从计算图中分离（不追踪梯度历史）
        #   requires_grad_(True): 重新启用梯度追踪
        weights_copy = weights.clone().detach().requires_grad_(True)

        # 创建 SGD 优化器，传入参数和学习率
        opt = SGD([weights_copy], lr=lr)

        res[lr] = []  # 初始化该学习率的损失记录列表

        for _ in range(10):  # 运行 10 个优化步骤
            opt.zero_grad()  # 梯度清零：每次迭代前必须清零，否则梯度会累积

            # 计算损失：权重的均方值 (L2 正则化的效果)
            # weights_copy**2: 逐元素平方
            # .mean(): 取平均值 → 得到一个标量损失值
            loss = (weights_copy**2).mean()

            print(loss.cpu().item())  # .item() 将单元素张量转为 Python 数字并打印

            loss.backward()  # 反向传播：计算损失对每个参数的梯度
            opt.step()  # 优化器更新参数：weights -= lr * gradient
            res[lr].append(loss.cpu().item())  # 记录当前损失值

    # 打印最终结果，便于比较不同学习率的效果
    pprint("learning_rate_tuning on SGD:")
    pprint(res)


# ============================================================================
# 子命令 6: 训练模型 (train_model)
# ============================================================================

@app.command()
def train_model(config: str):
    """
    训练神经网络语言模型

    参数：
        config: 配置文件路径（通常是 YAML 或 JSON 格式），包含：
            - 模型架构参数（层数、隐藏层维度、注意力头数等）
            - 训练超参数（学习率、批次大小、训练轮数等）
            - 数据路径配置
            - 日志和检查点保存路径

    这个命令是启动完整模型训练流程的入口
    """
    from src.nn import train  # 导入训练模块
    train.train(config)  # 调用训练函数，传入配置文件路径


# ============================================================================
# 子命令 7: 文本生成（解码）(decode)
# ============================================================================

@app.command()
def decode(
    model_path: Annotated[str, typer.Argument(help="Path to the trained model directory")],
    tokenizer_path: Annotated[str, typer.Argument(help="Path to the tokenizer directory")],
    prompt: Annotated[str, typer.Argument(help="Prompt text to start generation")],
    max_length: Annotated[int, typer.Argument(help="Maximum length of generated text")] = 100,
    temperature: Annotated[float, typer.Option(help="Sampling temperature")] = 1.0,
    p: Annotated[float, typer.Option(help="Nucleus sampling probability threshold")] = 0.9,
    checkpoint_number: Annotated[int | None, typer.Option(help="Checkpoint number to load")] = None,
):
    """
    使用训练好的语言模型进行文本生成

    文本生成过程（自回归生成）：
    1. 将 prompt（提示文本）编码为 token 序列
    2. 将 token 序列输入模型，预测下一个 token 的概率分布
    3. 从概率分布中采样一个 token
    4. 将新 token 添加到序列末尾
    5. 重复步骤 2-4 直到达到最大长度或遇到终止条件

    采样参数说明：
    - temperature (温度): 控制生成的随机性
        * < 1.0: 更确定的输出（倾向于选择高概率 token）
        * = 1.0: 标准采样
        * > 1.0: 更多样化的输出（概率分布更均匀）
    - p (nucleus sampling / top-p): 只从累积概率达到 p 的最小 token 集合中采样
        * 例如 p=0.9: 选择最可能的前若干 token，使它们概率之和 ≥ 0.9
        * 可以有效避免低概率的"噪音" token
    """
    from src.nn.decode import decode  # 导入解码（文本生成）函数

    generated_text = decode(
        prompt=prompt,  # 用户输入的提示文本
        model_path=model_path,  # 训练好的模型保存路径
        tokenizer_path=tokenizer_path,  # 分词器保存路径
        max_length=max_length,  # 生成文本的最大长度（token 数）
        temperature=temperature,  # 采样温度
        p=p,  # 核采样阈值
        checkpoint_number=checkpoint_number,  # 要加载的检查点编号（None 表示最新）
    )
    print("Generated Text:", generated_text)


# ============================================================================
# 程序入口点 (Main Entry Point)
# ============================================================================

if __name__ == "__main__":
    # 当直接运行此脚本时（而非作为模块导入），启动 Typer 应用
    # Typer 会自动解析命令行参数并调用相应的子命令函数
    #
    # 使用示例：
    #   python cli.py train-tokenizer corpus.txt 1000
    #   python cli.py compare-tokenizer save/corpus1 save/corpus2
    #   python cli.py check-compression-ratio save/corpus1 corpus.txt 20
    #   python cli.py encode-file save/corpus1 corpus.txt encoded.bin
    #   python cli.py learning-rate-tuning
    #   python cli.py train-model config.yaml
    #   python cli.py decode model_dir/ tokenizer_dir/ "Once upon a time" 200 --temperature 0.8
    app()