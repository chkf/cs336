"""
Fused Attention (融合注意力机制)
========================================================
这是 Flash Attention v2 算法的 Triton 实现
原始论文作者: Tri Dao (https://tridao.me/publications/flash2/flash2.pdf)

致谢: OpenAI kernel 团队

额外参考资料:
* 原始 Flash Attention 论文 (https://arxiv.org/abs/2205.14135)
* Rabe and Staats (https://arxiv.org/pdf/2112.05682v2.pdf)

==== 什么是 Flash Attention？====
传统 self-attention 需要 O(N^2) 的显存来存储完整的注意力矩阵，
而 Flash Attention 通过"分块计算"（tiling），把 Q、K、V 切分成小块，
每次只加载一小块到 GPU 的 SRAM（片上高速缓存）中计算，
从而将显存开销降低到 O(N)，同时因为减少了 HBM（显存）读写次数，速度也更快。

版本演进：
- Flash Attention v1: 提出了分块计算的思想
- Flash Attention v2: 优化了循环顺序（外层遍历 Q 块，内层遍历 K/V 块），
  减少了 non-matmul 操作，进一步提升了效率

==== 什么是 Triton？====
Triton 是 OpenAI 开发的 GPU 编程语言/编译器。
写 CUDA kernel 非常复杂（要手动管理线程、共享内存等），
Triton 让你用类似 Python 的语法写 GPU kernel，编译器自动优化。
@triton.jit 装饰器标记的函数会在 GPU 上执行。
"""

import pytest
import torch
import os

import triton
import triton.language as tl  # tl 是 Triton 的"语言"模块，提供 GPU 上的数学运算、内存操作等
from triton.tools.tensor_descriptor import TensorDescriptor  # 张量描述符，用于描述 GPU 上的张量布局

# ------------------------------------------------------------
# 自动检测当前 GPU 类型（NVIDIA CUDA 或 AMD HIP）
# ------------------------------------------------------------
# triton.runtime.driver.active.get_active_torch_device() 返回当前活跃的 PyTorch 设备对象
# 比如 cuda:0 或 hip:0
DEVICE = triton.runtime.driver.active.get_active_torch_device()


def is_hip():
    """判断当前是否是 AMD GPU（使用 HIP 后端）"""
    return triton.runtime.driver.active.get_current_target().backend == "hip"


def is_cuda():
    """判断当前是否是 NVIDIA GPU（使用 CUDA 后端）"""
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def supports_host_descriptor():
    """
    判断是否支持 host-side tensor descriptor（宿主端张量描述符）
    需要 CUDA 且计算能力 >= 9.0（即 Hopper 架构，如 H100）
    
    张量描述符（TensorDescriptor）是什么？
    - 它是一种在 CPU 端描述 GPU 张量形状/步幅的数据结构
    - 使用它可以让 Triton 编译器更好地优化内存访问模式
    - 只在较新的 GPU（Hopper 及以上）上支持
    """
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 9


def is_blackwell():
    """
    判断是否是 Blackwell 架构 GPU（计算能力 10.x，如 B200）
    Blackwell 是 NVIDIA 最新一代（2024年发布）的 GPU 架构
    """
    return is_cuda() and torch.cuda.get_device_capability()[0] == 10


def is_hopper():
    """
    判断是否是 Hopper 架构 GPU（计算能力 9.x，如 H100）
    Hopper 是 NVIDIA 2022年发布的 GPU 架构，首次支持 FP8 等特性
    """
    return is_cuda() and torch.cuda.get_device_capability()[0] == 9


# ================================================================
#                       前向传播（Forward Pass）
# ================================================================

@triton.jit  # 这个装饰器告诉 Triton 编译器：此函数是 GPU kernel，请编译为 GPU 机器码
def _attn_fwd_inner(
    # ---- 累加器和状态 ----
    acc,       # 输出累加器 [BLOCK_M, HEAD_DIM]，存储加权后的 V 的累加和
    l_i,       # 分母累加器 [BLOCK_M]，存储 softmax 分母的累加值（每个 row 的 exp sum）
    m_i,       # 当前最大值 [BLOCK_M]，用于 online softmax 的数值稳定技巧
    
    # ---- 查询向量 ----
    q,         # 查询 Q 的当前块 [BLOCK_M, HEAD_DIM]
    
    # ---- K 和 V 的张量描述符（Tensor Descriptor）----
    # 张量描述符是一种特殊的数据结构，它告诉 GPU：这个张量在显存中的形状、步幅、以及我们希望每次加载的块大小
    # 使用描述符可以更高效地加载数据（硬件自动处理地址计算）
    desc_k,    # K 矩阵的描述符
    desc_v,    # V 矩阵的描述符
    
    # ---- 偏移量和配置 ----
    offset_y,  # 当前 batch/head 在 K/V 中的起始偏移（因为 K/V 按 (batch*head, seq_len, head_dim) 展平存储）
    dtype,     # 计算使用的数据类型（fp16 或 fp8），由外部作为编译时常量传入（tl.constexpr）
    start_m,   # 当前 Q 块在所有 Q 块中的起始索引（第几个 BLOCK_M）
    qk_scale,  # QK 点积的缩放因子 = sm_scale * 1.44269504（1/ln2，用于以2为底的exp）
    
    # ---- 编译时常量（tl.constexpr）：这些值在编译时就确定，不能运行时改变 ----
    BLOCK_M: tl.constexpr,   # Q 的分块大小（一次处理多少行 Q）
    HEAD_DIM: tl.constexpr,  # 注意力头的维度（Q、K、V 每个 token 的向量维度）
    BLOCK_N: tl.constexpr,   # K/V 的分块大小（一次处理多少列 K/V）
    
    # ---- 阶段控制 ----
    STAGE: tl.constexpr,     # 当前处于哪个阶段（1=前段, 2=中间/对角线, 3=全部/causal=False时）
    
    # ---- 偏移量数组 ----
    offs_m: tl.constexpr,    # Q 块内 token 的局部偏移 [BLOCK_M]，例如 [0, 1, 2, ..., BLOCK_M-1]
    offs_n: tl.constexpr,    # K/V 块内 token 的局部偏移 [BLOCK_N]，例如 [0, 1, 2, ..., BLOCK_N-1]
    
    # ---- 上下文大小和硬件特性 ----
    N_CTX: tl.constexpr,     # 序列总长度（Context Length）
    warp_specialize: tl.constexpr,  # 是否启用 warp specialization（硬件优化技术）
    IS_HOPPER: tl.constexpr,        # 是否是 Hopper 架构 GPU
):
    """
    前向传播的内层循环 — 对 K 和 V 进行遍历，计算局部 softmax 并累加到 acc
    
    === Flash Attention 的核心思想（分块 Online Softmax）====
    
    传统的 softmax 需要三遍扫描：
      1. 找最大值 m = max(x)
      2. 计算 exp(x - m) 并求和 s = sum(exp(x - m))
      3. 归一化 y = exp(x - m) / s
    
    Online Softmax 可以在一次扫描中完成：
      - 维护当前最大值 m_i 和累加和 l_i
      - 每遇到一个新块，用"修正因子" alpha = exp2(m_old - m_new) 来修正之前的累加值
      - 最后一次性归一化
    
    这里的算法：
      对 K/V 按 BLOCK_N 分块遍历：
        1. 加载 K 块，计算 Q @ K^T 得到注意力分数 qk
        2. 更新局部最大值 m_ij = max(m_i, max(qk))
        3. 计算 exp(qk - m_ij)，即 softmax 的分子
        4. 计算修正因子 alpha = exp2(m_i - m_ij)（用旧的 m_i 和新的 m_ij）
        5. 用 alpha 修正之前的累加器 acc = acc * alpha
        6. 加载 V 块，计算新的加权和 p @ V 并累加到 acc
        7. 更新 l_i = l_i * alpha + sum(exp(qk))，m_i = m_ij
    """
    
    # ---- 根据 STAGE 确定 K/V 的遍历范围 ----
    # Flash Attention v2 将计算分为几个"阶段"（stage），用于支持 causal masking
    # 
    # Causal attention 的概念：
    #   在自回归语言模型中，第 i 个 token 只能"看到"第 0 到第 i 个 token（不能看到未来）
    #   这对应一个下三角矩阵的 mask
    #
    # 分阶段的原因：
    #   对于 causal attention，Q 块对应的 K/V 范围分为两段：
    #     - 前段（off-band）：第 0 到第 start_m*BLOCK_M 个 token（无条件可见）
    #     - 中间（on-band）：第 start_m*BLOCK_M 到第 (start_m+1)*BLOCK_M 个 token（需要 mask，对角线区域）
    #   非 causal attention 只有一个阶段：遍历全部 K/V
    if STAGE == 1:
        # 阶段1：处理 Q 块之前的所有 K/V（前段，off-band）
        # 这些 token 对当前 Q 块完全可见，不需要 mask
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        # 阶段2：处理 Q 块自身对应的 K/V（中间，on-band，即对角线区域）
        # 这个区域需要 causal mask：Q 的第 i 个 token 只能看到 K 的第 j 个 token，其中 j <= i
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        # tl.multiple_of(x, BLOCK_M) 告诉编译器 lo 是 BLOCK_M 的倍数，便于优化
        lo = tl.multiple_of(lo, BLOCK_M)
    else:
        # 阶段3（causal=False）：处理全部 K/V，从 0 到 N_CTX
        # 不需要任何 mask
        lo, hi = 0, N_CTX
    
    # ---- 计算 K 的起始偏移地址 ----
    # offset_y 是当前 batch/head 在展平的 K/V 中的起始位置
    # offsetk_y = offset_y + lo：跳过前面阶段已经处理过的部分
    offsetk_y = offset_y + lo
    
    # ---- 计算 V 的起始偏移地址 ----
    # FP8 数据类型特殊处理：V 是转置存储的（形状是 [HEAD_DIM, seq_len] 而不是 [seq_len, HEAD_DIM]）
    # 所以偏移量不同
    if dtype == tl.float8e5:
        # FP8 模式下，V 的维度是 [HEAD_DIM, y_dim]（转置了）
        # 偏移计算不同：offset_y * HEAD_DIM（因为在展平的 2D 空间中，每个 token 占 HEAD_DIM 个元素）
        offsetv_y = offset_y * HEAD_DIM + lo
    else:
        # 标准模式：V 维度是 [y_dim, HEAD_DIM]
        offsetv_y = offset_y + lo
    
    # ============================================================
    #  主循环：按 BLOCK_N 步长遍历 K/V 序列
    # ============================================================
    # tl.range(lo, hi, BLOCK_N) 在 GPU 上生成一个从 lo 到 hi、步长为 BLOCK_N 的循环
    # warp_specialize 是 GPU 优化参数，允许不同 warp 执行不同任务
    for start_n in tl.range(lo, hi, BLOCK_N, warp_specialize=warp_specialize):
        # 告诉编译器 start_n 是 BLOCK_N 的倍数（便于地址对齐优化）
        start_n = tl.multiple_of(start_n, BLOCK_N)
        
        # ---------- 步骤 1: 计算 QK^T（注意力分数）----------
        # 从描述符加载 K 的一个块 [BLOCK_N, HEAD_DIM]
        # desc_k.load([行偏移, 列偏移])，.T 表示转置
        # k 的形状：[BLOCK_N, HEAD_DIM] → 转置后为 [HEAD_DIM, BLOCK_N]
        k = desc_k.load([offsetk_y, 0]).T
        
        # tl.dot(q, k): 矩阵乘法
        # q 的形状：[BLOCK_M, HEAD_DIM]
        # k 的形状（转置后）：[HEAD_DIM, BLOCK_N]
        # qk 的结果形状：[BLOCK_M, BLOCK_N]
        # 这就是注意力分数矩阵的一个小块
        qk = tl.dot(q, k)
        
        # ---------- 步骤 2: 应用缩放和 causal mask（如果是阶段2）----------
        if STAGE == 2:
            # ---- Causal Mask（因果遮罩）----
            # 对于 阶段2（对角线区域），需要确保 Q 的第 i 个 token 只能看到 K 的第 j 个 token（j <= i）
            # 
            # offs_m[:, None] 形状是 [BLOCK_M, 1]，包含 Q 块内的全局索引
            # offs_n[None, :] 形状是 [1, BLOCK_N]，包含 K 块内的全局索引
            # start_n + offs_n 得到 K 块内 token 的全局序列位置
            # 
            # 比较 Q 索引 >= K 索引，得到下三角 mask
            # 例如 BLOCK_M=4, BLOCK_N=4, start_m*BLOCK_M=4, start_n=4：
            #   Q 索引: [4, 5, 6, 7]
            #   K 索引: [4, 5, 6, 7]
            #   mask:
            #     [[T, F, F, F],   # Q[4] 只能看 K[4]
            #      [T, T, F, F],   # Q[5] 只能看 K[4,5]
            #      [T, T, T, F],   # Q[6] 只能看 K[4,5,6]
            #      [T, T, T, T]]   # Q[7] 可以看 K[4,5,6,7]
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            
            # 应用 mask：
            # - 可见位置：qk * qk_scale（正常缩放）
            # - 不可见位置：设置 -1.0e6（一个极大的负数，经过 exp 后趋近于 0）
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
            
            # 计算当前块的最大值 m_ij
            # tl.max(qk, 1): 沿着 axis=1（K 维度）取最大值，得到 [BLOCK_M]
            # tl.maximum(m_i, ...): 和之前的最大值比较，更新
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            
            # 用新的最大值对 qk 做数值稳定处理
            qk -= m_ij[:, None]  # m_ij[:, None] 形状 [BLOCK_M, 1]，广播到 [BLOCK_M, BLOCK_N]
        else:
            # 阶段1 或 阶段3：不需要 mask
            # 先更新最大值 m_ij（这里的写法是先对 qk 缩放再取 max，等价于 max(qk) * scale）
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            # 缩放并减去最大值（online softmax 的数值稳定技巧）
            qk = qk * qk_scale - m_ij[:, None]
        
        # ---------- 步骤 3: 计算 softmax 分子 p = exp(qk) ----------
        # tl.math.exp2 是以 2 为底的指数函数（等价于 2^x）
        # 为什么用 exp2 而不是 exp？因为前面 qk_scale 已经乘了 1/ln(2)，
        # 这样 2^(qk_scaled) = e^(qk * sm_scale)，结果等价但计算更快（硬件对 exp2 有优化）
        # 
        # p 的形状：[BLOCK_M, BLOCK_N]，每行代表一个 query token 对 K 块内各 token 的注意力权重
        p = tl.math.exp2(qk)
        
        # ---------- 步骤 4: 计算修正因子 alpha ----------
        # 如果当前块的最大值 m_ij 比之前的最大值 m_i 更大，
        # 那么之前累加的 acc 和 l_i 需要被"缩小"（因为 exp 的值域变了）
        # 
        # alpha = exp2(m_i - m_ij):
        #   - 如果 m_ij > m_i（发现了更大的值），alpha < 1，之前的累加值被缩小
        #   - 如果 m_ij == m_i（没有发现更大的值），alpha = 1，累加值不变
        # 这是一个"修正因子"（correction factor）
        alpha = tl.math.exp2(m_i - m_ij)
        
        # 当前块的 softmax 分母（局部的 exp 和）
        l_ij = tl.sum(p, 1)  # 沿 axis=1 求和，得到 [BLOCK_M]
        
        # ---------- 步骤 5: 用修正因子更新之前的累加器 ----------
        # 这是 Flash Attention 的关键技巧：
        #   acc 存储的是 sum(softmax_score * V)，当我们发现新的最大值时，
        #   需要用 alpha 修正旧的累加值
        #
        # 特定的硬件优化分支（Hopper+ 的 warp_specialize 模式下 BLOCK_M=128, HEAD_DIM=128）：
        #   - 把 acc 沿 HEAD_DIM 维度拆成两半，分别乘 alpha
        #   - 这样可以利用 GPU 的硬件特性加速
        if not IS_HOPPER and warp_specialize and BLOCK_M == 128 and HEAD_DIM == 128:
            # acc 形状 [128, 128]
            BM: tl.constexpr = acc.shape[0]   # 128
            BN: tl.constexpr = acc.shape[1]   # 128
            # reshape 为 [128, 2, 64] → permute 为 [128, 64, 2] → split 为两个 [128, 64]
            acc0, acc1 = acc.reshape([BM, 2, BN // 2]).permute(0, 2, 1).split()
            # 分别乘 alpha
            acc0 = acc0 * alpha[:, None]
            acc1 = acc1 * alpha[:, None]
            # join → permute → reshape 还原为 [128, 128]
            acc = tl.join(acc0, acc1).permute(0, 2, 1).reshape([BM, BN])
        else:
            # 通用情况：直接广播乘法
            # alpha[:, None] 形状 [BLOCK_M, 1]，广播到 [BLOCK_M, HEAD_DIM]
            acc = acc * alpha[:, None]
        
        # ---------- 步骤 6: 加载 V 并计算加权和 ----------
        # 加载 V 的一个块
        # FP8 模式下 V 是转置存储的（形状 [HEAD_DIM, seq_len]），加载方式不同
        if dtype == tl.float8e5:
            # FP8: 加载 [HEAD_DIM, BLOCK_N]
            v = desc_v.load([0, offsetv_y]).T  # 加载后转置为 [BLOCK_N, HEAD_DIM]
        else:
            # 标准: 加载 [BLOCK_N, HEAD_DIM]
            v = desc_v.load([offsetv_y, 0])
        
        # 将 p 转换为计算所需的数据类型（fp16 或 fp8）
        # 注意：p 目前是 fp32（因为 exp2 返回 fp32），需要转成和 V 一致的类型做矩阵乘法
        p = p.to(dtype)
        
        # tl.dot(p, v, acc): acc = p @ v + acc（融合的矩阵乘加）
        # p: [BLOCK_M, BLOCK_N]，v: [BLOCK_N, HEAD_DIM]
        # 结果 [BLOCK_M, HEAD_DIM] 直接加到 acc 上
        # 这是 Triton 的"融合"操作 —— 矩阵乘法和加法在一个 kernel 中完成，不需要额外的内存读写
        acc = tl.dot(p, v, acc)
        
        # ---------- 步骤 7: 更新状态 m_i 和 l_i ----------
        # l_i 更新：旧的 l_i * alpha（修正）+ 新的 l_ij（当前块的 exp 和）
        l_i = l_i * alpha + l_ij
        # m_i 更新为当前最大值
        m_i = m_ij
        
        # ---------- 步骤 8: 移动偏移指针到下一个 K/V 块 ----------
        offsetk_y += BLOCK_N
        offsetv_y += BLOCK_N
    
    # 返回更新后的累加器、分母、最大值
    # 在外层函数 _attn_fwd 中会进行最终归一化：acc / l_i
    return acc, l_i, m_i


def _host_descriptor_pre_hook(nargs):
    """
    宿主端描述符预处理钩子（pre-hook）
    
    这个函数在 kernel 启动前被调用，用于设置 TensorDescriptor 的 block_shape。
    
    什么是 TensorDescriptor？
    - 它是描述 GPU 上张量的一种元数据结构
    - 包含：数据指针、形状(shape)、步幅(strides)、块形状(block_shape)
    - block_shape 告诉硬件："我计划每次加载这么大的块"，让硬件优化内存访问
    
    参数 nargs 是传递给 kernel 的命名参数字典，这个 hook 会修改其中的描述符。
    """
    BLOCK_M = nargs["BLOCK_M"]
    BLOCK_N = nargs["BLOCK_N"]
    HEAD_DIM = nargs["HEAD_DIM"]
    
    # 只处理真正的 TensorDescriptor 对象，如果是普通 tensor 则跳过
    if not isinstance(nargs["desc_q"], TensorDescriptor):
        return
    
    # 设置各张量的 block_shape：
    # - Q: 每次加载 [BLOCK_M, HEAD_DIM] 的块
    nargs["desc_q"].block_shape = [BLOCK_M, HEAD_DIM]
    
    # - V: FP8 模式下 V 是转置存储的 [HEAD_DIM, N]，所以 block_shape 也是转置的
    if nargs["FP8_OUTPUT"]:
        nargs["desc_v"].block_shape = [HEAD_DIM, BLOCK_N]
    else:
        nargs["desc_v"].block_shape = [BLOCK_N, HEAD_DIM]
    
    # - K 和 O: 和 Q 类似的布局
    nargs["desc_k"].block_shape = [BLOCK_N, HEAD_DIM]
    nargs["desc_o"].block_shape = [BLOCK_M, HEAD_DIM]


# ------------------------------------------------------------
# 自动调优配置（Autotuning Configs）
# ------------------------------------------------------------
# Triton 的 autotune 功能会自动尝试不同的配置组合，找到最快的那个
# 这些配置定义了不同的 BLOCK_M、BLOCK_N、流水线阶段数（num_stages）、warp 数量（num_warps）

# num_stages: GPU 流水线的阶段数
# - 更多的阶段可以隐藏内存延迟，但占用更多寄存器/SRAM
# - AMD GPU（HIP）只支持 1 个阶段
if is_hip():
    NUM_STAGES_OPTIONS = [1]
elif supports_host_descriptor():
    # 支持 host descriptor 的 GPU 可能需要更少的 stage（因为描述符已经优化了内存访问）
    NUM_STAGES_OPTIONS = [2, 3, 4]
else:
    NUM_STAGES_OPTIONS = [2, 3, 4]

# 生成所有配置组合的列表
configs = [
    triton.Config(
        {'BLOCK_M': BM, 'BLOCK_N': BN},  # kernel 参数
        num_stages=s,                      # 流水线阶段数
        num_warps=w,                       # warp 数量（每个 warp 有 32 个线程）
        pre_hook=_host_descriptor_pre_hook # 启动前的回调函数
    )
    for BM in [64, 128]       # Q 块大小：64 或 128
    for BN in [32, 64, 128]   # K/V 块大小：32、64 或 128
    for s in NUM_STAGES_OPTIONS
    for w in [4, 8]           # warp 数量：4 或 8（4*32=128 线程 或 8*32=256 线程）
]

if "PYTEST_VERSION" in os.environ:
    # 在测试环境中，只使用一个固定配置以保证可重复性
    configs = [
        triton.Config(dict(BLOCK_M=128, BLOCK_N=64), num_stages=2, num_warps=4, pre_hook=_host_descriptor_pre_hook),
    ]


def keep(conf):
    """
    配置过滤器：决定哪些 autotune 配置值得保留
    
    排除规则：在 Hopper GPU 上，如果 BLOCK_M * BLOCK_N < 128 * 128 且用了 8 个 warp，
    这种配置效率不高（因为块太小，warp 太多，调度开销大于收益）
    """
    BLOCK_M = conf.kwargs["BLOCK_M"]
    BLOCK_N = conf.kwargs["BLOCK_N"]
    return not (is_cuda() and torch.cuda.get_device_capability()[0] == 9 and BLOCK_M * BLOCK_N < 128 * 128
                and conf.num_warps == 8)


def prune_invalid_configs(configs, named_args, **kwargs):
    """
    提前剪枝函数：在 autotune 开始前过滤掉明显不合适的配置
    
    规则：
    1. BLOCK_M > N_CTX：Q 块大小不能超过序列总长度（否则只有不到一个块，浪费资源）
    2. 对于 causal attention（STAGE != 1）：BLOCK_M >= BLOCK_N（因为 causal 要求高效的对角线遍历）
    """
    N_CTX = kwargs["N_CTX"]
    STAGE = kwargs["STAGE"]

    return [
        conf for conf in configs
        if conf.kwargs.get("BLOCK_M", 0) <= N_CTX  # Q 块不能比序列长
        and (
            conf.kwargs.get("BLOCK_M", 0) >= conf.kwargs.get("BLOCK_N", 0)  # 对于 causal，Q块>=K块
            or STAGE == 1
        )
    ]


@triton.jit
def _maybe_make_tensor_desc(desc_or_ptr, shape, strides, block_shape):
    """
    辅助函数：如果传入的是普通指针，就创建一个张量描述符；如果已经是描述符，直接返回
    
    为什么要这样做？
    - 在支持 host descriptor 的 GPU 上，描述符在 CPU 端创建好传入
    - 在不支持的 GPU 上，传入的是普通 tensor，需要在 GPU kernel 内部创建描述符
    
    参数：
      desc_or_ptr: 可能是 TensorDescriptor 或普通 tensor 指针
      shape: 张量形状
      strides: 各维度的步幅（stride）
      block_shape: 期望的加载块大小
    """
    if isinstance(desc_or_ptr, tl.tensor_descriptor):
        # 已经是描述符，直接返回
        return desc_or_ptr
    else:
        # 是普通指针，在 GPU 上创建描述符
        return tl.make_tensor_descriptor(desc_or_ptr, shape, strides, block_shape)


# ------------------------------------------------------------
# 前向传播主函数
# ------------------------------------------------------------
# @triton.autotune 装饰器：
#   - configs: 要尝试的配置列表
#   - key: 决定使用哪个配置的关键参数（不同的 N_CTX, HEAD_DIM 等适用不同配置）
#   - prune_configs_by: 剪枝函数，提前过滤不合适的配置
@triton.autotune(
    configs=list(filter(keep, configs)),  # 先用 keep() 过滤掉无效配置
    key=["N_CTX", "HEAD_DIM", "FP8_OUTPUT", "warp_specialize"],
    prune_configs_by={'early_config_prune': prune_invalid_configs}
)
@triton.jit
def _attn_fwd(
    # ---- 运行时参数（在每次 kernel 调用时传入）----
    sm_scale,    # softmax 的缩放因子，通常是 1/sqrt(HEAD_DIM)
    M,           # 输出：log-sum-exp 值，用于反向传播（softmax_lse）
    Z,           # batch size（批次大小，一次处理多少个独立序列）
    H,           # 注意力头数量（num_heads）
    desc_q,      # Q 张量（或描述符）
    desc_k,      # K 张量（或描述符）
    desc_v,      # V 张量（或描述符）
    desc_o,      # Output（输出）张量（或描述符）
    N_CTX,       # 序列长度（context length）
    
    # ---- 编译时常量（在编译 kernel 时确定，不能在运行时改变）----
    HEAD_DIM: tl.constexpr,        # 每个注意力头的维度
    BLOCK_M: tl.constexpr,         # Q 的分块大小
    BLOCK_N: tl.constexpr,         # K/V 的分块大小
    FP8_OUTPUT: tl.constexpr,      # 是否使用 FP8 数据类型
    STAGE: tl.constexpr,           # 阶段配置（1=非causal, 3=causal）
    warp_specialize: tl.constexpr, # 是否启用 warp specialization
    IS_HOPPER: tl.constexpr,       # 是否是 Hopper GPU
):
    """
    前向传播的主 kernel 函数
    
    === 调用关系的层次结构 ===
    
    _attn_fwd (本函数)          ← 最外层，每个 GPU block 执行一次
      ├── 加载 Q 块到 SRAM
      ├── 如果有 STAGE 1：调用 _attn_fwd_inner(阶段1)  ← 处理前段 K/V
      ├── 如果有 STAGE 2：调用 _attn_fwd_inner(阶段2)  ← 处理对角线 K/V（causal mask）
      └── 最终归一化：acc /= l_i，写入输出
    
    === GPU 网格布局 ===
    一个 kernel 启动时会在 GPU 上创建很多"线程块"（thread blocks / CTAs）：
      - 第0维（program_id(0)）：不同的 Q 块（start_m = 0, 1, 2, ...）
      - 第1维（program_id(1)）：不同的 batch+head 组合（off_hz）
    这样每个线程块处理一个特定的 (batch, head, Q_block) 三元组
    """
    
    # 确定计算精度：FP8 或 FP16
    dtype = tl.float8e5 if FP8_OUTPUT else tl.float16
    
    # 断言 K 的分块大小不能超过注意力头维度
    # 因为 K 的维度是 [BLOCK_N, HEAD_DIM]，每个分块加载 BLOCK_N 行、HEAD_DIM 列
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    
    # ---- 解析 GPU 线程块索引 ----
    # program_id(0): 当前线程块在 Q 序列维度上的索引（第几个 Q 块）
    start_m = tl.program_id(0)
    # program_id(1): 当前线程块在 batch+head 维度上的索引
    off_hz = tl.program_id(1)
    
    # 从展平的 batch*head 索引中恢复原始的 batch 和 head 索引
    # 例如 Z=2, H=4, off_hz=5 → off_z=1 (第2个batch), off_h=1 (第2个头)
    off_z = off_hz // H  # 整除：batch 索引
    off_h = off_hz % H   # 取余：head 索引
    
    # ---- 构建张量描述符 ----
    # 总序列维度（展平了 batch 和 head）：
    # y_dim = batch_size * num_heads * seq_len
    # 这样把 4D 张量 [Z, H, N_CTX, HEAD_DIM] 展平为 2D [y_dim, HEAD_DIM]
    y_dim = Z * H * N_CTX
    
    # Q 描述符：形状 [y_dim, HEAD_DIM]，步幅 [HEAD_DIM, 1]（行优先存储）
    # block_shape 告诉硬件每次加载 [BLOCK_M, HEAD_DIM] 大小的块
    desc_q = _maybe_make_tensor_desc(
        desc_q,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_M, HEAD_DIM]
    )
    
    # V 描述符（FP8 模式特殊：V 是转置存储的）
    if FP8_OUTPUT:
        # FP8: V 存储为 [HEAD_DIM, y_dim]，步幅 [N_CTX, 1]
        # 为什么这样存？因为 FP8 的矩阵乘法要求第一个操作数是转置的
        desc_v = _maybe_make_tensor_desc(
            desc_v,
            shape=[HEAD_DIM, y_dim],
            strides=[N_CTX, 1],
            block_shape=[HEAD_DIM, BLOCK_N]
        )
    else:
        # 标准模式：V 存储为 [y_dim, HEAD_DIM]，和 Q、K 一样
        desc_v = _maybe_make_tensor_desc(
            desc_v,
            shape=[y_dim, HEAD_DIM],
            strides=[HEAD_DIM, 1],
            block_shape=[BLOCK_N, HEAD_DIM]
        )
    
    # K 描述符：和标准 V 一样 [y_dim, HEAD_DIM]
    desc_k = _maybe_make_tensor_desc(
        desc_k,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_N, HEAD_DIM]
    )
    
    # Output 描述符：和 Q 一样的形状和布局
    desc_o = _maybe_make_tensor_desc(
        desc_o,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_M, HEAD_DIM]
    )
    
    # ---- 计算地址偏移 ----
    # offset_y: 当前 (batch, head) 在展平张量中的起始位置
    # off_z * (N_CTX * H): 跳过前面的 batch
    # off_h * N_CTX: 跳过当前 batch 中前面的 head
    # 例如 Z=2, H=4, N_CTX=1024, off_z=1, off_h=2:
    #   offset_y = 1 * (1024*4) + 2 * 1024 = 4096 + 2048 = 6144
    offset_y = off_z * (N_CTX * H) + off_h * N_CTX
    
    # Q 和 Output 的偏移还需要加上 Q 块的位置
    # start_m * BLOCK_M: 跳过当前 head 中前面的 Q 块
    qo_offset_y = offset_y + start_m * BLOCK_M
    
    # ---- 初始化偏移数组 ----
    # offs_m: Q 块内的相对偏移 [0, 1, 2, ..., BLOCK_M-1]
    # 全局偏移 = start_m * BLOCK_M + offs_m
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    
    # offs_n: K/V 块内的相对偏移 [0, 1, 2, ..., BLOCK_N-1]
    offs_n = tl.arange(0, BLOCK_N)
    
    # ---- 初始化 online softmax 状态 ----
    # m_i: 当前找到的每行最大值，初始化为 -inf（负无穷）
    # 形状 [BLOCK_M]，dtype=tl.float32（softmax 计算通常用 fp32 保证精度）
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    
    # l_i: softmax 分母的累加和（每行 exp 的总和），初始化为 1.0
    # 为什么初始化为 1.0 而不是 0？因为在后续更新中 l_i = l_i * alpha + l_ij，
    # 如果初始为 0，第一轮 l_i * alpha = 0，l_i = 0 + l_ij = l_ij，正确
    # 实际上初始化为 0 也可以，这里初始化为 1 是惯例
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    
    # acc: 输出累加器 [BLOCK_M, HEAD_DIM]，初始化为全零
    # 这会累加 sum(softmax(QK^T) * V)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    
    # ---- 加载缩放因子 ----
    # qk_scale = sm_scale * 1.44269504
    # 1.44269504 = 1 / ln(2)，用于将以 e 为底的 softmax 转换为以 2 为底
    # 因为 GPU 硬件的 exp2 指令比 exp 快，所以用 exp2 替代 exp
    # 数学原理：e^(x * sm_scale) = 2^(x * sm_scale / ln(2)) = 2^(x * sm_scale * 1.4427)
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)，转换因子
    
    # ---- 加载 Q 块到 SRAM ----
    # Q 在整个内层循环中保持不变（驻留在 SRAM 中），这是一个重要的优化
    # desc_q.load([行偏移, 列偏移]) 加载 [BLOCK_M, HEAD_DIM] 大小的块
    q = desc_q.load([qo_offset_y, 0])
    
    # ============================================================
    #  分阶段执行内层循环
    # ============================================================
    # Flash Attention v2 根据 causal 设置使用不同的阶段策略：
    #
    # 非 causal (STAGE=1, 二进制 01):
    #   - STAGE & 1 = True  → 执行阶段1（实际传给 _attn_fwd_inner 的是 4-1=3）
    #   - STAGE & 2 = False → 不执行阶段2
    #   - 效果：处理全部 K/V（STAGE=3 在 _attn_fwd_inner 中表示 lo=0, hi=N_CTX）
    #
    # causal (STAGE=3, 二进制 11):
    #   - STAGE & 1 = True  → 执行阶段1（实际传给 _attn_fwd_inner 的是 4-3=1）
    #   - STAGE & 2 = True  → 执行阶段2（直接传 2）
    #   - 效果：先处理前段 K/V（无 mask），再处理对角线 K/V（有 causal mask）
    
    # ---- 阶段1: off-band（前段，无条件可见）----
    if STAGE & 1:
        # 调用内层函数，STAGE 参数为 4 - STAGE
        #   STAGE=3(causal) → 传入 1（处理前段）
        #   STAGE=1(non-causal) → 传入 3（处理全部）
        acc, l_i, m_i = _attn_fwd_inner(
            acc, l_i, m_i, q,
            desc_k, desc_v,
            offset_y, dtype, start_m, qk_scale,
            BLOCK_M, HEAD_DIM, BLOCK_N,
            4 - STAGE,       # 传给内层的阶段号
            offs_m, offs_n, N_CTX,
            warp_specialize, IS_HOPPER
        )
    
    # ---- 阶段2: on-band（对角线区域，需要 causal mask）----
    # 只在 causal attention 时执行
    if STAGE & 2:
        acc, l_i, m_i = _attn_fwd_inner(
            acc, l_i, m_i, q,
            desc_k, desc_v,
            offset_y, dtype, start_m, qk_scale,
            BLOCK_M, HEAD_DIM, BLOCK_N,
            2,               # 阶段2：对角线区域，需要 causal mask
            offs_m, offs_n, N_CTX,
            warp_specialize, IS_HOPPER
        )
    
    # ============================================================
    #  收尾（Epilogue）：最终归一化
    # ============================================================
    # 经过所有 K/V 块的遍历后，acc 存储的是 sum(exp(qk - m_final) * v)
    # l_i 存储的是 sum(exp(qk - m_final))
    #
    # 最终输出 = sum(exp(qk - m_final) * v) / sum(exp(qk - m_final))
    #          = acc / l_i
    #
    # 另外，为了反向传播需要，我们还要保存 log-sum-exp：
    #   LSE = m_final + log(sum(exp(qk - m_final)))
    #       = m_final + log(l_i)
    # 这里用 log2 然后换底：log(l_i) = log2(l_i) * ln(2)
    # 但实际上这里只是保存 m_i + log2(l_i)，反向传播时会相应处理
    
    # m_i = m_i + log2(l_i)：计算 log-sum-exp（用于反向传播）
    # 注意这里的 m_i 已经被更新为最终的全局最大值
    m_i += tl.math.log2(l_i)
    
    # acc = acc / l_i[:, None]：最终归一化
    # l_i[:, None] 形状 [BLOCK_M, 1]，广播到 [BLOCK_M, HEAD_DIM]
    acc = acc / l_i[:, None]
    
    # ---- 写回显存（HBM）----
    # 1. 保存 softmax_lse（log-sum-exp）到 M 数组，供反向传播使用
    # M 的形状是 [Z*H, N_CTX]，每个 token 存一个 LSE 值
    # m_ptrs 计算了写入地址
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i)
    
    # 2. 保存输出到 O 描述符
    # 将 acc 转换为目标精度（fp16 或 fp8）后写入
    desc_o.store([qo_offset_y, 0], acc.to(dtype))


# ================================================================
#                       反向传播（Backward Pass）
# ================================================================

@triton.jit
def _attn_bwd_preprocess(
    O,          # 前向传播的输出 [Z, H, N_CTX, HEAD_DIM]
    DO,         # 输出的梯度（从上游传回的梯度）[Z, H, N_CTX, HEAD_DIM]
    Delta,      # 输出：delta = sum(O * DO, axis=-1)，即逐元素的 output * grad_output 沿 head_dim 求和
    Z, H, N_CTX, # batch、头数、序列长度
    BLOCK_M: tl.constexpr,  # Q 方向的分块大小
    HEAD_DIM: tl.constexpr  # 注意力头维度
):
    """
    反向传播的预处理步骤：计算 delta = sum(O * DO)
    
    这个 kernel 的作用：
    Delta 是 Flash Attention 反向传播中的一个关键中间变量。
    根据数学推导，dS = P * (dP - D)，其中 D = rowsum(O * dO)，
    这里的 D 就是 delta（逐行的 output * grad_output 之和）。
    
    为什么要预处理？
    因为这个计算是逐行的（每个 token 独立），可以先并行算好，
    避免在后续的反向 kernel 中重复计算。
    """
    # 计算当前线程块负责处理的 token 偏移
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    off_hz = tl.program_id(1)                                    # batch*head 索引
    off_n = tl.arange(0, HEAD_DIM)                               # [HEAD_DIM]
    
    # ---- 加载 O 和 DO ----
    # O 的布局：[Z, H, N_CTX, HEAD_DIM]，展平为 [Z*H, N_CTX, HEAD_DIM]
    # off_hz * HEAD_DIM * N_CTX: 跳到对应的 batch*head
    # off_m[:, None] * HEAD_DIM: 跳到对应的 token（[:, None] 生成列向量 [BLOCK_M, 1]）
    # off_n[None, :]: 遍历 head_dim（[None, :] 生成行向量 [1, HEAD_DIM]）
    # 最终加载形状：[BLOCK_M, HEAD_DIM]
    o = tl.load(O + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :])
    do = tl.load(DO + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :]).to(tl.float32)
    
    # ---- 计算 delta = sum(O * DO, axis=1) ----
    # 逐元素乘（Hadamard product），然后沿 head_dim 求和
    # 结果形状：[BLOCK_M]（每个 token 一个标量 delta 值）
    delta = tl.sum(o * do, axis=1)
    
    # ---- 写回 Delta ----
    # Delta 的形状：[Z*H, N_CTX]
    tl.store(Delta + off_hz * N_CTX + off_m, delta)


@triton.jit
def _attn_bwd_dkdv(
    dk, dv,          # dK 和 dV 的累加器（输入/输出），形状 [BLOCK_N1, HEAD_DIM]
    Q, k, v, sm_scale, # Q、K、V 和 softmax scale
    DO,              # 输出的梯度，形状 [N_CTX, HEAD_DIM]
    M, D,            # M: log-sum-exp, D: delta（预处理计算的）
    stride_tok, stride_d,  # 步幅：token 维度和 head_dim 维度的步幅
    H, N_CTX,        # 头数、序列长度
    BLOCK_M1: tl.constexpr, # Q 的分块大小
    BLOCK_N1: tl.constexpr, # K/V 的分块大小（外层）
    HEAD_DIM: tl.constexpr, # 注意力头维度
    start_n, start_m, num_steps,  # K/V 起始位置、Q 起始位置、迭代步数
    MASK: tl.constexpr  # 是否需要 causal mask
):
    """
    反向传播中计算 dK 和 dV 的内层循环
    
    === 数学推导 ===
    
    前向: P = softmax(Q @ K^T * scale), O = P @ V
    
    反向（链式法则）：
      dV = P^T @ dO                           (1)  V 的梯度
      dP = dO @ V^T                           (2)  softmax 输出的梯度
      dS = P * (dP - rowsum(dP * P))          (3)  softmax 输入的梯度 (softmax 的反向传播)
         = P * (dP - D)                       其中 D = rowsum(O * dO) 已经预计算好
      dQ = dS @ K                             (4)  Q 的梯度
      dK = dS^T @ Q                           (5)  K 的梯度

    这个函数计算 dK 和 dV（公式 1 和 5）。

    === 循环结构 ===
    外层循环由 _attn_bwd 控制，遍历 K/V 的块（start_n）。
    本函数的内层循环遍历 Q 的块（从 start_m 开始，步长 BLOCK_M1）。
    这样设计的目的是最大化 K 和 V 在 SRAM 中的复用。
    """
    # 当前 Q 块内的偏移
    offs_m = start_m + tl.arange(0, BLOCK_M1)
    # 当前 K/V 块内的偏移
    offs_n = start_n + tl.arange(0, BLOCK_N1)
    # head_dim 的偏移
    offs_k = tl.arange(0, HEAD_DIM)

    # Q 的指针（注意：加载 Q 的转置）
    # Q 布局：[N_CTX, HEAD_DIM]
    # 我们希望加载 [HEAD_DIM, BLOCK_M1]（Q^T 的块）
    # offs_m[None, :]: [1, BLOCK_M1]（列索引）
    # offs_k[:, None]: [HEAD_DIM, 1]（行索引）
    qT_ptrs = Q + offs_m[None, :] * stride_tok + offs_k[:, None] * stride_d

    # dO 的指针，形状 [BLOCK_M1, HEAD_DIM]
    do_ptrs = DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d

    # 内层循环：遍历 Q 的块
    tl.static_assert(BLOCK_N1 % BLOCK_M1 == 0)
    curr_m = start_m
    step_m = BLOCK_M1

    for blk_idx in range(num_steps):
        # ---- 步骤 1: 加载 Q^T 块 ----
        qT = tl.load(qT_ptrs)  # [HEAD_DIM, BLOCK_M1]

        # ---- 步骤 2: 加载 M（log-sum-exp），用于重建 softmax ----
        offs_m = curr_m + tl.arange(0, BLOCK_M1)
        m = tl.load(M + offs_m)  # [BLOCK_M1]

        # ---- 步骤 3: 重计算 QK^T（前向的注意力分数）----
        # k 是 [BLOCK_N1, HEAD_DIM]，qT 是 [HEAD_DIM, BLOCK_M1]
        # qkT = k @ qT = [BLOCK_N1, BLOCK_M1]
        # 注意：这不是完整的 QK^T，只是当前 K 块和当前 Q 块的局部乘积
        qkT = tl.dot(k, qT)
        
        # ---- 步骤 4: 重计算 softmax（用保存的 LSE 做数值稳定）----
        # P = exp(QK^T * scale - LSE)
        # 这里 LSE 存储在 M 中，m[None, :] 广播为 [1, BLOCK_M1]
        # qkT - m[None, :] 实现数值稳定的 exp
        # 注意：QK^T 还没有乘以 scale！实际上 sm_scale 在前向已经乘入 qk_scale，
        # 这里 m 已经是 log-sum-exp（包含了 scale），所以直接 qkT - m 即可
        pT = tl.math.exp2(qkT - m[None, :])  # [BLOCK_N1, BLOCK_M1]
        
        # ---- 步骤 5: Causal mask（如果需要）----
        if MASK:
            # 下三角 mask：Q 索引 >= K 索引
            mask = (offs_m[None, :] >= offs_n[:, None])
            pT = tl.where(mask, pT, 0.0)
        
        # ---- 步骤 6: 加载 dO ----
        do = tl.load(do_ptrs)  # [BLOCK_M1, HEAD_DIM]
        
        # ---- 步骤 7: 累加 dV = P^T @ dO ----
        # pT: [BLOCK_N1, BLOCK_M1]（即 P 的转置 = P^T）
        # do: [BLOCK_M1, HEAD_DIM]
        # dv += P^T @ dO → [BLOCK_N1, HEAD_DIM]
        ppT = pT
        ppT = ppT.to(tl.float16)
        dv += tl.dot(ppT, do)
        
        # ---- 步骤 8: 加载 D（delta）----
        # D 是预计算的 rowsum(O * dO)，形状 [N_CTX]
        Di = tl.load(D + offs_m)  # [BLOCK_M1]
        
        # ---- 步骤 9: 计算 dP 和 dS ----
        # dP = dO @ V^T（在寄存器中计算）
        # v: [BLOCK_N1, HEAD_DIM], do: [BLOCK_M1, HEAD_DIM]
        # tl.trans(do): 转置为 [HEAD_DIM, BLOCK_M1]
        # v @ trans(do): [BLOCK_N1, BLOCK_M1] = dP^T（dP 的转置）
        dpT = tl.dot(v, tl.trans(do)).to(tl.float32)  # [BLOCK_N1, BLOCK_M1]
        
        # 计算 dS（softmax 输入的梯度）
        # dS = P * (dP - D)  ← 这是 softmax 反向传播的核心公式
        # 用我们的变量：dsT = pT * (dpT - Di[None, :])
        # pT: [BLOCK_N1, BLOCK_M1]，dpT: [BLOCK_N1, BLOCK_M1]
        # Di[None, :]: [1, BLOCK_M1]，广播到 [BLOCK_N1, BLOCK_M1]
        dsT = pT * (dpT - Di[None, :])
        dsT = dsT.to(tl.float16)
        
        # ---- 步骤 10: 累加 dK = dS^T @ Q ----
        # dsT: [BLOCK_N1, BLOCK_M1]
        # tl.trans(qT): 转置为 [BLOCK_M1, HEAD_DIM]
        # dk += dS^T @ Q = dsT @ trans(qT) → [BLOCK_N1, HEAD_DIM]
        dk += tl.dot(dsT, tl.trans(qT))
        
        # ---- 步骤 11: 移动指针到下一个 Q 块 ----
        curr_m += step_m
        qT_ptrs += step_m * stride_tok
        do_ptrs += step_m * stride_tok
    
    return dk, dv


@triton.jit
def _attn_bwd_dq(
    dq, q, K, V,      # dQ 累加器、Q 块、K、V
    do, m, D,          # 输出梯度、log-sum-exp、delta
    stride_tok, stride_d,  # 步幅
    H, N_CTX,          # 头数、序列长度
    BLOCK_M2: tl.constexpr,  # Q 的分块大小
    BLOCK_N2: tl.constexpr,  # K/V 的分块大小
    HEAD_DIM: tl.constexpr,  # 注意力头维度
    start_m, start_n, num_steps,  # Q 起始、K/V 起始、步数
    MASK: tl.constexpr  # 是否需要 causal mask
):
    """
    反向传播中计算 dQ 的内层循环
    
    数学公式：dQ = dS @ K（公式 4）
    其中 dS = P * (dP - D)，dP = dO @ V^T
    
    和 _attn_bwd_dkdv 不同，这个函数的外层遍历 Q 块、内层遍历 K/V 块。
    因为计算 dQ 需要完整的 K/V 信息，将 Q 块保持在 SRAM 中更高效。
    """
    # Q 块内的偏移
    offs_m = start_m + tl.arange(0, BLOCK_M2)
    # K/V 块内的偏移
    offs_n = start_n + tl.arange(0, BLOCK_N2)
    # head_dim 的偏移
    offs_k = tl.arange(0, HEAD_DIM)
    
    # K^T 的指针（用于加载 K 的转置块）
    # K 布局：[N_CTX, HEAD_DIM]
    # 我们要加载 K^T[HEAD_DIM, BLOCK_N2]
    kT_ptrs = K + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
    
    # V^T 的指针（用于加载 V 的转置块）
    vT_ptrs = V + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
    
    # ---- 加载 D（delta）----
    # D 是 rowsum(O * dO)，预计算好的，每个 token 一个值
    Di = tl.load(D + offs_m)  # [BLOCK_M2]
    
    tl.static_assert(BLOCK_M2 % BLOCK_N2 == 0)
    curr_n = start_n
    step_n = BLOCK_N2
    
    # 内层循环：遍历 K/V 的块
    for blk_idx in range(num_steps):
        # ---- 加载 K^T 和 V^T ----
        kT = tl.load(kT_ptrs)  # [HEAD_DIM, BLOCK_N2]
        vT = tl.load(vT_ptrs)  # [HEAD_DIM, BLOCK_N2]
        
        # ---- 重计算 QK^T ----
        # q: [BLOCK_M2, HEAD_DIM]，kT: [HEAD_DIM, BLOCK_N2]
        # qk = q @ kT = [BLOCK_M2, BLOCK_N2]
        qk = tl.dot(q, kT)
        
        # ---- 重计算 softmax（用保存的 LSE）----
        # p = exp(qk - m)，m 是前向保存的 log-sum-exp
        # m 形状 [BLOCK_M2, 1]，广播到 [BLOCK_M2, BLOCK_N2]
        p = tl.math.exp2(qk - m)
        
        # ---- Causal mask ----
        if MASK:
            offs_n = curr_n + tl.arange(0, BLOCK_N2)
            mask = (offs_m[:, None] >= offs_n[None, :])
            p = tl.where(mask, p, 0.0)
        
        # ---- 计算 dP = dO @ V^T ----
        # do: [BLOCK_M2, HEAD_DIM]，vT: [HEAD_DIM, BLOCK_N2]
        # dp = do @ vT = [BLOCK_M2, BLOCK_N2]
        dp = tl.dot(do, vT).to(tl.float32)
        
        # ---- 计算 dS = P * (dP - D) ----
        # softmax 反向传播的核心公式
        # P * (dP - D)：逐元素乘
        # D[:, None]: [BLOCK_M2, 1]，广播到 [BLOCK_M2, BLOCK_N2]
        ds = p * (dp - Di[:, None])
        ds = ds.to(tl.float16)
        
        # ---- 累加 dQ = dS @ K ----
        # ds: [BLOCK_M2, BLOCK_N2]，tl.trans(kT): [BLOCK_N2, HEAD_DIM]
        # dq += ds @ trans(kT) = [BLOCK_M2, HEAD_DIM]
        dq += tl.dot(ds, tl.trans(kT))
        
        # ---- 移动指针到下一个 K/V 块 ----
        curr_n += step_n
        kT_ptrs += step_n * stride_tok
        vT_ptrs += step_n * stride_tok
    
    return dq


@triton.jit
def _attn_bwd(
    Q, K, V, sm_scale,   # 前向输入和 softmax scale
    DO,                   # 输出的梯度
    DQ, DK, DV,          # 输出的梯度（Q、K、V 的梯度）
    M, D,                 # M: log-sum-exp, D: delta
    stride_z, stride_h, stride_tok, stride_d,  # 4维张量的各维度步幅
    H, N_CTX,             # 头数、序列长度
    BLOCK_M1: tl.constexpr,  # dK/dV 计算中的 Q 块大小
    BLOCK_N1: tl.constexpr,  # dK/dV 计算中的 K/V 块大小
    BLOCK_M2: tl.constexpr,  # dQ 计算中的 Q 块大小
    BLOCK_N2: tl.constexpr,  # dQ 计算中的 K/V 块大小
    BLK_SLICE_FACTOR: tl.constexpr,  # causal mask 时的块切片因子
    HEAD_DIM: tl.constexpr,   # 注意力头维度
    CAUSAL: tl.constexpr      # 是否 causal attention
):
    """
    反向传播的主 kernel
    
    GPU 网格布局：
      - program_id(0): K/V 分块索引（控制 dK/dV 的计算）
      - program_id(1): 始终为 0（保留维度）
      - program_id(2): batch*head 索引
    
    这个 kernel 同时计算 dK、dV 和 dQ。
    """
    LN2: tl.constexpr = 0.6931471824645996  # ln(2)，用于精度转换
    
    # 当前处理的 batch*head 索引
    bhid = tl.program_id(2)
    # 在 M/D 数组中的偏移（M/D 是 2D: [Z*H, N_CTX]）
    off_chz = (bhid * N_CTX).to(tl.int64)
    # 在 4D Q/K/V 张量中的 batch/head 偏移
    adj = (stride_h * (bhid % H) + stride_z * (bhid // H)).to(tl.int64)
    pid = tl.program_id(0)  # K/V 块索引
    
    # ---- 将指针移到对应的 batch 和 head ----
    Q += adj
    K += adj
    V += adj
    DO += adj
    DQ += adj
    DK += adj
    DV += adj
    M += off_chz
    D += off_chz
    
    # head_dim 的偏移数组
    offs_k = tl.arange(0, HEAD_DIM)
    
    # ==============================================================
    #  第一部分：计算 dK 和 dV
    # ==============================================================
    
    # K/V 的起始位置（第几个 K/V 块）
    start_n = pid * BLOCK_N1
    start_m = 0
    
    # causal 时要用更小的 Q 块（用于处理对角线区域）
    MASK_BLOCK_M1: tl.constexpr = BLOCK_M1 // BLK_SLICE_FACTOR
    offs_n = start_n + tl.arange(0, BLOCK_N1)
    
    # 初始化 dV 和 dK 累加器
    dv = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
    
    # ---- 加载 K 和 V 到 SRAM ----
    # 它们在 dK/dV 的内层循环中保持不变（复用！）
    # K/V 布局：[N_CTX, HEAD_DIM]
    # offs_n[:, None]: [BLOCK_N1, 1]（token 位置的列向量）
    # offs_k[None, :]: [1, HEAD_DIM]（head_dim 的行向量）
    # 结果形状：[BLOCK_N1, HEAD_DIM]
    k = tl.load(K + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)
    v = tl.load(V + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)
    
    # ---- Causal attention: 先处理对角线区域 ----
    if CAUSAL:
        start_m = start_n  # Q 从和 K 相同的位置开始（对角线）
        num_steps = BLOCK_N1 // MASK_BLOCK_M1
        dk, dv = _attn_bwd_dkdv(
            dk, dv, Q, k, v, sm_scale,
            DO, M, D,
            stride_tok, stride_d,
            H, N_CTX,
            MASK_BLOCK_M1, BLOCK_N1, HEAD_DIM,
            start_n, start_m, num_steps,
            MASK=True,
        )
        start_m += num_steps * MASK_BLOCK_M1
    
    # ---- 处理非 mask 区域（Q 在 K 之后的所有位置）----
    num_steps = (N_CTX - start_m) // BLOCK_M1
    dk, dv = _attn_bwd_dkdv(
        dk, dv, Q, k, v, sm_scale,
        DO, M, D,
        stride_tok, stride_d,
        H, N_CTX,
        BLOCK_M1, BLOCK_N1, HEAD_DIM,
        start_n, start_m, num_steps,
        MASK=False,
    )
    
    # ---- 写回 dV ----
    dv_ptrs = DV + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    tl.store(dv_ptrs, dv)
    
    # ---- 写回 dK ----
    # 注意：dK 需要乘以 sm_scale（因为 K 在前向被缩放过吗？实际上是为了数值等效）
    dk *= sm_scale
    dk_ptrs = DK + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    tl.store(dk_ptrs, dk)
    
    # ==============================================================
    #  第二部分：计算 dQ
    # ==============================================================
    
    start_m = pid * BLOCK_M2
    start_n = 0
    num_steps = N_CTX // BLOCK_N2
    
    MASK_BLOCK_N2: tl.constexpr = BLOCK_N2 // BLK_SLICE_FACTOR
    offs_m = start_m + tl.arange(0, BLOCK_M2)
    
    # ---- 加载 Q、初始化 dQ、加载 dO 和 M ----
    q = tl.load(Q + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)
    dq = tl.zeros([BLOCK_M2, HEAD_DIM], dtype=tl.float32)
    do = tl.load(DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)
    m = tl.load(M + offs_m)
    m = m[:, None]  # [BLOCK_M2, 1]，作为列向量用于广播
    
    # ---- Causal attention: 处理对角线区域 ----
    if CAUSAL:
        # 对角线区域的结束位置
        end_n = start_m + BLOCK_M2
        num_steps = BLOCK_M2 // MASK_BLOCK_N2
        dq = _attn_bwd_dq(
            dq, q, K, V,
            do, m, D,
            stride_tok, stride_d,
            H, N_CTX,
            BLOCK_M2, MASK_BLOCK_N2, HEAD_DIM,
            start_m, end_n - num_steps * MASK_BLOCK_N2, num_steps,
            MASK=True,
        )
        end_n -= num_steps * MASK_BLOCK_N2
        # 处理非 mask 区域
        num_steps = end_n // BLOCK_N2
        start_n = end_n - num_steps * BLOCK_N2
    
    # ---- 处理非 mask 区域 ----
    dq = _attn_bwd_dq(
        dq, q, K, V,
        do, m, D,
        stride_tok, stride_d,
        H, N_CTX,
        BLOCK_M2, BLOCK_N2, HEAD_DIM,
        start_m, start_n, num_steps,
        MASK=False,
    )
    
    # ---- 写回 dQ ----
    # dQ 需要乘以 ln(2)（因为前向用 exp2，反向的梯度链式法则需要 ln(2) 补偿）
    dq_ptrs = DQ + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    dq *= LN2
    tl.store(dq_ptrs, dq)


# ================================================================
#               PyTorch 自定义 autograd Function
# ================================================================

class _attention(torch.autograd.Function):
    """
    PyTorch 自定义 autograd 函数
    
    继承 torch.autograd.Function 并实现 forward 和 backward 静态方法，
    可以将我们的 Triton kernel 集成到 PyTorch 的自动微分系统中。
    
    使用方式：attention = _attention.apply
    之后就可以像普通 PyTorch 函数一样调用 attention(q, k, v, causal, sm_scale)
    它会自动支持梯度计算！
    """

    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale, warp_specialize=True):
        """
        前向传播
        
        参数：
          ctx: 上下文对象，用于在前向和反向之间传递信息
          q: Query 张量 [Z, H, N_CTX, HEAD_DIM]
          k: Key 张量 [Z, H, N_CTX, HEAD_DIM]
          v: Value 张量 [Z, H, N_CTX, HEAD_DIM]
          causal: 是否使用 causal mask
          sm_scale: softmax 缩放因子（通常是 1/sqrt(HEAD_DIM)）
          warp_specialize: 是否启用 warp specialization 优化
        
        返回：
          o: 注意力输出 [Z, H, N_CTX, HEAD_DIM]
        """
        # ---- 形状检查 ----
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        HEAD_DIM_V = v.shape[-1]
        
        # 确保 Q、K、V 的最后一个维度一致
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        # HEAD_DIM 必须是 16、32、64、128 或 256（GPU 硬件的对齐要求）
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}
        
        # ---- 分配输出张量 ----
        o = torch.empty_like(q)
        
        # ---- 确定阶段（stage）----
        # causal=True  → stage=3 (二进制 11，执行阶段1+阶段2)
        # causal=False → stage=1 (二进制 01，只执行阶段1)
        stage = 3 if causal else 1
        
        # ---- GPU 特定配置 ----
        extra_kern_args = {}
        if is_hip():
            # AMD GPU：设置每 EU 的 wave 数和 flush denormals
            waves_per_eu = 3 if HEAD_DIM_K <= 64 else 2
            extra_kern_args = {"waves_per_eu": waves_per_eu, "allow_flush_denorm": True}
        
        # ---- 分配 M（log-sum-exp）数组 ----
        # M 存储每个 token 的 LSE，用于反向传播
        M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        
        # ---- 创建张量描述符（如果支持）----
        # 对于支持 host descriptor 的 GPU，在 CPU 端创建描述符比在 GPU 端创建更高效
        # 但 Hopper + warp_specialize 的组合除外（不兼容）
        if supports_host_descriptor() and not (is_hopper() and warp_specialize):
            y_dim = q.shape[0] * q.shape[1] * q.shape[2]  # 展平后的总 token 数
            
            dummy_block = [1, 1]  # 占位的 block_shape，实际会在 pre_hook 中设置
            desc_q = TensorDescriptor(q, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)
            
            if q.dtype == torch.float8_e5m2:
                # FP8 模式：V 需要转置存储 [HEAD_DIM_K, y_dim]
                desc_v = TensorDescriptor(v, shape=[HEAD_DIM_K, y_dim], strides=[q.shape[2], 1],
                                          block_shape=dummy_block)
            else:
                desc_v = TensorDescriptor(v, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1],
                                          block_shape=dummy_block)
            
            desc_k = TensorDescriptor(k, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)
            desc_o = TensorDescriptor(o, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)
        else:
            # 不支持 host descriptor，直接传原始张量
            desc_q = q
            desc_v = v
            desc_k = k
            desc_o = o
        
        # ---- 设置自定义内存分配器（用于 scratch space）----
        # Triton 需要一些临时内存来做 autotune 和 kernel 执行
        def alloc_fn(size: int, align: int, _):
            return torch.empty(size, dtype=torch.int8, device="cuda")
        
        triton.set_allocator(alloc_fn)
        
        # ---- 定义网格（grid）函数 ----
        # grid 是一个 lambda，接收 autotune 选择的 META 参数，返回 GPU 线程块网格
        # - 第0维: Q 块的数量 = ceil(N_CTX / BLOCK_M)
        # - 第1维: batch*head 的数量 = Z * H
        def grid(META):
            return (triton.cdiv(q.shape[2], META["BLOCK_M"]), q.shape[0] * q.shape[1], 1)
        
        ctx.grid = grid
        
        # ---- Blackwell GPU 寄存器配置 ----
        # maxnreg: 每个线程的最大寄存器数量限制
        # 寄存器是 GPU 上最快的内存，但数量有限
        # 合理设置可以增加并行度
        if is_blackwell() and warp_specialize:
            if HEAD_DIM_K == 128 and q.dtype == torch.float16:
                extra_kern_args["maxnreg"] = 168
            else:
                extra_kern_args["maxnreg"] = 80
        
        # ---- 启动 Triton kernel！----
        # _attn_fwd[grid] 使用 autotune 找到的最佳配置启动 kernel
        _attn_fwd[grid](
            sm_scale, M,
            q.shape[0], q.shape[1],  # Z, H
            desc_q, desc_k, desc_v, desc_o,
            N_CTX=q.shape[2],
            HEAD_DIM=HEAD_DIM_K,
            FP8_OUTPUT=q.dtype == torch.float8_e5m2,
            STAGE=stage,
            warp_specialize=warp_specialize,
            IS_HOPPER=is_hopper(),
            **extra_kern_args
        )
        
        # ---- 保存反向传播需要的张量 ----
        # ctx.save_for_backward 是 PyTorch 的机制，在前向传播时保存张量，
        # 反向传播时通过 ctx.saved_tensors 取出
        ctx.save_for_backward(q, k, v, o, M)
        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.causal = causal
        
        return o

    @staticmethod
    def backward(ctx, do):
        """
        反向传播
        
        参数：
          ctx: 上下文对象（包含了 forward 时保存的信息）
          do: 输出的梯度（从上游传回的）[Z, H, N_CTX, HEAD_DIM]
        
        返回：
          各输入的梯度：dq, dk, dv, None, None, None, None
          （后四个 None 对应 causal, sm_scale, warp_specialize 等非张量参数的梯度）
        """
        # ---- 取出前向保存的张量 ----
        q, k, v, o, M = ctx.saved_tensors
        
        # ---- 确保梯度张量是连续的（内存布局要求）----
        assert do.is_contiguous()
        assert q.stride() == k.stride() == v.stride() == o.stride() == do.stride()
        
        # ---- 分配梯度张量 ----
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        
        BATCH, N_HEAD, N_CTX = q.shape[:3]
        PRE_BLOCK = 128  # 预处理的分块大小
        NUM_WARPS, NUM_STAGES = 4, 5
        BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 32, 128, 128, 32
        BLK_SLICE_FACTOR = 2  # causal mask 时的分块切片因子
        RCP_LN2 = 1.4426950408889634  # 1/ln(2)，用于 K 的预缩放
        
        # ---- 预缩放 K ----
        # K 乘以 sm_scale * (1/ln2)，这是为了在反向传播中复用前向的数值技巧
        arg_k = k
        arg_k = arg_k * (ctx.sm_scale * RCP_LN2)
        
        # ---- 步骤 1: 预处理，计算 delta = sum(O * dO) ----
        assert N_CTX % PRE_BLOCK == 0
        pre_grid = (N_CTX // PRE_BLOCK, BATCH * N_HEAD)
        delta = torch.empty_like(M)  # delta 和 M 形状相同 [BATCH, N_HEAD, N_CTX]
        _attn_bwd_preprocess[pre_grid](
            o, do,
            delta,
            BATCH, N_HEAD, N_CTX,
            BLOCK_M=PRE_BLOCK, HEAD_DIM=ctx.HEAD_DIM
        )
        
        # ---- 步骤 2: 主反向传播 ----
        # 同时计算 dK、dV 和 dQ
        grid = (N_CTX // BLOCK_N1, 1, BATCH * N_HEAD)
        _attn_bwd[grid](
            q, arg_k, v, ctx.sm_scale, do, dq, dk, dv,
            M, delta,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            N_HEAD, N_CTX,
            BLOCK_M1=BLOCK_M1, BLOCK_N1=BLOCK_N1,
            BLOCK_M2=BLOCK_M2, BLOCK_N2=BLOCK_N2,
            BLK_SLICE_FACTOR=BLK_SLICE_FACTOR,
            HEAD_DIM=ctx.HEAD_DIM,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
            CAUSAL=ctx.causal,
        )
        
        # 返回梯度：
        # q, k, v, causal(None), sm_scale(None), warp_specialize(None)
        return dq, dk, dv, None, None, None, None


# ---- 创建可直接调用的函数接口 ----
# torch.autograd.Function 通过 .apply 来调用
attention = _attention.apply

# ---- 检查是否支持 FP8 数据类型（PyTorch 版本要求）----
TORCH_HAS_FP8 = hasattr(torch, 'float8_e5m2')


# ================================================================
#                           测试代码
# ================================================================

@pytest.mark.parametrize("Z", [1, 4])
@pytest.mark.parametrize("H", [2, 48])
@pytest.mark.parametrize("N_CTX", [128, 1024, (2 if is_hip() else 4) * 1024])
@pytest.mark.parametrize("HEAD_DIM", [64, 128])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("warp_specialize", [False, True] if is_blackwell() else [False])
@pytest.mark.parametrize("mode", ["fwd", "bwd"])
@pytest.mark.parametrize("provider", ["triton-fp16"] + (["triton-fp8"] if TORCH_HAS_FP8 else []))
def test_op(Z, H, N_CTX, HEAD_DIM, causal, warp_specialize, mode, provider, dtype=torch.float16):
    """
    正确性测试函数
    
    使用参数化测试（parametrize）自动生成多个测试用例，
    对比 Triton 实现和 PyTorch 参考实现的结果。
    
    测试的参数组合：
      Z (batch): 1, 4
      H (num_heads): 2, 48
      N_CTX (seq_len): 128, 1024, 2048/4096
      HEAD_DIM: 64, 128
      causal: False, True
      warp_specialize: False (Blackwell 上还测试 True)
      mode: "fwd" (只测试前向), "bwd" (只测试反向)
      provider: "triton-fp16" 和 "triton-fp8"（如果支持）
    """
    if mode == "fwd" and "fp16" in provider:
        pytest.skip("避免重复运行前向计算。")
    if mode == "bwd" and "fp8" in provider:
        pytest.skip("FP8 的反向传播不支持。")
    
    # ---- 生成随机测试数据 ----
    torch.manual_seed(20)  # 固定随机种子，保证每次测试结果一致
    q = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE)
         .normal_(mean=0.0, std=0.5).requires_grad_())
    k = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE)
         .normal_(mean=0.0, std=0.5).requires_grad_())
    v = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE)
         .normal_(mean=0.0, std=0.5).requires_grad_())
    sm_scale = 0.5
    
    # ---- PyTorch 参考实现 ----
    ref_dtype = dtype
    if mode == "fwd" and "fp8" in provider:
        # FP8 测试时，用 FP32 做参考计算（更高精度）
        ref_dtype = torch.float32
    q = q.to(ref_dtype)
    k = k.to(ref_dtype)
    v = v.to(ref_dtype)
    
    # 标准 PyTorch attention 计算
    M = torch.tril(torch.ones((N_CTX, N_CTX), device=DEVICE))  # 下三角矩阵（全1）
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale  # Q @ K^T * scale
    if causal:
        p[:, :, M == 0] = float("-inf")  # 应用 causal mask（上三角设为 -inf）
    p = torch.softmax(p.float(), dim=-1)  # softmax（用 fp32 保证精度）
    p = p.to(ref_dtype)
    ref_out = torch.matmul(p, v).half()  # P @ V，转回 fp16 做比较
    
    if mode == "bwd":
        # 反向传播：计算参考梯度
        dout = torch.randn_like(q)         # 随机生成上游梯度
        ref_out.backward(dout)             # 反向传播
        ref_dv, v.grad = v.grad.clone(), None  # 保存 V 的梯度后清零
        ref_dk, k.grad = k.grad.clone(), None
        ref_dq, q.grad = q.grad.clone(), None
    
    # ---- Triton 实现 ----
    if mode == "fwd" and "fp8" in provider:
        # FP8 模式下 Q、K、V 需要转换为 FP8
        q = q.to(torch.float8_e5m2)
        k = k.to(torch.float8_e5m2)
        # V 需要转置，因为 FP8 的实现要求 V 按 [HEAD_DIM, seq_len] 存储
        v = v.permute(0, 1, 3, 2).contiguous()
        v = v.permute(0, 1, 3, 2)
        v = v.to(torch.float8_e5m2)
    
    tri_out = attention(q, k, v, causal, sm_scale, warp_specialize).half()
    
    if mode == "fwd":
        # 前向测试：比较输出
        # FP8 的容差更大（atol=3），因为 FP8 精度较低
        atol = 3 if "fp8" in provider else 1e-2
        torch.testing.assert_close(tri_out, ref_out, atol=atol, rtol=0)
        return
    
    # ---- 反向测试 ----
    tri_out.backward(dout)
    tri_dv, v.grad = v.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dq, q.grad = q.grad.clone(), None
    
    # 比较输出和梯度
    torch.testing.assert_close(tri_out, ref_out, atol=1e-2, rtol=0)
    
    rtol = 0.0
    # AMD MI200 GPU 的 FP16 矩阵乘精度较低，需要放宽相对容差
    if torch.version.hip is not None and triton.runtime.driver.active.get_current_target().arch == "gfx90a":
        rtol = 1e-2
    
    torch.testing.assert_close(tri_dv, ref_dv, atol=1e-2, rtol=rtol)
    torch.testing.assert_close(tri_dk, ref_dk, atol=1e-2, rtol=rtol)
    torch.testing.assert_close(tri_dq, ref_dq, atol=1e-2, rtol=rtol)


# ================================================================
#                       性能基准测试（Benchmark）
# ================================================================

# ---- 尝试导入 Flash Attention 官方实现以进行性能对比 ----
try:
    from flash_attn.flash_attn_interface import \
        flash_attn_qkvpacked_func as flash_attn_func
    HAS_FLASH = True
except BaseException:
    HAS_FLASH = False  # 没有安装 flash-attn 库，无法对比

BATCH, N_HEADS = 4, 32  # 固定 batch=4, heads=32

# ---- 生成 benchmark 配置 ----
configs = []
for HEAD_DIM in [64, 128]:
    for mode in ["fwd", "bwd"]:
        for causal in [True, False]:
            # 在 Hopper 和 Blackwell 上的 causal fwd 启用 warp_specialize
            enable_ws = mode == "fwd" and (is_blackwell() or (is_hopper() and not causal))
            for warp_specialize in [False, True] if enable_ws else [False]:
                configs.append(
                    triton.testing.Benchmark(
                        x_names=["N_CTX"],  # X 轴：序列长度
                        x_vals=[2**i for i in range(10, 15)],  # 1024, 2048, 4096, 8192, 16384
                        line_arg="provider",  # 不同线条对应不同实现
                        line_vals=["triton-fp16"] + (["triton-fp8"] if TORCH_HAS_FP8 else []) +
                        (["flash"] if HAS_FLASH else []),
                        line_names=["Triton [FP16]"] + (["Triton [FP8]"] if TORCH_HAS_FP8 else []) +
                        (["Flash-2"] if HAS_FLASH else []),
                        styles=[("red", "-"), ("blue", "-"), ("green", "-")],
                        ylabel="TFLOPS",  # Y 轴：万亿次浮点运算/秒
                        plot_name=f"fused-attention-batch{BATCH}-head{N_HEADS}-d{HEAD_DIM}-{mode}-causal={causal}-warp_specialize={warp_specialize}",
                        args={
                            "H": N_HEADS,
                            "BATCH": BATCH,
                            "HEAD_DIM": HEAD_DIM,
                            "mode": mode,
                            "causal": causal,
                            "warp_specialize": warp_specialize,
                        },
                    ))


@triton.testing.perf_report(configs)
def bench_flash_attention(BATCH, H, N_CTX, HEAD_DIM, causal, warp_specialize, mode, provider, device=DEVICE):
    """
    性能基准测试
    
    测量不同实现（Triton FP16、Triton FP8、Flash Attention）在不同序列长度下的 TFLOPS。
    
    TFLOPS = 每秒执行多少万亿次浮点运算
    这是衡量 GPU kernel 性能的标准指标。
    """
    assert mode in ["fwd", "bwd"]
    dtype = torch.float16
    
    # ---- Triton 实现测试 ----
    if "triton" in provider:
        # 创建随机输入
        q = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        k = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        v = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        
        # FP8 测试：转换数据格式
        if mode == "fwd" and "fp8" in provider:
            q = q.to(torch.float8_e5m2)
            k = k.to(torch.float8_e5m2)
            v = v.permute(0, 1, 3, 2).contiguous()
            v = v.permute(0, 1, 3, 2)
            v = v.to(torch.float8_e5m2)
        
        sm_scale = 1.3
        fn = lambda: attention(q, k, v, causal, sm_scale, warp_specialize)
        
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        
        # do_bench: 多次运行并测量时间（热身+平均）
        ms = triton.testing.do_bench(fn)
    
    # ---- Flash Attention 官方实现测试 ----
    if provider == "flash":
        # Flash Attention 使用打包的 QKV 格式 [BATCH, N_CTX, 3, H, HEAD_DIM]
        qkv = torch.randn((BATCH, N_CTX, 3, H, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        fn = lambda: flash_attn_func(qkv, causal=causal)
        
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        
        ms = triton.testing.do_bench(fn)
    
    # ---- 计算 FLOPS ----
    # 一个矩阵乘法的 FLOPS = 2 * BATCH * H * N_CTX * N_CTX * HEAD_DIM
    # （因为是 Q*K^T 和 P*V 两次乘法）
    flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX * HEAD_DIM
    total_flops = 2 * flops_per_matmul  # 两次矩阵乘法（QK^T 和 PV）
    
    if causal:
        # Causal attention 只需要计算一半的注意力矩阵（下三角）
        total_flops *= 0.5
    
    if mode == "bwd":
        # 反向传播的 FLOPS 约为前向的 2.5 倍
        # （2x 因为正向+反向各一次矩阵乘，0.5x 因为要重计算 softmax）
        total_flops *= 2.5
    
    # 转换为 TFLOPS: FLOPS / 时间 = (total_flops / 1e12) / (ms / 1000)
    # = total_flops * 1e-12 / (ms * 1e-3)
    return total_flops * 1e-12 / (ms * 1e-3)


# ================================================================
#                           主程序入口
# ================================================================
if __name__ == "__main__":
    # 运行性能基准测试
    # 只在支持的后 GPU 上工作（Ampere 及以上，即 Compute Capability >= 8.0）
    # save_path=".": 将结果图保存到当前目录
    # print_data=True: 打印测试数据到控制台
    bench_flash_attention.run(save_path=".", print_data=True)