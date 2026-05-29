"""
神经网络基础模块实现。

本文件实现了构建 Transformer 语言模型所需的基础神经网络层，包括：
- Linear: 线性变换层 (全连接层)
- Embedding: 词嵌入层
- RMSNorm: RMS 归一化层
- RotaryPositionalEmbedding: 旋转位置编码 (RoPE)
- MultiheadSelfAttention: 多头自注意力机制
"""

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module, Parameter

from . import functional as F
from .utils import generate_causal_mask, rotate_half


class Linear(Module):
    """
    线性变换层 (全连接层)。

    计算输出 = x @ W^T，其中 W 是权重矩阵。
    使用 Kaiming 风格的截断正态分布初始化权重，
    标准差为 sqrt(2 / (in_features + out_features))。
    """

    in_features: int      # 输入特征维度
    out_feature: int      # 输出特征维度
    weights: Parameter    # 权重矩阵，形状为 (out_features, in_features)

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """
        初始化线性层。

        Args:
            in_features: 输入特征数量
            out_features: 输出特征数量
            device: 参数所在的设备 (CPU/GPU)
            dtype: 参数的数据类型
        """
        super().__init__()

        self.in_features = in_features
        self.out_feature = out_features

        # 创建权重参数并初始化
        self.weights = Parameter(torch.empty((out_features, in_features), dtype=dtype, device=device))

        self._reset_params()

    def _reset_params(self):
        """使用截断正态分布初始化权重，标准差基于输入输出维度。"""
        std = (2 / (self.in_features + self.out_feature)) ** 0.5
        nn.init.trunc_normal_(self.weights, mean=0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: Tensor) -> Tensor:
        """前向传播：输入 x 与权重矩阵的转置相乘。"""
        return x @ self.weights.mT


class Embedding(Module):
    """
    词嵌入层。

    将 token ID 映射为稠密向量表示。
    查找表 (lookup table) 的形状为 (num_embeddings, embedding_dim)。
    """

    num_embedding: int      # 词汇表大小/嵌入数量
    embedding_dim: int      # 嵌入向量的维度
    embeds: Parameter       # 嵌入矩阵，形状为 (num_embeddings, embedding_dim)

    def __init__(
        self,
        num_embedding: int,
        embdding_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """
        初始化嵌入层。

        Args:
            num_embedding: 词汇表大小（嵌入数量）
            embdding_dim: 嵌入向量的维度
            device: 参数所在的设备
            dtype: 参数的数据类型
        """
        super().__init__()

        self.num_embedding = num_embedding
        self.embedding_dim = embdding_dim
        # 创建嵌入矩阵
        self.embeds = Parameter(torch.empty((num_embedding, embdding_dim), dtype=dtype, device=device))

    def _reset_param(self):
        """使用截断正态分布初始化嵌入矩阵。"""
        nn.init.trunc_normal_(self.embeds)

    def forward(self, token_ids: Tensor):
        """
        前向传播：根据 token ID 查找对应的嵌入向量。

        Args:
            token_ids: 输入 token ID 张量，形状为 (..., seq_len)

        Returns:
            嵌入向量张量，形状为 (..., seq_len, embedding_dim)
        """
        out_shape = (*token_ids.shape, self.embedding_dim)
        return self.embeds[token_ids.flatten(), :].reshape(out_shape)


class RMSNorm(Module):
    """
    RMS 归一化层 (Root Mean Square Layer Normalization)。

    计算公式：
        RMS(x) = sqrt(mean(x^2) + eps)
        output = (x / RMS(x)) * weight

    相比 LayerNorm，RMSNorm 去掉了均值中心化步骤，计算更高效。
    参考论文: "Root Mean Square Layer Normalization" (2019)
    """

    d_model: int        # 模型维度
    eps: float          # 防止除零的小常数
    weights: Parameter  # 可学习的缩放参数，形状为 (d_model,)

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        初始化 RMSNorm 层。

        Args:
            d_model: 模型维度（输入特征数）
            eps: 数值稳定性的小常数，默认为 1e-5
            device: 参数所在的设备
            dtype: 参数的数据类型
        """
        super().__init__()

        self.d_model = d_model
        self.eps = eps

        # 可学习的缩放权重，初始化为 1
        self.weights = Parameter(torch.empty((d_model,), dtype=dtype, device=device))

    def _reset_param(self):
        """将缩放权重初始化为全 1。"""
        nn.init.constant_(self.weights, 1)

    def forward(self, x: Tensor):
        """
        前向传播：对输入应用 RMS 归一化。

        Args:
            x: 输入张量，形状为 (..., d_model)

        Returns:
            归一化后的张量，形状与输入相同
        """
        dtype = x.dtype
        # 在 float32 精度下计算 RMS 以保证数值稳定性
        x = x.float()
        # 计算 RMS: sqrt(mean(x^2) + eps)
        rms = ((x * x).sum(-1, keepdim=True) / self.d_model + self.eps) ** 0.5
        # 归一化并恢复原始精度，再乘以可学习权重
        return (x / rms).to(dtype) * self.weights


class RotaryPositionalEmbedding(Module):
    """
    旋转位置编码 (Rotary Positional Embedding, RoPE)。

    通过对 query 和 key 向量施加旋转操作来编码位置信息。
    旋转角度基于频率 theta^(-2i/d_k)，每个维度对具有不同的旋转频率。

    参考论文: "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021)

    计算公式：
        RoPE(x, pos) = x * cos(pos * theta_i) + rotate_half(x) * sin(pos * theta_i)
    """

    theta: int               # 基础频率参数
    in_features: int         # 输入特征维度（必须为偶数）
    max_deq_len: int         # 支持的最大序列长度
    cos: Tensor              # 预计算的余弦值表，形状为 (max_seq_len, in_features)
    sin: Tensor              # 预计算的正弦值表，形状为 (max_seq_len, in_features)

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """
        初始化旋转位置编码。

        Args:
            theta: 基础频率参数（通常是 10000）
            d_k: 每个注意力头的维度（必须为偶数）
            max_seq_len: 支持的最大序列长度
            device: 参数所在的设备
        """
        super().__init__()
        self.in_features = d_k
        self.max_deq_len = max_seq_len

        # 计算每个维度对的频率：theta^(-2i/d_k)
        # 先计算偶数索引的频率，然后每个频率应用于相邻的 (偶数, 奇数) 维度对
        inv_freq = (
            theta ** (torch.arange(0, self.in_features, 2, dtype=torch.float32, device=device) / -self.in_features)
        ).unsqueeze(1)
        # 将每个频率复制到相邻的维度对上，形状变为 (1, in_features)
        inv_freq = inv_freq.expand(-1, 2).reshape(1, self.in_features)

        # 计算每个位置的角度：pos * theta^(-2i/d_k)
        angles = torch.arange(self.max_deq_len, device=device, dtype=torch.float32).unsqueeze(1)
        angles = angles @ inv_freq

        # 预计算所有位置的 cos 和 sin 值
        cos = angles.cos()
        sin = angles.sin()
        # cos, sin: 形状为 (max_seq_len, in_features)

        # 注册为缓冲区（不会被优化器更新，但会随模型移动设备）
        self.register_buffer("sin", sin, persistent=False)
        self.register_buffer("cos", cos, persistent=False)

    def forward(self, x: Tensor, position_ids: Tensor | None = None):
        """
        对输入 x 应用旋转位置编码。

        Args:
            x: 输入张量，形状为 (..., seq_len, d_k)
            position_ids: 位置索引，形状为 (seq_len,) 或 (batch_size, seq_len)。
                          如果为 None，则使用 0, 1, ..., seq_len-1。

        Returns:
            应用 RoPE 后的张量，形状与输入相同
        """
        input_dtype = x.dtype
        if position_ids is None:
            # 默认使用连续位置 0, 1, ..., seq_len-1
            position_ids = torch.arange(x.shape[-2], device=x.device)
        cos, sin = self.cos[position_ids, :], self.sin[position_ids, :]
        # RoPE 公式: x * cos + rotate_half(x) * sin
        return (x * cos + rotate_half(x) * sin).to(input_dtype)


class MultiheadSelfAttention(Module):
    """
    多头自注意力机制 (Multi-head Self-Attention)。

    支持可选的旋转位置编码 (RoPE)。
    实现了因果掩码 (causal mask)，确保每个位置只能关注其之前的位置。

    结构：
        1. QKV 投影：将输入投影为 Query、Key、Value
        2. 可选的 RoPE 位置编码（应用于 Q 和 K）
        3. 缩放点积注意力（带因果掩码）
        4. 输出投影
    """

    embed_dim: int             # 输入/输出特征维度
    num_heads: int             # 注意力头数
    head_dim: int              # 每个头的维度 = embed_dim // num_heads
    theta: float | None        # RoPE 的 theta 参数（如果不使用 RoPE 则为 None）
    max_seq_len: int | None    # 最大序列长度（如果不使用 RoPE 则为 None）

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        theta: float | None = None,
        max_seq_len: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        初始化多头自注意力层。

        Args:
            embed_dim: 嵌入维度（必须能被 num_heads 整除）
            num_heads: 注意力头的数量
            theta: RoPE 的基础频率参数。如果为 None，则不使用位置编码。
            max_seq_len: 最大序列长度。仅当 theta 不为 None 时使用。
            device: 参数所在的设备
            dtype: 参数的数据类型

        Raises:
            AssertionError: 如果 embed_dim 不能被 num_heads 整除
            AssertionError: 如果 theta 和 max_seq_len 只提供了一个（必须同时提供或同时不提供）
        """
        super().__init__()
        assert embed_dim % num_heads == 0, f"期望 embed_dim 能被 num_heads 整除, 但得到 embed_dim={embed_dim}, num_heads={num_heads}"

        assert not ((theta is None) ^ (max_seq_len is None)), (
            "theta 和 max_seq_len 必须同时提供或同时不提供"
        )

        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        self.theta = theta
        self.max_seq_len = max_seq_len

        # QKV 投影层：将 embed_dim 投影到 3 * embed_dim（Q、K、V 拼接）
        self.qkv_proj = Linear(embed_dim, 3 * embed_dim, device=device, dtype=dtype)
        # 输出投影层：将多头输出投影回 embed_dim
        self.out_proj = Linear(embed_dim, embed_dim, device=device, dtype=dtype)

        # 可选的旋转位置编码
        self.rotery_embed = None
        if theta is not None and max_seq_len is not None:
            self.rotery_embed = RotaryPositionalEmbedding(
                theta=theta, d_k=self.head_dim, max_seq_len=max_seq_len, device=device
            )

    def _reset_params(self):
        """重置 QKV 投影层和输出投影层的参数。"""
        self.qkv_proj._reset_params()
        self.out_proj._reset_params()

    def forward(self, x: Tensor):
        """
        前向传播：执行多头自注意力计算。

        Args:
            x: 输入张量，形状为 (batch_size, seq_len, embed_dim)

        Returns:
            注意力输出，形状为 (batch_size, seq_len, embed_dim)

        计算流程：
            1. 通过 QKV 投影生成 Q、K、V
            2. 将 Q、K、V 重塑为多头形式 (batch_size, num_heads, seq_len, head_dim)
            3. 如果启用 RoPE，对 Q 和 K 应用旋转位置编码
            4. 生成因果掩码并计算缩放点积注意力
            5. 拼接所有头的输出
            6. 通过输出投影层
        """
        seq_len = x.shape[-2]
        # 1. QKV 投影
        qkv: Tensor = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, -1)

        # 2. 重塑为多头形式
        last_two_dim = (self.num_heads, self.head_dim)
        q, k, v = q.unflatten(-1, last_two_dim), k.unflatten(-1, last_two_dim), v.unflatten(-1, last_two_dim)
        # 转置为 (batch_size, num_heads, seq_len, head_dim)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # 3. 可选的 RoPE 位置编码
        if self.rotery_embed is not None:
            q, k = self.rotery_embed(q), self.rotery_embed(k)

        # 4. 生成因果掩码（上三角矩阵，屏蔽未来位置）
        causal_mask = generate_causal_mask(seq_len, device=x.device)

        # 5. 缩放点积注意力
        attn_out = F.scaled_dot_production_attention(q, k, v, causal_mask)
        # 6. 拼接多头输出: (batch_size, seq_len, embed_dim)
        attn_out = attn_out.transpose(1, 2).flatten(-2, -1)

        # 7. 输出投影
        out = self.out_proj(attn_out)
        return out