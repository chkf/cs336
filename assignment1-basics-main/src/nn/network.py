import torch
import torch.nn as nn
from torch import Tensor
from .basic import Linear, RotaryPositionalEmbedding, RMSNorm, Embedding
from . import functional as F


class SwiGLU(nn.Module):
    def __init__(self,
                 d_model: int,
                 d_ff: int
                 ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        self.linear1 = Linear(
            self.d_model,
            self.d_ff,
        )
        self.linear2 = Linear(
            self.d_ff,
            self.d_model,
        )
        self.linear3 = Linear(
            self.d_model,
            self.d_ff,
        )

    def _reset_params(self):
        self.linear1._reset_params()
        self.linear2._reset_params()
        self.linear3._reset_params()

    def forward(self, x: Tensor) -> Tensor:
        gate = self.linear1(x)
        silu = gate * torch.sigmoid(gate)
        value = self.linear3(x)

        return self.linear2(silu * value)


class MultiheadSelfAttention(nn.Module):
    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 theta: float | None = None,
                 max_seq_len: int | None = None) -> None:
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.theta = theta
        self.max_seq_len = max_seq_len

        self.head_dim = d_model // num_heads

        assert self.d_model == self.head_dim * self.num_heads

        self.qkv_proj = Linear(self.d_model, 3*self.d_model)
        self.out_proj = Linear(self.d_model, self.d_model)

        self.rotery_embed = None
        if theta is not None and max_seq_len is not None:
            self.rotery_embed = RotaryPositionalEmbedding(theta, self.head_dim, max_seq_len)

    def _reset_params(self):
        self.qkv_proj._reset_params()
        self.out_proj._reset_params()

    def forward(self, x: Tensor) -> Tensor:
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, -1)

        last_two_dim = (self.num_heads, self.head_dim)
        q = q.unflatten(-1, last_two_dim).transpose(1, 2)
        k = k.unflatten(-1, last_two_dim).transpose(1, 2)
        v = v.unflatten(-1, last_two_dim).transpose(1, 2)

        if self.rotery_embed is not None:
            q, k = self.rotery_embed(q), self.rotery_embed(k)

        causal_mask = F.generate_causal_mask(x.shape[-2], device=x.device)

        attn_out = F.scaled_dot_production_attention(q, k, v, causal_mask).transpose(1, 2).flatten(-2, -1)

        return self.out_proj(attn_out)


# TODO: to finish
class MoE(nn.Module):
    def __init__(self,
                 d_model: int,
                 d_ff: int,
                 num_experts: int,
                 top_k: int = 1,
                 router_jitter: float = 0.0,
                 z_loss_coef: float = 1e-3,
                 lb_loss_coef: float = 1e-1) -> None:
        super().__init__()

        self.router = Linear(d_model, num_experts)
        self.experts = nn.ModuleList()
        for i in range(num_experts):
            self.experts.append(SwiGLU(d_model, d_ff))
        self.num_experts = num_experts
        self.top_k = top_k
        self.router_jitter = router_jitter
        self.z_loss_coef = z_loss_coef
        self.lb_loss_coef = lb_loss_coef

    @staticmethod
    def _z_loss(x: Tensor) -> Tensor:
        log_sum_exp = torch.logsumexp(x, dim=-1)
        z_loss = torch.mean(log_sum_exp**2)
        return z_loss

    @staticmethod
    def _load_balance_loss(router_probs: Tensor,
                           topk_idx: Tensor,
                           num_experts: int) -> Tensor:
        p = router_probs.mean(dim=(0, 1))

        dispatch = nn.functional.one_hot(topk_idx, num_experts).to(router_probs.dtype)
        f = dispatch.mean(dim=(0, 1, 2))

        return num_experts * torch.sum(p * f)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        logits = self.router(x)
        if self.router_jitter > 0.0 and self.training:
            noise = torch.randn_like(logits) * self.router_jitter
            logits = logits + noise

        router_probs = torch.softmax(logits, dim=-1)

        topk_logits, topk_idx = torch.topk(logits, self.top_k)
        topk_gates = F.softmax(topk_logits)

        z_loss = self._z_loss(logits)
        lb_loss = self._load_balance_loss(router_probs, topk_idx, self.num_experts)

        x_flat = x.flatten(0, 1)
        topk_gates_flat = topk_gates.flatten(0, 1)
        out = torch.zeros_like(x_flat)

        for i, expert in enumerate(self.experts):
            mask = (topk_idx == i).any(dim=-1)

            mask_flat = mask.flatten(0, 1)
            input = x_flat[mask_flat]

            expert_out = expert(input)

            gated_expert_out = expert_out * topk_gates_flat[mask_flat, i]

            out[mask_flat] += gated_expert_out

        return out.reshape(x.shape), z_loss, lb_loss


class TransformerBlock(nn.Module):
    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 d_ff: int,
                 max_seq_len: int,
                 theta: float) -> None:
        super().__init__()

        self.norm1 = RMSNorm(d_model)
        self.mha = MultiheadSelfAttention(d_model, num_heads, theta, max_seq_len)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)

    def _reset_params(self):
        self.norm1._reset_params()
        self.mha._reset_params()
        self.norm2._reset_params()
        self.ffn._reset_params()

    def forward(self, x: Tensor) -> Tensor:
        x = self.mha(self.norm1(x)) + x
        x = self.ffn(self.norm2(x)) + x
        return x


class TransformerLM(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 max_seq_len: int,
                 d_model: int,
                 num_layers: int,
                 num_heads: int,
                 d_ff: int,
                 theta: float) -> None:
        super().__init__()

        self.embed = Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([])

        for _ in range(num_layers):
            self.layers.append(TransformerBlock(d_model, num_heads, d_ff, max_seq_len, theta))

        self.post_norm = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

    def _reset_params(self):
        self.embed._reset_params()
        for layer in self.layers:
            layer._reset_params()
        self.post_norm._reset_params()
        self.lm_head._reset_params()

    def forward(self, x: Tensor) -> Tensor:
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        x = self.post_norm(x)
        logits = self.lm_head(x)
        return logits
