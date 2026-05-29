import math
import torch
from torch import Tensor


def softmax(x: Tensor, dim: int = -1):
    x_temp = x - torch.amax(x, dim=dim, keepdim=True)
    exp = x_temp.exp()
    return exp/exp.sum(dim=dim, keepdim=True)


def scaled_dot_production_attention(query: Tensor,
                                    key: Tensor,
                                    value: Tensor,
                                    attn_mask: Tensor | None = None):
    *_, head_dim = query.shape
    
    score = (query @ key.mT) / head_dim**0.5
    if attn_mask is not None:
        score = torch.where(attn_mask, score, torch.full_like(score, -torch.inf))

    return softmax(score) @ value


def generate_causal_mask(seq_len: int, device: torch.device):
    mask = torch.ones((seq_len, seq_len), dtype=torch.bool, device=device)
    mask = mask.tril(0)
    return mask


def cross_entropy_loss(logits: Tensor, target: Tensor) -> Tensor:
    logits = logits.to(torch.float32)
    logits = logits - torch.amax(logits, -1, keepdim=True)

    target = target.unsqueeze(-1)

    loss = torch.log(torch.exp(logits).sum(-1, keepdim= True)) - torch.gather(logits, -1, target)

    return loss.mean()

def get_lr_cosine_schedule(t: int, lr_max: float, lr_min: float, t_warmup: int, t_cosine: int) -> float:
    if t < t_warmup:
        lr = lr_max * t / t_warmup
    elif t_warmup <= t <= t_cosine:
        lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * (t - t_warmup) / (t_cosine - t_warmup)))
    else:
        lr = lr_min
    return lr


@torch.no_grad()
def gradient_clipping(params, max_norm: float, eps=1e-6) -> None:
    grads: tuple = tuple(p.grad for p in params if p.grad is not None)
    square_sum = torch.tensor(0.0, device=grads[0].device)

    for g in grads:
        square_sum += torch.sum(g**2)
    norm_all = square_sum.sqrt()
    if norm_all > max_norm:
        clip_coef = max_norm / (norm_all + eps)
        for g in grads:
            g.mul_(clip_coef)