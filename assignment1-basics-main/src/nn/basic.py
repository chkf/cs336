import torch
import torch.nn as nn
from torch import Tensor


class Linear(nn.Module):
    def __init__(self, 
                 in_features: int,
                 out_features: int,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.weights = nn.Parameter(torch.empty((out_features, in_features), dtype=dtype, device=device))

        self._reset_params()

    def _reset_params(self):
        std = (2 / (self.in_features + self.out_features)) ** 0.5
        nn.init.trunc_normal_(self.weights, mean=0, std=std, a=-3*std, b=3*std)

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weights.mT
    

class Embedding(nn.Module):
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None) -> None:
        super().__init__()
        self.num_embedding = num_embeddings
        self.embedding_dim = embedding_dim
        self.embeds = nn.Parameter(torch.empty((num_embeddings, embedding_dim), dtype=dtype, device=device))

        self._reset_params()

    def _reset_params(self):
        nn.init.trunc_normal_(self.embeds, a=-3, b=3)

    def forward(self, token_ids: Tensor):
        out_shape = (*token_ids.shape, self.embedding_dim)
        return self.embeds[token_ids.flatten(), :].reshape(out_shape)
    

class RMSNorm(nn.Module):
    def __init__(self, 
                 d_model: int,
                 eps: float = 1e-5,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None) -> None:
        super().__init__()

        self.d_model = d_model
        self.eps = eps

        self.weights = nn.Parameter(torch.empty((d_model), dtype=dtype, device=device))

        self._reset_param()

    def _reset_param(self):
        nn.init.constant_(self.weights, 1)

    def forward(self, x: Tensor):
        dtype = x.dtype
        x = x.to(torch.float32)
        rms = ((x * x).sum(-1, keepdim=True)/self.d_model +self.eps) ** 0.5
        return (x/rms).to(dtype) * self.weights