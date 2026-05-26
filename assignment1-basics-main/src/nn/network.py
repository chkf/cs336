import torch
import torch.nn as nn
from torch import Tensor
from .basic import Linear

class SwiGLU(nn.Module):
    def __init__(self, 
                 in_features: int,
                 ff_features: int,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.d_model = in_features
        self.d_ff = ff_features

        self.linear1 = Linear(
            self.d_model,
            self.d_ff,
            device=device,
            dtype=dtype
        )
        self.linear2 = Linear(
            self.d_model,
            self.d_ff,
            device=device,
            dtype=dtype
        )
        self.linear3 = Linear(
            self.d_ff,
            self.d_model,
            device=device,
            dtype=dtype
        )

    def _reset_params(self):
        pass

    def forward(self, x: Tensor):
        gate = self.linear1(x)
        silu = gate * torch.sigmoid(gate)
        value = self.linear3(x)

        return self.linear2(silu * value)