from collections.abc import Callable
from typing import Optional
import torch
import math


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1

        return loss


class AdamW(torch.optim.Optimizer):
    def __init__(self,
                 params,
                 lr: float,
                 weight_decay: float,
                 betas: tuple[float, float],
                 eps: float) -> None:

        defaults = {
            "alpha": lr,
            "beta1": betas[0],
            "beta2": betas[1],
            "lambda": weight_decay,
            "eps": eps
        }
        super().__init__(params, defaults)

        self.current_step = 0

    @torch.no_grad()
    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()

        self.current_step += 1
        t = self.current_step

        for group in self.param_groups:
            alpha = group["alpha"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            lambda_ = group["lambda"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                grad = p.grad

                alpha_t = alpha * math.sqrt(1 - beta2**t) / (1 - beta1**t)

                p.data = p - alpha * lambda_ * p

                m = state.get("m", torch.zeros_like(grad))
                v = state.get("v", torch.zeros_like(grad))

                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad * grad

                state["m"] = m
                state["v"] = v

                p.data = p - alpha_t * m / (v**0.5 + eps)

        return loss


if __name__ == "__main__":
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=1)
    for t in range(100):
        opt.zero_grad()
        loss = (weights**2).mean()
        print(loss.cpu().item())
        loss.backward()
        opt.step()
