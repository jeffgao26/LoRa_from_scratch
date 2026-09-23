"""LoRA linear layer.

Wraps a frozen base weight W0 (d_out x d_in) with a trainable low-rank update
B @ A, where A is (r x d_in) and B is (d_out x r), r << min(d_in, d_out).

Forward pass:
    h = W0 @ x + (alpha / r) * B @ A @ x

Reference: Hu et al. 2021, https://arxiv.org/abs/2106.09685, Section 4.1.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int,
        alpha: float = 1.0,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = alpha

        # Frozen base weight, same shape convention as nn.Linear.
        self.weight = nn.Parameter(torch.empty(out_features, in_features), requires_grad=False)
        self.bias = nn.Parameter(torch.empty(out_features), requires_grad=False) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            bound = 1 / (in_features ** 0.5)
            nn.init.uniform_(self.bias, -bound, bound)

        # Trainable low-rank update. B is zero-initialized so B @ A = 0 at
        # the start of training -- the layer initially matches the base layer.
        self.A = nn.Parameter(torch.empty(r, in_features))
        self.B = nn.Parameter(torch.zeros(out_features, r))
        nn.init.normal_(self.A, mean=0.0, std=1.0 / r ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias)
        lora_update = (self.alpha / self.r) * F.linear(F.linear(x, self.A), self.B)
        return base + lora_update

    def merge(self) -> torch.Tensor:
        """Return the effective merged weight W0 + (alpha/r) * B @ A.

        Useful for verifying correctness and for the no-extra-latency
        deployment trick described in the paper (Section 4.1).
        """
        return self.weight + (self.alpha / self.r) * (self.B @ self.A)
