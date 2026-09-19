"""Baseline linear layer wrapper.

Before adding LoRA, we establish a plain wrapper around nn.Linear that behaves
identically to nn.Linear. This gives us a stable interface: LoRALinear (added
later) will subclass or mirror this so swapping between "plain" and "LoRA"
layers is a drop-in change.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear(nn.Module):
    """A plain linear layer: y = x @ W.T + b, matching nn.Linear exactly."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None

        self.reset_parameters()

    def reset_parameters(self):
        # Match nn.Linear's default init (kaiming uniform for weight, uniform for bias).
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / (fan_in ** 0.5)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)
