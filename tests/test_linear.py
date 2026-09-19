import torch
import torch.nn as nn

from lora.linear import Linear


def test_matches_nn_linear():
    torch.manual_seed(0)
    ref = nn.Linear(8, 4)
    ours = Linear(8, 4)

    # Copy weights so we're comparing the same function, not just the same init scheme.
    with torch.no_grad():
        ours.weight.copy_(ref.weight)
        ours.bias.copy_(ref.bias)

    x = torch.randn(3, 8)
    assert torch.allclose(ref(x), ours(x))


if __name__ == "__main__":
    test_matches_nn_linear()
    print("ok")
