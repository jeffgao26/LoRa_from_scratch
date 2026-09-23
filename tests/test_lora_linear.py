import torch

from lora.lora_linear import LoRALinear


def test_matches_base_layer_at_init():
    """B is zero-initialized, so LoRALinear should behave like the frozen
    base layer before any training happens."""
    torch.manual_seed(0)
    layer = LoRALinear(in_features=8, out_features=4, r=2, alpha=4.0)

    x = torch.randn(3, 8)
    base_only = torch.nn.functional.linear(x, layer.weight, layer.bias)
    assert torch.allclose(layer(x), base_only)


def test_only_A_and_B_receive_gradients():
    torch.manual_seed(0)
    layer = LoRALinear(in_features=8, out_features=4, r=2, alpha=4.0)

    x = torch.randn(3, 8, requires_grad=False)
    out = layer(x)
    out.sum().backward()

    assert layer.weight.grad is None
    assert layer.bias.grad is None
    assert layer.A.grad is not None
    assert layer.B.grad is not None


def test_merge_matches_forward():
    torch.manual_seed(0)
    layer = LoRALinear(in_features=8, out_features=4, r=2, alpha=4.0)

    # Perturb B away from zero so the LoRA path actually contributes.
    with torch.no_grad():
        layer.B.copy_(torch.randn_like(layer.B))

    x = torch.randn(5, 8)
    unmerged = layer(x)

    merged_weight = layer.merge()
    merged = torch.nn.functional.linear(x, merged_weight, layer.bias)

    assert torch.allclose(unmerged, merged, atol=1e-6)


if __name__ == "__main__":
    test_matches_base_layer_at_init()
    test_only_A_and_B_receive_gradients()
    test_merge_matches_forward()
    print("ok")
