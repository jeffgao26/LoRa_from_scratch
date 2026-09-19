# LoRA from scratch

Implementing Low-Rank Adaptation (LoRA) from the paper (Hu et al., 2021) step by step, to learn
the mechanics rather than just use a library.

## Plan

- [ ] `LoRALinear`: a linear layer with frozen base weights + trainable low-rank update
- [ ] Tests: init equivalence, gradient isolation, weight merging
- [ ] Inject LoRA into a small transformer block
- [ ] Train on a toy task, compare against full fine-tuning
- [ ] Rank ablation

## Layout

- `lora/` — library code
- `tests/` — unit tests
- `examples/` — training scripts / experiments
