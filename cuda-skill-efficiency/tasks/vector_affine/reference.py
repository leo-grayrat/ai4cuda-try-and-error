import torch


def run(x: torch.Tensor) -> torch.Tensor:
    return x * 1.75 + 0.25
