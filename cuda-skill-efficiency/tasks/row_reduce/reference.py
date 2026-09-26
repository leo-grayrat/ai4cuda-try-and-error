import torch


def run(x: torch.Tensor) -> torch.Tensor:
    return x.sum(dim=1)
