import torch


def run(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x, dim=1)
