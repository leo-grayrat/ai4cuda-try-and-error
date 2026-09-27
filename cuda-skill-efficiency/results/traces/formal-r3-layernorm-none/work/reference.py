import torch
import torch.nn.functional as F


def run(x: torch.Tensor) -> torch.Tensor:
    return F.layer_norm(x, (x.shape[1],), eps=1e-5)
