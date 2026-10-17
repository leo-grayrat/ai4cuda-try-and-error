import torch
import triton
import triton.language as tl

from reference import MODEL   # the weights, fixed by seed


def run(x):
    """Replace this with a Triton implementation. x is a CUDA tensor."""
    raise NotImplementedError
