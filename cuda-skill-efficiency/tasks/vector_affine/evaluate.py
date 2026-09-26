import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from eval_common import evaluate
from reference import run as reference


def make_input(shape):
    return torch.randn(shape, dtype=torch.float32, device="cuda")


print(json.dumps(evaluate(Path(sys.argv[1]), reference, make_input, [(1024,), (65536,), (1_048_576,)])))
