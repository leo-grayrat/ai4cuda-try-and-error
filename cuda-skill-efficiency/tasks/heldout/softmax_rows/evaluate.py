import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from eval_common import evaluate
from reference import run as reference


def make_input(shape):
    scale = 12.0 if shape[1] == 769 else 1.0
    return torch.randn(shape, dtype=torch.float32, device="cuda") * scale


print(json.dumps(evaluate(Path(sys.argv[1]), reference, make_input, [(8, 128), (64, 769), (256, 2048)])))
