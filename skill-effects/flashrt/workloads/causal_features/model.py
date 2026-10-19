"""Fixed computation specification using the pinned upstream convolution class."""
import ast
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


def build_layers(repository, config, device):
    source_path = Path(repository) / 'flashrt/layers/audio.py'
    node = next(node for node in ast.parse(source_path.read_text(encoding='utf-8')).body
                if isinstance(node, ast.ClassDef) and node.name == 'CausalConv1d')
    namespace = {'nn': nn, 'F': F, 'torch': torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source_path), 'exec'), namespace)
    conv = namespace['CausalConv1d']
    torch.manual_seed(config['weight_seed'])
    channels, hidden = config['channels'], config['hidden']
    return nn.ModuleList([conv(channels, hidden, 5), conv(hidden, hidden, 5),
                          conv(hidden, channels, 3)]).to(device).eval()


class ReferencePipeline:
    def __init__(self, repository, config):
        self.device = config['device']
        self.layers = build_layers(repository, config, self.device)
        self.reset()

    def reset(self):
        self.histories = [None] * len(self.layers)

    @torch.inference_mode()
    def step(self, features):
        value = torch.from_numpy(features.copy()).unsqueeze(0).to(self.device)
        for index, layer in enumerate(self.layers):
            history = self.histories[index]
            joined = value if history is None else torch.cat([history, value], dim=-1)
            result = layer(joined)
            if history is not None:
                result = result[:, :, history.shape[-1]:]
            self.histories[index] = joined[:, :, -layer._time_causal_padding[0]:].clone()
            value = F.silu(result)
        return value.squeeze(0).cpu().numpy()


@torch.inference_mode()
def full_sequence_oracle(repository, config, features):
    """CPU full-sequence evaluation, independent of chunk caches and IPC."""
    value = torch.from_numpy(features.copy()).unsqueeze(0)
    for layer in build_layers(repository, config, 'cpu'):
        value = F.silu(layer(value))
    return value.squeeze(0).numpy()
