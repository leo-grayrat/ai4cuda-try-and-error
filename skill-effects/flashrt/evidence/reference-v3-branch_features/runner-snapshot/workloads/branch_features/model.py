"""A different dependency graph under the same fixed streaming IPC contract."""
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
    return nn.ModuleDict({'a0': conv(channels, hidden, 5), 'a1': conv(hidden, channels, 3),
                          'b0': conv(channels, hidden, 3), 'b1': conv(hidden, channels, 5)}).to(device).eval()


def gate_and_merge(features, branch_a, branch_b):
    gate = torch.sigmoid(features.mean(dim=1, keepdim=True))
    return gate * branch_a + (1 - gate) * branch_b + 0.1 * features


class ReferencePipeline:
    def __init__(self, repository, config):
        self.device = config['device']
        self.layers = build_layers(repository, config, self.device)
        self.reset()

    def reset(self):
        self.histories = {name: None for name in self.layers}

    def convolution(self, name, features):
        layer, history = self.layers[name], self.histories[name]
        joined = features if history is None else torch.cat([history, features], dim=-1)
        out = layer(joined)
        if history is not None:
            out = out[:, :, history.shape[-1]:]
        self.histories[name] = joined[:, :, -layer._time_causal_padding[0]:].clone()
        return out

    @torch.inference_mode()
    def step(self, features):
        values = torch.from_numpy(features.copy()).unsqueeze(0).to(self.device)
        branch_a = self.convolution('a1', F.silu(self.convolution('a0', values)))
        branch_b = self.convolution('b1', F.silu(self.convolution('b0', values)))
        return gate_and_merge(values, branch_a, branch_b).squeeze(0).cpu().numpy()


@torch.inference_mode()
def full_sequence_oracle(repository, config, features):
    values = torch.from_numpy(features.copy()).unsqueeze(0)
    layers = build_layers(repository, config, 'cpu')
    branch_a = layers['a1'](F.silu(layers['a0'](values)))
    branch_b = layers['b1'](F.silu(layers['b0'](values)))
    return gate_and_merge(values, branch_a, branch_b).squeeze(0).numpy()
