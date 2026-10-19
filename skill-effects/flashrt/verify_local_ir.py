"""Compare original IR execution with outputs captured through the app adapter."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repository', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    repository = args.repository.resolve()
    base = Path(__file__).resolve().parent
    sys.path[:0] = [str(base / 'workloads'), str(repository)]
    import numpy as np
    import torch
    import torch.nn.functional as F
    from causal_features.model import build_layers
    from flashrt.ir import (IRGraph, IREdge, IROperator, OpType, SequentialExecutor,
                            StateObject, TensorPort, summarize_graph)

    result_path = args.run / 'result.json'
    evidence = json.loads(result_path.read_text(encoding='utf-8'))
    assert evidence['status'] == 'reference_run_passed', 'Need an independently completed reference run'
    for name, expected_hash in evidence['source_sha256'].items():
        if name.replace('\\', '/').startswith('workloads/'):
            assert sha(base / name) == expected_hash, 'Task computation changed after the reference run'
    for name in ['fixtures.npz', 'outputs.npz', 'config.json']:
        assert sha(args.run / name) == evidence['artifacts'][name], f'Changed app evidence: {name}'
    config = evidence['config']
    torch.set_num_threads(2)
    layers = build_layers(repository, config, 'cpu')

    class CausalStage(IROperator):
        def __init__(self, index, layer):
            self.layer, self.history = layer, f'history_{index}'
            super().__init__(name=f'conv_{index}', op_type=OpType.EXPOSED,
                             inputs=[TensorPort('features')], outputs=[TensorPort('features')],
                             state_reads=[self.history], state_writes=[self.history])

        def execute(self, inputs, context, state):
            current, previous = inputs['features'], state.get(self.history)
            value = current if previous is None else torch.cat([previous, current], dim=-1)
            out = self.layer(value)
            if previous is not None:
                out = out[:, :, previous.shape[-1]:]
            state.set(self.history, value[:, :, -self.layer._time_causal_padding[0]:].clone())
            return {'features': F.silu(out)}

    graph = IRGraph(name='causal_features_reference', external_inputs=[TensorPort('features')],
                    external_outputs=[TensorPort('features')])
    for index, layer in enumerate(layers):
        graph.add_operator(CausalStage(index, layer))
        graph.add_state(StateObject(name=f'history_{index}'))
        graph.chunk_ops.append(f'conv_{index}')
        if index:
            graph.add_edge(IREdge(f'conv_{index-1}', 'features', f'conv_{index}', 'features'))
    if config.get('task') == 'branch_features':
        from branch_ir import build_graph
        graph = build_graph(repository, config)
    executor = SequentialExecutor(graph)
    rows = []
    with np.load(args.run / 'fixtures.npz') as fixtures, np.load(args.run / 'outputs.npz') as outputs, torch.inference_mode():
        for row in evidence['requests']:
            index = row['index']
            if index == 0 or row['reset_before']:
                executor.init_state({name: None for name in graph.state_objects})
            inputs = fixtures[f'input_{index}']
            pieces = []
            for position in range(0, config['length'], config['chunk']):
                features = torch.from_numpy(inputs[:, position:position + config['chunk']].copy()).unsqueeze(0)
                pieces.append(executor.run_chunk({'features': features})['features'].squeeze(0).numpy())
            actual, reference = np.concatenate(pieces, axis=-1), outputs[f'output_{index}']
            matches = bool(np.allclose(actual, reference, atol=3e-5, rtol=3e-4))
            rows.append({'request_index': index, 'reset_before': row['reset_before'], 'numerical_match': matches,
                         'max_abs_difference': float(np.max(np.abs(actual - reference))), 'atol': 3e-5, 'rtol': 3e-4})
    result = {'status': 'ir_reference_comparison_passed' if all(row['numerical_match'] for row in rows) else 'ir_reference_comparison_failed',
              'reference_result_sha256': sha(result_path), 'verifier_sha256': sha(Path(__file__)),
              'source_sha256': {name: sha(repository / name) for name in
                                ['flashrt/ir/graph.py', 'flashrt/ir/executor.py', 'flashrt/ir/analysis.py', 'flashrt/layers/audio.py']},
              'scope': 'Original CPU IR executor versus reference backend outputs captured through original IPC',
              'requests': rows, 'analysis': summarize_graph(graph),
              'analysis_validation_label': 'Structural only; numerical comparisons are recorded separately above'}
    if config.get('task') == 'branch_features':
        result['branch_ir_sha256'] = sha(base / 'branch_ir.py')
        result['branch_model_sha256'] = sha(base / 'workloads/branch_features/model.py')
    (args.run / 'ir-validation.json').write_bytes((json.dumps(result, indent=2) + '\n').encode('utf-8'))
    print(json.dumps({'status': result['status'], 'requests_compared': len(rows)}))
    return 0 if result['status'] == 'ir_reference_comparison_passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
