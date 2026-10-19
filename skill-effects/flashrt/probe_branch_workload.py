"""Validate the second task's state semantics and original executable IR."""
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repository', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = parser.parse_args()
    base = Path(__file__).resolve().parent
    sys.path[:0] = [str(base / 'workloads'), str(args.repository.resolve()), str(base.parent)]
    import numpy as np
    import torch
    import torch.nn.functional as F
    from branch_features.model import ReferencePipeline, build_layers, full_sequence_oracle
    from flashrt.ir import IRGraph, IREdge, IROperator, OpType, TensorPort, StateObject, SequentialExecutor, summarize_graph
    from gpu_lock import gpu_lock

    class Convolution(IROperator):
        def __init__(self, name, layer, activate):
            self.layer, self.activate, self.history = layer, activate, 'history_' + name
            super().__init__(name=name, op_type=OpType.EXPOSED, inputs=[TensorPort('features')],
                             outputs=[TensorPort('features')], state_reads=[self.history], state_writes=[self.history])

        def execute(self, inputs, context, state):
            current, history = inputs['features'], state.get(self.history)
            joined = current if history is None else torch.cat([history, current], dim=-1)
            value = self.layer(joined)
            if history is not None:
                value = value[:, :, history.shape[-1]:]
            state.set(self.history, joined[:, :, -self.layer._time_causal_padding[0]:].clone())
            return {'features': F.silu(value) if self.activate else value}

    class Gate(IROperator):
        def execute(self, inputs, context, state):
            return {'gate': torch.sigmoid(inputs['features'].mean(dim=1, keepdim=True))}

    class Merge(IROperator):
        def execute(self, inputs, context, state):
            gate = inputs['gate']
            return {'features': gate * inputs['a'] + (1 - gate) * inputs['b'] + 0.1 * inputs['features']}

    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    random, rows, analyses = np.random.default_rng(2017), [], []
    lock = gpu_lock(base.parent / 'runs/native-gpu.lock', timeout=120) if args.device == 'cuda' else nullcontext()
    with lock, torch.inference_mode():
        for length, chunk in [(1043, 127), (33, 1), (3, 2)]:
            config = dict(channels=16, hidden=32, length=length, chunk=chunk, weight_seed=703, device=args.device)
            pipeline = ReferencePipeline(args.repository, config)
            layers = build_layers(args.repository, config, 'cpu')
            graph = IRGraph(name='branch_features', external_inputs=[TensorPort('features')], external_outputs=[TensorPort('features')])
            for name, layer in layers.items():
                graph.add_operator(Convolution(name, layer, name.endswith('0')))
                graph.add_state(StateObject(name='history_' + name))
            graph.add_operator(Gate(name='gate', op_type=OpType.EXPOSED, inputs=[TensorPort('features')], outputs=[TensorPort('gate')]))
            graph.add_operator(Merge(name='merge', op_type=OpType.EXPOSED,
                                     inputs=[TensorPort(x) for x in ['features', 'a', 'b', 'gate']], outputs=[TensorPort('features')]))
            graph.chunk_ops = [*layers.keys(), 'gate', 'merge']
            for source, source_port, target, target_port in [
                ('a0', 'features', 'a1', 'features'), ('b0', 'features', 'b1', 'features'),
                ('a1', 'features', 'merge', 'a'), ('b1', 'features', 'merge', 'b'), ('gate', 'gate', 'merge', 'gate')]:
                graph.add_edge(IREdge(source, source_port, target, target_port))
            executor = SequentialExecutor(graph)
            initial = {'history_' + name: None for name in layers}
            executor.init_state(initial)
            first = random.standard_normal((16, length), dtype=np.float32)
            second = random.standard_normal((16, length), dtype=np.float32)
            joined = np.concatenate([first, second], axis=-1)
            expected = full_sequence_oracle(args.repository, config, joined)
            request_checks = []
            for index, features in enumerate([first, second, first]):
                if index == 2:
                    pipeline.reset()
                    executor.init_state(initial)
                desired = expected[:, index * length:(index + 1) * length] if index < 2 else expected[:, :length]
                actual, ir_output = [], []
                for position in range(0, length, chunk):
                    values = features[:, position:position + chunk]
                    actual.append(pipeline.step(values))
                    tensor = torch.from_numpy(values.copy()).unsqueeze(0)
                    ir_output.append(executor.run_chunk({'features': tensor})['features'].squeeze(0).numpy())
                actual, ir_output = np.concatenate(actual, axis=-1), np.concatenate(ir_output, axis=-1)
                request_checks.append({'request': index, 'reset_before': index == 2,
                                       'streamed_matches_cpu_full': bool(np.allclose(actual, desired, atol=3e-5, rtol=3e-4)),
                                       'original_ir_matches_cpu_full': bool(np.allclose(ir_output, desired, atol=3e-5, rtol=3e-4)),
                                       'streamed_max_abs_difference': float(np.max(np.abs(actual - desired)))})
            wrong = []
            for position in range(0, length, chunk):
                pipeline.reset()
                wrong.append(pipeline.step(first[:, position:position + chunk]))
            missing_history_detected = not np.allclose(np.concatenate(wrong, axis=-1), expected[:, :length], atol=3e-5, rtol=3e-4)
            features = torch.from_numpy(first.copy()).unsqueeze(0)
            branch_a = layers['a1'](F.silu(layers['a0'](features)))
            gate = torch.sigmoid(features.mean(dim=1, keepdim=True))
            missing_branch = (gate * branch_a + 0.1 * features).squeeze(0).numpy()
            missing_branch_detected = not np.allclose(missing_branch, expected[:, :length], atol=3e-5, rtol=3e-4)
            assert all(row['streamed_matches_cpu_full'] and row['original_ir_matches_cpu_full'] for row in request_checks)
            assert missing_history_detected and missing_branch_detected
            rows.append({'config': config, 'requests': request_checks, 'missing_history_detected': missing_history_detected,
                         'missing_branch_detected': missing_branch_detected})
            analyses.append(summarize_graph(graph))
        if args.device == 'cuda':
            torch.cuda.synchronize()
    result = {'scope': 'Second-task numerical and original-IR development validation; no model-treatment effect claim',
              'status': 'branch_workload_checks_passed', 'device': args.device, 'cases': rows, 'analysis': analyses,
              'source_sha256': {str(path.relative_to(base)): hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in [Path(__file__), *sorted((base / 'workloads/branch_features').glob('*.py'))]},
              'upstream_audio_sha256': hashlib.sha256((args.repository / 'flashrt/layers/audio.py').read_bytes()).hexdigest()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes((json.dumps(result, indent=2) + '\n').encode())
    print(json.dumps({'status': result['status'], 'cases': len(rows), 'request_checks': sum(len(row['requests']) for row in rows),
                      'deliberate_omissions_detected': 2 * len(rows)}))


if __name__ == '__main__':
    main()
