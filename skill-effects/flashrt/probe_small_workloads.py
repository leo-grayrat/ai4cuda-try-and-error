"""Feasibility probes for local tasks, not formal skill-comparison trials.

Executes the pinned CausalConv1d class without unrelated heavyweight imports,
and exercises the pinned tensor channel across Windows subprocesses.
"""
import argparse
import ast
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time
import uuid


def tensor_writer(repository, name, result):
    sys.path.insert(0, repository)
    import numpy as np
    from flashrt.channels.shm_channel.tensor_buffer import SharedTensorBuffer
    buffer = SharedTensorBuffer(name, (4,), 8, create=False, timeout_s=5)
    try:
        buffer.write(np.arange(12, dtype=np.float32).reshape(3, 4))
        result.put({'written': buffer.num})
    finally:
        buffer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repository', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    repository = args.repository.resolve()
    sys.path.insert(0, str(repository))
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional
    from flashrt.channels.shm_channel.tensor_buffer import SharedTensorBuffer
    from flashrt.ir import (IRGraph, IREdge, IROperator, OpType, SequentialExecutor,
                            StateObject, TensorPort, summarize_graph)

    source_path = repository / 'flashrt/layers/audio.py'
    source = source_path.read_text(encoding='utf-8')
    node = next(node for node in ast.parse(source).body
                if isinstance(node, ast.ClassDef) and node.name == 'CausalConv1d')
    namespace = {'torch': torch, 'nn': nn, 'F': functional}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source_path), 'exec'), namespace)
    causal_conv = namespace['CausalConv1d']
    torch.manual_seed(703)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rows = []
    analysis_summaries = []

    class ChunkConv(IROperator):
        def __init__(self, index, layer):
            self.layer = layer
            self.history_name = f'history_{index}'
            super().__init__(name=f'conv_{index}', op_type=OpType.EXPOSED,
                             inputs=[TensorPort('features')], outputs=[TensorPort('features')],
                             state_reads=[self.history_name], state_writes=[self.history_name])

        def execute(self, inputs, context, state):
            value = inputs['features']
            history = state.get(self.history_name)
            joined = value if history is None else torch.cat([history, value], dim=-1)
            out = self.layer(joined)
            if history is not None:
                out = out[:, :, history.shape[-1]:]
            state.set(self.history_name, joined[:, :, -self.layer._time_causal_padding[0]:].clone())
            return {'features': functional.silu(out)}

    with torch.inference_mode():
        for channels, hidden, length, chunk in [(16, 32, 1024, 128), (16, 32, 1043, 127), (128, 256, 16384, 512)]:
            layers = nn.ModuleList([causal_conv(channels, hidden, 5), causal_conv(hidden, hidden, 5),
                                    causal_conv(hidden, channels, 3)]).to(device).eval()
            inputs = torch.randn(1, channels, length, device=device)
            if device == 'cuda':
                torch.cuda.reset_peak_memory_stats()
            expected = inputs
            for layer in layers:
                expected = functional.silu(layer(expected))
            histories = [None] * len(layers)
            pieces = []
            for position in range(0, length, chunk):
                value = inputs[:, :, position:position + chunk]
                for index, layer in enumerate(layers):
                    history = histories[index]
                    joined = value if history is None else torch.cat([history, value], dim=-1)
                    out = layer(joined)
                    if history is not None:
                        out = out[:, :, history.shape[-1]:]
                    histories[index] = joined[:, :, -layer._time_causal_padding[0]:].clone()
                    value = functional.silu(out)
                pieces.append(value)
            actual = torch.cat(pieces, dim=-1)
            difference = (expected - actual).abs()
            matches = torch.allclose(expected, actual, atol=3e-5, rtol=3e-4)
            graph = IRGraph(name='causal_audio_chain', external_inputs=[TensorPort('features')],
                            external_outputs=[TensorPort('features')])
            for index, layer in enumerate(layers):
                graph.add_state(StateObject(name=f'history_{index}'))
                graph.add_operator(ChunkConv(index, layer))
                graph.chunk_ops.append(f'conv_{index}')
                if index:
                    graph.add_edge(IREdge(f'conv_{index-1}', 'features', f'conv_{index}', 'features'))
            executor = SequentialExecutor(graph)
            initial_state = {f'history_{index}': None for index in range(len(layers))}
            executor.init_state(initial_state)
            ir_output = torch.cat([executor.run_chunk({'features': inputs[:, :, pos:pos + chunk]})['features']
                                   for pos in range(0, length, chunk)], dim=-1)
            ir_matches = torch.allclose(expected, ir_output, atol=3e-5, rtol=3e-4)
            # Reset the same executor, rather than hiding reset bugs behind reconstruction.
            executor.init_state(initial_state)
            reset_output = torch.cat([executor.run_chunk({'features': inputs[:, :, pos:pos + chunk]})['features']
                                      for pos in range(0, length, chunk)], dim=-1)
            reset_matches = torch.allclose(expected, reset_output, atol=3e-5, rtol=3e-4)
            bad_pieces = []
            for position in range(0, length, chunk):
                bad = inputs[:, :, position:position + chunk]
                for layer in layers:
                    bad = functional.silu(layer(bad))
                bad_pieces.append(bad)
            bad_output = torch.cat(bad_pieces, dim=-1)
            omission_detected = not torch.allclose(expected, bad_output, atol=3e-5, rtol=3e-4)
            analysis_summaries.append(summarize_graph(graph))
            rows.append({'channels': channels, 'hidden': hidden, 'length': length, 'chunk': chunk,
                         'parameters': sum(p.numel() for p in layers.parameters()),
                         'chunks_emitted': len(pieces), 'max_abs_difference': difference.max().item(),
                         'numerical_match': matches, 'atol': 3e-5, 'rtol': 3e-4,
                         'original_ir_executor_matches': ir_matches, 'same_executor_reset_matches': reset_matches,
                         'missing_history_counterexample_detected': omission_detected,
                         'peak_allocated_bytes': torch.cuda.max_memory_allocated() if device == 'cuda' else None})
            assert matches and ir_matches and reset_matches and omission_detected, rows[-1]
            del layers, inputs, expected, actual, pieces, histories, difference, out, joined, value, history
            del executor, graph, ir_output, reset_output, bad_pieces, bad_output, bad

    context = mp.get_context('spawn')
    queue = context.Queue()
    name = 'flashrt_probe_' + uuid.uuid4().hex
    buffer = SharedTensorBuffer(name, (4,), 8, create=True)
    process = context.Process(target=tensor_writer, args=(str(repository), name, queue))
    try:
        process.start()
        process.join(20)
        if process.is_alive():
            raise TimeoutError('Tensor writer did not finish')
        assert process.exitcode == 0
        message = queue.get(timeout=2)
        consumed, values = buffer.read(0, 3)
        assert consumed == 3 and np.array_equal(values, np.arange(12, dtype=np.float32).reshape(3, 4))
        shared_memory_result = {'child_exitcode': process.exitcode, 'child_message': message,
                                'frames_read': consumed, 'values_match': True}
    finally:
        if process.is_alive():
            process.terminate()
            process.join(5)
        if process.is_alive():
            process.kill()
            process.join()
        buffer.unlink()
        buffer.close()
        queue.close()
        queue.join_thread()
    shared_memory_result['child_alive_after_cleanup'] = process.is_alive()
    hashes = {name: hashlib.sha256((repository / name).read_bytes()).hexdigest() for name in
              ['flashrt/layers/audio.py', 'flashrt/channels/shm_channel/tensor_buffer.py',
               'flashrt/ir/graph.py', 'flashrt/ir/executor.py', 'flashrt/ir/analysis.py']}
    result = {'scope': 'Local workload feasibility; not a full app or a skill-treatment effect estimate',
              'device': device, 'gpu': torch.cuda.get_device_name() if device == 'cuda' else None,
              'torch': torch.__version__, 'source_sha256': hashes,
              'class_source_lines': [node.lineno, node.end_lineno],
              'class_loading': 'Exact original class AST; unrelated imports excluded, no class body edits',
              'weights': 'Seeded fixed random initialization, no pretrained quality claim or weights download',
              'audio_cases': rows, 'original_ir_analysis': analysis_summaries,
              'tensor_channel': shared_memory_result}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes((json.dumps(result, indent=2) + '\n').encode('utf-8'))
    print(json.dumps({'device': device, 'audio_cases_passed': len(rows),
                      'largest_peak_allocated_bytes': max(row['peak_allocated_bytes'] or 0 for row in rows),
                      'cross_process_channel_matches': True, 'child_alive_after_cleanup': process.is_alive()}))


if __name__ == '__main__':
    main()
