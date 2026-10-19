"""Fixed IPC contract. One complete feature sequence in, ordered chunks out."""
import math
import time

import numpy as np

from flashrt.channels.shm_channel.control_buffer import SharedControlBuffer
from flashrt.channels.shm_channel.tensor_buffer import SharedTensorBuffer


def open_buffers(config, create):
    prefix = config['prefix']
    common = dict(create=create, timeout_s=60)
    buffers = {}
    try:
        buffers['control'] = SharedControlBuffer(prefix, **common)
        buffers['input'] = SharedTensorBuffer(prefix + '_input', (config['channels'], config['length']), 1, **common)
        capacity = math.ceil(config['length'] / config['chunk']) + 4
        buffers['output'] = SharedTensorBuffer(prefix + '_output', (config['channels'], config['chunk']), capacity, **common)
        buffers['metadata'] = SharedTensorBuffer(prefix + '_metadata', (3,), capacity, dtype=np.int64, **common)
        return buffers
    except BaseException:
        close_buffers(buffers, unlink=create)
        raise


def close_buffers(buffers, unlink=False):
    for buffer in reversed(list(buffers.values())):
        if unlink:
            buffer.unlink()
        buffer.close()


class FeatureAdapter:
    def __init__(self, config):
        self.config = config
        self.buffers = open_buffers(config, create=False)
        self.consumed = 0
        self.pending = None

    def start(self):
        if not self.buffers['control'].send(1, timeout_s=30):
            raise TimeoutError('Start was not acknowledged')

    def reset(self):
        if self.pending is not None:
            raise RuntimeError('Drain the pending request before reset')
        if not self.buffers['control'].send(3, timeout_s=30):
            raise TimeoutError('Reset was not acknowledged')
        # Returning to opcode 1 makes repeated resets distinguishable using
        # the original level-valued control channel without private fields.
        self.start()

    def submit(self, features):
        if self.pending is not None:
            raise RuntimeError('Only one request may be in flight')
        values = np.asarray(features, dtype=np.float32)
        if values.shape != (self.config['channels'], self.config['length']):
            raise ValueError('Input shape differs from the configured contract')
        request_id = self.buffers['input'].write(values) - 1
        published_ns = time.perf_counter_ns()
        self.pending = {'id': request_id, 'chunk': 0}
        return request_id, published_ns

    def receive(self, timeout_s=30):
        if self.pending is None:
            raise RuntimeError('No request is pending')
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            next_cursor, metadata = self.buffers['metadata'].read(self.consumed, 1)
            if metadata is None:
                time.sleep(0.0002)
                continue
            if next_cursor != self.consumed + 1:
                raise RuntimeError('Output ring overwrote unconsumed metadata')
            request_id, chunk_index, valid = map(int, metadata[0])
            if (request_id, chunk_index) != (self.pending['id'], self.pending['chunk']):
                raise RuntimeError('Request or chunk order differs from the contract')
            expected_valid = min(self.config['chunk'], self.config['length'] - chunk_index * self.config['chunk'])
            if valid != expected_valid or valid <= 0:
                raise RuntimeError('Invalid output length')
            output_cursor, data = self.buffers['output'].read(self.consumed, 1)
            if output_cursor != next_cursor or data is None:
                raise RuntimeError('Output payload was not published before metadata')
            observed_ns = time.perf_counter_ns()
            self.consumed = next_cursor
            self.pending['chunk'] += 1
            if (chunk_index + 1) * self.config['chunk'] >= self.config['length']:
                self.pending = None
            return data[0, :, :valid], observed_ns
        raise TimeoutError('No output chunk arrived before timeout')

    def terminate(self):
        if not self.buffers['control'].send(2, timeout_s=10):
            raise TimeoutError('Termination was not acknowledged')

    def close(self):
        close_buffers(self.buffers)
