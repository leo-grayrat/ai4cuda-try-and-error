"""Strictly sequential reference worker for the two-branch task."""
import argparse
import json
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repository', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repository.resolve()))
    import numpy as np
    import torch
    from branch_features.adapter import open_buffers, close_buffers
    from branch_features.model import ReferencePipeline

    config = json.loads(args.config.read_text(encoding='utf-8'))
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    pipeline = ReferencePipeline(args.repository, config)
    buffers = open_buffers(config, create=True)
    consumed, last_command, active = 0, 0, False
    try:
        print(json.dumps({'ready': True, 'device': config['device'], 'task': 'branch_features'}), flush=True)
        while True:
            opcode = buffers['control'].recv()
            if opcode != last_command:
                last_command = opcode
                if opcode == 2:
                    buffers['control'].commit()
                    break
                if opcode == 3:
                    pipeline.reset()
                    active = False
                    buffers['control'].commit()
                elif opcode == 1:
                    active = True
                    buffers['control'].commit()
                elif opcode != 0:
                    raise ValueError(f'Unknown control opcode: {opcode}')
            if not active:
                time.sleep(0.0002)
                continue
            next_cursor, features = buffers['input'].read(consumed, 1)
            if features is None:
                time.sleep(0.0002)
                continue
            if next_cursor != consumed + 1:
                raise RuntimeError('Input overwritten before processing')
            request_id, consumed = consumed, next_cursor
            for index, position in enumerate(range(0, config['length'], config['chunk'])):
                output = pipeline.step(features[0, :, position:position + config['chunk']])
                padded = np.zeros((config['channels'], config['chunk']), dtype=np.float32)
                padded[:, :output.shape[-1]] = output
                buffers['output'].write(padded)
                buffers['metadata'].write(np.array([request_id, index, output.shape[-1]], dtype=np.int64))
    finally:
        close_buffers(buffers, unlink=True)
        print(json.dumps({'closed': True}), flush=True)


if __name__ == '__main__':
    main()
