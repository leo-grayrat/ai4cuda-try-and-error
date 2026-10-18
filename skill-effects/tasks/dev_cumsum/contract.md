# dev_cumsum

Implement the Model in model.py as ModelNew. Support contiguous float32 CUDA inputs, constructor settings and shapes in the following configurations, including different random seeds, non-default streams and repeated calls. Preserve state_dict compatibility, including randomized affine/linear parameters. Do not modify inputs; previously returned outputs must remain unchanged after later calls.

```json
[
  {
    "batch_size": 257,
    "input_shape": [
      4099
    ],
    "dim": 1
  },
  {
    "batch_size": 17,
    "input_shape": [
      513
    ],
    "dim": 1
  }
]
```
