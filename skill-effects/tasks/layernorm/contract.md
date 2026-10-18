# layernorm

Implement the Model in model.py as ModelNew. Support contiguous float32 CUDA inputs, constructor settings and shapes in the following configurations, including different random seeds, non-default streams and repeated calls. Preserve state_dict compatibility, including randomized affine/linear parameters. Do not modify inputs; previously returned outputs must remain unchanged after later calls.

```json
[
  {
    "batch_size": 16,
    "features": 64,
    "dim1": 256,
    "dim2": 256
  },
  {
    "batch_size": 3,
    "features": 17,
    "dim1": 5,
    "dim2": 33
  },
  {
    "batch_size": 1,
    "features": 1,
    "dim1": 1,
    "dim2": 257
  }
]
```
