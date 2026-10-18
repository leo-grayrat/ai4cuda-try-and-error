# gemm_chain

Implement the Model in model.py as ModelNew. Support contiguous float32 CUDA inputs, constructor settings and shapes in the following configurations, including different random seeds, non-default streams and repeated calls. Preserve state_dict compatibility, including randomized affine/linear parameters. Do not modify inputs; previously returned outputs must remain unchanged after later calls.

```json
[
  {
    "batch_size": 1024,
    "in_features": 8192,
    "out_features": 8192,
    "multiplier": 2.0,
    "negative_slope": 0.1
  },
  {
    "batch_size": 77,
    "in_features": 257,
    "out_features": 389,
    "multiplier": 1.3,
    "negative_slope": 0.2
  },
  {
    "batch_size": 1,
    "in_features": 33,
    "out_features": 17,
    "multiplier": -0.7,
    "negative_slope": 0.03
  }
]
```
