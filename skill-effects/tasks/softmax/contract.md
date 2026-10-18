# softmax

Implement the Model in model.py as ModelNew. Support contiguous float32 CUDA inputs, constructor settings and shapes in the following configurations, including different random seeds, non-default streams and repeated calls. Preserve state_dict compatibility, including randomized affine/linear parameters. Do not modify inputs; previously returned outputs must remain unchanged after later calls.

```json
[
  {
    "batch_size": 128,
    "dim": 393216
  },
  {
    "batch_size": 17,
    "dim": 393217
  },
  {
    "batch_size": 7,
    "dim": 33
  }
]
```

Additional softmax distribution checks: row sums must match 1 with atol=rtol=1e-4, and each row's L1 error versus the reference must be <=1e-3. Elementwise atol=rtol=1e-2 alone is insufficient for very wide probability vectors.
