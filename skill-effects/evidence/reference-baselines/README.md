这里是第一版 12 组参考基线，保留作开发记录。其中 Sigmoid 和 Softmax 的输入尺寸已被修改，Softmax 的 4,096 列不能代表当前保留题的 393,216 列归约。

当前索引见上级 `current-reference-baselines.json`，宽度修正的数据见 `reference-baselines-wide/`，补充分布验证见 `softmax-distribution-validation/`。LayerNorm 与 GEMM 链的原始尺寸未变，其数据仍在当前索引中。
