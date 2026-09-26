# 真实 Agent 轨迹的首个 2×2 性能归因（2026-09-24）

## 对象与构造

[AdaExplore Level 3 公开轨迹](https://huggingface.co/datasets/VanishD/AdaExplore_Traces)中，任务 `3_1` 的节点 `7 → 8 → 9` 有显式父节点，归档评测均标为正确，且三次记录均在原环境的设备编号 1 上。归档运行时间约为 9.83、9.24、4.77 ms；这些数字仅用于选样，不与本机数字混算。

- `base` = 节点 7：MLP 的两处激活使用自定义 Triton 原地 ReLU。
- `A` = 节点 8：改用 PyTorch `out.relu_()`。
- `B` = 在节点 7 上单独加入节点 9 的两个 TF32 后端开关；这是本实验补造的反事实版本，不是 Agent 轨迹原节点。
- `AB` = 节点 9：A 与 TF32 开关并用。构造脚本检查“节点 8 + B”与归档节点 9 的可执行 AST 一致。

[`prepare_adaexplore_mlp.py`](../scripts/prepare_adaexplore_mlp.py) 校验归档 SHA-256、父节点、原评测正确性和组合代码；候选源码与原归档保存在忽略版本控制的 `.cache/`。原任务完整保留：批量 128、输入和隐藏层宽度 16384、两个隐藏层、输出宽度 8192。

## 本机重测

设备为 RTX 5060 Laptop GPU；KernelBench 修订 `423217d9fda91e0c2d67e4a43bf62f96f6d104f1`，Python 3.13.5、PyTorch 2.11.0+cu128、Windows Triton 3.8.0.post28。使用上游正确性检查（每候选每轮 5 个输入）和 CUDA Event 计时（每候选每轮 30 次），四候选五轮改变评测顺序，合计 600 条原始计时样本。20 次候选评测均为 `5/5` 正确。每个候选由独立进程评测，记录保存在忽略版本控制的 `runs/adaexplore-mlp-{0..4}.json`，同名 `.assets` 内保存源码和评测器快照。

| 候选 | 五轮中位延迟的均值 | 相对 base 的平均加速比 |
|---|---:|---:|
| base | 24.802 ms | 1.000× |
| A | 23.573 ms | 1.052× |
| B | 12.807 ms | 1.937× |
| AB | 12.686 ms | 1.955× |

用 `J(K) = -log(该轮中位延迟)` 定义收益。五轮的均值：A 的单独贡献 `0.0508`，B 的单独贡献 `0.6609`，交互项 `J(AB)-J(A)-J(B)+J(base) = -0.0413`。交互在五轮中均为负，范围 `[-0.0527, -0.0319]`。把交互项各分一半的 Shapley 对数信用为 A `0.0302`、B `0.6403`；这种均分是明确的分摊约定，不代表能识别唯一的物理原因。五轮均观察到 A、B、AB 均快于 base，且 AB 略快于 B。

## 正确性补查与解释边界

TF32 开关是进程全局状态：上游评测在创建候选后运行参考模型，所以 B/AB 的参考模型也会受开关影响。为排除这种共同改变掩盖误差，用 [`check_fixed_reference.py`](../scripts/check_fixed_reference.py) 额外让原始参考模型强制关闭 TF32、B/AB 开启 TF32，再以相同初始化和五个输入比较。B、AB 各五次都满足 KernelBench 的 `atol=rtol=1e-2`；最大绝对差在约 `1.26e-4` 至 `1.32e-4`。原始逐次结果保存在 `.cache/adaexplore-mlp/fixed-{B,AB}.json`。

这是一道 MLP 在一张 GPU 上的受控案例。主要收益来自切换矩阵乘法的精度/后端选项，不能直接解释为 Triton ReLU 内核本身的优化效果，更不能推广到其他题目或 GPU。后续要在不同算子和输入规模上重复同样的父链、独立反事实及固定参考检查，才能研究性能信用分配的规律。

## 复现

将 Level 3 归档放在 `.cache/adaexplore-l3.tar.gz`，KernelBench 放在 `.cache/KernelBench`；归档校验值为 `602732f6132b7a93f9e4a9ce572b303321e60e89542591436003f0dbd0e66ee5`。

```powershell
.venv-kb\Scripts\python.exe scripts\prepare_adaexplore_mlp.py `
  .cache\adaexplore-l3.tar.gz .cache\KernelBench\KernelBench\level3\1_MLP.py .cache\adaexplore-mlp
$env:TRITON_CACHE_DIR = (Join-Path (Get-Location) '.cache\triton')
foreach ($seed in 0,1,2,3,4) {
  .venv-kb\Scripts\python.exe lab.py run .cache\adaexplore-mlp\manifest.json `
    "runs\adaexplore-mlp-$seed.json" --kernelbench-root .cache\KernelBench `
    --evaluator-python .venv-kb\Scripts\python.exe --order-seed $seed
}
.venv-kb\Scripts\python.exe lab.py credit-series (0..4 | ForEach-Object { "runs\adaexplore-mlp-$_.json" })
```

固定参考检查分别对 B 和 AB 执行 `scripts/check_fixed_reference.py <reference.py> <candidate.py>`。已有运行记录不会被覆盖；复现时使用新的输出文件名。
