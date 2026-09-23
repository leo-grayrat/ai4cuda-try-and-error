# Kernel 优化实验：信用分配与跨 GPU 排名

本仓库只负责**记录候选版本、分析 2×2 反事实修改、比较跨 GPU 排名**。正确性和运行时间由已有评测器负责；正式 KernelBench 实验调用其原有 `eval_kernel_against_ref`。这里的 Numba AXPY 题仅用于确认记录与分析链路能运行，不代表 Agent 优化或 kernel 特有的学习现象。

## 当前状态

- 第 1 项（性能信用分配）：已用上游 KernelBench 评测器在向量加法、逐行求和两道题上分别完成 base、A、B、AB 的 CUDA 先导实验。每题三轮，每轮每候选 30 次原始计时；结果见 [先导实验记录](results/pilot.md)。研究阶段还需要把候选换成真实 Agent 轨迹中可独立重放的修改，并扩展到更多题型。
- 第 2 项（跨 GPU 泛化）：能保存设备身份和候选源码哈希，拒绝拿同一 GPU 或不同代码伪装成跨设备对比；第二台 GPU 可用时，在其上运行同一清单并执行 `compare`。目前只有一块 RTX 5060，因此没有跨 GPU 结果。

真实候选数据的可用性与评测风险见[公开档案核查](results/source-audit.md)。现有 KernelBench 先导实验只能说明本地流程跑通；正式研究需增加更广的输入与初始化检查，并复核异常加速。

## 本机最小运行

Windows PowerShell 中，使用 NVIDIA 维护的 `numba-cuda`，不要使用系统环境里旧版 Numba 内置 CUDA 后端：

```powershell
$env:UV_CACHE_DIR = (Join-Path (Get-Location) '.cache\uv')
uv venv --python 3.13 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements-smoke.txt
.venv\Scripts\python.exe lab.py run experiments/axpy-2x2.json runs/axpy-local.json
.venv\Scripts\python.exe lab.py credit runs/axpy-local.json
.venv\Scripts\python.exe -m pytest -q
```

`runs/` 不入库。每次运行生成一份 JSON 和同名 `.assets` 文件夹，快照保存清单及源代码；已有记录不会被覆盖。运行记录保留候选配置、代码哈希、随机种子、设备 UUID、驱动、正确性、每次测量及中位时间。测量单位是毫秒。`credit` 使用 `-log(中位延迟)` 作为分数，输出 A、B 的单独贡献和 `J(AB)-J(A)-J(B)+J(base)`；这四份程序必须从**同一基线**构造，且都通过正确性检查。单个样例和一次测量不能支持普遍的因果结论。

要从快照重新运行 Numba 样例，显式指定保存的评测器代码：

```powershell
.venv\Scripts\python.exe lab.py run runs/axpy-local.assets/manifest.json runs/axpy-replay.json `
  --evaluator-source runs/axpy-local.assets/evaluator.py
```

## 接入 KernelBench

采用[上游 KernelBench](https://github.com/ScalingIntelligence/KernelBench) 的题目与评测，不在此重写正确性和计时。建议在其支持的 Python 3.10 CUDA 环境里安装上游项目。`lab.py run` 的 `kernelbench` 模式会逐个调用上游 `eval_kernel_against_ref`，把结果合并为本仓库的运行记录。

准备一个清单，`evaluator` 设为 `kernelbench`；`reference_path` 是包含 `Model`、`get_init_inputs`、`get_inputs` 的 PyTorch 原题文件；四个 `source_path` 分别是包含 `ModelNew` 的候选文件。路径相对清单。示例结构见 `experiments/kernelbench-template.json`。A、B、AB 要由实验者检查为真正对应两项可组合修改的代码；工具不自动拼接补丁。

```powershell
<KernelBench环境的python> lab.py run experiments\your-case.json runs\gpu-a.json `
  --kernelbench-root .cache\KernelBench `
  --evaluator-python <KernelBench环境的python>
<KernelBench环境的python> lab.py credit runs\gpu-a.json
```

上游仓库在建立此脚手架时的版本是 `423217d9fda91e0c2d67e4a43bf62f96f6d104f1`。`kernelbench_adapter.py` 调用上游正确性检查和 CUDA Event 计时，并在上游计算统计量前截取原始计时样本。这个截取点依赖当前上游版本；换版本后应重新验证。

本机已在 Python 3.13 + PyTorch 2.11 + CuPy 14.2 上实测 `experiments/kernelbench-add/manifest.json`。上游 `pyproject.toml` 声明 Python 3.10；这里通过源码导入进行验证，不能据此推断上游完整项目在 Python 3.13 获得支持。复现本机先导实验还需使评测 Python 能导入 `kernelbench.eval` 和 `cupy`，并把 `CUPY_CACHE_DIR` 指向可写目录。启动示例：

本机评测环境以已有的 CUDA 版 PyTorch 为基础，在项目内虚拟环境加入 `requirements-kernelbench-pilot.txt`，并从上游仓库 `src` 导入 KernelBench。无需为此重新安装或训练 Agent。

```powershell
$env:CUPY_CACHE_DIR = (Join-Path (Get-Location) '.cache\cupy')
.venv-kb\Scripts\python.exe lab.py run experiments/kernelbench-add/manifest.json runs/gpu-a.json `
  --kernelbench-root .cache/KernelBench --evaluator-python .venv-kb\Scripts\python.exe
.venv-kb\Scripts\python.exe lab.py credit runs/gpu-a.json
```

重复实验可加 `--order-seed 1`、`--order-seed 2` 改变四个候选的评测顺序，并用 `lab.py credit-series <多份运行记录>` 汇总每轮贡献及轮间波动。

新运行会把 KernelBench 适配器也保存到 `.assets/evaluator.py`。从同一份快照重跑时指定它，且继续使用记录对应的上游 KernelBench 修订：

```powershell
.venv-kb\Scripts\python.exe lab.py run runs/gpu-a.assets/manifest.json runs/gpu-a-replay.json `
  --kernelbench-root .cache/KernelBench --evaluator-python .venv-kb\Scripts\python.exe `
  --evaluator-source runs/gpu-a.assets/evaluator.py
```

[KernelBlaster](https://github.com/NVlabs/KernelBlaster) 已在每个 `trajectory_*` 目录保存逐步 `.cu` 候选和日志，适合用来**寻找**真实的优化修改。但它使用 `init.cu`、`driver.cpp` 格式；上面的 KernelBench Python 适配器接收 `ModelNew` 文件，不能直接把这些 `.cu` 文件交给它。正式实验可选择沿用 KernelBlaster 自己的编译与评测环境重测，或先在 KernelBench 格式中构造同一组受控候选。连续两步修改也不能自动当作独立的 A、B：必须额外构造并验证“只做 B”的版本。

## 第二台 GPU 可用时

把**相同的清单和候选源码**带到第二台机器，在那边运行 `lab.py run` 得到 `gpu-b.json`，然后执行：

```powershell
python lab.py compare runs\gpu-a.json runs\gpu-b.json
```

输出候选排名相关性、两两排名翻转率、A 卡冠军在 B 卡的名次。这个比较要求实验 ID、工作负载配置及每个候选源码哈希一致。绝对延迟不能单凭不同 GPU 的数值直接解释为“迁移能力”；后续还需在多题目、多次重测的基础上分析。
