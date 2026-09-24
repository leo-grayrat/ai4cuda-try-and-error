# 从代码定位 Kernel 优化动作

核心问题是：一段 Agent 生成代码里，哪些少量修改真正改变了运行时间？先从父子源码找出优化动作和普通代码，再用已有 KernelBench 正确性与计时入口检查撤销该动作的后果。定位只是待验证的假设；未通过正确性检查的撤销版本没有可解释的性能贡献。早期 2×2 与跨 GPU 工具仍保留供核查，但不是当前研究主线。

## 当前状态

- 代码优先试点：真实 AdaExplore 父子版本已接入源码定位、动作撤销、普通代码对照和本机 KernelBench 重测；具体筛查与限制见[试点记录](results/code-first-pilot.md)。公开档案没有原始生成 token，不能用于 token 学习标签。
- 早期信用分配核查：向量加法、逐行求和的[合成先导实验](results/pilot.md)，以及 MLP 的 2×2 和 TF32 测量检查见[真实轨迹实验](results/adaexplore-mlp.md)。这些不能代替多任务代码定位结论。
- 跨 GPU 工具：仍可保存设备身份和比较排名。目前只有一块 RTX 5060，跨 GPU 实验暂停。

真实候选数据的可用性与评测风险见[公开档案核查](results/source-audit.md)。AdaExplore Level 2/3 的节点关系已审计；早期 2×2 核查只完成一条可独立构造两项修改的真实链，不能代替当前多任务代码定位评测。

## 代码优先试点入口

从档案里选定**明确父子关系**且档案标记两者正确的一条修改。准备命令读取两份源码，按源码位置和语法上下文列出变化，写出父版本、子版本、最多三个待检验优化动作的撤销版本，以及一个普通代码对照。`screen-manifest.json` 只含父子两个版本，适合先核对本机正确性；只有父子都正确再运行完整 `manifest.json`。每个动作初始状态都是 `unverified`；大规模联动修改可能只能作为整体检验。

```powershell
.venv-kb\Scripts\python.exe -m scripts.prepare_credit_case `
  .cache\adaexplore-l2.tar.gz 2_55 30 34 `
  .cache\KernelBench\KernelBench\level2\55_Matmul_MaxPool_Sum_Scale.py `
  runs\cases\example
.venv-kb\Scripts\python.exe lab.py run runs\cases\example\manifest.json runs\example-r0.json `
  --kernelbench-root .cache\KernelBench --evaluator-python .venv-kb\Scripts\python.exe --order-seed 0
.venv-kb\Scripts\python.exe -m scripts.summarize_credit_case `
  runs\cases\example\trace.json runs\example-r0.json --output runs\example-summary.json
```

重复测量时为每轮换 `--order-seed`，并把所有记录一起传给汇总命令。汇总前核对相同候选源码、参考题、评测器版本、工作量和 GPU。数值是“在子版本中撤销动作”相对于原子版本的条件效应；无法把相互依赖的源码行分别认领提速。运行记录保存在忽略目录 `runs/`，不会覆盖已有文件。

新采集的模型轨迹可用 `python -m scripts.record_credit_response <trace.json> <child.py> <generation.json> <output.json>` 追加原始回复、采样 token ID 以及生成时 tokenizer 给出的字符区间。`generation.json` 需包含 `raw_response`、`token_ids` 和 `token_offsets`。只有子源码在回复中完整且唯一出现时才建立源码到 token 的对应；删除行、改写后才应用的源码或无法匹配的回复会明确标为未映射。公开 AdaExplore 档案没有这些数据，不能事后补造。

进入训练的门槛是先在任务留出集比较代码定位与简单差异基线，确认相同阅读预算下关键动作召回、普通代码误报和额外测量成本有稳定优势；否则停在归因实验。当前尚未达到这个门槛。

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

`runs/` 不入库。每次运行生成一份 JSON 和同名 `.assets` 文件夹，快照保存清单及源代码；已有记录不会被覆盖。运行记录保留候选配置、代码哈希、随机种子、设备 UUID、驱动、正确性、每次测量及中位时间。源码哈希在计算前只把 CRLF 换行转换为 LF，以便比较 Windows 与 Linux 上的同一份文件。测量单位是毫秒。`credit` 使用 `-log(中位延迟)` 作为分数，输出 A、B 的单独贡献、交互项 `J(AB)-J(A)-J(B)+J(base)`，以及将交互项各分一半的 Shapley 对数信用；这四份程序必须从**同一基线**构造，且都通过正确性检查。单个样例和一次测量不能支持普遍的因果结论。

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
