# 正式实验原始轨迹

本目录保存首轮 18 次正式运行的原始事件、完整提示、运行元数据、源码版本、最终候选和评测输出。每个 `formal-r*` 子目录对应一次运行；`manifest.json` 记录文件的字节数与 SHA-256，便于核查复制结果。

文件来自本机 `cuda-skill-efficiency/runs/formal-r*`，仅排除了 Triton 编译缓存、Python 字节码与缓存目录。这些缓存不是模型轨迹或评测证据。本机 `runs/` 仍保留原样，不因入库而删除。

从仓库中的轨迹重新生成汇总：在 `cuda-skill-efficiency/` 目录运行 `python summarize_formal.py`。脚本核查任务、规则、评测与事件计数，并写出 `results/formal-summary.json`。
