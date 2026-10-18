# 调查记录

## 相关研究

1. Gloaguen 等，*Evaluating AGENTS.md: Are Repository-Level Context Files Helpful for Coding Agents?*，2026。作者研究站点和论文报告，仓库上下文文件会改变探索和测试行为，额外要求也带来推理成本。需要在本研究里分别记录任务质量、工具行动和费用，不能把读了文档当成获得收益。
   - 作者站点：https://www.sri.inf.ethz.ch/publications/gloaguen2026agentsmd
   - 论文：https://arxiv.org/abs/2602.11988
   - 作者代码：https://github.com/eth-sri/agentbench
2. Harada 等，*When Instructions Multiply: Measuring and Estimating LLM Capabilities of Multiple Instructions Following*，2025。ManyIFEval 和 StyleMBPP 分别研究多项文字和编程要求，显示多个要求同时满足的难度。其任务和规则类型与完整 CUDA 开发不同，本研究借用逐条评分和共同规则分析方法。
   - 论文：https://aclanthology.org/2025.findings-emnlp.896/
   - 作者代码：https://github.com/kenoharada/Multiple-Instructions-Following
3. CUDA Agent 真实 skill、示例实现和编译验证工具按提交冻结。skill 同时含硬约束、示例、优化建议、迭代要求、重复强调，不能只按文件 token 长度理解它。
   - 原仓库：https://github.com/BytedTsinghua-SIA/CUDA-Agent
   - 冻结提交：见 `sources/provenance.json`。

以上为本轮回源调查的初步设计依据，论文报告的结果不作为本项目的实测结果。

## 本机核查（2026-10-19）

- 当前 GPU：RTX 5060 Laptop，8 GB；驱动 617.42。
- Windows CUDA Toolkit：12.9；MSVC：14.44.35207。
- `.venv-kb` 和系统 Python 均可导入 PyTorch 2.11.0+cu128 并访问 GPU；`.venv` 未装 torch。
- OpenCode：1.18.12。
- WSL 的 Ubuntu-22.04 和 Ubuntu-D 可启动并看到 GPU，但默认 Python 均没有 torch，PATH 内未找到 nvcc。
- 下一项实测：Windows 原始 CUDA C++ 示例的编译、正确性及 compiled baseline。原版编译器使用 Linux 参数和 `.so` 文件名，需要明确、最小的平台适配。

## 原始研究对象与材料

用户再次明确起点后，重新检查了早期聊天检索结果及 FlashRT 原仓库。FlashRT 的 `AGENTS.md` 当前为 1,749 行，原文和提交哈希已保存在 `sources/FlashRT.*`。GitHub 按该文件路径返回的提交记录只有初始提交，因此当前公开历史尚不能提供其逐条追加过程。文件内容与公开提交历史的可观测事实分开记录；若要推断作者如何逐次写出这些规则，需要额外来源。

这份文件面向 FlashRT 应用部署、IR 和流式性能，不能原样贴到 CUDA Agent 的 kernel 题中。将其作为现实累积文档的内容审查对象，效果测试必须在匹配工作流中进行，或明确标注经领域映射的构造实验。

## 环境实测进展

`runs/environment-01/`：原始例子的 Windows 适配编译遇到工具输出编码错误。该 API 会话的 OEM 解码映射到不可用的 cp1。

`runs/environment-02/`：将编译器输出的解码设为 UTF-8 并替换不可解码字节后，三份原始 C++/CUDA 源码编译成功（20.42 秒），原版正确性验证返回 0（13.55 秒）。原版 profiling 也返回 0，但日志报告 CUPTI 初始化失败，三个计时全为 0；这项计时无效。继续用 CUDA event 测量真实调用，并保持此处失败日志原样。

随后 CUDA event 探针完成三种实现的交错测量，各保存 7 组、每组 100 次完整调用的结果。各组均为正数；eager、compiled 和示例扩展的中位调用时间分别约为 26.92、43.33、12.46 微秒。此处输入只有上游示例的 128 个元素，数值用于验证测量链路；正式任务将另外验证稳定性、形状和真实 kernel 活动记录。可复核的原始样本保存在 `evidence/environment-02/`。
