# 本地工具调用采集补充（2026-10-18）

本次接续的要求是把本地工具调用采集与 DeepSeek 正文采集的互补关系写进论文。核对从已有的本地响应开始：将生成 token 按顺序拼回文本，解析工具参数，再与同次运行保存的工具事件逐项比较。

三个目录共保留 195 条响应、26,529 个生成 token。其中 33 条响应含 40 个可解析的代码参数，参数内覆盖 3,162 个生成 token，全部带有有限的 logprob。40 个参数字符串均与对应 write 或 edit 事件的输入一致。核对表、原文件行号、代码内容、token 索引和概率保存在 `../local_tool_evidence/audit.json`，中文说明见 `../local_tool_evidence/REPORT.zh.md`。

旧搜索在单个 token 中查找完整的 triton、def run 等字符串，因此复现得到零匹配。重建完整文本后，同一批记录中的代码参数可以直接解析。本次据此纠正了稿件中将本地路线整体记为失败的说明。

论文的更新分布在以下位置：引言的采集约束、相关工作的概率可得性、方法中的采集接口和模型选择、结果 4.8 节，以及讨论 5.7 节。方法部分说明如何读取本地工具参数，结果部分列出目录级核对数字和两例原始记录，讨论部分说明两条路线分别覆盖正文输出和工具参数输出。

术语沿用已有稿件：DeepSeek API、Qwen3-4B-AWQ、vLLM、reply-body collection（正文采集）、tool-call collection（工具调用采集）。正文采集负责 Families A/B 的主要实验；本地记录提供工具参数代码概率的补充证据。

重建命令：

```powershell
python lowprob-attribution/scripts/audit_local_tool_logprobs.py
python lowprob-attribution/scripts/build_v2_paper.py
```
