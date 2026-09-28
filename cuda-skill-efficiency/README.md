# CUDA agent 的 skill 效率与执行责任

研究问题：在相同 CUDA 开发任务、模型和工具条件下，一份 skill 占用的上下文与它带来的任务收益、后续执行 token 成本是什么关系？只有发现明确瓶颈后，才试验把含义清楚的规则交给权限、流程或检查器。

本目录与 [`action-credit/`](../action-credit/README.md) 是两项正交研究。这里不使用对方的性能信用标签，也不把静态源码档案当成逐轮 agent 轨迹。

## 目前材料与边界

- [冻结前的实验约定](protocol.md)：定义对照、记录字段、可行性门槛和后续研究的启动条件。
- [任务与规则材料清单](materials.md)：列出两项任务、两份本项目提示及其操作信息对应关系。
- [Codex CLI 采集可行性记录](feasibility.md)：保留第一轮试跑、环境差异和暂停原因。
- [OpenCode + DeepSeek V4.1 Flash 试跑记录](opencode-pilot.md)：记录可用的逐步事件、已完成对照和仍未完成的试验。
- [第二轮隔离采集试跑](isolated-pilot.md)：记录隔离与评测修复、两任务三条件的采集门槛结果。
- [封存的正式对照约定](formal-protocol.md)：规定新任务、重复顺序、资源上限和报告口径。
- [首轮正式结果](results/formal-results.md)：18 条运行的质量、token 成本、异常轨迹和第一阶段判断；[JSON 汇总](results/formal-summary.json)可由原始记录重建。
- 首轮选用与本机环境匹配的 CUDA kernel 开发任务。`skills/full.md` 是为这些任务撰写的第一版完整规则；`skills/concise.md` 保留同一批可操作要求。二者是 **本项目实验材料**，不冒称 CUDA Agent 官方 skill 的压缩版。
- [CUDA Agent 的公开 skill](https://github.com/BytedTsinghua-SIA/CUDA-Agent/blob/main/agent_workdir/SKILL.md) 描述 CUDA C++ 扩展与其专属工作目录；[FlashRT 的 AGENTS.md](https://github.com/Infini-AI-Lab/FlashRT/blob/main/AGENTS.md) 描述另一类应用部署。两者可供规则类型审查，只有任务、环境与规则相配时才进入效果对照。
- CUDA Agent 的公开仓库提供示例工作目录；[issue #3](https://github.com/BytedTsinghua-SIA/CUDA-Agent/issues/3) 请求中间轨迹，[issue #18](https://github.com/BytedTsinghua-SIA/CUDA-Agent/issues/18) 请求 agent 框架细节。因此本项目只用自己新采集的逐轮记录回答效率问题。

OpenCode 已能采集逐步 token 用量和工具反馈；两项开发任务通过采集门槛，另有两项封存任务完成首轮重复对照。当前没有证据显示这批任务中的附加通用 skill 带来稳定的质量与成本优势，结果范围和异常见正式报告。18 次正式运行的原始事件、提示、源码版本和评测记录保存在 [`results/traces/`](results/traces/README.md)，可用汇总脚本重新核查；本机 `runs/` 还包含未入库的编译缓存与试跑记录。
