# 研究对象纠正与旧实验停止

2026-10-19，回读用户提供的两份原始思考：

- D:/File/Markdown/Retire the Abstractions 等四篇文章.md
- D:/File/Markdown/haozhi reply.md

其中“论文中的图论/结构化 IR 方法，与仓库内长篇试错告诫的贡献如何区分”“几千行文档的遗漏、定量压倒定性、优化及拆分”的直接研究对象明确是 FlashRT。CUDAAgent 对应另一条关于训练、低概率 token 与程序级信用分配的讨论；权限与检查器的动机另与 HFSV 讨论相连。

本项目早在本地时间 06:00 已下载 FlashRT 的 1,749 行 AGENTS.md，并保存于 sources/FlashRT.AGENTS.md；来源见 sources/FlashRT.provenance.json。随后选择 376 行、3,392 token 的 CUDAAgent 文档作为正式主体，未解决研究对象与动机的冲突。用户一度记混名称，执行者未回查已有原始思考，反而将错误替换写成“恢复主线”。此前声称正式方向已对齐的判断撤回。

protocol/v1 和 formal-series-01 保留为 CUDAAgent 文档处理的既有记录，不再作为上述 FlashRT 研究问题的正式主实验。已完成七个槽位，第八个槽位运行时中断。正式控制器 exec 会话 63522 与后处理会话 90044 已终止，另经进程查询确认未剩对应控制器、后处理队列或已记录的第八轮子进程。旧 campaign/state 文件是停止前快照，不应据其 running/waiting 字段自动续跑。

没有启动替代实验。后续应先据原文核对 FlashRT 实际指令层级、适用任务、图结构要求及确定性检查的边界，再判断如何检验文档整理、拆分和执行机制。不能把 FlashRT 文档直接移植到不匹配的 CUDAAgent 小任务，也不能再构造一遍长篇追加过程来替代真实对象。
