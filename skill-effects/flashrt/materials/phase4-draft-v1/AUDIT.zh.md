# Phase 4 整理草稿的来源对照

本稿替换根 AGENTS.md 第 547—1181 行，其余原文与三份分目录说明保持原样。英文要求合并为六组，供下一步逐项复核；尚未冻结为正式条件。

| 原文位置 | 整理稿位置 | 保留的核心条件 |
| --- | --- | --- |
| 547—593 | Exploration and queue；Work scope and persistence | 十种为下限、修 bug 不算新方案、覆盖多种策略及组合、清空队列、无会话期限 |
| 594—618 | Work scope and persistence | 实现和调试成本不得成为轮换或跳过理由 |
| 619—638 | Work scope and persistence | 只读文件复制到可写后端，注明来源，保留数值对照 |
| 639—768 | Terminal statuses and removal evidence | 三种终态、严格环境阻塞条件、真实尝试、具体证据与代理自写错误责任 |
| 769—851 | Investigating silent or slow workers | 静默不等于死锁、进程与缓存诊断、单进程对照、等待真实进展、逐个尝试绕过路径 |
| 852—868 | Work scope and persistence | 日志、README 与最终状态禁止未来会话交接说法及其改写 |
| 869—902 | Terminal statuses and removal evidence | 保留指定质疑段落原句、分类证据要求和状态纠正 |
| 903—927 | Exploration and queue | 初始队列、每项字段、测量间更新与新增假设 |
| 928—972 | Terminal statuses and removal evidence | 唯一移除理由、修 IR 并重验、变换合法性失败测试 |
| 973—990 | Exploration and queue | 每个方案独立实现和测量，关联或缩放版本不能替代 |
| 991—1047 | Exploration and queue | 后续想法同次编辑入队、影响排序、早期胜利继续探索、先独立后组合 |
| 1048—1066 | Exploration and queue | 不同指标各胜且互不支配时，关闭独立项前强制加入组合 |
| 1067—1112 | Work scope and persistence；Exploration and queue | 查已有抽象、无支持也需实现、不可按预测跳过、引用其他方案预测须入队 |
| 1113—1160 | Each iteration, in order | 决定、实现、正确性、性能、迭代的顺序；进下一项前双证据门槛 |
| 1161—1181 | Each iteration, in order | 参考后端为固定基线、全部终态才能结束、needs-followup 必须先测量 |

本次只处理密集重复段，尚未整理整个说明家族。表中的范围覆盖全部替换区间，但范围覆盖只方便追溯；逐项语义保留仍需核对。

第二轮核对修复了两处具体变化：根文件完成检查引用的六个原始标题在初稿中被重组标题替代，现已恢复，构建时检查这些入口；初稿将“日志不得出现几种指定措辞”改成了“不得用这些措辞推迟”，弱化了原限制，现恢复原限制强度。其他重复表达仍按表中位置合并。

需要保留并记录的原文张力：迭代步骤要求确认 PASS 后再晋升，同时终态和进入下一项门槛允许保存 FAIL；原文还要求失败方案有实测 benchmark。草稿均予保留，后续评测须分别记录“已测量”和“可晋升”，不能把两者混成一个成功标记。

原说明要求穷尽新增队列，研究试验仍需统一外部预算。预算中止要单独记录当时的未完成内容，不能通过改写终态来满足完成率。
