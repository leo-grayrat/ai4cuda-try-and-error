# AI4CUDA 研究试验仓库

本仓库包含两条独立的研究线：

- [优化动作定位与性能信用分配](action-credit/README.md)：原有代码、实验与结果已整体归档到 `action-credit/`。
- [CUDA agent 的 skill 效率与执行责任](cuda-skill-efficiency/README.md)：新研究的任务、规则、轨迹与实验约定单独维护。

两条方向迄今做过什么、为什么都没有完成原始验证，见 [阶段性复盘](RESEARCH_RETROSPECTIVE.md)。第一条线后续可能的 `program transformation` 表示方向见 [`research-notes/program-transformation-action-space.md`](research-notes/program-transformation-action-space.md)。

**简而言之AI牛大了，我只是做简单的探索性试跑实验都死掉了，而且都是中途扎进技术细节一去不复返，任务偏移到姥姥家了，最终成果只能说是没用，无论是证实还是证伪，最后还给我拉一堆AI防御性文案**

旧研究已有的本机虚拟环境 `.venv*`、下载缓存 `.cache/` 和原始记录 `runs/` 留在仓库根目录，均不入库。目录迁移不改变这些历史文件；旧研究的新运行可在其子目录下生成自己的 `runs/`。
