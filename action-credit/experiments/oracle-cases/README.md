# Oracle cases

这里暂时不存放案例 JSON。

原因不是缺少 CUDA 父子版本，而是当前公开 AdaExplore 档案缺少本实验真正需要的 token 级生成记录。把 child source 包进一段后补的“模型回复”，再用自定义正则切成 token，会制造并不存在的训练轨迹，因此已经删除。

未来一个可进入这里的真实案例至少需要同时保存：

- 实际模型的原始回复；
- 当次生成所用 tokenizer / tokenizer 版本；
- 真实 token id 与字符 offset；
- parent / child source 及二者对应关系；
- 人工 oracle 决策 span，且 span 坐标必须相对于原始回复；
- 正确性与性能证据；
- 若要与 CUDA-Agent-style GAE 比较，还需真实 Critic value 或逐 token advantage。

早期 experiments/kernelbench-add/ 和 experiments/kernelbench-row-sum/ 是人为构造的 smoke test，可以继续用于测试评测链路，但不能改名后作为“真实 Agent oracle cases”。
