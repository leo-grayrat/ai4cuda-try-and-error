这个方向值得认真拆，因为我刚才那段话其实把三个层次揉在了一起：**“程序变换是一种更好的表示”**、**“搜索时先决定做什么再写代码”**、**“RL 里把变换而不是 token 当 action”**。它们有关，但不是一回事。真正有研究价值的地方，首先得把第一层坐实，后两层才有资格往上长。

### 先把核心命题说得更精确

现在大多数 CUDA 生成 Agent 的基本动作，形式上仍然是：

```math
a_t = \text{下一个 token}
```

一次完整的“优化尝试”其实是几千个这样的动作拼成一份程序，然后编译、验证、测速，才拿到反馈。

但人看 CUDA 优化时并不是这么理解的。比如某个 child 相比 parent 做了这样一件事：

> 每个线程从处理 1 个元素改成处理 2 个元素。

真正的程序改动可能分散在两个地方：

```cpp id="mfdwil"
int j = i + stride;
if (j < n) out[j] = ...
```

同时 host 端还得把 block 数从大约

```math
N/T
```

改成

```math
N/(2T).
```

从文本 diff 看，这是两块相距很远的代码；从程序优化角度看，它们却是 **一个决定**：

```math
\text{work-per-thread}:1\rightarrow2
```

甚至缺了其中任何一块，整个动作都没有正确实现。

这时候一个比较自然的数学对象其实是：

```math
a=(\tau,\ell,\theta)
```

其中 $`\tau`$ 是变换类型，$`\ell`$ 是作用位置，$`\theta`$ 是参数。上面这个例子可以粗略写成：

```math
a=
(\text{increase-work-per-thread},
\text{elementwise kernel},
k=2).
```

然后真正的 CUDA 代码只是这个 action 的一种 **realization**：

```math
a \longrightarrow \Delta P.
```

关键就是这里的多对多关系：

```math
\boxed{
\text{一个语义 action}
\rightarrow
\text{很多 token、很多代码位置}
}
```

同时也可能：

```math
\boxed{
\text{完全不同的代码写法}
\rightarrow
\text{同一个语义 action}
}
```

这才是我觉得它比原来 semantic-credit 想法更根本的原因。

原来我们还在问：

> 一次优化散落在 40 个 token 上，到底该把 reward 给哪些 token？

现在可以反问：

> **为什么训练系统非得把这 40 个 token 当成 40 个独立 action？它们本来就是一个程序决策的实现。**

---

### 但“优化动作”绝不能只是几个漂亮的中文标签

这里很容易又掉进另一个坑。

如果 action space 只是：

> 融合  
> tiling  
> vectorization  
> memory optimization  
> thread optimization

那几乎没用。因为“做 tiling”离实际 CUDA 代码还差十万八千里。

真正有意义的 action 必须是 **参数化的程序变换**。

比如不是：

```math
\text{tile}
```

而更接近：

```math
\text{tile}(
\text{reduction axis},
128,
\text{map to block/warp}
)
```

不是：

```math
\text{fuse}
```

而是：

```math
\text{fuse}(
\text{producer}=A,
\text{consumer}=B,
\text{materialization removed}
)
```

不是：

```math
\text{change launch config}
```

而是：

```math
\text{set blockDim}=256.
```

这里其实和传统编译优化、TVM/Halide 那种 schedule space 有一点血缘关系。编译器本来就知道“程序文本”和“优化变换”不是同一个层次。

所以这件事情 **本身绝不是前无古人的概念**。如果最后只是：

> “我们发现 compiler pass 比 token 更语义化。”

那没论文价值。

真正可能有意思的是：

> **对于 LLM 驱动的 CUDA 优化，把搜索/学习界面从整段代码生成提升到参数化 transformation，是否真的显著改变搜索效率和学习效率？**

这个才是我们需要证明的。

---

### 这又恰好解释了我们之前碰到的“代码定位”为什么那么难

回头看之前 `code_credit.py` 那堆 regex，其实我们一直隐隐在试图从代码里回答：

> 哪几行代码是优化？

但这个问题本身可能就问歪了。

一个优化 action 完全可以横跨：

- kernel body；
- shared memory 声明；
- host launch 参数；
- index calculation。

所以你很难找到一段连续的“优化代码”。

更合理的对象可能是：

```math
a
\rightarrow
\{e_1,e_2,e_3,\ldots\}
```

一个 action 对应一组 distributed edits。

这就把我们以前那个很烦的问题解释通了：

> 为什么简单 locator 总是在漏？

因为我们试图从 **文本局部性** 恢复 **程序决策单位**，而两者本来就不是一一对应。

所以这个方向和之前 semantic credit 其实不是完全割裂的。

它像是在说：

> 我们之前想把 token 重新归组以后再分 credit；  
> 更进一步，干脆让“组”直接成为 action。

这样 credit assignment 问题会被大幅简化：

```math
\text{选择 transformation}
\rightarrow
\text{实现}
\rightarrow
\text{测速}
```

性能反馈天然就可以记在 transformation 上。

当然，如果连续做了 $`A,B,C`$ 三个 transformation，仍然会有高层 credit assignment 问题。但至少问题从：

```math
3000\text{ 个 token 谁负责}
```

降成了：

```math
3\text{ 个程序决策谁负责}.
```

这不是彻底消灭 credit assignment，但问题规模和语义性完全不同。

---

### 不过我认为现在最重要的不是马上做“两阶段 Agent”

这里有一个非常危险的伪阳性：

> 让 LLM “先想优化方案，再写代码”，效果比直接写代码好。

这根本不足以证明 transformation action space 有价值。

因为它也可能只是普通的：

```math
\text{plan first}
```

或者多给了模型一些 reasoning token。

所以如果以后真做实验，至少得区分四种东西。我只列这一组，因为这正好能把问题拆干净：

1. **Direct**：直接生成优化后的完整 kernel。
2. **Free-plan**：先自由文本说准备怎么优化，再写 kernel。
3. **Structured-action**：必须先从结构化 transformation space 中选出参数化动作，再据此实现。
4. **Oracle-action**：人直接告诉它正确的 transformation，只测试“已知做什么以后，代码能不能实现出来”。

这四个实验其实回答四个完全不同的问题。

如果：

```math
\text{Free-plan}\approx\text{Structured-action}>\text{Direct}
```

那我们大概率只是证明：

> 先规划比较好。

没什么意思。

如果：

```math
\text{Structured-action}>\text{Free-plan}>\text{Direct}
```

才开始说明：

> **结构化 action 本身提供了额外 inductive bias。**

如果：

```math
\text{Oracle-action}\gg\text{Structured-action}
```

则说明 action abstraction 很有用，但是：

> **最大困难在于选对 transformation。**

如果连：

```math
\text{Oracle-action}\approx\text{Direct}
```

那这个方向基本就可以杀掉了——连“做什么”都告诉模型了，依旧没提高成功率，说明 strategy/realization 分离根本没有帮忙。

这其实是个非常漂亮的 falsification structure。

---

### AdaExplore 那些 parent→child 轨迹真正能帮我们什么？

这里也需要纠正我刚才略显粗糙的说法。

它们不能直接拿来证明：

> “这就是正确的 transformation action space。”

但是它们非常适合回答一个更前置的问题：

```math
\boxed{
\text{成功的 CUDA 优化轨迹到底能不能被少量重复出现的程序变换解释？}
}
```

这件事本身就值得先看。

假设我们人工读 30～50 个真实 successful parent→child edge。

如果最后发现，每个 child 都是一套完全独一无二的重构：

> 这个改了算法，  
> 那个重写数据结构，  
> 下一个直接调用新库，  
> 再下一个整个 kernel 架构推翻重来，

根本聚不成稳定 transformation vocabulary，那么：

```math
\text{program-transformation action space}
```

可能就是个幻想。

反过来，如果我们发现大量 edge 能不断落回一些重复模式：

```math
A_1,A_2,\ldots,A_{12}
```

比如不同任务里不断出现：

> 增加每线程工作量；  
> 消除中间 materialization；  
> 合并两个 kernel；  
> 改 shared-memory tiling；  
> warp reduction；  
> vectorized load/store；  
> 改 block mapping；

而且 **同一个 action 的代码实现差异很大**，那就出现了一个很强的现象：

```math
\text{code diff entropy 很高}
```

但：

```math
\text{transformation entropy 较低}.
```

这才是“action abstraction”最值得研究的经验基础。

甚至这里可以定义一个很直观的问题：

```math
\phi:\Delta P\rightarrow a
```

我们能不能找到一个足够简单的 $`\phi`$，让大量成功程序变换被压缩到一个小 action vocabulary？

注意，我这里说的仍然是 **人工标注**。

不写 classifier。

不写 locator。

不写 AST pipeline。

就我们自己坐在那里看几十个。

因为此时真正需要知道的是：

> **这种结构到底存在不存在。**

不是：

> “机器能不能自动把它识别出来。”

后者是另一个研究问题。

---

### 这里还有一个非常重要的数据泄漏问题

如果我们已经看过某个 parent→child，然后人工知道：

> “这题正确动作是 fuse。”

随后在同一个 parent 上对模型说：

> “请选择一个 action，再实现。”

然后它选到了 fuse，这不能说明什么——因为如果我们的 action catalog 或 prompt 设计是根据这题调出来的，很容易暗含答案。

所以真正实验应该是：

```math
\text{一些 trajectory}
\rightarrow
\text{建立 transformation vocabulary}
```

然后 **冻结它**。

再去新的 task 上测试：

```math
P_{\text{held-out}}
\rightarrow
\text{选择 action}
\rightarrow
\text{实现}
\rightarrow
\text{测速}.
```

这里真正想看的是：

> 从过去成功优化中抽象出来的 action space，能不能帮助新的 kernel？

否则就只是把答案换了个名字喂回模型。

---

### action 的粒度会是这个方向最难的理论问题之一

这点我现在觉得特别关键。

太粗：

```math
\text{memory optimization}
```

没意义。

太细：

```math
\text{把第 172 行的 i 改成 i+stride}
```

又退化成 code diff。

真正有价值的 action 应该处在中间：

```math
\boxed{
\text{比 token/code edit 抽象}
\quad\land\quad
\text{又足够具体到能约束实现}
}
```

这很像找一个“自然坐标系”。

比如：

> “一线程处理两个元素”

我觉得是一个不错的 action。

因为它：

- 有明确语义；
- 会影响硬件执行；
- 能跨不同 kernel 重复出现；
- 会要求多个代码位置协调变化；
- 同时又比“提高 ILP”具体很多。

而：

> “提高 ILP”

就太抽象了。

所以最终可能不是 10 个名词标签，而是一个很小的 typed DSL：

```math
\text{IncreaseWorkPerThread}(axis,k)
```

```math
\text{Tile}(axis,size,mapping)
```

```math
\text{Fuse}(producer,consumer)
```

```math
\text{EliminateMaterialization}(site)
```

```math
\text{Vectorize}(axis,width)
```

……

但这是 **如果现象存在以后** 才值得形式化。

现在别写 DSL。

否则又会变成我们的老朋友：

> 一晚上设计 47 个 action type，写 schema，写 validator，写 parser，然后研究问题一点没动。

---

### 它跟普通“让 Agent 多想一步”最大的区别在哪里？

这是我觉得必须守住的界线。

普通 plan-first 是：

```math
P
\rightarrow
\text{自然语言 reasoning}
\rightarrow
P'
```

中间那个 reasoning 没有明确可学习的结构。

我们想讨论的东西应该是：

```math
P_t
\xrightarrow{a_t}
P_{t+1}
```

其中 $`a_t`$ 是一个 **可复用、可比较、可统计、可以跨任务重复出现的程序变换对象**。

于是搜索空间就从：

```math
\mathcal V^L
```

——长度 $`L`$ 的 token 序列——

变成：

```math
\mathcal A\times\mathcal R(a)
```

前面先在一个小得多的 transformation space $`\mathcal A`$ 里决定“做什么”，后面再解决 realization。

如果 $`|\mathcal A|`$ 真比所有可能代码序列小很多，而且过去经验可以对 $`\mathcal A`$ 泛化，那么这个结构才真正有意义。

---

### 这条路真正可能带来的学习优势是什么？

我觉得不是“生成代码会更容易”这么简单。

更重要的是经验可以在 action 层共享。

假设模型以前在 reduction A 上学到：

```math
\text{warp-level reduction}
```

是一个有价值的动作。

另一个 reduction B 的 CUDA 实现可能长得完全不一样。

token 级经验很难直接复用：

```math
\text{code}_A \not\approx \text{code}_B.
```

但 transformation 层可以：

```math
a_A=a_B=\text{WarpReduce}.
```

于是过去的经验实际上变成：

```math
Q(P,a)
```

而不是：

```math
Q(P,\text{某一串具体 token}).
```

这才是“更可迁移”真正应该指的东西。

不是跨 GPU。

不是 CUDA→AMD。

而是：

> **同一个目标平台上，不同 kernel 之间能不能共享“什么程序变换通常值得尝试”的经验。**

这个我觉得就比之前说 cross-GPU 合理太多了。

---

### 但它也有几个很可能直接把方向打死的地方

最危险的一个是：**真正好的 CUDA 优化可能根本不是离散可分的 transformation。**

很多强优化可能是：

> 一开始重新设计整个 kernel。

它不是：

```math
A\rightarrow B\rightarrow C
```

而更像：

```math
P\rightarrow P'
```

一次整体重构。

这时候强行说：

> 里面包含 tile + fuse + layout + vectorize 四个 action。

可能只是事后解释。

而且这些 action 强耦合：

```math
\text{tile size}
```

取决于 layout；

layout 又取决于 thread mapping；

thread mapping 又和 shared memory footprint 一起决定 occupancy。

如果这样的话，把它拆成 action sequence 反而可能破坏整体联合搜索。

第二个危险是：

> LLM 本来内部就已经在隐式做这些 abstraction。

也就是说，我们外部强行加一层：

```math
\text{select transformation}
```

可能只是在把模型本来隐式完成的 reasoning 显式打印出来，没有减少真正困难。

这就是为什么前面那个 **Free-plan control** 很重要。

第三个危险是：

> transformation 选择容易，realization 才是最难的。

比如人告诉模型：

> “用 warp-level reduction。”

模型照样写不出正确、高性能实现。

如果大量任务都是这样，那么 high-level action 根本不是瓶颈。

这时 Oracle-action 实验就会直接暴露出来。

---

### 所以我现在对这条路的评价其实是

它比我们之前那个“给关键 token 多一点 reward”更值得研究，**不是因为它更复杂，而是因为它把问题往程序语义的正确层级上移了一层。**

之前是：

```math
\text{已有 token action space}
\rightarrow
\text{怎样更聪明地分 credit}
```

现在是：

```math
\boxed{
\text{token action space 本身是不是选错了坐标系？}
}
```

但我不会现在就说：

> “这肯定是我们的主线。”

因为它有一个非常明确、也非常容易杀掉的前置条件：

```math
\boxed{
\text{真实成功 CUDA 优化能否被稳定地压缩成重复出现的 transformation？}
}
```

如果我们人工看几十个真实 trajectory，发现根本压不出来，或者必须发明越来越细的 action 才能覆盖，那么这条路就应该死。

反过来，如果真的出现一个很漂亮的现象：

> 30 个成功 child，看起来代码千奇百怪，但大部分其实反复在做十来种程序决策；

那时候我会觉得——**好，这回我们终于不是为了做论文而设计一个 abstraction，而是数据本身逼着我们承认这里存在一个比 token 更自然的结构。**

这两者差别非常大。