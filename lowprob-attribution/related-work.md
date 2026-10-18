# 相关证据：闭源 API 在逐 token 概率上的限制

本文件汇总公开可查的讨论与研究，用来支持实验报告里关于观测数据可得性的边界说明。
结论先写在这里：**拿不到完整逐 token 概率（尤其是工具调用参数与推理内容）不是本项目的操作失误，而是厂商主动设置的策略边界；多方证据把这归因于防蒸馏。**

---

## 一、用户层面的抱怨：OpenAI 论坛，持续两年未解决

**"Can I use logprobs & function calling at the same time?"**
<https://community.openai.com/t/can-i-use-logprobs-function-calling-at-the-same-time/579561>

- 楼主（2024-01-09）实测：带 `functions` 时设 `logprobs=True, top_logprobs=5`，返回 `logprobs=ChoiceLogprobs(content=None)`；**去掉 function 后 logprobs 恢复正常**。
- 论坛常驻回复（同日）：

  > They seem turned off when a function is employed. The likely reason is that **OpenAI doesn't want the AI method of sending to a tool recipient revealed**.

- 后续在 2024-03、2025-06、2025-07 连续有人追问是否计划支持 tools/function calling 的 logprobs，**至今无官方答复，也未支持**。

这条与我们自己的实测完全一致：DeepSeek 在 `finish_reason=tool_calls` 的那一轮，`logprobs.content` 里没有 arguments。

---

## 二、19 模型实测（datallmlab，2026-10 快照）

**"Chat Completions API: Which Parameters Actually Work (Tested on 19 Models)"**
<https://www.datallmlab.com/blog/chat-completions-api.html>

- `logprobs: true` 在 **GPT-6 Sol / GPT-6 Luna / GPT-6.1 Sol** 上返回 400，错误为 OpenAI 原文：

  ```json
  {"message": "logprobs are not supported with reasoning models.",
   "type": "invalid_request_error", "code": "unsupported_parameter"}
  ```

- **Claude、Gemini、Grok、Devstral、MiMo V2.6 Flash、Solar Mini 4：接受 `logprobs` 参数但返回 none** —— 静默忽略，不报错。
- **8 个推理模型"把预算花在 reasoning 上、返回空 content"**：gpt-oss-120b、Gemini 3.8 Flash、Gemini 3.1 Pro Preview、GLM-5.3、Kimi K3、DeepSeek V4.1 Flash、Qwen3.8 Max 0902、MiMo V2.6 Flash（`status=200`、`finish_reason=length`）。
  → 与我们踩到的"`max_tokens` 过小导致响应停在 reasoning、`content` 为空、`logprobs` 为 None"是同一类陷阱。
- Grok 4.7：44 completion tokens 中 39 个是 reasoning。

---

## 三、学术论文：三篇正面讨论「闭源 API 做科学推断」的合法性

### 1. How Open Must Language Models be to Enable Reliable Scientific Inference?（2026）

<https://arxiv.org/abs/2606.32008> 之外的独立条目，摘要页：<https://lacuna.tiptreesystems.com/work/how-open-must-language-models-be-to-enable-reliable-scientific-inference/wrk_c7d83a1d87376ae24eca2159acd2aa0b>

提出三个障碍：

- **Versioning Problem**：闭源模型持续静默更新、旧版被弃用，精确复现不可能（文中举例：同一个"gpt-4"在不同月份版本上，简单基准表现可差 60%）。
- **Credit Assignment Problem**：商业系统是复合体——隐藏 system prompt、安全过滤、web search、未公开解码策略——无法判断行为来自模型还是工程封装。
- **Information Restriction Problem**：闭源 API 通常只返回最终文本，**隐藏完整概率分布与内部激活**。

TL;DR 原文：把闭源模型当作科学研究的仪器，"generally unfit for rigorous scientific inference"。

### 2. Surrogate Fidelity: When Can Open LLMs Explain Closed Ones?（Meta）

<https://arxiv.org/abs/2606.32008>

开篇即概括现状：**"the APIs for most widely deployed language models at best expose log-probabilities over output tokens"**。提出用 log-odds 作为 API 兼容的标量读出，跨 Llama / Qwen / GPT / Gemini 四族 11 个模型评估替代保真度。

### 3. Log Probability Tracking of LLM APIs（INRIA，OpenReview 已接收）

<https://openreview.net/forum?id=hFxivbAgVP>

用首 token 的 logprobs 监控 API 漂移（监控 189 个端点、检出 37 次服务变化）。审稿意见明确指出该方法的根本局限：

> Only APIs that provide logprobs can be tracked. **This is ultimately the largest limitation. Most providers do not provide logprobs as these can be used to exfiltrate model details.**

所引依据是 Carlini et al.《Stealing Part of a Production Language Model》——从概率分布可反推模型规模等信息。

---

## 四、行业评论：这是政策选择，不是技术限制

**O'Reilly Radar《Don't Blame the Model》**
<https://www.oreilly.com/radar/dont-blame-the-model/>

- 根因判断：

  > The case for not including logprobs and reasoning tokens is similar. **The risk of distillation increases with the amount of information that the API returns.**

- 各家现状（文中点名）：Google 只给 top-20；**OpenAI 从 GPT-5 起不再提供任何 logprobs**；**Anthropic 从未提供过**；reasoning tokens 方面 Anthropic / Google / OpenAI 都只给摘要。
- 明确主张："Model providers should resume providing at least the top 20 logprobs, **over the entire output (reasoning included)**"。
- 结论句：

  > **The decision to withhold these features is a policy choice, not a technical limitation.**

- 本地方案对照："For local open weight models ... The entire reasoning trace is always available and **logprobs are fully transparent**."

---

## 五、国内讨论

### CSDN《面试官问："模型一本正经胡说时，logprobs 抓得到吗？"》（2026）

<https://blog.csdn.net/m0_59163425/article/details/162587309>

直接从工程选型角度点出同一问题：

> 工程上第一个坑……**你最想要置信度的那类场景，恰恰拿不到 logprob**。截至 2026 年 4 月，推理模型（OpenAI 的 o 系列、GPT-5 系列的思考模式这一类）在接口上明确不支持 logprobs、top_logprobs，连同 temperature、top_p 一起被禁掉了。……越是复杂推理、越值得你去核验可信度的回答，越没有 logprob 给你看。

文中给出的替代思路（可作本项目 limitations 的对照）：

- **序列级聚合**：对整句 token logprob 做聚合时，**min 比 mean 更灵敏**——单个关键 token 心虚会拉低 min，而均值会被大量高概率常规 token 稀释。
- **语义熵**：Farquhar et al., *Nature* (2024)，同题高温采样约 5 次，按语义聚类后算簇熵；代价是推理成本约 5 倍。

### 《推理模型的输出概率与置信度》

<https://www.bbccb.com/html/20260916/48555.html>

> 部分推理模型（如 o1 系列早期版本）**出于产品考虑不开放 logprobs**，选型时要确认所用模型和接口版本是否支持该参数。

### 具体报错文本（theneuralbase，2026-04 核实）

<http://theneuralbase.com/reasoning-models/errors/openai-o1-logprobs-not-supported>

```text
400 - {'error': {'message': "'logprobs' is not supported for this model. 'logprobs' can only be
set to 'true' for 'gpt-4o', 'gpt-4o-mini', 'gpt-4.1', and 'gpt-4.1-mini'.",
'type': 'invalid_request_error', 'param': 'logprobs', 'code': 'invalid_parameter_value'}}
```

原因说明：推理模型使用对用户不透明的内部思考过程，"the API doesn't expose token-level probabilities because they're generated during hidden reasoning phases, not in the final response"。

---

## 六、Reddit / 社媒：本地工具调用的可靠性讨论

**r/LocalLLaMA："Local Tool Calling Divides Our Community: Breakthrough or Collective Cope?"**（66 upvotes / 113 comments，经 theagenttimes 转述）
<https://theagenttimes.com/articles/local-tool-calling-divides-our-community-breakthrough-or-col-a0eff21e>

虽然不是直接谈 logprobs，但给出的判断与本项目经验一致：

> inference runtime inconsistencies — **not model capability** — emerging as the primary bottleneck for local agent autonomy.

- 典型现象：Ollama issue #11608（2025-07 至 2026-03）中模型在聊天窗口里吐出原始 JSON 而不执行工具调用；open-multi-agent issue #15 报告本地模型经常不符合 OpenAI 的 `tool_calls` 线格式。
- 反例：xLAM-2 8B（Berkeley Function Calling Leaderboard 第一）在 Hodges 的测试里只有 15%，原因是 **chat template 与 LM Studio 不兼容**。
- 结论句：**"a model's raw capability matters less than whether the serving layer correctly formats and routes tool calls"**。

**vLLM 工具调用的常见配置坑**（对将来切 vLLM 有直接参考价值）
<https://infinisynapse.com/zh/blog/vllm-tool-calling-reddit>

- `tool_choice="auto"` 需要服务端带 `--enable-auto-tool-choice` 与匹配的 `--tool-call-parser`，否则直接 400。
- **"Parser mismatch is the silent failure (empty tool_calls)"**；"the gap is almost always parser/template config, not the base model"。

---

## 七、厂商文档里写明的限制（本项目已逐条核对）

| 提供方 | 逐 token 概率 | 明确文本 / 备注 |
|---|---|---|
| OpenAI | GPT-4 系可用；**GPT-5 系与推理模型弃用** | 400：`logprobs are not supported with reasoning models` |
| Anthropic | **从未提供** | 带参直接报 `Unrecognized request argument supplied: logprobs` |
| Google Gemini | 支持但 **3.x 已弃用** | `logprobs` 参数标注 deprecated |
| xAI Grok | **grok-4.20 及更新静默忽略** | 官方模型页原文 |
| Meta（Muse Spark） | **不支持，返回 HTTP 400** | `dev.meta.ai/docs/reasoning`："Log probabilities: Not supported." |
| DeepSeek | 支持（`content` + `reasoning_content`） | 本项目实测 |
| 智谱 GLM | **官方不支持** | 参数表无该字段 |
| MiniMax | **官方不支持** | 参数表无该字段 |
| Moonshot Kimi | 支持（0–20） | 返在 `message.logprobs` |
| 腾讯混元 | 支持（0–20） | 需 `logprobs=true` |
| 阿里 Qwen | 支持但仅 **0–5** | 官方明文："思考阶段生成的内容（`reasoning_content`）不会返回对数概率。" |

---

## 八、对本项目的直接含义

1. **工具调用参数的概率在所有 OpenAI 兼容接口上都不存在**，因为 `logprobs` 对象的形状是 `{content[], refusal[]}`，结构上无此位置。这不是厂商差异。
2. **推理内容（reasoning）的概率在多数接口上也被排除**；DeepSeek 是少数会在独立的 `logprobs.reasoning_content` 键里给出的。
3. 多方把根因指向**防蒸馏**（返回信息量越大、越易被复制），并明确这是 policy choice 而非技术限制。
4. 因此，若要同时获得 reasoning 与工具参数的完整逐 token 概率，**自部署推理是唯一路径**——本地方案下"整个推理轨迹可见、logprobs 完全透明"。
5. 转向自部署时需注意：vLLM 的 `tool_choice="auto"` 必须配 `--enable-auto-tool-choice` 与匹配的 `--tool-call-parser`，parser 不匹配会静默返回空 `tool_calls`。
