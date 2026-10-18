# 本地工具参数代码概率核对

本报告只读取已有记录，没有启动模型、调用 API 或重新采样。

## 核对方法

先把每条记录的 content_logprobs 按生成顺序拼成文本，再解析其中的工具调用。仅将 write.content 和 edit.newString 中包含 import、def 或 Triton 装饰器的字符串计为代码参数。代码是否正确执行另行判断；注释和格式字符仍保留在参数内。

| 目录 | 响应记录 | 全部 token | 含代码参数的响应 | 代码参数 token | 有有限概率值 |
|---|---:|---:|---:|---:|---:|
| runs | 49 | 7182 | 19 | 898 | 898 |
| runs-triton | 118 | 15987 | 12 | 1862 | 1862 |
| runs-migrated | 28 | 3360 | 2 | 402 | 402 |

合计 195 条响应、26,529 个 token；其中 33 条响应含可解析的代码参数，覆盖 3,162 个生成 token。

旧检查逐个 token 搜完整关键词，命中 0 次。这个零与现存代码并不矛盾：triton 可拆成 tr 和 iton，def run 可拆成 def 和 run。空的 tool_call_text 也不能否定 content_logprobs 中的工具调用文本。

40/40 个代码参数字符串与同一运行的工具事件输入完全一致；audit.json 同时列出对应事件行号和执行状态。

未解析的工具块：3。这些块不计入代码参数数量。重复 request_id：0。

## 可逐项复查的例子

完整代码、生成 token、概率、原文件行号和 SHA-256 保存在 audit.json。以下片段均直接来自工具参数，不是任务提示或事后补写。

### runs-migrated/row_reduce-full-00/logprobs.jsonl，第 4 行

模型：qwen3-4b；工具：write；参数：content。

```python
import triton
import triton.clib
import numpy as np

@triton.jit
def kernel(x, N):
    # Compute the sum of each row in the matrix x
    # x is a 2D array with shape (N, N)
    # Each row is summed and returned as a float32 value
    # This kernel is designed to process each row and return the sum
    # The kernel is launched with the number of blocks equal to N
    # and each block processes one row
```

### runs-triton/row_reduce-full-07/logprobs.jsonl，第 1 行

模型：qwen3-4b；工具：edit；参数：newString。

```python
import triton
import tritoneti

@triton.jit
def kernel(data, N, ROWS, COLS):
    # Compute the row index
    pid = triton.page_id()
    # Compute the row index
    row = pid // COLS
    # Compute the column index
    col = pid % COLS
    # Load the value from the matrix
```

### runs-triton/row_reduce-full-07/logprobs.jsonl，第 2 行

模型：qwen3-4b；工具：edit；参数：newString。

```python
import triton
import tritoneti

@triton.jit
def kernel(data, N, ROWS, COLS):
    # Compute the row index
    pid = triton.page_id()
    # Compute the row index
    row = pid // COLS
    # Compute the column index
    col = pid % COLS
    # Load the value from the matrix
```

### runs-triton/row_reduce-full-08/logprobs.jsonl，第 1 行

模型：qwen3-4b；工具：edit；参数：newString。

```python
from triton import kernel, launch
```

## 两条采集路线如何互补

本地 vLLM 将工具参数作为生成 token 保存在 content_logprobs 中。把 token 按顺序重建后，可以直接读取 write.content、edit.newString 中的代码，同时保留每个生成 token 的概率。DeepSeek API 路线则让代码写在回复正文，以正文概率字段完成采集。

两条路线分别覆盖工具参数输出和正文输出。这里核对的 40 个代码参数及其 3,162 个生成 token，提供了工具调用路线已经实际采到代码概率的证据。
