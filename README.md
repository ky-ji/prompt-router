# Prompt Router

一个很小的固定 prompt 路由器：你给它一句中文命令，它从一组固定英文 prompt 里找出语义最接近的一条。

默认模式是本地多语言 embedding，不需要 `OPENAI_API_KEY`。第一次运行会通过 `sentence-transformers` 下载模型。

## 30 秒上手

```bash
cd /Users/jky/workspace/project/prompt-router
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[local]"

prompt-router route examples/prompts.json "帮我总结一下这段内容"
```

默认会在 prompt 文件旁边自动生成索引文件，例如：

```text
examples/prompts.prompt-index.json
```

之后再次运行会复用索引；如果 prompt 文件内容变了，索引会自动重建。

## Prompt 库格式

`prompts.json` 只是普通字符串数组，不需要结构化字段：

```json
[
  "Summarize the following text into concise bullet points.",
  "Translate the following text into English.",
  "Rewrite the following text to make it clearer and more professional."
]
```

## 查看分数

```bash
prompt-router route examples/prompts.json "帮我总结一下这段内容" --json
```

输出里会包含：

- `prompt`：最佳 prompt
- `score`：最高相似度
- `confident`：是否足够确定
- `reason`：`ok`、`below_threshold` 或 `low_margin`
- `candidates`：候选 prompt 和分数

如果你希望“不确定”时让命令返回失败码：

```bash
prompt-router route examples/prompts.json "随便聊聊天" --strict
```

## 高级用法

显式构建索引：

```bash
prompt-router build examples/prompts.json prompt-index.json
```

显式匹配已有索引：

```bash
prompt-router match prompt-index.json "帮我总结一下这段内容"
```

索引会记录 provider、model 和 prompt 文件 hash。`match` 默认读取索引里的 provider/model，避免手动传错。

## 使用 OpenAI embedding

本地模式不需要 key。如果你想用 OpenAI embedding：

```bash
export OPENAI_API_KEY="你的 API key"

prompt-router route examples/prompts.json "帮我总结一下这段内容" \
  --provider openai \
  --model text-embedding-3-small
```

也可以显式 build：

```bash
prompt-router build examples/prompts.json prompt-index.json --provider openai
```

## Python API

```python
from prompt_router import route_command

result = route_command(
    "examples/prompts.json",
    "帮我总结一下这段内容",
)

print(result.prompt)
print(result.confident, result.reason)
```

需要自己控制索引路径时：

```python
result = route_command(
    "examples/prompts.json",
    "帮我总结一下这段内容",
    index_path="prompt-index.json",
)
```

## 常见问题

**必须要 OpenAI API key 吗？**

不需要。默认 `local` provider 使用本地 `sentence-transformers` 模型。

**第一次为什么比较慢？**

第一次本地运行会下载模型。之后模型会走本地缓存。

**prompt 更新后要手动 rebuild 吗？**

用 `route` 不需要。它会检测 prompt 文件 hash，变化后自动重建索引。

**如何处理不确定结果？**

用 `--json` 读取 `confident` 和 `reason`，或者加 `--strict` 让低置信度结果返回非 0 exit code。

**没有安装 local extra 会怎样？**

本地模式会提示：

```bash
pip install -e ".[local]"
```

## 测试

```bash
cd /Users/jky/workspace/project/prompt-router
PYTHONPATH=src python3 -m unittest discover -s tests
python3 -m compileall -q src
```
