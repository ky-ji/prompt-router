# Prompt Router

一个很小的固定 prompt 路由器：你给机器人一句中文命令，它从一组固定英文 prompt 里找出语义最接近的一条。

核心流程：

```text
固定 prompts.json
-> prompt-router build 预先生成 embedding 索引
-> 用户中文命令
-> prompt-router match 只计算命令 embedding
-> 输出最相似 prompt
```

## Prompt 库格式

`prompts.json` 是普通字符串数组，不需要结构化字段：

```json
[
  "Summarize the following text into concise bullet points.",
  "Translate the following text into English.",
  "Rewrite the following text to make it clearer and more professional."
]
```

## 安装

```bash
cd /Users/jky/workspace/project/prompt-router
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

配置 OpenAI API key：

```bash
export OPENAI_API_KEY="你的 API key"
```

## 使用

先为固定 prompt 库构建索引：

```bash
prompt-router build examples/prompts.json prompt-index.json
```

匹配中文命令：

```bash
prompt-router match prompt-index.json "帮我总结一下这段内容"
```

默认只输出命中的 prompt。需要分数和候选项时：

```bash
prompt-router match prompt-index.json "帮我总结一下这段内容" --json
```

可调两个置信度参数：

```bash
prompt-router match prompt-index.json "帮我整理一下" --threshold 0.45 --margin 0.03 --json
```

- `threshold`：最高分低于该值时，结果会标记为 `below_threshold`。
- `margin`：第一名和第二名差距小于该值时，结果会标记为 `low_margin`。

即使不确定，CLI 仍会输出最相似 prompt；如果你用 `--json`，可以读取 `confident` 和 `reason` 做二次确认。

## 不安装也可运行

```bash
cd /Users/jky/workspace/project/prompt-router
PYTHONPATH=src python3 -m prompt_router.cli build examples/prompts.json prompt-index.json
PYTHONPATH=src python3 -m prompt_router.cli match prompt-index.json "帮我总结一下这段内容" --json
```

## 测试

```bash
cd /Users/jky/workspace/project/prompt-router
PYTHONPATH=src python3 -m unittest discover -s tests
```
