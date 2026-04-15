# Agent Project Template

一个标准的 Lagent Agent 项目的参考结构。

## 目录结构

```
my-agent/
├── config.py          # 唯一入口：完整的 PyConfig（必须）
├── agent.py           # 自定义 Agent 子类（可选）
├── actions.py         # 自定义 Actions（可选）
├── memory.py          # 自定义 Memory（可选）
├── prompts.py         # System prompt 和常量（可选）
└── skills/            # 技能文件目录（可选）
    └── review.md
```

## 核心原则

1. **config.py 是唯一的完整表达** — 所有 Agent 配置最终都是一个
   `agent_config = dict(type=..., ...)` 的 PyConfig dict
2. **没有隐式假设** — 模型、Actions、Memory 的参数全部显式声明
3. **create_object() 递归展开** — 嵌套 dict 中的每个 `type=...`
   都会被 `create_object()` 自动实例化

## 使用方式

```python
# 方式 1: AgentLoader 自动发现
# 把项目放在 workspace/agents/my-agent/ 下即可

# 方式 2: 手动注册
from lagent.interclaw.services.agent import AgentService
service = AgentService()
service.register("my-agent", agent_config=config.agent_config)

# 方式 3: 直接实例化
from lagent.utils import create_object
agent = create_object(config.agent_config)
response = await agent("Hello")
```

## 简化形式

如果你的 Agent 只需要换个 prompt 和 model，可以用 AGENT.md 代替整个项目：

```markdown
---
name: my-agent
description: A simple agent
model: gpt-4
max_turns: 50
---
You are a helpful assistant.
```

但 AGENT.md 是语法糖，需要 preset/defaults 层展开为完整 PyConfig。
