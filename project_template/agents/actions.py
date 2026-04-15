"""自定义 Actions 示例。

Action 是 Agent 与外部世界交互的接口。每个 @tool_api 方法
都会被自动转换为 LLM 可调用的工具。

两种模式:
  - 单工具 Action: 只有一个 run() 方法
  - 工具箱 (Toolkit): 多个 @tool_api 方法，LLM 看到的工具名
    是 "ClassName.method_name" 格式

注意: 不要在 Action 文件中使用 `from __future__ import annotations`，
否则会破坏 @tool_api 的类型解析。
"""

from typing import Annotated, Optional

from lagent.actions.base_action import BaseAction, tool_api
from lagent.schema import ActionReturn, ActionStatusCode


class FileAnalyzer(BaseAction):
    """示例：文件分析工具箱（Toolkit 模式）。

    LLM 看到的工具列表:
      - FileAnalyzer.count_lines
      - FileAnalyzer.search_pattern
    """

    @tool_api
    def count_lines(
        self,
        file_path: Annotated[str, "要分析的文件路径"],
    ) -> ActionReturn:
        """统计文件行数。"""
        try:
            with open(file_path) as f:
                count = sum(1 for _ in f)
            return ActionReturn(
                result=[{"type": "text", "content": f"{count} lines"}],
                state=ActionStatusCode.SUCCESS,
            )
        except Exception as e:
            return ActionReturn(
                errmsg=str(e),
                state=ActionStatusCode.API_ERROR,
            )

    @tool_api
    def search_pattern(
        self,
        file_path: Annotated[str, "要搜索的文件路径"],
        pattern: Annotated[str, "正则表达式模式"],
        max_results: Annotated[int, "最大返回数量"] = 10,
    ) -> ActionReturn:
        """在文件中搜索匹配正则表达式的行。"""
        import re

        try:
            matches = []
            with open(file_path) as f:
                for i, line in enumerate(f, 1):
                    if re.search(pattern, line):
                        matches.append(f"L{i}: {line.rstrip()}")
                        if len(matches) >= max_results:
                            break
            result = "\n".join(matches) if matches else "No matches found"
            return ActionReturn(
                result=[{"type": "text", "content": result}],
                state=ActionStatusCode.SUCCESS,
            )
        except Exception as e:
            return ActionReturn(
                errmsg=str(e),
                state=ActionStatusCode.API_ERROR,
            )


class HealthCheck(BaseAction):
    """示例：单工具 Action（非 Toolkit 模式）。

    LLM 看到的工具名就是 "HealthCheck"。
    """

    @tool_api
    def run(
        self,
        service_url: Annotated[str, "要检查的服务 URL"],
        timeout: Annotated[int, "超时秒数"] = 5,
    ) -> ActionReturn:
        """检查一个 HTTP 服务是否可达。"""
        import urllib.request

        try:
            req = urllib.request.Request(service_url, method="HEAD")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                status = resp.status
            return ActionReturn(
                result=[{"type": "text", "content": f"OK (HTTP {status})"}],
                state=ActionStatusCode.SUCCESS,
            )
        except Exception as e:
            return ActionReturn(
                result=[{"type": "text", "content": f"FAIL: {e}"}],
                state=ActionStatusCode.SUCCESS,
            )
