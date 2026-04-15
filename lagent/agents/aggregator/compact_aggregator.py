"""Aggregator for CompactAgent.

Takes formatted_messages (list[dict]) from policy's aggregator
as the input message content, and assembles them into:
  [system: COMPACT_PROMPT, user: formatted conversation text]
"""

from typing import Dict, List, Optional, Tuple

from lagent.memory import Memory


class CompactAggregator:
    """Aggregator that formats list[dict] messages into readable text for compaction.

    When CompactAgent receives a message whose content is list[dict]
    (the formatted_messages from policy's aggregator), this aggregator
    converts it into a human-readable conversation transcript and
    prepends the compact prompt as system message.
    """

    def aggregate(
        self,
        messages: Memory,
        name: str,
        parser=None,
        system_instruction: str = None,
        tools: List[Dict] = None,
    ) -> Tuple[List[Dict[str, str]], Optional[List[Dict]]]:
        _messages = []

        # System message: the compact prompt (passed as template/system_instruction)
        if system_instruction:
            _messages.append(dict(role='system', content=system_instruction))

        # Find the input message with formatted_messages as content
        all_msgs = messages.get_memory()
        for msg in all_msgs:
            content = msg.content
            if isinstance(content, list):
                # list[dict] from policy's aggregator → format as readable text
                formatted = self._format_messages(content)
                _messages.append(dict(role='user', content=formatted))
            elif isinstance(content, str) and content:
                _messages.append(dict(role='user', content=content))
        latest_env_info = None
        for message in all_msgs:
            if getattr(message, 'env_info', None) is not None:
                latest_env_info = message.env_info

        tools_to_use = tools
        if latest_env_info and latest_env_info.get("tools"):
            tools_to_use = latest_env_info.get("tools")
            
        return _messages, tools_to_use


    @staticmethod
    def _format_messages(messages: List[Dict]) -> str:
        """Convert list[dict] messages to readable conversation text."""
        lines = []
        for msg in messages:
            role = msg.get('role', 'unknown').upper()
            content = msg.get('content', '')
            if content is None:
                content = ''

            # Include tool calls if present
            tool_calls = msg.get('tool_calls', [])
            if tool_calls:
                tool_names = []
                for tc in tool_calls:
                    if isinstance(tc, dict):
                        func = tc.get('function', {})
                        tool_names.append(func.get('name', 'unknown'))
                content += f" [tool_calls: {', '.join(tool_names)}]"

            if content:
                lines.append(f"{role}: {content}")

        return '\n'.join(lines)
