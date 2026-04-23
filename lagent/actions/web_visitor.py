import asyncio
import json
import re
import traceback
import warnings
from typing import Any, List

from transformers import AutoTokenizer

from lagent.actions import AsyncActionMixin, BaseAction
from lagent.schema import ActionStatusCode, ActionValidCode, AgentMessage
from lagent.utils import create_object


def extract_last_json(text: str) -> dict | None:
    """
    Extracts the last valid JSON object from a string.
    Handles Markdown code blocks (```json ... ```) and raw JSON strings.
    """
    try:
        # 1. Try to find JSON within Markdown code blocks first
        # Look for ```json ... ``` or just ``` ... ```
        code_block_pattern = re.compile(r'```(?:json)?\s*(\{.*?\})\s*```', re.DOTALL)
        matches = code_block_pattern.findall(text)
        if matches:
            return json.loads(matches[-1])

        # 2. If no code blocks, try to find the last outermost pair of braces
        # This regex looks for { ... } lazily but we want the last one.
        # A simple approach for nested JSON is tricky with regex,
        # so we scan from right to left for the last '}' and find its matching '{'.

        stack, end_idx = 0, -1
        # Reverse search to find the last valid JSON structure
        for i in range(len(text) - 1, -1, -1):
            char = text[i]
            if char == '}':
                if stack == 0:
                    end_idx = i
                stack += 1
            elif char == '{':
                if stack > 0:
                    stack -= 1
                    if stack == 0 and end_idx != -1:
                        # Found a potential outermost JSON object
                        candidate = text[i : end_idx + 1]
                        try:
                            return json.loads(candidate)
                        except json.JSONDecodeError:
                            # If this chunk isn't valid, reset and keep searching backwards
                            # (or you might decide to stop here depending on strictness)
                            stack, end_idx = 0, -1
        return None
    except Exception:
        return None


class WebVisitor(AsyncActionMixin, BaseAction):

    EXTRACTION_PROMPT = """Please process the following webpage content and user goal to extract relevant information:

## **Webpage Content** 
{webpage_content}

## **User Goal**
{goal}

## **Task Guidelines**
1. **Content Scanning for Rationale**: Locate the **specific sections/data** directly related to the user's goal within the webpage content
2. **Key Extraction for Evidence**: Identify and extract the **most relevant information** from the content, you never miss any important information, output the **full original context** of the content as far as possible, it can be more than three paragraphs.
3. **Summary Output for Summary**: Organize into a concise paragraph with logical flow, prioritizing clarity and judge the contribution of the information to the goal.

**Final Output Format using JSON format has "rational", "evidence", "summary" fields**
"""

    def __init__(
        self,
        browse_tool: BaseAction | dict,
        llm: Any,
        max_browse_attempts: int = 3,
        max_extract_attempts: int = 3,
        sleep_interval: int = 3,
        truncate_browse_response_length: int | None = None,
        tokenizer_path: str | None = None,
        name: str = 'visit',
    ):
        super().__init__(
            description={
                'name': name,
                'description': 'Visit webpage(s) and return the summary of the content.',
                'parameters': [
                    {
                        'name': 'url',
                        'type': ['STRING', 'ARRAY'],
                        "items": {"type": "string"},
                        "minItems": 1,
                        'description': 'The URL(s) of the webpage(s) to visit. Can be a single URL or an array of URLs.',
                    },
                    {'name': 'goal', 'type': 'STRING', 'description': 'The goal of the visit for webpage(s).'},
                ],
                'required': ['url', 'goal'],
            }
        )
        browse_tool = create_object(browse_tool)
        assert not browse_tool.is_toolkit and browse_tool.description['required'] == [
            'url'
        ], "browse_tool must be a single-tool action with only 'url' as required argument."
        self.browse_tool = browse_tool
        self.llm = create_object(llm)
        self.max_browse_attempts = max_browse_attempts
        self.max_extract_attempts = max_extract_attempts
        self.sleep_interval = sleep_interval
        self.truncate_browse_response_length = truncate_browse_response_length
        self.tokenizer = (
            AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True) if tokenizer_path else None
        )
        if self.truncate_browse_response_length is not None and self.tokenizer is None:
            warnings.warn(
                'truncate_browse_response_length is set but tokenizer_path is not provided. '
                'The raw webpage content will be truncated by characters instead of tokens.'
            )

    async def run(self, url: str | List[str], goal: str) -> str:
        if isinstance(url, str):
            url = [url]

        async def _inner_call(single_url: str) -> str:
            try:
                return await self._read_webpage(single_url, goal)
            except Exception as e:
                return f"Error fetching {single_url}: {str(e)}"

        response = await asyncio.gather(*[_inner_call(single_url) for single_url in url])
        return "\n=======\n".join(response).strip()

    async def _read_webpage(self, url: str, goal: str) -> str:
        tool_response = compressed = None
        return_template = (
            f"The useful information in {url} for user goal {goal} as follows: \n\n"
            f"Evidence in page: \n{{evidence}}\n\nSummary: \n{{summary}}\n\n"
        )
        for _ in range(self.max_browse_attempts):
            resp = await self.browse_tool({'url': url})
            if resp.valid == ActionValidCode.OPEN and resp.state == ActionStatusCode.SUCCESS:
                tool_response = resp.format_result()
                break
            await asyncio.sleep(self.sleep_interval)
        else:
            return return_template.format(
                evidence="The provided webpage content could not be accessed. Please check the URL or file format.",
                summary="The webpage content could not be processed, and therefore, no information is available.",
            )

        if self.truncate_browse_response_length is not None:
            tool_response = (
                self.tokenizer.decode(
                    self.tokenizer.encode(
                        tool_response,
                        max_length=self.truncate_browse_response_length,
                        truncation=True,
                        add_special_tokens=False,
                    )
                )
                if self.tokenizer is not None
                else tool_response[: self.truncate_browse_response_length]
            )

        for _ in range(self.max_extract_attempts):
            try:
                prompt = self.EXTRACTION_PROMPT.format(webpage_content=tool_response, goal=goal)
                llm_response = await self.llm.chat([{'role': 'user', 'content': prompt}])
                if llm_response and not isinstance(llm_response, str):
                    llm_response = (
                        llm_response.content
                        if isinstance(llm_response, AgentMessage)
                        else llm_response.choices[0].message.content
                    )
                if not llm_response or len(llm_response) < 10:
                    tool_response = tool_response[: int(len(tool_response) * 0.7)]
                    continue
                compressed = extract_last_json(llm_response)
                if isinstance(compressed, dict) and all(
                    key in compressed for key in ['rational', 'evidence', 'summary']
                ):
                    break
            except Exception:
                print(f"Error in extracting information: {traceback.format_exc()}")
                await asyncio.sleep(self.sleep_interval)
        else:
            return return_template.format(
                evidence="Failed to extract relevant information from the webpage content.",
                summary="The webpage content could not be processed, and therefore, no information is available.",
            )
        return return_template.format(evidence=compressed['evidence'], summary=compressed['summary'])
