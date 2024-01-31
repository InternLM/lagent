# flake8: noqa
import ast
import platform
from typing import Dict, List, Optional, Tuple, Union

from jsonschema import Draft7Validator

from lagent.actions import ActionExecutor
from lagent.llms.base_api import BaseAPIModel
from lagent.llms.base_llm import BaseModel
from lagent.schema import ActionReturn, ActionStatusCode, AgentReturn
from .base_agent import BaseAgent

DEFAULT_TRIGGERING_PROMPT = ('Determine exactly one command to use based on '
                             'the given goals and the progress you have made '
                             'so far, and respond using the JSON schema '
                             'specified previously:')

DEFAULT_PREFIX = """You are {ai_name}, {role_description}. Your decisions must always be made independently without seeking user assistance. Play to your strengths as an LLM and pursue simple strategies with no legal complications.
The OS you are running on is: {os_info}
## Constraints
You operate within the following constraints:
1. ~4000 word limit for short term memory. Your short term memory is short, so immediately save important information to files.
2. 'If you are unsure how you previously did something or want to recall past events, thinking about similar events will help you remember.
3. No user assistance
4. Exclusively use the commands listed below e.g. command_name
## Commands
You have access to the following commands:
{tool_description}
## Resources
You can leverage access to the following resources:
1. Internet access for searches and information gathering.
2. Long Term memory management.', 'File output.', 'Command execution
## Best practices
1. Continuously review and analyze your actions to ensure you are performing to the best of your abilities.
2. Constructively self-criticize your big-picture behavior constantly.
3. Reflect on past decisions and strategies to refine your approach.
4. Every command has a cost, so be smart and efficient. Aim to complete tasks in the least number of steps.
## Goals
For your task, you must fulfill the following goals:
{ai_goals}
"""

DEFAULT_CALL_PROTOCOL = """Respond strictly with JSON. The JSON should be compatible with the TypeScript type `Response` from the following:
```ts
interface Response {
    thoughts: {
        // Thoughts
        text: string;
        reasoning: string;
        // Short markdown-style bullet list that conveys the long-term plan
        plan: string;
        // Constructive self-criticism
        criticism: string;
        // Summary of thoughts to say to the user
        speak: string;
    };
    command: {
        name: string;
        args: Record<string, any>;
    };
}
```
"""
DEFAULT_SCHEMA = {
    '$schema': 'http://json-schema.org/draft-07/schema#',
    'type': 'object',
    'properties': {
        'thoughts': {
            'type': 'object',
            'properties': {
                'text': {
                    'type': 'string',
                    'description': 'thoughts'
                },
                'reasoning': {
                    'type': 'string'
                },
                'plan': {
                    'type':
                    'string',
                    'description':
                    '- short bulleted\n- list that conveys\n- long-term plan'
                },
                'criticism': {
                    'type': 'string',
                    'description': 'constructive self-criticism'
                },
                'speak': {
                    'type': 'string',
                    'description': 'thoughts summary to say to user'
                }
            },
            'required': ['text', 'reasoning', 'plan', 'criticism', 'speak'],
            'additionalProperties': False
        },
        'command': {
            'type': 'object',
            'properties': {
                'name': {
                    'type': 'string'
                },
                'args': {
                    'type': 'object'
                }
            },
            'required': ['name', 'args'],
            'additionalProperties': False
        }
    },
    'required': ['thoughts', 'command'],
    'additionalProperties': False
}


class AutoGPTProtocol:
    """A wrapper of AutoGPT prompt which manages the response from LLM and
    generate desired prompts in a AutoGPT format.

    Args:
        ai_name (str): the name of the agent, default to 'AutoGPT'
        role_description (str): description of the role, e.g., System, User
        prefix (str): the prefix prompt for AutoGPT
        call_protocol (str): the request prompt which defines the protocol
            of return format from LLM.
        valid_schema (dict): defines the schema of the return format.
        triggering_prompt (str): the predefined trigger prompt.
    """

    def __init__(self,
                 ai_name: Optional[str] = 'AutoGPT',
                 role_description: Optional[str] = '',
                 prefix: str = DEFAULT_PREFIX,
                 call_protocol: str = DEFAULT_CALL_PROTOCOL,
                 valid_schema: str = DEFAULT_SCHEMA,
                 triggering_prompt: str = DEFAULT_TRIGGERING_PROMPT) -> None:
        self.ai_name = ai_name
        self.role_description = role_description
        self.prefix = prefix
        self.call_protocol = call_protocol
        self.valid_schema = valid_schema
        self.triggering_prompt = triggering_prompt

    def parse(self, response: str,
              action_executor: ActionExecutor) -> Tuple[str, str]:
        """Parse the action returns in a AutoGPT format.

        Args:
            response (str): The response from LLM with AutoGPT format.
            action_executor (ActionExecutor): Action executor to
                provide no_action/finish_action name.

        Returns:
            tuple: the return value is a tuple contains:
                - action (str): the extracted action name.
                - action_input (str): the corresponding action input.
        """
        try:
            if response.startswith('```') and response.endswith('```'):
                # Discard the first and last ```, then re-join in case the response naturally included ```
                response = '```'.join(response.split('```')[1:-1])
            response = ast.literal_eval(response)
            validator = Draft7Validator(self.valid_schema)
            valid = True
            if errors := sorted(
                    validator.iter_errors(response), key=lambda e: e.path):
                valid = False
            if not valid:
                return action_executor.no_action, 'Validation of response failed:\n  ' + ';\n  '.join(
                    [str(e) for e in errors])
            try:
                if 'command' not in response:
                    return action_executor.no_action, "Missing 'command' object in JSON"
                if not isinstance(response, dict):
                    return action_executor.no_action, f'The previous message sent was not a dictionary {response}'
                command = response['command']
                if not isinstance(command, dict):
                    return action_executor.no_action, "'command' object is not a dictionary"
                if 'name' not in command:
                    return action_executor.no_action, "Missing 'name' field in 'command' object"
                command_name = command['name']
                # Use an empty dictionary if 'args' field is not present in 'command' object
                arguments = command.get('args', {})
                return command_name, arguments
            except Exception as e:
                return action_executor.no_action, repr(e)
        except SyntaxError as e:
            return action_executor.no_action, f'Your response could not be parsed: {repr(e)} \nRemember to only respond using the specified format above!'

    def format(self, goal: str, inner_history: List[Dict],
               action_executor: ActionExecutor) -> List[Dict]:
        """Generate the AutoGPT format prompt.

        Args:
            goal (str): The user request.
            inner_history (List[Dict]): The log in the current run.
            action_executor (ActionExecutor): the action manager to
                execute actions.
        Returns:
            List[Dict]: AutoGPT format prompt.
        """
        import distro
        formatted_data = []
        os_name = platform.system()
        os_info = (
            platform.platform(terse=True)
            if os_name != 'Linux' else distro.name(pretty=True))
        prefix = self.prefix.format(
            ai_name=self.ai_name,
            role_description=self.role_description,
            tool_description=action_executor.get_actions_info(),
            ai_goals=goal,
            os_info=os_info,
        )
        formatted_data.append(dict(role='system', content=prefix))
        formatted_data.append(dict(role='system', content=self.call_protocol))
        formatted_data += inner_history
        formatted_data.append(
            dict(role='user', content=self.triggering_prompt))
        return formatted_data

    def format_response(self, action_return) -> dict:
        """Format the final response at current step.

        Args:
            action_return (ActionReturn): return value of the current action.

        Returns:
            dict: the final response at current step.
        """
        if action_return.state == ActionStatusCode.SUCCESS:
            response = f'Command {action_return.type} returned: {response.format_result()}'
        else:
            response = action_return.errmsg
        return dict(role='system', content=response)


class AutoGPT(BaseAgent):
    """An implementation of AutoGPT (https://github.com/Significant-
    Gravitas/Auto-GPT)

    Args:
        llm (BaseModel or BaseAPIModel): a LLM service which can chat
            and act as backend.
        action_executor (ActionExecutor): an action executor to manage
            all actions and their response.
        protocol (ReActProtocol): a wrapper to generate prompt and
            parse the response from LLM / actions.
        max_turn (int): the maximum number of trails for LLM to generate
            plans that can be successfully parsed by ReWOO protocol.
    """

    def __init__(self,
                 llm: Union[BaseModel, BaseAPIModel],
                 action_executor: ActionExecutor,
                 protocol: AutoGPTProtocol = AutoGPTProtocol(),
                 max_turn: int = 2):
        self.max_turn = max_turn
        super().__init__(
            llm=llm, action_executor=action_executor, protocol=protocol)

    def chat(self, goal: str, **kwargs) -> AgentReturn:
        inner_history = []
        agent_return = AgentReturn()
        default_response = 'Sorry that I cannot answer your question.'
        for _ in range(self.max_turn):
            prompt = self._protocol.format(
                goal=goal,
                inner_history=inner_history,
                action_executor=self._action_executor)
            response = self._llm.chat(prompt, **kwargs)
            inner_history.append(dict(role='assistant', content=response))
            action, action_input = self._protocol.parse(
                response, self._action_executor)
            action_return: ActionReturn = self._action_executor(
                action, action_input)
            agent_return.actions.append(action_return)
            if action_return.type == self._action_executor.finish_action.name:
                agent_return.response = action_return.format_result()
                return agent_return
            inner_history.append(self._protocol.format_response(action_return))
        agent_return.inner_steps = inner_history
        agent_return.response = default_response
        return agent_return
