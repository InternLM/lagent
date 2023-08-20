# flake8: noqa
import ast
import platform

from jsonschema import Draft7Validator

from etangent.actions import ActionExecutor
from etangent.schema import ActionReturn, ActionStatusCode, AgentReturn
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

    def __init__(self,
                 ai_name='AutoGPT',
                 role_description='',
                 prefix=DEFAULT_PREFIX,
                 call_protocol=DEFAULT_CALL_PROTOCOL,
                 valid_schema=DEFAULT_SCHEMA,
                 triggering_prompt=DEFAULT_TRIGGERING_PROMPT) -> None:
        self.ai_name = ai_name
        self.role_description = role_description
        self.prefix = prefix
        self.call_protocol = call_protocol
        self.valid_schema = valid_schema
        self.triggering_prompt = triggering_prompt

    def parse(self, response, action_executor: ActionExecutor):
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

    def format(self, goal, inner_history, action_executor: ActionExecutor):
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

    def format_response(self, action_return):
        if action_return.state == ActionStatusCode.SUCCESS:
            response = action_return.result['text']
            response = f'Command {action_return.type} returned: {response}'
        else:
            response = action_return.errmsg
        return response


class AutoGPT(BaseAgent):

    def __init__(self,
                 llm,
                 action_executor: ActionExecutor,
                 prompter=AutoGPTProtocol(),
                 max_turn=2):
        self.max_turn = max_turn
        super().__init__(
            llm=llm, action_executor=action_executor, prompter=prompter)

    def chat(self, goal):
        self._inner_history = []
        agent_return = AgentReturn()
        default_response = '对不起，我无法回答你的问题'
        for _ in range(self.max_turn):
            prompt = self._prompter.format(
                goal=goal,
                inner_history=self._inner_history,
                action_executor=self._action_executor)
            response = self._llm.generate_from_template(prompt, 512)
            self._inner_history.append(
                dict(role='assistant', content=response))
            action, action_input = self._prompter.parse(
                response, self._action_executor)
            action_return: ActionReturn = self._action_executor(
                action, action_input)
            agent_return.actions.append(action_return)
            if action_return.type == self._action_executor.finish_action.name:
                agent_return.response = action_return.result['text']
                return agent_return
            self._inner_history.append(
                dict(
                    role='system',
                    content=self._prompter.format_response(action_return)))
        agent_return.response = default_response
        return agent_return
