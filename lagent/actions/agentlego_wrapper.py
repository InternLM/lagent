from typing import List, Union, Optional
from copy import deepcopy
from lagent import BaseAction, ActionReturn, ActionStatusCode
from lagent.actions.parser import ParseError, JsonParser

from agentlego.tools.remote import RemoteTool
from agentlego.parsers import DefaultParser


class AgentLegoToolkit(BaseAction):
    """A wrapper to align with the interface of iLagent tools."""

    def __init__(self,
                 type: str,
                 url: Optional[str] = None,
                 text: Optional[str] = None,
                 spec_dict: Optional[dict] = None,
                 parser=JsonParser,
                 enable: bool = True):

        if url is not None:
            spec = dict(url=url)
        elif text is not None:
            spec = dict(text=text)
        else:
            assert spec_dict is not None
            spec = dict(spec_dict=spec_dict)
        api_list = [api for api in RemoteTool.from_openapi(**spec)]
        api_desc = []
        for api in api_list:
            api.set_parser(DefaultParser)
            desc = deepcopy(api.toolmeta.__dict__)
            if not desc.get('parameters') and desc.get('inputs'):
                parameters = [vars(param) for param in desc['inputs']]
                for param in parameters:
                    if 'type' in param and isinstance(param['type'], object):
                        param['type'] = param['type'].__name__
                desc['parameters'] = parameters
                desc.pop('inputs')
            if not desc.get('return_data') and desc.get('outputs'):
                return_data = [vars(param) for param in desc['outputs']]
                for param in return_data:
                    if 'type' in param and isinstance(param['type'], object):
                        param['type'] = param['type'].__name__
                desc['return_data'] = return_data
                desc.pop('outputs')
            if not desc.get('required'):
                desc['required'] = [param['name'] for param in desc['parameters']]
            api_desc.append(desc)
        if len(api_list) > 1:
            tool_description = dict(
                name=type,
                api_list=api_desc
            )
            self.name2func = {f'{api.name}': api for api in api_list}
        else:
            tool_description = api_desc[0]
            self.name2func = {'run': api_list[0]}
        super().__init__(
            description=tool_description,
            parser=parser,
            enable=enable
        )

    @property
    def is_toolkit(self):
        return 'api_list' in self.description

    def __call__(self, inputs: str, name='run') -> ActionReturn:
        fallback_args = {'inputs': inputs, 'name': name}
        if not self.name2func.get(name):
            return ActionReturn(
                fallback_args,
                type=self.name,
                errmsg=f'invalid API: {name}',
                state=ActionStatusCode.API_ERROR)
        try:
            inputs = self._parser.parse_inputs(inputs, name)
        except ParseError as exc:
            return ActionReturn(
                fallback_args,
                type=self.name,
                errmsg=exc.err_msg,
                state=ActionStatusCode.ARGS_ERROR)
        outputs = self.name2func.get(name)(**inputs)
        try:
            outputs = self.name2func.get(name)(**inputs)
        except Exception as exc:
            return ActionReturn(
                inputs,
                type=self.name,
                errmsg=str(exc),
                state=ActionStatusCode.API_ERROR)
        if isinstance(outputs, ActionReturn):
            action_return = outputs
            if not action_return.args:
                action_return.args = inputs
            if not action_return.type:
                action_return.type = self.name
        else:
            result = self._parser.parse_outputs(outputs)
            action_return = ActionReturn(inputs, type=self.name, result=result)
        return action_return

