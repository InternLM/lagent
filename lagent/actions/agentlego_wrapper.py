from typing import List, Optional
import types
from copy import deepcopy
from lagent import BaseAction
from lagent.actions.parser import JsonParser

from agentlego.tools.remote import RemoteTool
from agentlego.parsers import DefaultParser


class AgentLegoToolkit(BaseAction):
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
        if url is not None and not url.endswith('.json'):
            api_list = [RemoteTool(url)]
        else:
            api_list = [api for api in RemoteTool.from_openapi(**spec)]
        api_desc = []
        for api in api_list:
            api.set_parser(DefaultParser)
            desc = deepcopy(api.toolmeta.__dict__)
            description_normalization(desc)
            api_desc.append(desc)
        if len(api_list) > 1:
            tool_description = dict(
                name=type,
                api_list=api_desc
            )
            self.add_method(api_list)
        else:
            tool_description = api_desc[0]
            setattr(self, 'run', api_list[0])
        super().__init__(
            description=tool_description,
            parser=parser,
            enable=enable
        )

    @property
    def is_toolkit(self):
        return 'api_list' in self.description

    def add_method(self, funcs):
        for func in funcs:
            setattr(self, func.name, func)


def description_normalization(desc: dict):
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
