from typing import Optional

# from agentlego.parsers import DefaultParser
from agentlego.tools.remote import RemoteTool

from lagent import BaseAction
from lagent.actions.parser import JsonParser


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
            api_list = [RemoteTool.from_url(url).to_lagent()]
        else:
            api_list = [
                api.to_lagent() for api in RemoteTool.from_openapi(**spec)
            ]
        api_desc = []
        for api in api_list:
            api_desc.append(api.description)
        if len(api_list) > 1:
            tool_description = dict(name=type, api_list=api_desc)
            self.add_method(api_list)
        else:
            tool_description = api_desc[0]
            setattr(self, 'run', api_list[0].run)
        super().__init__(
            description=tool_description, parser=parser, enable=enable)

    @property
    def is_toolkit(self):
        return 'api_list' in self.description

    def add_method(self, funcs):
        for func in funcs:
            setattr(self, func.name, func.run)
