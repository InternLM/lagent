from typing import Type

from .action_executor import ActionExecutor
from .arxiv_search import ArxivSearch
from .base_action import TOOL_REGISTRY, BaseAction, tool_api
from .bing_browser import BingBrowser
from .bing_map import BINGMap
from .builtin_actions import FinishAction, InvalidAction, NoAction
from .google_scholar_search import GoogleScholar
from .google_search import GoogleSearch
from .ipython_interactive import IPythonInteractive
from .ipython_interpreter import IPythonInterpreter
from .ipython_manager import IPythonInteractiveManager
from .parser import BaseParser, JsonParser, TupleParser
from .ppt import PPT
from .python_interpreter import PythonInterpreter

__all__ = [
    'BaseAction', 'ActionExecutor', 'InvalidAction', 'FinishAction',
    'NoAction', 'BINGMap', 'ArxivSearch', 'FinishAction', 'GoogleSearch',
    'GoogleScholar', 'IPythonInterpreter', 'IPythonInteractive',
    'IPythonInteractiveManager', 'PythonInterpreter', 'PPT', 'BaseParser',
    'JsonParser', 'TupleParser', 'tool_api', 'list_tools', 'get_tool_cls',
    'get_tool', 'BingBrowser'
]


def list_tools(with_class: bool = False):
    """List available tools.

    Args:
        with_class (bool): whether to return the action class along
            with its name. Defaults to ``False``.

    Returns:
        list: all action names
    """
    return list(TOOL_REGISTRY.items()) if with_class else list(
        TOOL_REGISTRY.keys())


def get_tool_cls(specifier: str) -> Type[BaseAction]:
    """Get the action class.

    Args:
        specifier (:class:`str`): tool name

    Returns:
        Type[BaseAction]: action class
    """
    return TOOL_REGISTRY.get_class(specifier)


def get_tool(specifier: str, *args, **kwargs) -> BaseAction:
    """Instantiate an action.

    Args:
        specifier (str): tool name
        args: positional arguments passed to the action's ``__init__`` method
        kwargs: keyword arguments passed to the action's ``__init__`` method

    Returns:
        :class:`BaseAction`: action object
    """
    return TOOL_REGISTRY.get(specifier, *args, **kwargs)
