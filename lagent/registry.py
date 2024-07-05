import inspect
from abc import ABCMeta
from typing import Any, Dict, Optional, Union

from class_registry import AutoRegister, ClassRegistry


def is_class(obj):
    return inspect.isclass(obj)


class RegistryMeta(ABCMeta):
    """Metaclass of tools."""

    def __new__(mcs, name, base, attrs):
        attrs.setdefault('__name__', name)
        return super().__new__(mcs, name, base, attrs)


AGENT_REGISTRY = ClassRegistry('__name__', unique=True)
LLM_REGISTRY = ClassRegistry('__name__', unique=True)
HOOK_REGISTRY = ClassRegistry('__name__', unique=True)
MEMORY_REGISTRY = ClassRegistry('__name__', unique=True)
PARSER_REGISTRY = ClassRegistry('__name__', unique=True)
AGGREGATOR_REGISTRY = ClassRegistry('__name__', unique=True)
TOOL_REGISTRY = ClassRegistry('__name__', unique=True)


class ObjectFactory:

    @staticmethod
    def create(config: Union[Dict, Any],
               registry: Optional[ClassRegistry] = None):
        if config is None:
            return None
        if is_class(config):
            return config
        assert isinstance(config, dict) and 'type' in config

        config = config.copy()
        obj_type = config.pop('type')
        if isinstance(obj_type, str):
            assert registry is not None
            obj_type = registry.get_class(obj_type)
        obj = obj_type(**config)
        return obj


__all__ = [
    'ObjectFactory',
    'AutoRegister',
    'RegistryMeta',
    'is_class',
    'AGENT_REGISTRY',
    'LLM_REGISTRY',
    'TOOL_REGISTRY',
    'HOOK_REGISTRY',
    'MEMORY_REGISTRY',
    'PARSER_REGISTRY',
    'AGGREGATOR_REGISTRY',
]
