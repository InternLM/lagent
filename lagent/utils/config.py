import collections.abc
import importlib.util
import os
import types

from benedict import benedict


class ConfigDict(dict):

    def __getattr__(self, item):
        if item in self:
            return self[item]
        raise AttributeError(f"'ConfigDict' object has no attribute '{item}'")

    def __setattr__(self, key, value):
        self[key] = value


class Config:

    @staticmethod
    def fromfile(file_path):
        config_dict = ConfigDict()
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f'Config file not found: {file_path}')

        # Load the configuration file as a module
        spec = importlib.util.spec_from_file_location('config_module', file_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)

        # Function to convert nested dictionaries to ConfigDict recursively
        def convert_to_config_dict(d):
            if isinstance(d, dict):

                config_dict = ConfigDict()
                for key, value in d.items():
                    if isinstance(value, dict):
                        config_dict[key] = convert_to_config_dict(value)
                    else:
                        config_dict[key] = value
                return config_dict
            else:
                return d

        # Retrieve all attributes (variables) from the module
        for attribute_name in dir(config_module):
            if not attribute_name.startswith('__'):
                config_dict[attribute_name] = convert_to_config_dict(getattr(config_module, attribute_name))
        for key, value in list(config_dict.items()):
            if isinstance(value, (types.FunctionType, types.ModuleType)):
                config_dict.pop(key)
        return config_dict


def to_native_types(obj: benedict):
    """
    递归地将任何嵌套的自定义对象（如 benedict）和集合
    转换为 Python 的原生类型（dict, list, tuple, set 等）。
    """
    # 1. 最优先处理类字典对象
    if isinstance(obj, dict):
        return {key: to_native_types(value) for key, value in obj.items()}

    # 2. 处理非字符串的可迭代对象 (list, tuple, set 等)
    #    必须先排除 str 和 bytes，因为它们也是可迭代的，但我们不希望遍历其字符。
    if isinstance(obj, collections.abc.Iterable) and not isinstance(obj, (str, bytes)):
        # type(obj) -> 获取原始容器的类型 (如 list, tuple, 或 set)
        # (...)     -> 使用该类型的构造函数，重新创建一个新的、已转换内容的容器。
        return type(obj)(to_native_types(item) for item in obj)

    # 3. 如果不是以上任何情况（如 int, str, bool, None 等），直接返回值
    return obj
