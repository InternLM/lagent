import json
from ast import literal_eval
from typing import Any


class ParseError(Exception):
    """Parsing exception class"""

    def __init__(self, err_msg: str):
        self.err_msg = err_msg


class BaseParser:
    """Base parser to process inputs and outputs of actions.

    Args:
        action (:class:`BaseAction`): action to validate

    Attributes:
        PARAMETER_DESCRIPTION (:class:`str`): declare the input format which
            LLMs should follow when generating arguments for decided tools.
    """

    PARAMETER_DESCRIPTION: str = ''

    def __init__(self, action):
        self.action = action
        self._api2param = {}
        self._api2required = {}
        # perform basic argument validation
        if action.description:
            for api in action.description.get('api_list',
                                              [action.description]):
                name = (f'{action.name}.{api["name"]}'
                        if self.action.is_toolkit else api['name'])
                required_parameters = set(api['required'])
                all_parameters = {j['name'] for j in api['parameters']}
                if not required_parameters.issubset(all_parameters):
                    raise ValueError(
                        f'unknown parameters for function "{name}": '
                        f'{required_parameters - all_parameters}')
                if self.PARAMETER_DESCRIPTION:
                    api['parameter_description'] = self.PARAMETER_DESCRIPTION
                api_name = api['name'] if self.action.is_toolkit else 'run'
                self._api2param[api_name] = api['parameters']
                self._api2required[api_name] = api['required']

    def parse_inputs(self, inputs: str, name: str = 'run') -> dict:
        """parse inputs LLMs generate for the action

        Args:
            inputs (:class:`str`): input string extracted from responses
            
        Returns:
            :class:`dict`: processed input
        """
        inputs = {self._api2param[name][0]['name']: inputs}
        return inputs

    def parse_outputs(self, outputs: Any) -> dict:
        """parser outputs returned by the action

        Args:
            outputs (:class:`Any`): raw output of the action

        Returns:
            :class:`dict`: processed output
        """
        if isinstance(outputs, dict):
            outputs = json.dumps(outputs, ensure_ascii=False)
        elif not isinstance(outputs, str):
            outputs = str(outputs)
        return {'text': outputs}


class JsonParser(BaseParser):
    """Json parser to convert input string into a dictionary.

    Args:
        action (:class:`BaseAction`): action to validate
    """

    PARAMETER_DESCRIPTION = '如果调用该工具，你必须使用Json格式 {key: value} 传参，其中key为参数名称'

    def parse_inputs(self, inputs: str, name: str = 'run') -> dict:
        try:
            inputs = json.loads(inputs)
        except json.JSONDecodeError as exc:
            raise ParseError(f'invalid json format: {inputs}') from exc
        input_keys = set(inputs)
        all_keys = {param['name'] for param in self._api2param[name]}
        if not input_keys.issubset(all_keys):
            raise ParseError(f'unknown arguments: {input_keys - all_keys}')
        required_keys = set(self._api2required[name])
        if not input_keys.issuperset(required_keys):
            raise ParseError(
                f'missing required arguments: {required_keys - input_keys}')
        return inputs


class TupleParser(BaseParser):
    """Tuple parser to convert input string into a tuple.

    Args:
        action (:class:`BaseAction`): action to validate
    """

    PARAMETER_DESCRIPTION = '如果调用该工具，你必须使用Tuple格式 (arg1, arg2, arg3) 传参，且参数是有序的'

    def parse_inputs(self, inputs: str, name: str = 'run') -> dict:
        try:
            inputs = literal_eval(inputs)
        except Exception as exc:
            raise ParseError(f'invalid tuple format: {inputs}') from exc
        if len(inputs) < len(self._api2required[name]):
            raise ParseError(
                f'API takes {len(self._api2required[name])} required positional '
                f'arguments but {len(inputs)} were given')
        if len(inputs) > len(self._api2param[name]):
            raise ParseError(
                f'API takes {len(self._api2param[name])} positional arguments '
                f'but {len(inputs)} were given')
        inputs = {
            self._api2param[name][i]['name']: item
            for i, item in enumerate(inputs)
        }
        return inputs
