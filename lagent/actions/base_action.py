from typing import Optional, Type

from lagent.actions.parser import BaseParser, JsonParser, ParseError
from lagent.schema import ActionReturn


class BaseAction:
    """Base class for all actions.

    Args:
        description (:class:`Optional[dict]`): The description of the action.
            Defaults to ``None``.
        parser (:class:`Type[BaseParser]`): The parser class to process the
            action's inputs and outputs. Defaults to :class:`JsonParser``.
        enable (:class:`bool`): Whether the action is enabled. Defaults to
            ``True``.

    Examples:

        * simple tool

        .. code-block:: python
        
            class Bold(BaseAction):
                def run(self, text):
                    return '**' + text + '**'

            desc = dict(
                name='bold',
                description='make text bold',
                parameters=[dict(name='text', type='STRING', description='input text')],
                required=['text'],
            )
            action = Bold(desc)

        * toolkit with multiple APIs

        .. code-block:: python
        
            class Calculator(BaseAction):
                def add(self, a, b):
                    return a + b
                    
                def sub(self, a, b):
                    return a - b

            desc = dict(
                name='calculate',
                description='perform arithmetic operations',
                api_list=[
                    dict(
                        name='add',
                        descrition='addition operation',
                        parameters=[
                            dict(name='a', type='NUMBER', description='augend'),
                            dict(name='b', type='NUMBER', description='addend'),
                        ],
                        required=['a', 'b'],
                    ),
                    dict(
                        name='sub',
                        description='subtraction operation',
                        parameters=[
                            dict(name='a', type='NUMBER', description='minuend'),
                            dict(name='b', type='NUMBER', description='subtrahend'),
                        ],
                        required=['a', 'b'],
                    )
                ]
            )
            action = Calculator(desc)
    """

    def __init__(self,
                 description: Optional[dict] = None,
                 parser: Type[BaseParser] = JsonParser,
                 enable: bool = True):
        self._description = description.copy() if description else {}
        self._name = self._description.get('name', self.__class__.__name__)
        self._enable = enable
        self._is_toolkit = 'api_list' in self._description
        self._parser = parser(self)

    def __call__(self, inputs: str, name='run') -> ActionReturn:
        fallback_args = {'inputs': inputs, 'name': name}
        if not hasattr(self, name):
            return ActionReturn(
                fallback_args, type=self.name, errmsg=f'invalid API: {name}')
        try:
            inputs = self._parser.parse_inputs(inputs, name)
        except ParseError as exc:
            return ActionReturn(
                fallback_args, type=self.name, errmsg=exc.err_msg)
        try:
            outputs = getattr(self, name)(**inputs)
        except Exception as exc:
            return ActionReturn(inputs, type=self.name, errmsg=str(exc))
        if isinstance(outputs, ActionReturn):
            action_return = outputs
        else:
            result = self._parser.parse_outputs(outputs)
            action_return = ActionReturn(inputs, type=self.name, result=result)
        return action_return

    def run(self):
        return NotImplementedError

    @property
    def name(self):
        return self._name

    @property
    def enable(self):
        return self._enable

    @property
    def description(self):
        return self._description

    @property
    def is_toolkit(self):
        return self._is_toolkit

    def __repr__(self):
        return f'{self.description}'

    __str__ = __repr__
