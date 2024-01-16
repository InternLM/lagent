from typing import Optional, Type

from lagent.actions.parser import BaseParser, ParseError
from lagent.schema import ActionReturn


class BaseAction:
    """Base class for all actions.

    Args:
        description (:class:`Optional[dict]`): The description of the action.
            Defaults to ``None``.
        parser (:class:`Type[BaseParser]`): The parser class to process the
            action's inputs and outputs. Defaults to :class:`BaseParser``.
        enable (:class:`bool`): Whether the action is enabled. Defaults to
            ``True``.

    Examples:

        * simple tool

        .. code-block:: python

            desc = dict(
                name='highlight',
                description='highlight a piece of text',
                parameters=[dict(name='text', type='STRING', description='input text')],
                required=['text'],
            )
            action = BaseAction(desc)

        * complex tool with multiple methods

        .. code-block:: python

            desc = dict(
                name='calculate',
                description='a calulator to perform arithmetic operations',
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
            action = BaseAction(desc)
    """

    def __init__(self,
                 description: Optional[dict] = None,
                 parser: Type[BaseParser] = BaseParser,
                 enable: bool = True):
        self._description = description or {}
        self._name = self._description.get('name', self.__class__.__name__)
        self._enable = enable
        self._nested = 'api_list' in self._description
        self._parser = parser(self)

    def __call__(self, inputs: str, name='run') -> ActionReturn:
        try:
            inputs = self._parser.parse_inputs(inputs, name)
        except ParseError as exc:
            action_return = ActionReturn(
                args={'inputs': inputs}, type=self.name, errmsg=exc.err_msg)
            return action_return
        action_return = ActionReturn(args=inputs, type=self.name)
        outputs = getattr(self, name)(**inputs)
        result = self._parser.parse_outputs(outputs)
        action_return.result = result
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
    def nested(self):
        return self._nested

    def __repr__(self):
        return f'{self.description}'

    __str__ = __repr__
