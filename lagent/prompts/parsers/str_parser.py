import string
from typing import Any


class StrParser:

    def __init__(
        self,
        template: str = '',
        **format_field,
    ):
        fields = {item[1] for item in string.Formatter().parse(template) if item[1] is not None}
        if not fields.issubset(format_field.keys()):
            raise ValueError(
                'not all required fields of "template" are provided, missing '
                f'{fields - format_field.keys()}. Please pass them as keyword arguments.'
            )
        self.template = template
        self.format_field = format_field

    def format_instruction(self) -> Any:
        format_data = {key: self.format_to_string(value) for key, value in self.format_field.items()}
        return self.template.format(**format_data)

    def format_to_string(self, format_model: Any) -> str:
        return format_model

    def format_response(self, parsed: dict) -> str:
        raise NotImplementedError

    def parse_response(self, data: str) -> str:
        return data
