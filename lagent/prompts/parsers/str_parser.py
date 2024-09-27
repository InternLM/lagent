from typing import Any


class StrParser:

    def __init__(
        self,
        template: str = '',
        **format_field,
    ):
        self.template = template
        self.format_field = format_field

    def format_instruction(self) -> Any:
        format_data = {
            key: self.format_to_string(value)
            for key, value in self.format_field.items()
        }
        return self.template.format(**format_data)

    def format_to_string(self, format_model: Any) -> str:
        return format_model

    def format_response(self, parsed: dict) -> str:
        raise NotImplementedError

    def parse_response(self, data: str) -> str:
        return data
