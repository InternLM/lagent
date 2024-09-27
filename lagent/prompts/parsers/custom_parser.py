import re
from typing import Any, Dict, Union

from pydantic import BaseModel, ValidationError

from lagent.prompts.parsers.str_parser import StrParser


class CustomFormatParser(StrParser):

    def _extract_fields_with_metadata(
            self, model: BaseModel) -> Dict[str, Dict[str, Any]]:
        fields_metadata = {}
        for field_name, field in model.model_fields.items():
            fields_metadata[field_name] = {
                'annotation': field.annotation,
                'default': field.default
                if field.default is not None else '<required>',
                'comment': field.description if field.description else ''
            }
        return fields_metadata

    def format_to_string(self, format_model: BaseModel) -> str:
        fields = self._extract_fields_with_metadata(format_model)
        formatted_str = ''
        for field_name, metadata in fields.items():
            comment = metadata.get('comment', '')
            field_annotation = metadata['annotation'].__name__ if metadata[
                'annotation'] is not None else 'Any'
            if comment:
                formatted_str += f'<!-- {comment} -->\n'
            formatted_str += f'<{field_name} type="{field_annotation}">{metadata["default"] if metadata["default"] != "<required>" else ""}</{field_name}>\n'
        return formatted_str

    def parse_response(self, data: str) -> Union[dict, BaseModel]:
        pattern = re.compile(r'(<!--\s*(.*?)\s*-->)?\s*<(\w+)[^>]*>(.*?)</\3>',
                             re.DOTALL)
        matches = pattern.findall(data)

        data_dict = {}
        for _, comment_text, key, value in matches:
            if comment_text:
                self.fields[key]['comment'] = comment_text.strip()
            data_dict[key] = value

        model = self.default_format
        if self.unknown_format and not self._is_valid_format(
                data_dict, self.default_format):
            model = self.unknown_format

        return model.model_validate(data_dict)

    def _is_valid_format(self, data: Dict, format_model: BaseModel) -> bool:
        try:
            format_model.model_validate(data)
            return True
        except ValidationError:
            return False


if __name__ == '__main__':
    # Example usage
    class DefaultFormat(BaseModel):
        name: str
        age: int

    class UnknownFormat(BaseModel):
        title: str
        year: int

    template = """如果了解该问题请按照一下格式回复
                    ```html
                    {format}
                    ```
                    否则请回复
                    ```html
                        {unknown_format}
                        ```
                        """
    parser = CustomFormatParser(
        template, default_format=DefaultFormat, unknown_format=UnknownFormat)

    # Example data
    response = '''
    <!-- User's full name -->
    <name type="str">John Doe</name>
    <!-- User's age -->
    <age type="int">30</age>
    '''

    result = parser.parse_response(response)
    print(result)
