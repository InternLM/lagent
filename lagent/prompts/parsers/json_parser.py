import json
from typing import Any, Dict, List, Union, get_args, get_origin

from pydantic import BaseModel, Field
from pydantic_core import PydanticUndefined

from lagent.prompts.parsers.str_parser import StrParser


def get_field_type_name(field_type):
    # 获取字段类型的起源类型（对于复合类型，如 List、Dict 等）
    origin = get_origin(field_type)
    if origin:
        # 获取复合类型的所有参数
        args = get_args(field_type)
        # 重新构建类型名称，例如 List[str] 或 Optional[int]
        args_str = ', '.join([get_field_type_name(arg) for arg in args])
        return f'{origin.__name__}[{args_str}]'
    # 如果不是复合类型，直接返回类型的名称
    elif hasattr(field_type, '__name__'):
        return field_type.__name__
    else:
        return str(field_type)  # 处理一些特殊情况，如来自未知库的类型


# class JSONParser(BaseParser):
class JSONParser(StrParser):

    def _extract_fields_with_metadata(
            self, model: BaseModel) -> Dict[str, Dict[str, Any]]:
        fields_metadata = {}
        for field_name, field in model.model_fields.items():
            fields_metadata[field_name] = {
                'annotation': field.annotation,
                'default': field.default
                if field.default is not PydanticUndefined else '<required>',
                'comment': field.description if field.description else ''
            }

            # 类型检查，以支持 BaseModel 的子类
            origin = get_origin(field.annotation)
            args = get_args(field.annotation)
            if origin is None:
                # 不是复合类型，直接检查是否为 BaseModel 的子类
                if isinstance(field.annotation, type) and issubclass(
                        field.annotation, BaseModel):
                    fields_metadata[field_name][
                        'fields'] = self._extract_fields_with_metadata(
                            field.annotation)
            else:
                # 是复合类型，检查其中是否有 BaseModel 的子类
                for arg in args:
                    if isinstance(arg, type) and issubclass(arg, BaseModel):
                        fields_metadata[field_name][
                            'fields'] = self._extract_fields_with_metadata(arg)
                        break
        return fields_metadata

    def _format_field(self,
                      field_name: str,
                      metadata: Dict[str, Any],
                      indent: int = 1) -> str:
        comment = metadata.get('comment', '')
        field_type = get_field_type_name(
            metadata['annotation']
        ) if metadata['annotation'] is not None else 'Any'
        default_value = metadata['default']
        indent_str = '    ' * indent
        formatted_lines = []

        if comment:
            formatted_lines.append(f'{indent_str}// {comment}')

        if 'fields' in metadata:
            formatted_lines.append(f'{indent_str}"{field_name}": {{')
            for sub_field_name, sub_metadata in metadata['fields'].items():
                formatted_lines.append(
                    self._format_field(sub_field_name, sub_metadata,
                                       indent + 1))
            formatted_lines.append(f'{indent_str}}},')
        else:
            if default_value == '<required>':
                formatted_lines.append(
                    f'{indent_str}"{field_name}": "{field_type}",  // required'
                )
            else:
                formatted_lines.append(
                    f'{indent_str}"{field_name}": "{field_type}",  // default: {default_value}'
                )

        return '\n'.join(formatted_lines)

    def format_to_string(self, format_model) -> str:
        fields = self._extract_fields_with_metadata(format_model)
        formatted_lines = []
        for field_name, metadata in fields.items():
            formatted_lines.append(self._format_field(field_name, metadata))

        # Remove the trailing comma from the last line
        if formatted_lines and formatted_lines[-1].endswith(','):
            formatted_lines[-1] = formatted_lines[-1].rstrip(',')

        return '{\n' + '\n'.join(formatted_lines) + '\n}'

    def parse_response(self, data: str) -> Union[dict, BaseModel]:
        # Remove comments
        data_no_comments = '\n'.join(
            line for line in data.split('\n')
            if not line.strip().startswith('//'))
        try:
            data_dict = json.loads(data_no_comments)
            parsed_data = {}

            for field_name, value in self.format_field.items():
                if self._is_valid_format(data_dict, value):
                    model = value
                    break

            self.fields = self._extract_fields_with_metadata(model)

            for field_name, value in data_dict.items():
                if field_name in self.fields:
                    metadata = self.fields[field_name]
                    if value in [
                            'str', 'int', 'float', 'bool', 'list', 'dict'
                    ]:
                        if metadata['default'] == '<required>':
                            raise ValueError(
                                f"Field '{field_name}' is required but not provided"
                            )
                        parsed_data[field_name] = metadata['default']
                    else:
                        parsed_data[field_name] = value

            return model.model_validate(parsed_data).dict()
        except json.JSONDecodeError:
            raise ValueError('Input string is not a valid JSON.')

    def _is_valid_format(self, data: dict, format_model: BaseModel) -> bool:
        try:
            format_model.model_validate(data)
            return True
        except Exception:
            return False


if __name__ == '__main__':

    # Example usage
    class DefaultFormat(BaseModel):
        name: List[str] = Field(description='Name of the person')
        age: int = Field(description='Age of the person')

    class UnknownFormat(BaseModel):
        title: str
        year: int

    TEMPLATE = """如果了解该问题请按照一下格式回复
    ```json
    {format}
    ```
    否则请回复
    ```json
    {unknown_format}
    ```
    """

    parser = JSONParser(
        template=TEMPLATE,
        default_format=DefaultFormat,
        unknown_format=UnknownFormat,
    )

    # Example data
    data = '''
    {
        "name": ["John Doe"],
        "age": 30
    }
    '''
    print(parser.format())
    result = parser.parse_response(data)
    print(result)
