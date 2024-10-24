import re
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Optional, Union

import jinja2
from pydantic import BaseModel


class PromptTemplate:
    """prompt templates.

    Args:
        template (str): The template string.
        variables (Optional[Union[Dict[str, str], BaseModel, Any]]): Variables for the template.
        format_type (str): The format type of the template ('json' or 'jinja').

    """

    def __init__(self, template: str, format_type: str = 'json') -> None:
        self.template = template
        self.format_type = format_type

    def _convert_to_dict(
        self, variables: Optional[Union[Dict[str, str], BaseModel, Any]]
    ) -> Dict[str, str]:
        """
        Convert variables to a dictionary.

        Args:
            variables (Optional[Union[Dict[str, str], BaseModel, Any]]):
                Variables to convert.

        Returns:
            Dict[str, str]: The converted dictionary.

        Raises:
            ValueError: If the variables type is unsupported.
        """
        if variables is None:
            return {}
        if isinstance(variables, BaseModel):
            return variables.dict()
        if is_dataclass(variables):
            return asdict(variables)
        if isinstance(variables, dict):
            return variables
        raise ValueError(
            'Unsupported variables type. Must be a dict, BaseModel, or '
            'dataclass.')

    def parse_template(self, template: str) -> Dict[str, str]:
        """
        Extract variables from the template.

        Args:
            template (str): The template string.

        Returns:
            Dict[str, str]: A dictionary of variables with None values.
        """
        if self.format_type == 'jinja':
            variables = re.findall(r'\{\{(.*?)\}\}', template)

        elif self.format_type == 'json':
            variables = re.findall(r'\{(.*?)\}', template)
            variables = [var for var in variables if '{' not in var]
        else:
            variables = []
        return {var.strip(): None for var in variables}

    def format_json(self, template: str, variables: Dict[str, str]) -> str:
        """
        Format the JSON template.

        Args:
            template (str): The JSON template string.
            variables (Dict[str, str]): The variables to fill in the template.

        Returns:
            str: The formatted JSON string.

        Raises:
            ValueError: If the template is not a valid JSON.
        """
        try:
            return template.format(**variables)
        except KeyError as e:
            raise ValueError('Invalid JSON template') from e

    def format_jinja(self, template: str, variables: Dict[str, str]) -> str:
        """
        Format the Jinja template.

        Args:
            template (str): The Jinja template string.
            variables (Dict[str, str]): The variables to fill in the template.

        Returns:
            str: The formatted Jinja string.

        Raises:
            ValueError: If the template is not a valid Jinja template.
        """
        try:
            jinja_template = jinja2.Template(template)
            return jinja_template.render(variables)
        except jinja2.TemplateError as e:
            raise ValueError('Invalid Jinja template') from e

    def _update_variables_with_info(self) -> Dict[str, str]:
        """
        Update variables dictionary with action_info and agents_info.

        Returns:
            Dict[str, str]: The updated variables dictionary.
        """
        variables = self.variables.copy()
        if 'action_info' not in variables and self.actions_info:
            variables['action_info'] = self.actions_info
        if 'agents_info' not in variables and self.agents_info:
            variables['agents_info'] = self.agents_info
        return variables

    def _check_variables_match(self, parsed_variables: Dict[str, str],
                               variables: Dict[str, str]) -> None:
        """
        Check if all keys in variables are present in parsed_variables.

        Args:
            parsed_variables (Dict[str, str]): The parsed variables from
                the template.
            variables (Dict[str, str]): The variables to check.

        Raises:
            ValueError: If any key in variables is not present in
                parsed_variables.
        """
        if not all(key in parsed_variables for key in variables.keys()):
            raise ValueError(
                'Variables keys do not match the template variables')

    def format(
        self,
        **kwargs: Optional[Union[Dict[str, str], BaseModel, Any]],
    ) -> Any:
        self.variables = kwargs
        return str(self)

    def __str__(self) -> Any:
        """
        Call the template formatting based on format_type.

        Returns:
            Any: The formatted template.

        Raises:
            ValueError: If the format_type is unsupported.
        """
        parsed_variables = self.parse_template(self.template)
        updated_variables = self._update_variables_with_info()
        self._check_variables_match(parsed_variables, updated_variables)

        if self.format_type == 'json':
            return self.format_json(self.template, updated_variables)
        elif self.format_type == 'jinja':
            return self.format_jinja(self.template, updated_variables)
        else:
            raise ValueError('Unsupported format type')

    @property
    def actions_info(self) -> Optional[Dict[str, Any]]:
        """Get the action information."""
        return getattr(self, '_action_info', None)

    @actions_info.setter
    def actions_info(self, value: Dict[str, Any]) -> None:
        """Set the action information."""
        self._action_info = value

    @property
    def agents_info(self) -> Optional[Dict[str, Any]]:
        """Get the agent information."""
        return getattr(self, '_agents_info', None)

    @agents_info.setter
    def agents_info(self, value: Dict[str, Any]) -> None:
        """Set the agent information."""
        self._agents_info = value
