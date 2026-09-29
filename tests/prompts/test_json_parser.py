from unittest import TestCase

from pydantic import BaseModel

from lagent.prompts.parsers.json_parser import JSONParser


class Choice(BaseModel):
    kind: str = 'other'


class TestJSONParser(TestCase):

    def test_an_answer_named_list_is_not_replaced_by_the_default(self):
        parser = JSONParser(template='{format}', format=Choice)
        parsed = parser.parse_response('{"kind": "list"}')
        self.assertEqual(parsed['kind'], 'list')

        echoed = parser.parse_response('{"kind": "str"}')
        self.assertEqual(echoed['kind'], 'other')

        plain = parser.parse_response('{"kind": "hello"}')
        self.assertEqual(plain['kind'], 'hello')
