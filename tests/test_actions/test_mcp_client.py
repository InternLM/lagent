from types import SimpleNamespace

from lagent.actions.mcp_client import AsyncMCPClient


class StubMCPClient(AsyncMCPClient):

    async def _fetch_tool_metadata(self):
        return [
            SimpleNamespace(
                name='search',
                description='Search with a prebound credential.',
                inputSchema={
                    'properties': {
                        'query': {
                            'type': 'string',
                            'description': 'Search query.',
                        },
                        'api_key': {
                            'type': 'string',
                            'description': 'Credential.',
                        },
                        'limit': {
                            'type': 'integer',
                            'description': 'Result limit.',
                        },
                    },
                    'required': ['query', 'api_key'],
                },
            ),
        ]


def test_mcp_extra_args_are_removed_from_required_schema():
    client = StubMCPClient(
        server_type='http',
        url='https://example.invalid/mcp',
        extra_args={'api_key': 'secret'},
    )

    assert [param['name'] for param in client.description['parameters']] == [
        'query',
        'limit',
    ]
    assert client.description['required'] == ['query']
