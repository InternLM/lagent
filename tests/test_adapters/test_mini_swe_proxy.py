import asyncio
import sys
from types import ModuleType, SimpleNamespace

from lagent.adapters.mini_swe_agent import MiniSWEAgentAdapter
from lagent.adapters.proxy import SessionClient


def test_mini_swe_uses_versioned_proxy_base(monkeypatch):
    captured = {}

    def model_factory(*, config):
        captured.update(config)
        return object()

    fake_agent = SimpleNamespace(run=lambda *args, **kwargs: {'submission': 'done'})
    modules = {
        'minisweagent': {},
        'minisweagent.agents': {'get_agent': lambda *args, **kwargs: fake_agent},
        'minisweagent.config': {'get_config_from_spec': lambda name: {}},
        'minisweagent.environments': {'get_environment': lambda *args, **kwargs: object()},
        'minisweagent.models': {'get_model': model_factory},
        'minisweagent.utils': {},
        'minisweagent.utils.serialize': {'recursive_merge': lambda base, config, override: config},
    }
    for name, attributes in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    proxy = SessionClient(real_api_key='EMPTY', real_base_url='http://upstream.test/v1',
                          session_id='mini-test', port=12345)
    adapter = MiniSWEAgentAdapter(model='model', proxy=proxy)
    assert asyncio.run(adapter.run_external_async('task')) == 'done'
    assert captured['model_kwargs']['api_base'] == proxy.openai_base_url
    assert captured['model_kwargs']['api_base'].endswith('/v1')
