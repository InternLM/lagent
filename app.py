import asyncio
import json
import logging
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Union

import janus
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from lagent.schema import AgentStatusCode


@dataclass
class InvalidConfig:
    type: Optional[str] = None
    cfg: Optional[dict] = None
    details: Optional[str] = None


def init_agent(cfg):
    from datetime import datetime

    import lagent.actions as action_factory
    import lagent.agents as agent_factory
    import lagent.llms as llm_factory
    from lagent.actions import ActionExecutor
    from lagent.agents.mindsearch_prompt import GRAPH_PROMPT_CN, searcher_input_template_cn, searcher_system_prompt_cn
    from lagent.llms import INTERNLM2_META

    def init_module(module_cfg, module_factory):
        try:
            module_type = module_cfg.pop('type')
            module_class = getattr(module_factory, module_type)
            module = module_class(**module_cfg)
            return module
        except Exception as exc:
            logging.exception(str(exc))
            return InvalidConfig(
                type=module_type, cfg=module_cfg, details=str(exc))

    def init_executor(module_cfg):
        plugin_executor = None
        interpreter_executor = None
        if 'plugin' in module_cfg:
            actions = []
            plugin_cfg = module_cfg.pop('plugin')
            for ac_cfg in plugin_cfg:
                ac = init_module(ac_cfg, action_factory)
                if isinstance(ac, InvalidConfig):
                    return ac
                actions.append(ac)
            plugin_executor = ActionExecutor(actions=actions)
        if 'interpreter' in module_cfg:
            interpreter_cfg = module_cfg.pop('interpreter')
            ci = init_module(interpreter_cfg, action_factory)
            if isinstance(ci, InvalidConfig):
                return ci
            interpreter_executor = ActionExecutor(actions=[ci])
        return plugin_executor, interpreter_executor

    llm_cfg = cfg.get('llm', None)
    if not llm_cfg:
        llm_cfg = dict(
            type='LMDeployClient',
            model_name='internlm2-chat-7b',
            url=os.environ.get('LLM_URL', 'http://localhost:23333'),
            meta_template=INTERNLM2_META,
            max_new_tokens=4096,
            top_p=0.8,
            top_k=1,
            temperature=0.8,
            repetition_penalty=1.02,
            stop_words=['<|im_end|>'])
    llm = init_module(llm_cfg, llm_factory)
    if isinstance(llm, InvalidConfig):
        return llm
    if cfg.get('type', None) is None:
        cfg['type'] = 'MindSearchAgent'
    searcher_cfg = cfg.get('searcher', None)
    if searcher_cfg:
        cfg.pop('searcher')
    if cfg['type'] == 'MindSearchAgent' and searcher_cfg is None:
        searcher_cfg = dict(
            llm=llm,
            plugin=[dict(type='BingBrowser', api_key='Your API Key')],
            protocol=dict(
                type='MindSearchProtocol',
                meta_prompt=datetime.now().strftime(
                    'The current date is %Y-%m-%d.'),
                plugin_prompt=searcher_system_prompt_cn,
            ),
            template=searcher_input_template_cn)
    if searcher_cfg:
        # searcher initialization
        if 'type' in searcher_cfg:
            searcher_cfg.pop('type')
        if isinstance(searcher_cfg['llm'], dict):
            searcher_llm = init_module(searcher_cfg['llm'], llm_factory)
            if isinstance(searcher_llm, InvalidConfig):
                return searcher_llm
            searcher_cfg['llm'] = searcher_llm
        searcher_protocol = init_module(searcher_cfg['protocol'],
                                        agent_factory)
        if isinstance(searcher_protocol, InvalidConfig):
            return searcher_protocol
        searcher_cfg['protocol'] = searcher_protocol
        executors = init_executor(searcher_cfg)
        if isinstance(executors, InvalidConfig):
            return executors
        searcher_cfg['plugin_executor'] = executors[0]
        cfg['searcher_cfg'] = searcher_cfg
    if cfg.get('protocol', None) is None:
        cfg['protocol'] = dict(
            type='MindSearchProtocol',
            meta_prompt=datetime.now().strftime(
                'The current date is %Y-%m-%d.'),
            interpreter_prompt=GRAPH_PROMPT_CN,
            response_prompt='请根据上文内容对问题给出详细的回复')
    if cfg.get('max_turn', None) is None:
        cfg['max_turn'] = 10
    # agent initialization
    cfg['llm'] = llm
    protocol = init_module(cfg['protocol'], agent_factory)
    if isinstance(protocol, InvalidConfig):
        return protocol
    cfg['protocol'] = protocol
    executors = init_executor(cfg)
    if isinstance(executors, InvalidConfig):
        return executors
    if isinstance(executors[0], ActionExecutor):
        cfg['plugin_executor'] = executors[0]
    if isinstance(executors[1], ActionExecutor):
        cfg['interpreter_executor'] = executors[1]
    agent = init_module(cfg, agent_factory)
    return agent


# agent = os.environ.get('agent_cfg', dict())

app = FastAPI(docs_url='/')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'])


class GenerationParams(BaseModel):
    inputs: Union[str, List[Dict]]
    agent_cfg: Dict = dict()


@app.post('/solve')
async def run(request: GenerationParams):
    inputs = request.inputs
    agent = init_agent(request.agent_cfg)
    if not inputs:
        raise HTTPException(status_code=400, detail='Inputs are required')
    if isinstance(agent, InvalidConfig):
        raise InvalidConfig(**agent)

    async def generate():
        try:
            queue = janus.Queue()

            # 使用 run_in_executor 将同步生成器包装成异步生成器
            def sync_generator_wrapper():
                try:
                    for response in agent.stream_chat(
                            inputs, return_mode='dict'):
                        queue.sync_q.put(response)
                except KeyError as e:
                    logging.error(f'KeyError in sync_generator_wrapper: {e}')

            async def async_generator_wrapper():
                loop = asyncio.get_event_loop()
                loop.run_in_executor(None, sync_generator_wrapper)
                while True:
                    response = await queue.async_q.get()
                    yield response
                    if not isinstance(
                            response,
                            tuple) and response.state == AgentStatusCode.END:
                        break

            async for response in async_generator_wrapper():
                if isinstance(response, tuple):
                    agent_return, node_name = response
                else:
                    agent_return = response
                    node_name = None
                response_json = json.dumps(
                    dict(
                        response=asdict(agent_return), current_node=node_name),
                    ensure_ascii=False)
                yield {'data': response_json}
                # yield f'data: {response_json}\n\n'
        except Exception as exc:
            msg = 'An error occurred while generating the response.'
            logging.exception(msg)
            response_json = json.dumps(
                dict(error=dict(msg=msg, details=str(exc))),
                ensure_ascii=False)
            yield {'data': response_json}
            # yield f'data: {response_json}\n\n'

    return EventSourceResponse(generate())


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000, log_level='info')
