import json
import logging
import os
from dataclasses import asdict

from flask import Flask, Response, request

app = Flask(__name__)


def init_agent():
    from datetime import datetime

    from ilagent.agents.python_web import GRAPH_PROMPT_CN, GraphPlanAgent, WebSearchAgent

    from lagent.agents.internlm2_agent import Internlm2Protocol
    from lagent.llms import INTERNLM2_META, LMDeployClient

    protocol_cfg = os.environ.get('protocol_cfg', None)
    if not protocol_cfg:
        protocol_cfg = dict(
            type=Internlm2Protocol,
            interpreter_prompt=GRAPH_PROMPT_CN,
            meta_prompt=datetime.now().strftime(
                'The current date is %Y-%m-%d.'))
    llm_cfg = os.environ.get('llm_cfg', None)
    if not llm_cfg:
        llm_cfg = dict(
            type=LMDeployClient,
            model_name='internlm2-chat-7b',
            url='http://22.8.24.123:23333',
            meta_template=INTERNLM2_META,
            max_new_tokens=4096,
            top_p=0.8,
            top_k=1,
            temperature=0.8,
            repetition_penalty=1.0,
            stop_words=['<|im_end|>'])
    searcher_cfg = os.environ.get('searcher_cfg', None)
    if not searcher_cfg:
        searcher_cfg = dict(type=WebSearchAgent, prompt_lan='cn')
    agent_cfg = os.environ.get('agent_cfg', dict())
    llm = llm_cfg.pop('type')(**llm_cfg)
    protocol = protocol_cfg.pop('type')(**protocol_cfg)
    agent_cfg['llm'] = llm
    agent_cfg['protocol'] = protocol
    agent_cfg['searcher_cfg'] = searcher_cfg
    agent = GraphPlanAgent(**agent_cfg)
    return agent


graph_plan_agent = init_agent()


@app.route('/solve', methods=['POST'])
def run():
    data = request.get_json()
    inputs = data.get('inputs')
    assert inputs

    def generate():
        for response in graph_plan_agent.stream_chat(
                inputs, return_mode='dict'):
            if isinstance(response, tuple):
                agent_return, node_name = response
            else:
                agent_return = response
                node_name = None
            yield json.dumps(
                dict(response=asdict(agent_return), current_node=node_name),
                ensure_ascii=False)

    try:
        response_iterator = Response(generate(), mimetype='application/json')
    except RuntimeError as exc:
        msg = 'An error occurred while generating the response.'
        logging.exception(msg)
        return json.dumps(
            dict(error=dict(msg=msg, details=str(exc)),
                 ensure_ascii=False)), 500
    else:
        return response_iterator


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
