import json

import gradio as gr

from lagent.schema import AgentStatusCode


# Initialize the agent
def init_agent():
    from datetime import datetime

    from lagent.actions import ActionExecutor, BingBrowser
    from lagent.agents.mindsearch_agent import MindSearchAgent, MindSearchProtocol
    # from lagent.agents.mindsearch_prompt import GRAPH_PROMPT_EN
    # from lagent.agents.mindsearch_prompt import searcher_input_template_en
    # from lagent.agents.mindsearch_prompt import searcher_system_prompt_en
    from lagent.agents.mindsearch_prompt import GRAPH_PROMPT_CN, searcher_input_template_cn, searcher_system_prompt_cn
    from lagent.llms import INTERNLM2_META, LMDeployClient

    llm = LMDeployClient(
        model_name='internlm2-chat-7b',
        url='http://22.8.24.123:23333',
        meta_template=INTERNLM2_META,
        max_new_tokens=4096,
        top_p=0.8,
        top_k=1,
        temperature=0.8,
        repetition_penalty=1.0,
        stop_words=['<|im_end|>'])

    agent = MindSearchAgent(
        llm=llm,
        searcher_cfg=dict(
            llm=llm,
            plugin_executor=ActionExecutor(BingBrowser('Your API Key')),
            protocol=MindSearchProtocol(
                meta_prompt=datetime.now().strftime(
                    'The current date is %Y-%m-%d.'),
                plugin_prompt=searcher_system_prompt_cn,
            ),
            template=searcher_input_template_cn),
        protocol=MindSearchProtocol(
            meta_prompt=datetime.now().strftime(
                'The current date is %Y-%m-%d.'),
            interpreter_prompt=GRAPH_PROMPT_CN,
            response_prompt='请根据上文内容对问题给出详细的回复'),
        max_turn=10)
    return agent


plan_agent = init_agent()

PLANNER_HISTORY = []
SEARCHER_HISTORY = []


def rst_mem(history_planner: list, history_searcher: list):
    '''
    Reset the chatbot memory.
    '''
    history_planner = []
    history_searcher = []
    if PLANNER_HISTORY:
        PLANNER_HISTORY.clear()
    return history_planner, history_searcher


def format_response(gr_history, agent_return):
    if agent_return.state in [
            AgentStatusCode.STREAM_ING, AgentStatusCode.ANSWER_ING
    ]:
        gr_history[-1][1] = agent_return.response
    elif agent_return.state == AgentStatusCode.PLUGIN_START:
        thought = gr_history[-1][1].split('```')[0]
        if agent_return.response.startswith('```'):
            gr_history[-1][1] = thought + '\n' + agent_return.response
    elif agent_return.state == AgentStatusCode.PLUGIN_END:
        thought = gr_history[-1][1].split('```')[0]
        if isinstance(agent_return.response, dict):
            gr_history[-1][
                1] = thought + '\n' + f'```json\n{json.dumps(agent_return.response, ensure_ascii=False, indent=4)}\n```'
    elif agent_return.state == AgentStatusCode.PLUGIN_RETURN:
        assert agent_return.inner_steps[-1]['role'] == 'environment'
        item = agent_return.inner_steps[-1]
        gr_history.append([
            None,
            f"```json\n{json.dumps(item['content'], ensure_ascii=False, indent=4)}\n```"
        ])
        gr_history.append([None, ''])
    return


def predict(history_planner, history_searcher, max_new_tokens, top_p,
            temperature):
    global PLANNER_HISTORY
    PLANNER_HISTORY.append(dict(role='user', content=history_planner[-1][0]))
    new_search_turn = True
    for resp in plan_agent.stream_chat(
            PLANNER_HISTORY,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            temperature=temperature):
        if isinstance(resp, tuple):
            agent_return, node_name = resp
            if node_name in ['root', 'response']:
                continue
            agent_return = agent_return.nodes[node_name]['detail']
            if new_search_turn:
                history_searcher.append([agent_return.content, ''])
                new_search_turn = False
            format_response(history_searcher, agent_return)
            if agent_return.state == AgentStatusCode.END:
                new_search_turn = True
            yield history_planner, history_searcher
        else:
            agent_return = resp
            new_search_turn = True
            format_response(history_planner, agent_return)
            if agent_return.state == AgentStatusCode.END:
                PLANNER_HISTORY = agent_return.inner_steps
            yield history_planner, history_searcher
    return history_planner, history_searcher


with gr.Blocks() as demo:
    gr.HTML("""<h1 align="center">WebAgent Gradio Simple Demo</h1>""")
    with gr.Row():
        with gr.Column(scale=2, min_width=20):
            with gr.Accordion('gen setting', elem_id='Accordion'):
                max_new_tokens = gr.Slider(
                    0,
                    32768,
                    value=4096,
                    step=1.0,
                    label='Max tokens',
                    interactive=True)
                top_p = gr.Slider(
                    0,
                    1,
                    value=0.8,
                    step=0.01,
                    label='Top P',
                    interactive=True)
                temperature = gr.Slider(
                    0,
                    1,
                    value=0.8,
                    step=0.01,
                    label='Temperature',
                    interactive=True)
        with gr.Column(scale=10):
            with gr.Row():
                with gr.Column():
                    planner = gr.Chatbot(
                        label='planner',
                        height=700,
                        show_label=True,
                        show_copy_button=True,
                        bubble_full_width=False,
                        render_markdown=True)
                with gr.Column():
                    searcher = gr.Chatbot(
                        label='searcher',
                        height=700,
                        show_label=True,
                        show_copy_button=True,
                        bubble_full_width=False,
                        render_markdown=True)
            with gr.Row():
                user_input = gr.Textbox(
                    show_label=False,
                    placeholder='inputs...',
                    lines=5,
                    container=False)
            with gr.Row():
                with gr.Column(scale=2):
                    submitBtn = gr.Button('Submit')
                with gr.Column(scale=1, min_width=20):
                    emptyBtn = gr.Button('Clear History')

    def user(query, history):
        return '', history + [[query, '']]

    submitBtn.click(
        user, [user_input, planner], [user_input, planner], queue=False).then(
            predict, [planner, searcher, max_new_tokens, top_p, temperature],
            [planner, searcher])
    emptyBtn.click(
        rst_mem, [planner, searcher], [planner, searcher], queue=False)

demo.queue()
demo.launch(
    server_name='127.0.0.1', server_port=7882, inbrowser=True, share=True)
