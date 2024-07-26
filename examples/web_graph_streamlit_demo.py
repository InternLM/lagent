import json
import tempfile

import streamlit as st
from pyvis.network import Network

from lagent.schema import AgentStatusCode


# Function to create the network graph
def create_network_graph(nodes, adjacency_list):
    net = Network(
        height='500px', width='60%', bgcolor='white', font_color='black')
    for node_id, node_data in nodes.items():
        if node_id in ['root', 'response']:
            title = node_data.get('content', node_id)
        else:
            title = node_data['detail'].content
        net.add_node(
            node_id, label=node_id, title=title, color='#FF5733', size=25)
    for node_id, neighbors in adjacency_list.items():
        for neighbor in neighbors:
            if neighbor in nodes:
                net.add_edge(node_id, neighbor)
    net.show_buttons(filter_=['physics'])
    return net


# Function to draw the graph and return the HTML file path
def draw_graph(net):
    path = tempfile.mktemp(suffix='.html')
    net.save_graph(path)
    return path


# Initialize Streamlit session state
if 'queries' not in st.session_state:
    st.session_state['queries'] = []
    st.session_state['responses'] = []
    st.session_state['graphs_html'] = []
    st.session_state['nodes_list'] = []
    st.session_state['adjacency_list_list'] = []
    st.session_state['history'] = []
    st.session_state['already_used_keys'] = list()

# Set up page layout
st.set_page_config(layout='wide')
st.title('MindSearch-思索')


# Function to update chat
def update_chat(query):
    with st.chat_message('user'):
        st.write(query)
    if query not in st.session_state['queries']:
        # Mock data to simulate backend response
        # response, history, nodes, adjacency_list
        st.session_state['queries'].append(query)
        st.session_state['responses'].append([])
        history = None
        # 暂不支持多轮
        message = [dict(role='user', content=query)]
        for resp in st.session_state['agent'].stream_chat(message):
            if isinstance(resp, tuple):
                agent_return, node_name = resp
                if node_name in ['root', 'response']:
                    continue
            else:
                agent_return = resp
                node_name = None
            nodes = agent_return.nodes
            adjacency_list = agent_return.adjacency_list
            response = agent_return.response
            history = agent_return.inner_steps
            if nodes:
                net = create_network_graph(nodes, adjacency_list)
                graph_html_path = draw_graph(net)
                with open(graph_html_path, encoding='utf-8') as f:
                    graph_html = f.read()
            else:
                graph_html = None
            if 'graph_placeholder' not in st.session_state:
                st.session_state['graph_placeholder'] = st.empty()
            if 'expander_placeholder' not in st.session_state:
                st.session_state['expander_placeholder'] = st.empty()
            if graph_html:
                with st.session_state['expander_placeholder'].expander(
                        'Show Graph', expanded=False):
                    st.session_state['graph_placeholder']._html(
                        graph_html, height=500)
            if 'container_placeholder' not in st.session_state:
                st.session_state['container_placeholder'] = st.empty()
            with st.session_state['container_placeholder'].container():
                if 'columns_placeholder' not in st.session_state:
                    st.session_state['columns_placeholder'] = st.empty()
                col1, col2 = st.session_state['columns_placeholder'].columns(
                    [2, 1])
                with col1:
                    if 'planner_placeholder' not in st.session_state:
                        st.session_state['planner_placeholder'] = st.empty()
                    if 'session_info_temp' not in st.session_state:
                        st.session_state['session_info_temp'] = ''
                    if not node_name:
                        if agent_return.state in [
                                AgentStatusCode.STREAM_ING,
                                AgentStatusCode.ANSWER_ING
                        ]:
                            st.session_state['session_info_temp'] = response
                        elif agent_return.state == AgentStatusCode.PLUGIN_START:
                            thought = st.session_state[
                                'session_info_temp'].split('```')[0]
                            if agent_return.response.startswith('```'):
                                st.session_state[
                                    'session_info_temp'] = thought + '\n' + response
                        elif agent_return.state == AgentStatusCode.PLUGIN_RETURN:
                            assert agent_return.inner_steps[-1][
                                'role'] == 'environment'
                            st.session_state[
                                'session_info_temp'] += '\n' + agent_return.inner_steps[
                                    -1]['content']
                        st.session_state['planner_placeholder'].markdown(
                            st.session_state['session_info_temp'])
                        if agent_return.state == AgentStatusCode.PLUGIN_RETURN:
                            st.session_state['responses'][-1].append(
                                st.session_state['session_info_temp'])
                            st.session_state['session_info_temp'] = ''
                    else:
                        st.session_state['planner_placeholder'].markdown(
                            st.session_state['responses'][-1][-1] if
                            not st.session_state['session_info_temp'] else st.
                            session_state['session_info_temp'])
                with col2:
                    if 'selectbox_placeholder' not in st.session_state:
                        st.session_state['selectbox_placeholder'] = st.empty()
                    if 'searcher_placeholder' not in st.session_state:
                        st.session_state['searcher_placeholder'] = st.empty()
                    # st.session_state['searcher_placeholder'].markdown('')
                    if node_name:
                        selected_node_key = f"selected_node_{len(st.session_state['queries'])}_{node_name}"
                        if selected_node_key not in st.session_state:
                            st.session_state[selected_node_key] = node_name
                        if selected_node_key not in st.session_state[
                                'already_used_keys']:
                            selected_node = st.session_state[
                                'selectbox_placeholder'].selectbox(
                                    'Select a node:',
                                    list(nodes.keys()),
                                    key=f'key_{selected_node_key}',
                                    index=list(nodes.keys()).index(node_name))
                            st.session_state['already_used_keys'].append(
                                selected_node_key)
                        else:
                            selected_node = node_name
                        st.session_state[selected_node_key] = selected_node
                        if selected_node in nodes:
                            node = nodes[selected_node]
                            agent_return = node['detail']
                            node_info_key = f'{selected_node}_info'
                            if 'node_info_temp' not in st.session_state:
                                st.session_state[
                                    'node_info_temp'] = f'### {agent_return.content}'
                            if node_info_key not in st.session_state:
                                st.session_state[node_info_key] = []
                            if agent_return.state in [
                                    AgentStatusCode.STREAM_ING,
                                    AgentStatusCode.ANSWER_ING
                            ]:
                                st.session_state[
                                    'node_info_temp'] = agent_return.response
                            elif agent_return.state == AgentStatusCode.PLUGIN_START:
                                thought = st.session_state[
                                    'node_info_temp'].split('```')[0]
                                if agent_return.response.startswith('```'):
                                    st.session_state[
                                        'node_info_temp'] = thought + '\n' + agent_return.response
                            elif agent_return.state == AgentStatusCode.PLUGIN_END:
                                thought = st.session_state[
                                    'node_info_temp'].split('```')[0]
                                if isinstance(agent_return.response, dict):
                                    st.session_state[
                                        'node_info_temp'] = thought + '\n' + f'```json\n{json.dumps(agent_return.response, ensure_ascii=False, indent=4)}\n```'
                            elif agent_return.state == AgentStatusCode.PLUGIN_RETURN:
                                assert agent_return.inner_steps[-1][
                                    'role'] == 'environment'
                                st.session_state[node_info_key].append(
                                    ('thought',
                                     st.session_state['node_info_temp']))
                                st.session_state[node_info_key].append(
                                    ('observation',
                                     agent_return.inner_steps[-1]['content']))
                            st.session_state['searcher_placeholder'].markdown(
                                st.session_state['node_info_temp'])
                            if agent_return.state == AgentStatusCode.END:
                                st.session_state[node_info_key].append(
                                    ('answer',
                                     st.session_state['node_info_temp']))
                                st.session_state['node_info_temp'] = ''
        if st.session_state['session_info_temp']:
            st.session_state['responses'][-1].append(
                st.session_state['session_info_temp'])
            st.session_state['session_info_temp'] = ''
        # st.session_state['responses'][-1] = '\n'.join(st.session_state['responses'][-1])
        st.session_state['graphs_html'].append(graph_html)
        st.session_state['nodes_list'].append(nodes)
        st.session_state['adjacency_list_list'].append(adjacency_list)
        st.session_state['history'] = history


def display_chat_history():
    for i, query in enumerate(st.session_state['queries'][-1:]):
        # with st.chat_message('assistant'):
        if st.session_state['graphs_html'][i]:
            with st.session_state['expander_placeholder'].expander(
                    'Show Graph', expanded=False):
                st.session_state['graph_placeholder']._html(
                    st.session_state['graphs_html'][i], height=500)
            with st.session_state['container_placeholder'].container():
                col1, col2 = st.session_state['columns_placeholder'].columns(
                    [2, 1])
                with col1:
                    st.session_state['planner_placeholder'].markdown(
                        st.session_state['responses'][-1][-1])
                with col2:
                    selected_node_key = st.session_state['already_used_keys'][
                        -1]
                    st.session_state['selectbox_placeholder'] = st.empty()
                    selected_node = st.session_state[
                        'selectbox_placeholder'].selectbox(
                            'Select a node:',
                            list(st.session_state['nodes_list'][i].keys()),
                            key=f'replay_key_{i}',
                            index=list(st.session_state['nodes_list'][i].keys(
                            )).index(st.session_state[selected_node_key]))
                    st.session_state[selected_node_key] = selected_node
                    if selected_node not in [
                            'root', 'response'
                    ] and selected_node in st.session_state['nodes_list'][i]:
                        node_info_key = f'{selected_node}_info'
                        for item in st.session_state[node_info_key]:
                            if item[0] in ['thought', 'answer']:
                                st.session_state[
                                    'searcher_placeholder'] = st.empty()
                                st.session_state[
                                    'searcher_placeholder'].markdown(item[1])
                            elif item[0] == 'observation':
                                st.session_state[
                                    'observation_expander'] = st.empty()
                                with st.session_state[
                                        'observation_expander'].expander(
                                            'Results'):
                                    st.write(item[1])
                        # st.session_state['searcher_placeholder'].markdown(st.session_state[node_info_key])


def clean_history():
    st.session_state['queries'] = []
    st.session_state['responses'] = []
    st.session_state['graphs_html'] = []
    st.session_state['nodes_list'] = []
    st.session_state['adjacency_list_list'] = []
    st.session_state['history'] = []
    st.session_state['already_used_keys'] = list()
    for k in st.session_state:
        if k.endswith('placeholder') or k.endswith('_info'):
            del st.session_state[k]


# Initialize the agent
def init_agent():
    from datetime import datetime

    from lagent.actions import ActionExecutor, BingBrowser
    from lagent.agents.mindsearch_agent import MindSearchAgent, MindSearchProtocol
    # from lagent.agents.mindsearch_prompt import GRAPH_PROMPT_EN
    # from lagent.agents.mindsearch_prompt import searcher_input_template_en
    # from lagent.agents.mindsearch_prompt import searcher_system_prompt_en
    from lagent.agents.mindsearch_prompt import (FINAL_RESPONSE_CN,
                                                 GRAPH_PROMPT_CN,
                                                 searcher_context_template_cn,
                                                 searcher_input_template_cn,
                                                 searcher_system_prompt_cn)
    # from lagent.agents.mindsearch_prompt import GRAPH_PROMPT_EN, searcher_input_template_en, searcher_system_prompt_en, searcher_context_template_en, FINAL_RESPONSE_EN
    from lagent.llms import INTERNLM2_META, LMDeployClient

    # from lagent.llms import GPTAPI

    llm = LMDeployClient(
        model_name='internlm2-chat-7b',
        url='http://22.8.69.5:23333',
        meta_template=INTERNLM2_META,
        max_new_tokens=4096,
        top_p=0.8,
        top_k=1,
        temperature=0.8,
        repetition_penalty=1.0,
        stop_words=['<|im_end|>'])

    # api_key = 'Your API Key'
    # gpt4_allen = dict(
    #     type=GPTAPI,
    #     model_type='gpt-4-turbo',
    #     key=api_key,
    #     query_per_second=50,
    #     max_new_tokens=4096,
    #     retry=1000)
    # gpt4o_allen = dict(
    #     type=GPTAPI,
    #     model_type='gpt-4o',
    #     key=api_key,
    #     query_per_second=50,
    #     max_new_tokens=4096,
    #     retry=1000)

    # llm = gpt4_allen.pop('type')(**gpt4_allen)
    # gpt4o_llm = gpt4o_allen.pop('type')(**gpt4o_allen)
    # llm = gpt4o_llm
    agent = MindSearchAgent(
        llm=llm,
        searcher_cfg=dict(
            llm=llm,
            plugin_executor=ActionExecutor(BingBrowser('YOUR BING API KEY')),
            protocol=MindSearchProtocol(
                meta_prompt=datetime.now().strftime(
                    'The current date is %Y-%m-%d.'),
                plugin_prompt=searcher_system_prompt_cn,
            ),
            template=dict(
                input=searcher_input_template_cn,
                context=searcher_context_template_cn)),
        protocol=MindSearchProtocol(
            meta_prompt=datetime.now().strftime(
                'The current date is %Y-%m-%d.'),
            interpreter_prompt=GRAPH_PROMPT_CN,
            response_prompt=FINAL_RESPONSE_CN),
        max_turn=10)
    return agent


# Main function to run the Streamlit app
def main():
    st.sidebar.title('Model Control')
    if 'agent' not in st.session_state:
        st.session_state['agent'] = init_agent()

    col1, col2 = st.columns([4, 1])
    with col1:
        user_input = st.chat_input('Enter your query:')
    with col2:
        if st.button('Clear History'):
            clean_history()
    if user_input:
        update_chat(user_input)
    display_chat_history()


if __name__ == '__main__':
    main()
