import copy
import hashlib
import json
import os
from typing import Dict, List, Union
import re
import streamlit as st
import requests

from lagent.actions import ActionExecutor, ArxivSearch, IPythonInterpreter
from lagent.prompts.parsers import PluginParser
from lagent.agents.stream import INTERPRETER_CN, META_CN, PLUGIN_CN, AgentForInternLM, get_plugin_prompt
from lagent.llms.base_api import BaseAPILLM
from lagent.schema import AgentStatusCode
from lagent.llms import GPTAPI

# 替换为自己的授权令牌
YOUR_TOKEN_HERE = ""

class SessionState:
    """管理会话状态的类。"""

    def init_state(self):
        """初始化会话状态变量。"""
        st.session_state['assistant'] = []  # 助手消息历史
        st.session_state['user'] = []  # 用户消息历史
        # 初始化插件列表
        action_list = [
            ArxivSearch(),
        ]
        st.session_state['plugin_map'] = {action.name: action for action in action_list}
        st.session_state['model_map'] = {}  # 存储模型实例
        st.session_state['model_selected'] = None  # 当前选定模型
        st.session_state['plugin_actions'] = set()  # 当前激活插件
        st.session_state['history'] = []  # 聊天历史

    def clear_state(self):
        """清除当前会话状态。"""
        st.session_state['assistant'] = []
        st.session_state['user'] = []
        st.session_state['model_selected'] = None
        st.session_state['file'] = set()
        if 'chatbot' in st.session_state:
            st.session_state['chatbot']._session_history = []


class StreamlitUI:
    """管理 Streamlit 界面的类。"""

    def __init__(self, session_state: SessionState):
        self.session_state = session_state
        self.plugin_action = []  # 当前选定的插件
        # 初始化提示词
        self.meta_prompt = META_CN
        self.da_prompt = INTERPRETER_CN
        self.plugin_prompt = PLUGIN_CN
        self.init_streamlit()

    def init_streamlit(self):
        """初始化 Streamlit 的 UI 设置。"""
        st.set_page_config(
            layout='wide',
            page_title='lagent-web',
            page_icon='./docs/imgs/lagent_icon.png'
        )
        st.header(':robot_face: :blue[Lagent] Web Demo ', divider='rainbow')
        st.sidebar.title('模型控制')
        st.session_state['file'] = set()  # 存储上传文件列表
        st.session_state['ip'] = None  # 初始化模型 IP

    def setup_sidebar(self):
        """设置侧边栏，选择模型和插件。"""
        # 模型名称和 IP 输入框
        model_name = st.sidebar.text_input('模型名称：', value='internlm2.5-latest')
        model_ip = st.sidebar.text_input('模型IP：', value='127.0.0.1:23333')

        # 提示词设置
        self.meta_prompt = st.sidebar.text_area('系统提示词', value=META_CN)
        self.da_prompt = st.sidebar.text_area('数据分析提示词', value=INTERPRETER_CN)
        self.plugin_prompt = st.sidebar.text_area('插件提示词', value=PLUGIN_CN)

        # 插件选择
        plugin_name = st.sidebar.multiselect(
            '插件选择',
            options=list(st.session_state['plugin_map'].keys()),
            default=[],
        )
        da_flag = st.sidebar.checkbox('数据分析', value=False)

        # 根据选择的插件生成插件操作列表
        self.plugin_action = [st.session_state['plugin_map'][name] for name in plugin_name]

        # 动态生成插件提示
        if self.plugin_action:
            self.plugin_prompt = get_plugin_prompt(self.plugin_action)

        # 清空对话按钮
        if st.sidebar.button('清空对话', key='clear'):
            self.session_state.clear_state()

        uploaded_file = st.sidebar.file_uploader('上传文件')  # 文件上传

        return model_name, model_ip, self.plugin_action, uploaded_file

    def initialize_chatbot(self, model_name, plugin_action):
        """初始化 GPTAPI 实例作为 chatbot。"""
        self.meta_prompt = [
            {"role": "system", "content": self.meta_prompt, "api_role": "system"},
            {"role": "user", "content": "", "api_role": "user"},
            {"role": "assistant", "content": "", "api_role": "assistant"},
            {"role": "environment", "content": "", "api_role": "environment"}
        ]

        api_model = GPTAPI(
            model_type=model_name,
            api_base="https://internlm-chat.intern-ai.org.cn/puyu/api/v1/chat/completions",
            key=YOUR_TOKEN_HERE,
            meta_template=self.meta_prompt,
            max_new_tokens=512,
            temperature=0.8,
            top_p=0.9
        )
        return api_model

    def render_user(self, prompt: str):
        """渲染用户输入内容。"""
        with st.chat_message('user'):
            st.markdown(prompt)

    def render_assistant(self, agent_return):
        """渲染助手响应内容。"""
        print("agent_return", agent_return)
        with st.chat_message('assistant'):
            content = getattr(agent_return, "content", str(agent_return))
            st.markdown(content if isinstance(content, str) else str(content))


def main():
    """主函数，运行 Streamlit 应用。"""
    if 'ui' not in st.session_state:
        session_state = SessionState()
        session_state.init_state()
        st.session_state['ui'] = StreamlitUI(session_state)
    else:
        st.set_page_config(
            layout='wide',
            page_title='lagent-web',
            page_icon='./docs/imgs/lagent_icon.png'
        )
        st.header(':robot_face: :blue[Lagent] Web Demo ', divider='rainbow')

    # 设置侧边栏并获取模型和插件信息
    model_name, model_ip, plugin_action, uploaded_file = st.session_state['ui'].setup_sidebar()
    plugins = [dict(type=f"lagent.actions.{plugin.__class__.__name__}") for plugin in plugin_action]

    if (
        'chatbot' not in st.session_state or
        model_name != st.session_state['chatbot'].model_type or
        'last_plugin_action' not in st.session_state or
        plugin_action != st.session_state['last_plugin_action']
    ):
        st.session_state['chatbot'] = st.session_state['ui'].initialize_chatbot(model_name, plugin_action)
        st.session_state['last_plugin_action'] = plugin_action  # 更新插件状态

        # 初始化 AgentForInternLM
        st.session_state['agent'] = AgentForInternLM(
            llm=st.session_state['chatbot'],
            plugins=plugins,
            output_format=dict(
                type=PluginParser,
                template=PLUGIN_CN,
                prompt=get_plugin_prompt(plugin_action)
            )
        )
        # 清空对话历史
        st.session_state['session_history'] = []

    if 'agent' not in st.session_state:
        st.session_state['agent'] = None

    agent = st.session_state['agent']
    for prompt, agent_return in zip(st.session_state['user'], st.session_state['assistant']):
        st.session_state['ui'].render_user(prompt)
        st.session_state['ui'].render_assistant(agent_return)

    # 处理用户输入
    if user_input := st.chat_input(''):
        st.session_state['ui'].render_user(user_input)
        res = agent(user_input, session_id=0)
        st.session_state['ui'].render_assistant(res)

        # 更新会话状态
        st.session_state['user'].append(user_input)
        st.session_state['assistant'].append(copy.deepcopy(res))

        # 处理文件上传
        if uploaded_file and uploaded_file.name not in st.session_state['file']:
            st.session_state['file'].add(uploaded_file.name)
            file_bytes = uploaded_file.read()
            file_path = os.path.join("tmp_dir", hashlib.md5(file_bytes).hexdigest())
            with open(file_path, 'wb') as tmpfile:
                tmpfile.write(file_bytes)
            st.markdown(f"文件已上传：{uploaded_file.name}")

    st.session_state['last_status'] = AgentStatusCode.END


if __name__ == '__main__':
    os.makedirs("tmp_dir", exist_ok=True)
    main()
