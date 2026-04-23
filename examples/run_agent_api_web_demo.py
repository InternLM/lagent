import copy
import os
from typing import List
import streamlit as st
from lagent.actions import ArxivSearch, WeatherQuery
from lagent.prompts.parsers import PluginParser
from lagent.agents.stream import INTERPRETER_CN, META_CN, PLUGIN_CN, AgentForInternLM, get_plugin_prompt
from lagent.llms import GPTAPI

class SessionState:
    """管理会话状态的类。"""

    def init_state(self):
        """初始化会话状态变量。"""
        st.session_state['assistant'] = []  # 助手消息历史
        st.session_state['user'] = []  # 用户消息历史
        # 初始化插件列表
        action_list = [
            ArxivSearch(),
            WeatherQuery(),
        ]
        st.session_state['plugin_map'] = {action.name: action for action in action_list}
        st.session_state['model_map'] = {}  # 存储模型实例
        st.session_state['model_selected'] = None  # 当前选定模型
        st.session_state['plugin_actions'] = set()  # 当前激活插件
        st.session_state['history'] = []  # 聊天历史
        st.session_state['api_base'] = None  # 初始化API base地址

    def clear_state(self):
        """清除当前会话状态。"""
        st.session_state['assistant'] = []
        st.session_state['user'] = []
        st.session_state['model_selected'] = None


class StreamlitUI:
    """管理 Streamlit 界面的类。"""

    def __init__(self, session_state: SessionState):
        self.session_state = session_state
        self.plugin_action = []  # 当前选定的插件
        # 初始化提示词
        self.meta_prompt = META_CN
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

    def setup_sidebar(self):
        """设置侧边栏，选择模型和插件。"""
        # 模型名称和 API Base 输入框
        model_name = st.sidebar.text_input('模型名称：', value='internlm2.5-latest')
        # 注意，如果采用硅基流动API，模型名称需要更改为：internlm/internlm2_5-7b-chat 或者 internlm/internlm2_5-20b-chat
        # ================================== 硅基流动的API ==================================
        # api_base = st.sidebar.text_input(
        #     'API Base 地址：', value='https://api.siliconflow.cn/v1/chat/completions'
        # )
        # ================================== 浦语官方的API ==================================
        api_base = st.sidebar.text_input(
            'API Base 地址：', value='https://internlm-chat.intern-ai.org.cn/puyu/api/v1/chat/completions'
        )
        # ==================================================================================
        
        plugin_name = st.sidebar.multiselect(
            '插件选择',
            options=list(st.session_state['plugin_map'].keys()),
            default=[],
        )

        # 根据选择的插件生成插件操作列表
        self.plugin_action = [st.session_state['plugin_map'][name] for name in plugin_name]

        if self.plugin_action:
            self.plugin_prompt = get_plugin_prompt(self.plugin_action)

        if st.sidebar.button('清空对话', key='clear'):
            self.session_state.clear_state()

        return model_name, api_base, self.plugin_action

    def initialize_chatbot(self, model_name, api_base, plugin_action):
        """初始化 GPTAPI 实例作为 chatbot。"""
        token = os.getenv("token")
        if not token:
            st.error("糟糕，未检测到环境变量 `token`，请设置环境变量，例如 `export token='your_token_here'` 后重新运行 X﹏X")
            st.stop()  # 停止运行应用
            
        meta_prompt = [
            {"role": "system", "content": self.meta_prompt, "api_role": "system"},
            {"role": "user", "content": "", "api_role": "user"},
            {"role": "assistant", "content": self.plugin_prompt, "api_role": "assistant"},
            {"role": "environment", "content": "", "api_role": "environment"}
        ]

        api_model = GPTAPI(
            model_type=model_name,
            api_base=api_base,
            key=token,
            meta_template=meta_prompt,
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

    model_name, api_base, plugin_action = st.session_state['ui'].setup_sidebar()
    plugins = [dict(type=f"lagent.actions.{plugin.__class__.__name__}") for plugin in plugin_action]

    if (
        'chatbot' not in st.session_state or
        model_name != st.session_state['chatbot'].model_type or
        'last_plugin_action' not in st.session_state or
        plugin_action != st.session_state['last_plugin_action'] or
        api_base != st.session_state['api_base']    
    ):
        # 更新 Chatbot
        st.session_state['chatbot'] = st.session_state['ui'].initialize_chatbot(model_name, api_base, plugin_action)
        st.session_state['last_plugin_action'] = plugin_action
        st.session_state['api_base'] = api_base 

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

        st.session_state['user'].append(user_input)
        st.session_state['assistant'].append(copy.deepcopy(res))

    st.session_state['last_status'] = None


if __name__ == '__main__':
    main()
