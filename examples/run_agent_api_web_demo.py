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

YOUR_TOKEN_HERE = ""      # 请注意，这里要替换为自己实际授权令牌！！！

class CustomAPILLM(BaseAPILLM):
    """自定义的 API LLM 类，用于调用外部 API 进行文本生成。"""

    def __init__(self, model_type, meta_template=None, **gen_params):
        super().__init__(model_type, meta_template=meta_template, **gen_params)

    def call_api(self, messages):
        """调用外部 API 并返回响应结果。"""
        url = 'https://internlm-chat.intern-ai.org.cn/puyu/api/v1/chat/completions'
        headers = {
            'Content-Type': 'application/json',
            "Authorization": "Bearer " + YOUR_TOKEN_HERE  
        }
        data = {
            "model": self.model_type,
            "messages": messages,
            "n": 1,
            "temperature": self.gen_params['temperature'],
            "top_p": self.gen_params['top_p'],
            "stream": False,
        }
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"API 调用失败，状态码: {response.status_code}")

    def generate(self, inputs: Union[str, List[str]], **gen_params) -> Union[str, List[str]]:
        """调用外部 API。"""
        if isinstance(inputs, str):
            inputs = [{"role": "user", "content": inputs}]
        elif isinstance(inputs, list) and isinstance(inputs[0], str):
            inputs = [{"role": "user", "content": text} for text in inputs]

        # 调用 call_api 并返回响应
        response = self.call_api(inputs)
        content = response["choices"][0]["message"]["content"]

        if len(inputs) == 1:
            return content
        else:
            return [content]

class SessionState:
    """管理会话状态的类。"""

    def init_state(self):
        """初始化会话状态变量。"""
        st.session_state['assistant'] = []
        st.session_state['user'] = []
        action_list = [
            ArxivSearch(),
        ]
        st.session_state['plugin_map'] = {
            action.name: action for action in action_list
        }
        st.session_state['model_map'] = {}
        st.session_state['model_selected'] = None
        st.session_state['plugin_actions'] = set()
        st.session_state['history'] = []

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
        self.init_streamlit()
        self.session_state = session_state
        self.plugin_action = []  # 插件操作列表
        # 初始化提示词
        self.meta_prompt = META_CN
        self.da_prompt = INTERPRETER_CN
        self.plugin_prompt = PLUGIN_CN

    def init_streamlit(self):
        """初始化 Streamlit 的 UI 设置。"""
        st.set_page_config(
            layout='wide',
            page_title='lagent-web',
            page_icon='./docs/imgs/lagent_icon.png')
        st.header(':robot_face: :blue[Lagent] Web Demo ', divider='rainbow')
        st.sidebar.title('模型控制')
        st.session_state['file'] = set()
        st.session_state['ip'] = None

    def setup_sidebar(self):
        """设置侧边栏，选择模型和插件。"""
        model_name = st.sidebar.text_input('模型名称：', value='internlm2.5-latest')
        self.meta_prompt = st.sidebar.text_area('系统提示词', value=META_CN)
        self.da_prompt = st.sidebar.text_area('数据分析提示词', value=INTERPRETER_CN)
        self.plugin_prompt = st.sidebar.text_area('插件提示词', value=PLUGIN_CN)
        model_ip = st.sidebar.text_input('模型IP：', value='127.0.0.1:23333')

        # 确保 model_map 已初始化
        if model_name not in st.session_state['model_map']:
            st.session_state['model_map'][model_name] = self.call_api

        model = st.session_state['model_map'][model_name]

        # 添加插件选择
        plugin_name = st.sidebar.multiselect(
            '插件选择',
            options=list(st.session_state['plugin_map'].keys()),
            default=[],
        )
        da_flag = st.sidebar.checkbox('数据分析', value=False)

        # 创建插件操作列表
        self.plugin_action = [
            st.session_state['plugin_map'][name] for name in plugin_name
        ]

        # 动态生成插件提示
        if self.plugin_action:
            self.plugin_prompt = get_plugin_prompt(self.plugin_action)

        # 初始化或更新 chatbot
        if 'chatbot' in st.session_state:
            if self.plugin_action:
                st.session_state['chatbot'].plugin_executor = ActionExecutor(
                    actions=self.plugin_action)
            else:
                st.session_state['chatbot'].plugin_executor = None

            if da_flag:
                st.session_state['chatbot'].interpreter_executor = ActionExecutor(
                    actions=[IPythonInterpreter()])
            else:
                st.session_state['chatbot'].interpreter_executor = None

            # 更新提示词
            st.session_state['chatbot'].meta_prompt = self.meta_prompt
            st.session_state['chatbot'].plugin_prompt = self.plugin_prompt
            st.session_state['chatbot'].interpreter_prompt = self.da_prompt

        # 清空对话按钮
        if st.sidebar.button('清空对话', key='clear'):
            self.session_state.clear_state()

        uploaded_file = st.sidebar.file_uploader('上传文件')

        return model_name, model, self.plugin_action, uploaded_file, model_ip

    def call_api(self, prompt="你是一个机器人"):
        """使用外部 API 请求生成响应（用于模型初始化）。"""
        url = 'https://internlm-chat.intern-ai.org.cn/puyu/api/v1/chat/completions'
        headers = {
            'Content-Type': 'application/json',
            "Authorization": "Bearer " + YOUR_TOKEN_HERE
        }
        data = {
            "model": "internlm2.5-latest",
            "messages": [{"role": "assistant", "content": prompt}],
            "n": 1,
            "temperature": 0.8,
            "top_p": 0.9,
            "stream": False,
        }
        response = requests.post(url, headers=headers, json=data)
        return response

    def initialize_chatbot(self, model_name, plugin_action):
        """使用 CustomAPILLM 初始化 chatbot。"""
        # meta_template 是一个包含字典的列表，并包含所有角色
        self.meta_prompt = [
            {"role": "system", "content": self.meta_prompt, "api_role": "system"},
            {"role": "user", "content": "", "api_role": "user"},
            {"role": "assistant", "content": "", "api_role": "assistant"}
        ]

        # 使用 CustomAPILLM 类
        api_model = CustomAPILLM(
            model_type=model_name,
            meta_template=self.meta_prompt,
            max_new_tokens=512,
            temperature=0.8,
            top_p=0.9
        )
        return api_model

    def render_user(self, prompt: str):
        """渲染用户的输入。"""
        with st.chat_message('user'):
            st.markdown(prompt)

    def render_assistant(self, agent_return):
        """渲染助手的响应，包括处理插件的结果。"""
        with st.chat_message('assistant'):
            if hasattr(agent_return, "content"):
                content = agent_return.content
            else:
                content = str(agent_return)

            if isinstance(content, list):
                content = '\n'.join(content)
            elif not isinstance(content, str):
                content = str(content)

            st.markdown(content)

            json_match = re.search(r'\{.*\}', content)
            if json_match:
                json_string = json_match.group()
                try:
                    action_data = json.loads(json_string)
                    plugin_name = action_data.get('name')
                    parameters = action_data.get('parameters', {})

                    # 提取插件的基本名称
                    base_plugin_name = plugin_name.split('.')[0]

                    if base_plugin_name in [action.name for action in self.plugin_action]:
                        plugin = st.session_state['plugin_map'][base_plugin_name]

                        # 根据插件类型调用不同的方法
                        if base_plugin_name == "ArxivSearch":
                            arxiv_results = plugin.get_arxiv_article_information(parameters.get('query', ''))
                            # 解析和显示 Arxiv 信息
                            results = arxiv_results.get('content', '').split('\n\n')
                            for result in results:
                                lines = result.split('\n')
                                if len(lines) >= 4:
                                    published = lines[0].replace('Published: ', '').strip()
                                    title = lines[1].replace('Title: ', '').strip()
                                    authors = lines[2].replace('Authors: ', '').strip()
                                    summary = ' '.join(lines[3:]).replace('Summary: ', '').strip()

                                    st.markdown(f"  **标题**: {title}")
                                    st.markdown(f"  **作者**: {authors}")
                                    st.markdown(f"  **发表日期**: {published}")
                                    st.markdown(f"  **摘要**: {summary}\n")
                                else:
                                    st.warning("无法解析论文信息，格式不正确。")
                        else:
                            st.warning(f"未找到插件: {base_plugin_name}")
                    else:
                        st.warning(f"未找到插件: {base_plugin_name}")
                except json.JSONDecodeError:
                    st.error("无法解析 action 中的 JSON 数据，请检查其格式是否正确。")

    def render_plugin_args(self, action):
        """渲染插件的参数。"""
        action_name = action.type
        args = action.args
        parameter_dict = dict(name=action_name, parameters=args)
        parameter_str = 'json\n' + json.dumps(
            parameter_dict, indent=4, ensure_ascii=False) + '\n'
        st.markdown(parameter_str)

    def render_interpreter_args(self, action):
        """渲染解释器的参数。"""
        st.info(action.type)
        st.markdown(action.args['text'])

    def render_action(self, action):
        """渲染动作，包括思考过程和结果。"""
        st.markdown(action.thought)
        if action.type == 'IPythonInterpreter':
            self.render_interpreter_args(action)
        elif action.type == 'FinishAction':
            pass
        else:
            self.render_plugin_args(action)
        self.render_action_results(action)

    def render_action_results(self, action):
        if isinstance(action.result, dict):
            if 'text' in action.result:
                st.markdown('\n' + action.result['text'] + '\n')
            if 'image' in action.result:
                for image_path in action.result['image']:
                    image_data = open(image_path, 'rb').read()
                    st.image(image_data, caption='Generated Image')
            if 'video' in action.result:
                video_data = action.result['video']
                video_data = open(video_data, 'rb').read()
                st.video(video_data)
            if 'audio' in action.result:
                audio_data = action.result['audio']
                audio_data = open(audio_data, 'rb').read()
                st.audio(audio_data)
        elif isinstance(action.result, list):
            for item in action.result:
                if item['type'] == 'text':
                    st.markdown('\n' + item['content'] + '\n')
                elif item['type'] == 'image':
                    image_data = open(item['content'], 'rb').read()
                    st.image(image_data, caption='Generated Image')
                elif item['type'] == 'video':
                    video_data = open(item['content'], 'rb').read()
                    st.video(video_data)
                elif item['type'] == 'audio':
                    audio_data = open(item['content'], 'rb').read()
                    st.audio(audio_data)
        if action.errmsg:
            st.error(action.errmsg)

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

    # 设置侧边栏并获取模型和插件
    model_name, model, plugin_action, uploaded_file, _ = st.session_state['ui'].setup_sidebar()

    # 初始化 chatbot 和 agent
    if 'chatbot' not in st.session_state or model_name != st.session_state['chatbot'].model_type:
        st.session_state['chatbot'] = st.session_state['ui'].initialize_chatbot(model_name, plugin_action)
        plugins = [
            dict(type='lagent.actions.ArxivSearch'),
        ]

        # 创建 AgentForInternLM 实例并存储在 session_state 中
        st.session_state['agent'] = AgentForInternLM(
            llm=st.session_state['chatbot'],
            plugins=plugins,
            output_format=dict(
                type=PluginParser,
                template=PLUGIN_CN,
                prompt=get_plugin_prompt(plugins)
            )
        )
        # 清空会话历史
        st.session_state['session_history'] = []

    if 'agent' not in st.session_state:
        st.session_state['agent'] = None

    agent = st.session_state['agent']
    for prompt, agent_return in zip(st.session_state['user'], st.session_state['assistant']):
        st.session_state['ui'].render_user(prompt)
        st.session_state['ui'].render_assistant(agent_return)

    if user_input := st.chat_input(''):
        with st.container():
            st.session_state['ui'].render_user(user_input)
        res = agent(user_input, session_id=0)
        st.session_state['ui'].render_assistant(res)
        st.session_state['assistant'].append(copy.deepcopy(res))

        if uploaded_file and uploaded_file.name not in st.session_state['file']:
            st.session_state['file'].add(uploaded_file.name)
            file_bytes = uploaded_file.read()
            file_type = uploaded_file.type
            if 'image' in file_type:
                st.image(file_bytes, caption='Uploaded Image')
            elif 'video' in file_type:
                st.video(file_bytes, caption='Uploaded Video')
            elif 'audio' in file_type:
                st.audio(file_bytes, caption='Uploaded Audio')
            postfix = uploaded_file.name.split('.')[-1]
            prefix = hashlib.md5(file_bytes).hexdigest()
            filename = f'{prefix}.{postfix}'
            file_path = os.path.join(root_dir, filename)
            with open(file_path, 'wb') as tmpfile:
                tmpfile.write(file_bytes)
            file_size = os.stat(file_path).st_size / 1024 / 1024
            file_size = f'{round(file_size, 2)} MB'
            user_input = [
                dict(role='user', content=user_input),
                dict(role='user', content=json.dumps(dict(path=file_path, size=file_size)), name='file')
            ]
        else:
            user_input = [dict(role='user', content=user_input)]

    st.session_state['last_status'] = AgentStatusCode.END

if __name__ == '__main__':
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    root_dir = os.path.join(root_dir, 'tmp_dir')
    os.makedirs(root_dir, exist_ok=True)
    main()
