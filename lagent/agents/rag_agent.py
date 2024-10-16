from typing import Any, List, Dict
import yaml

from .agent import Agent
from ..schema import AgentMessage
from lagent.rag.pipeline import BaseProcessor
from lagent.utils.util import create_object
from lagent.rag.utils import replace_variables_in_prompt


class BaseAgent(Agent):
    def __init__(self, processors_config: Dict,
                 **kwargs):
        super().__init__(memory={}, **kwargs)
        self.external_memory = None
        self.llm = None
        self.embedder = None
        self.tokenizer = None

        self.processors = self.init_processors(processors_config)
        # processors_config='[InitGrpah(), Load_DB(), DocParser(path=xx), ChunckSpliter(), ..., ToDB(), SaveGraph()]'

    def init_external_memory(self, data):

        processors = self.processors
        for processor in processors:
            data = processor.run(data)

        self.external_memory = data

        return data

    def forward(self, **kwargs) -> Any:
        raise NotImplemented

    def init_processors(self, processors_config: Dict):
        dependencies = self.init_dependencies(processors_config.get('dependencies', {}))

        # 实例化处理器
        processors = []
        for proc_conf in processors_config.get('processors', []):
            # 解析依赖项
            resolved_config = self.resolve_dependencies(proc_conf.get('params', {}), dependencies)
            # 创建处理器实例
            processor = create_object({'type': proc_conf['type'], **resolved_config})
            processors.append(processor)
        return processors

    def init_processors_from_yaml(self, config_path: str) -> List[BaseProcessor]:
        """initial processors from yaml"""
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)

        # 初始化依赖项
        dependencies = self.init_dependencies(config.get('dependencies', {}))

        # 实例化处理器
        processors = []
        for proc_conf in config.get('processors', []):
            # 解析依赖项
            resolved_config = self.resolve_dependencies(proc_conf.get('params', {}), dependencies)
            # 创建处理器实例
            processor = create_object({'type': proc_conf['type'], **resolved_config})
            processors.append(processor)
        return processors

    def init_dependencies(self, dependencies_config: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """根据配置初始化依赖项"""
        dependencies = {}
        for dep_name, dep_conf in dependencies_config.items():
            dependency = create_object(dep_conf)
            dependencies[dep_name] = dependency

        # 全局使用统一的依赖项
        if 'llm' in dependencies:
            self.llm = dependencies['llm']
        if 'embedder' in dependencies:
            self.embedder = dependencies['embedder']
        if 'tokenizer' in dependencies:
            self.tokenizer = dependencies['tokenizer']

        return dependencies

    def resolve_dependencies(self, params: Dict[str, Any], dependencies: Dict[str, Any]) -> Dict[str, Any]:
        """解析处理器配置中的依赖项引用"""
        resolved_params = params.copy()
        for key, value in params.items():
            if isinstance(value, str) and value in dependencies:
                resolved_params[key] = dependencies[value]
        return resolved_params

    def prepare_prompt(self, knowledge: str, query: str, prompt: str):

        prompt_variables = {
            'External_Knowledge': knowledge,
            'Query': query
        }

        prompt = replace_variables_in_prompt(prompt=prompt, prompt_variables=prompt_variables)

        return prompt
