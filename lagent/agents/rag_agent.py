from typing import Any, List, Dict
import yaml
import inspect

from .agent import Agent
from ..schema import AgentMessage
from lagent.rag.pipeline import BaseProcessor
from lagent.utils.util import create_object
from lagent.rag.pipeline import Pipeline
from lagent.llms import BaseAPILLM, DeepseekAPI
from lagent.rag.nlp import SentenceTransformerEmbedder, SimpleTokenizer
from lagent.rag.utils import replace_variables_in_prompt


class BaseAgent(Agent):
    def __init__(self,
                 processors_config: List,
                 llm: BaseAPILLM=dict(type=DeepseekAPI),
                 embedder=dict(type=SentenceTransformerEmbedder),
                 tokenizer=dict(type=SimpleTokenizer),
                 **kwargs):
        super().__init__(memory={}, **kwargs)
        self.external_memory = None
        self.llm = create_object(llm)
        self.embedder = create_object(embedder)
        self.tokenizer = create_object(tokenizer)

        self.processors = self.init_processors(processors_config)

    def init_external_memory(self, data):

        processors = self.processors
        pipeline = Pipeline()
        for processor in processors:
            pipeline.add_processor(processor)

        self.external_memory = pipeline.run(data)

        return self.external_memory

    def forward(self, **kwargs) -> Any:
        raise NotImplemented

    def init_processors(self, processors_config: List):
        processors = []
        for processor_config in processors_config:
            if isinstance(processor_config, dict):
                processor = create_object(processor_config)
            elif isinstance(processor_config, object):
                processor = processor_config
            else:
                raise ValueError
            processors.append(processor)

        return processors

    def prepare_prompt(self, knowledge: str, query: str, prompt: str):

        prompt_variables = {
            'External_Knowledge': knowledge,
            'Query': query
        }

        prompt = replace_variables_in_prompt(prompt=prompt, prompt_variables=prompt_variables)

        return prompt
