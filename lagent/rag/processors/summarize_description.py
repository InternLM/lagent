from typing import Dict, List, Optional, Union

from lagent.rag.prompts import DEFAULT_SUMMATY_PROMPT
from lagent.rag.settings import DEFAULT_LLM_MAX_TOKEN
from lagent.rag.nlp import SimpleTokenizer as Tokenizer
from lagent.rag.doc import Storage
from lagent.rag.utils import replace_variables_in_prompt, tuple_to_str
from lagent.rag.schema import MultiLayerGraph
from lagent.rag.pipeline import register_processor, BaseProcessor


def merge_description(descri_list: List[str]):
    merged_descri = []
    for descri in descri_list:
        if descri not in merged_descri:
            merged_descri.append(descri)

    return merged_descri


@register_processor
class DescriptionSummarizer(BaseProcessor):
    name = 'DescriptionSummarizer'

    def __init__(self,
                 llm,
                 storage: Union[Storage, Dict, None] = None,
                 summarization_prompt: Optional[str] = None,
                 max_tokens: Optional[int] = None,
                 **kwargs
                 ):
        super().__init__(name='DescriptionSummarizer')
        self.llm = llm
        self.prompt = summarization_prompt or DEFAULT_SUMMATY_PROMPT
        if storage is not None:
            if isinstance(storage, Storage):
                self.storage = storage
            else:
                self.storage = Storage(**storage)
        else:
            self.storage = Storage()
        self.max_tokens = max_tokens or DEFAULT_LLM_MAX_TOKEN

    def run(self, graph: MultiLayerGraph, **kwargs) -> MultiLayerGraph:
        res = {}
        id_map_items = {}
        summarized_layer = graph.add_layer('summarized_entity_layer')
        entity_layer = graph.layers['entity_layer']
        entities = entity_layer.get_nodes()
        relationships = entity_layer.get_edges()

        node_description_list = []
        for entity in entities:
            node_description_list = entity['description'].split(' | ')
            node_name = entity['id']
            node_description_list = merge_description(descri_list=node_description_list)
            if len(node_description_list) == 1:
                entity['description'] = node_description_list[0]
                continue
            summary = self.summarize_descriptions(node_name, node_description_list)
            entity['description'] = summary

        self.storage.put('summarized_entity', entities)

        rela_description_list = []
        for rela in relationships:
            rela_description_list = rela['description'].split(' | ')
            rela_name = (rela['source'], rela['target'])
            rela_name = tuple_to_str(rela_name)
            rela_description_list = merge_description(rela_description_list)
            if len(relationships) == 1:
                rela['description'] = rela_description_list[0]
                continue
            summary = self.summarize_descriptions(rela_name, rela_description_list)
            rela['description'] = summary

        self.storage.put('summarized_relationships', relationships)

        for entity in entities:
            entity_id = entity.pop('id')
            summarized_layer.add_node(entity_id, **entity)
        for rela in relationships:
            source = rela.pop('source')
            target = rela.pop("target")
            summarized_layer.add_edge(source, target, **rela)

        return graph

    def summarize_descriptions(self, name: str | tuple[str, str], descr_list: List[str]) -> str:

        if len(descr_list) == 1:
            return descr_list[0]

        available_token = self.max_tokens

        tokenizer = Tokenizer()

        available_token -= tokenizer.get_token_num(self.prompt)
        input_descriptions = []
        res = None
        i = 0
        while i < len(descr_list):
            descr = descr_list[i]
            token_num = tokenizer.get_token_num(descr)
            if token_num <= available_token:
                input_descriptions.append(descr)
                available_token -= token_num
                i += 1
            else:
                if not input_descriptions:
                    continue

                res = self.summary_from_llm(name, input_descriptions)
                if i == len(descr_list) - 1:
                    break
                else:
                    input_descriptions = [res]
                    available_token = self.max_tokens - tokenizer.get_token_num(self.prompt)
        if res is None:
            res = self.summary_from_llm(name, input_descriptions)
        return res

    def summary_from_llm(self, name, descr_list: List[str]) -> str:

        descriptions = ".".join(descr_list)
        prompt_varaibles = {
            'node_name': name,
            'description_list': descriptions
        }
        prompt = replace_variables_in_prompt(self.prompt, prompt_varaibles)

        # TODO: create messages
        messages = [{"role": "user", "content": prompt}]

        response = self.llm.chat(messages)

        return response
