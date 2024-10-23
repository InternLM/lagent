from typing import Dict, List, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

from lagent.rag.prompts import DEFAULT_SUMMATY_PROMPT
from lagent.rag.settings import DEFAULT_LLM_MAX_TOKEN
from lagent.rag.nlp import SimpleTokenizer as Tokenizer
from lagent.rag.doc import Storage
from lagent.llms import DeepseekAPI, BaseAPILLM
from lagent.utils import create_object
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
                 llm: BaseAPILLM = dict(type=DeepseekAPI),
                 storage: Storage = dict(type=Storage),
                 summarization_prompt: Optional[str] = None,
                 max_tokens: Optional[int] = None,
                 **kwargs
                 ):
        super().__init__(name='DescriptionSummarizer')
        self.llm = create_object(llm)
        self.prompt = summarization_prompt or DEFAULT_SUMMATY_PROMPT
        self.storage = create_object(storage)
        self.max_tokens = max_tokens or DEFAULT_LLM_MAX_TOKEN

    def run(self, graph: MultiLayerGraph, **kwargs) -> MultiLayerGraph:
        summarized_layer = graph.add_layer('summarized_entity_layer')
        entity_layer = graph.layers['entity_layer']
        entities = entity_layer.get_nodes()
        relationships = entity_layer.get_edges()

        summarized_entities = []
        summarized_relationships = []

        def summarize_entity(entity: Dict) -> Dict:
            try:
                node_description = entity.get('description', '')
                if not node_description:
                    raise ValueError(f"Entity {entity.get('id', 'Unknown ID')} is missing 'description' key.")
                node_description_list = merge_description(descri_list=entity['description'].split(' | '))
                node_name = entity['id']
                if len(node_description_list) == 1:
                    entity['description'] = node_description_list[0]
                    return entity

                summary = self.summarize_descriptions(node_name, node_description_list)
                entity['description'] = summary
                return entity

            except Exception as e:
                raise ValueError(f"Error processing entity {entity['id']}: {e}")

        def summarize_relationship(rela: Dict) -> Dict:
            try:
                rela_description_list = merge_description(descri_list=rela['description'].split(' | '))
                rela_name = tuple_to_str((rela['source'], rela['target']))
                if len(rela_description_list) == 1:
                    rela['description'] = rela_description_list[0]
                    return rela

                summary = self.summarize_descriptions(rela_name, rela_description_list)
                rela['description'] = summary
                return rela

            except Exception as e:
                raise ValueError(f"Error processing relationship {rela['source']} -> {rela['target']}: {e}")

        with ThreadPoolExecutor(max_workers=4) as executor:
            entity_futures = {executor.submit(summarize_entity, entity): entity for entity in entities}
            relationship_futures = {executor.submit(summarize_relationship, rela): rela for rela in relationships}

            for future in as_completed(entity_futures):
                entity = entity_futures[future]
                try:
                    summarized_entity = future.result()
                    summarized_entities.append(summarized_entity)
                except Exception as e:
                    raise ValueError

            for future in as_completed(relationship_futures):
                rela = relationship_futures[future]
                try:
                    summarized_rela = future.result()
                    summarized_relationships.append(summarized_rela)
                except Exception as e:
                    raise ValueError
        for entity in summarized_entities:
            entity_id = entity.pop('id')
            summarized_layer.add_node(entity_id, **entity)

        for rela in summarized_relationships:
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

        messages = [{"role": "user", "content": prompt}]

        response = self.llm.chat(messages)

        return response
