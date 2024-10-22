from typing import Any, List, Dict
import re

from .rag_agent import BaseAgent
from lagent.rag.schema import Node, CommunityReport, Community, Chunk, DocumentDB
from lagent.rag.nlp import SentenceTransformerEmbedder, SimpleTokenizer
from lagent.llms import DeepseekAPI, BaseAPILLM
from lagent.rag.prompts import KNOWLEDGE_PROMPT
from lagent.rag.processors import (DocParser, ChunkSplitter, EntityExtractor, DescriptionSummarizer,
                                   CommunitiesDetector, CommunityReportsExtractor, BuildDatabase, SaveGraph)
from lagent.rag.nlp import FaissDatabase


class GraphRagAgent(BaseAgent):
    def __init__(self,
                 llm: BaseAPILLM = dict(type=DeepseekAPI),
                 embedder=dict(type=SentenceTransformerEmbedder),
                 tokenizer=dict(type=SimpleTokenizer),
                 processors_config:
                 List = [dict(type=DocParser), dict(type=ChunkSplitter), dict(type=EntityExtractor),
                         dict(type=DescriptionSummarizer), dict(type=CommunitiesDetector),
                         dict(type=CommunityReportsExtractor), dict(type=BuildDatabase), dict(type=SaveGraph)],
                 **kwargs):
        super().__init__(llm=llm, embedder=embedder, tokenizer=tokenizer, processors_config=processors_config, **kwargs)

    def forward(self, query: str, **kwargs) -> Any:
        """
           Processes the input query and returns a response generated by the language model.

           This method performs the following steps:
               1. Initializes external memory.
               2. Performs similarity searches to identify top relevant entities.
               3. Builds community and chunk contexts based on the retrieved entities.
               4. Prepares the final prompt and interacts with the language model to generate a response.

           Args:
               query (str): The input query string to be processed.
               **kwargs (Any): Additional keyword arguments, such as prompts, token limits, and weighting factors.

           Returns:
               Any: The response generated by the language model based on the processed query and contexts.
        """

        memory = self.external_memory
        prompt = kwargs.get('prompts', KNOWLEDGE_PROMPT)
        tokenizer = self.tokenizer

        max_ref_token = kwargs.get('max_ref_token', 4096)
        w_community = kwargs.get('w_community', 0.4)
        w_text = kwargs.get('w_text', 0.6)
        max_ref_token = max_ref_token - tokenizer.get_token_num(prompt)

        dict_chunks = memory.layers['chunk_layer'].get_nodes()
        chunks: List[Chunk] = []
        for chunk in dict_chunks:
            chunks.append(Chunk.dict_to_chunk(chunk))

        entities: List[Node] = []

        assert 'summarized_entity_layer' in memory.layers
        dict_entities = memory.layers['summarized_entity_layer'].get_nodes()
        for entity in dict_entities:
            entities.append(Node.dict_to_node(entity))

        top_k = kwargs.get('top_k', 8)
        entities_db = memory.layers_db['summarized_entity_layer']

        selected_entities = []
        selected_entities = entities_db.similarity_search_with_score(query, k=top_k)
        selected_entities = [(en[0].metadata['id'], en[1]) for en in selected_entities]

        # build community_reports context
        dict_commu_rep = memory.layers['community_report_layer'].get_nodes()
        community_reports: List[CommunityReport] = []
        for commu in dict_commu_rep:
            community_reports.append(CommunityReport(
                community_id=commu['community_id'],
                level=commu['level'],
                report=commu['report'],
                structured_report=commu['structured_report']
            ))

        dict_commu = memory.layers['community_layer'].get_nodes()
        communities: List[Community] = []
        for commu in dict_commu:
            communities.append(Community(
                community_id=commu['id'],
                level=commu['level'],
                nodes_id=commu['nodes_id']
            ))

        community_context = self.build_community_contexts(entities_with_score=selected_entities,
                                                          community_reports=community_reports,
                                                          max_tokens=int(max_ref_token * w_community),
                                                          communities=communities)

        chunk_content = self.build_chunk_contexts(entities_with_score=selected_entities,
                                                  entities=entities,
                                                  chunks=chunks,
                                                  max_tokens=int(max_ref_token * w_text))

        final_search_contents = f'{community_context}\n{chunk_content}'

        result = self.prepare_prompt(query=query, knowledge=final_search_contents, prompt=prompt)

        messages = [{"role": "user", "content": result}]

        response = self.llm.chat(messages)

        return response

    def initialize_chunk_faiss(self, chunks: List[Chunk]) -> FaissDatabase:
        documents = [
            DocumentDB(
                id=chunk.id,
                content=chunk.content,
                metadata=chunk.metadata
            )
            for chunk in chunks
        ]

        embedding_function = self.embedder

        # create faiss index
        faiss_db = FaissDatabase.from_documents(documents, embedding_function)

        return faiss_db

    def add_entities_to_faiss(self, entities: List[Node], db: FaissDatabase):
        # Create a list of DocumentDB instances from entities
        documents = [
            DocumentDB(
                id=entity.id,
                content=entity.description,
                metadata={
                    "id": entity.id,
                    "entity_type": entity.entity_type
                }
            )
            for entity in entities
        ]

        db.add_documents(documents)

    def build_community_contexts(self, entities_with_score: List[tuple[str, float]],
                                 community_reports: List[CommunityReport], max_tokens: int,
                                 communities: List[Community]) -> str:
        """
           Constructs the community context based on selected entities, community reports, and communities.

           This context is built by aggregating relevant community reports and ensuring
           that the total number of tokens does not exceed the specified maximum.

           Args:
               entities_with_score (List[tuple[str, float]]): A list of tuples containing entity IDs and their
                    corresponding scores.
               community_reports (List[CommunityReport]): A list of CommunityReport instances.
               max_tokens (int): The maximum number of tokens allowed for the community context.
               communities (List[Community]): A list of Community instances.

           Returns:
               str: The constructed community context as a string.
        """

        selected_entities = {}
        for entity in entities_with_score:
            selected_entities[entity[0]] = entity
        selected_commu = {}
        for community in communities:
            for node_id in community.nodes_id:
                if node_id in selected_entities:
                    selected_commu[community.community_id] = selected_commu.get(community.community_id, 0) + \
                                                             selected_entities[node_id][1]
        selected_commu = dict(sorted(selected_commu.items(), key=lambda item: item[1], reverse=True))

        id_map_reports = {}
        for community_report in community_reports:
            id_map_reports[community_report.community_id] = community_report

        selected_reports = [id_map_reports[community_id] for community_id in selected_commu.keys()]

        # Trim reports to fit within the token limit
        result = [f'------reports------' + '\n']
        tokenizer = SimpleTokenizer()
        for selected_report in selected_reports:
            content = selected_report.report
            token_num = tokenizer.get_token_num(content)
            if token_num <= max_tokens:
                result.append(content)
                max_tokens = max_tokens - token_num
            else:
                tmp = ''
                sentences = re.split(r'\. |。', content)
                j = 0
                while j < len(sentences):
                    sentence = sentences[j].strip()
                    num = tokenizer.get_token_num(sentence)
                    if num == 0:
                        continue
                    if num <= max_tokens:
                        tmp = f'{tmp}.{sentence}'
                        max_tokens = max_tokens - num
                        j += 1
                    else:
                        break
                result.append(tmp)
                break

        return '\n'.join(result)

    def build_chunk_contexts(self, entities_with_score: List[tuple[str, float]], entities: List[Node],
                             chunks: List[Chunk], max_tokens: int):
        """
            Constructs the chunk context based on selected entities and chunks.

            This context aggregates relevant chunks associated with the selected entities
            while ensuring that the total number of tokens does not exceed the specified maximum.

            Args:
                entities_with_score (List[tuple[str, float]]): A list of tuples containing entity IDs and their corresponding scores.
                entities (List[Node]): A list of Node instances representing entities.
                chunks (List[Chunk]): A list of Chunk instances to be considered.
                max_tokens (int): The maximum number of tokens allowed for the chunk context.

            Returns:
                str: The constructed chunk context as a string.
        """
        id_map_entities = {}
        for entity in entities:
            id_map_entities[entity.id] = entity

        id_map_chunks = {}
        for chunk in chunks:
            id_map_chunks[chunk.id] = chunk

        selected_chunks: Dict[str, Dict] = {}
        for entity_with_score in entities_with_score:
            entity = id_map_entities[entity_with_score[0]]
            for chunk_id in entity.source_id:
                if chunk_id not in selected_chunks:
                    selected_chunks[chunk_id] = {}
                selected_chunks[chunk_id]['score'] = selected_chunks[chunk_id].get('score', 0) + entity_with_score[1]

        selected_chunks = dict(
            sorted(selected_chunks.items(), key=lambda item: item[1]["score"], reverse=True)
        )

        result = [f'------texts------' + '\n']
        tokenizer = SimpleTokenizer()
        for chunk_id, selected_chunk in selected_chunks.items():
            content = id_map_chunks[chunk_id].content
            token_num = id_map_chunks[chunk_id].token_num
            if token_num <= max_tokens:
                result.append(content)
                max_tokens -= token_num
            else:
                tmp = ''
                sentences = re.split(r'\. |。', content)
                j = 0
                while j < len(sentences):
                    sentence = sentences[j].strip()
                    num = tokenizer.get_token_num(sentence)
                    if num == 0:
                        continue
                    if num <= max_tokens:
                        tmp = f'{tmp}.{sentence}'
                        max_tokens = max_tokens - num
                        j += 1
                    else:
                        break
                result.append(tmp)

        return '\n'.join(result)
