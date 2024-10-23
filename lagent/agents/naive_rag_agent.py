import re
from typing import Any, List

from .rag_agent import BaseAgent
from lagent.rag.prompts import KNOWLEDGE_PROMPT
from lagent.rag.schema import Document, Chunk, DocumentDB
from lagent.llms import DeepseekAPI, BaseAPILLM
from lagent.rag.processors import ChunkSplitter, DocParser, BuildDatabase, SaveGraph
from lagent.rag.nlp import SentenceTransformerEmbedder, SimpleTokenizer
from lagent.rag.settings import DEFAULT_LLM_MAX_TOKEN


class NaiveRAGAgent(BaseAgent):
    def __init__(self,
                 llm: BaseAPILLM = dict(type=DeepseekAPI),
                 embedder=dict(type=SentenceTransformerEmbedder),
                 tokenizer=dict(type=SimpleTokenizer),
                 processors_config:
                 List = [dict(type=DocParser), dict(type=ChunkSplitter),
                         dict(type=BuildDatabase), dict(type=SaveGraph)],
                 **kwargs):
        super().__init__(llm=llm, embedder=embedder, tokenizer=tokenizer, processors_config=processors_config, **kwargs)

    def forward(self, query: str, **kwargs) -> Any:
        memory = self.external_memory
        tokenizer = self.tokenizer

        prompt = kwargs.get('prompts', KNOWLEDGE_PROMPT)

        max_ref_token = kwargs.get('max_ref_token', DEFAULT_LLM_MAX_TOKEN)
        max_ref_token = max_ref_token - tokenizer.get_token_num(prompt)

        dict_chunks = memory.layers['chunk_layer'].get_nodes()
        chunks: List[Chunk] = []
        for chunk in dict_chunks:
            chunks.append(Chunk.dict_to_chunk(chunk))

        top_k = kwargs.get('top_k', 3)
        chunks_db = memory.layers_db['chunk_layer']
        results = chunks_db.similarity_search_with_score(query, k=top_k)
        search_contents = [result[0].content for result in results]

        # TODO:find better ways to trim the context
        text = '\n'.join(search_contents)
        final_search_contents = self.trim_context(text, max_ref_token)

        result = self.prepare_prompt(query=query, knowledge=final_search_contents, prompt=prompt)

        messages = [{"role": "user", "content": result}]
        response = self.llm.chat(messages)

        return response

    def trim_context(self, text, max_ref_token):
        tokenizer = self.tokenizer
        token_num = tokenizer.get_token_num(text)
        if token_num <= max_ref_token:
            return text
        paras = text.split('\n')
        available_num = max_ref_token
        result = []
        for para in paras:
            para = para.strip()
            token_num = tokenizer.get_token_num(para)
            if token_num <= available_num:
                result.append(para)
                available_num -= token_num
            else:
                tmp = ''
                sentences = re.split(r'\. |。', para)
                j = 0
                while j < len(sentences):
                    sentence = sentences[j].strip()
                    num = tokenizer.get_token_num(sentence)
                    if num == 0:
                        continue
                    if num <= available_num:
                        tmp = f'{tmp}.{sentence}'
                        max_tokens = available_num - num
                        j += 1
                    else:
                        break
                result.append(tmp)
        return '\n'.join(result)
