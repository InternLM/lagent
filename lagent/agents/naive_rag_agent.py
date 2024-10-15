from typing import Any, List

from .rag_agent import BaseAgent
from lagent.rag.prompts import KNOWLEDGE_PROMPT
from lagent.rag.schema import Document, Chunk
from lagent.rag.processors import ChunkSplitter
from lagent.rag.nlp import SentenceTransformerEmbedder
from lagent.rag.nlp import FaissDatabase, DocumentDB


class NaiveRAGAgent(BaseAgent):
    def __init__(self, processors_config, **kwargs):
        super().__init__(processors_config=processors_config, **kwargs)

    def forward(self, query: str, **kwargs) -> Any:

        embedder = self.embedder or SentenceTransformerEmbedder()
        memory = self.external_memory
        tokenizer = self.tokenizer

        prompt = kwargs.get('prompts', KNOWLEDGE_PROMPT)

        max_ref_token = kwargs.get('max_ref_token', 2000)
        w_community = kwargs.get('w_community', 0.4)
        w_text = kwargs.get('w_text', 0.6)
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
        paras = text.split('\n')
        temp_doc = Document(
            content=[{'page_num': 1, 'content': {'text': paras}}],
            id='search_result',
            metadata={'source': 'retrieve transform'}
        )
        chunk_extractor = ChunkSplitter({'chunk_size': int(max_ref_token * 0.5)})
        temp_chunks = chunk_extractor.split_into_chunks(temp_doc)
        temp_chunks_db = self.initialize_chunk_faiss(temp_chunks)
        final_results = temp_chunks_db.similarity_search_with_score(query, k=2)
        final_search_contents = ['------texts------' + '\n']
        for result in final_results:
            final_search_contents.append(result[0].content)
        final_search_contents = '\n'.join(final_search_contents)

        num = tokenizer.get_token_num(final_search_contents)
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

        faiss_db = FaissDatabase.from_documents(documents, embedding_function)

        return faiss_db
