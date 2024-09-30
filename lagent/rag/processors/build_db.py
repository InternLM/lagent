from lagent.rag.schema import MultiLayerGraph
from lagent.rag.nlp import FaissDatabase, DocumentDB
from lagent.rag.pipeline import BaseProcessor, register_processor
from lagent.rag.nlp import SentenceTransformerEmbedder
from lagent.rag.schema import Chunk, Node

from typing import Optional, List


@register_processor
class BuildDatabase(BaseProcessor):
    name = 'BuildDatabase'

    def __init__(self, embedder: Optional = None):
        super().__init__(name='BuildDatabase')

        self.embedder = embedder or SentenceTransformerEmbedder()

    def run(self, data: MultiLayerGraph) -> MultiLayerGraph:
        if 'chunk_layer' in data.layers:
            if 'chunk_layer' not in data.layers_db:
                chunks = data.layers['chunk_layer'].get_nodes()
                chunks = [Chunk.dict_to_chunk(chunk) for chunk in chunks]
                db = self.initialize_chunk_faiss(chunks)
                data.layers_db['chunk_layer'] = db

        if 'summarized_entity_layer' in data.layers:
            if 'summarized_entity_layer' not in data.layers_db:
                entities = data.layers['summarized_entity_layer'].get_nodes()
                entities = [Node.dict_to_node(entity) for entity in entities]
                db = self.initialize_entity_faiss(entities)
                data.layers_db['summarized_entity_layer'] = db

        return data

    def initialize_chunk_faiss(self, chunks: List[Chunk]) -> FaissDatabase:
        """
        目前默认使用FAISS 向量数据库。初始化 FAISS 向量数据库。如果存在已保存的 FAISS 索引，则加载它；否则，创建一个新的。

        :return: FAISS 向量数据库实例
        """
        # 创建文档列表
        documents = [
            DocumentDB(
                id=chunk.id,
                content=chunk.content,
                metadata=chunk.metadata
            )
            for chunk in chunks
        ]

        # 初始化嵌入函数（使用 HuggingFaceEmbeddings）
        embedding_function = self.embedder

        # 创建 FAISS 索引
        faiss_db = FaissDatabase.from_documents(documents, embedding_function)

        # # 保存 FAISS 索引
        # faiss_db.save_local(f'{self.db_index_path}_chunks.pkl')
        # print(f"FAISS index created and saved to {self.db_index_path}_chunks.pkl")

        return faiss_db

    def initialize_entity_faiss(self, entities: List[Node]) -> FaissDatabase:
        """
        目前默认使用FAISS 向量数据库。初始化 FAISS 向量数据库。如果存在已保存的 FAISS 索引，则加载它；否则，创建一个新的。

        :return: FAISS 向量数据库实例
        """

        # entities_with_embeddings = self.generate_embeddings(entities)

        # 创建文档列表
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

        # 初始化嵌入函数（使用 HuggingFaceEmbeddings）
        embedding_function = self.embedder

        # 创建 FAISS 索引
        faiss_db = FaissDatabase.from_documents(documents=documents, emedder=embedding_function)

        # # 保存 FAISS 索引
        # faiss_db.save_local(f'{self.db_index_path}_entities')
        # print(f"FAISS index created and saved to {self.db_index_path}_entities.")

        return faiss_db
