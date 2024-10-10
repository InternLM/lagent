import operator
import os
import pickle

import numpy as np
import faiss
from typing import List, Tuple, Optional, Union, Callable, Dict, Any

from lagent.rag.nlp.sentence_transformer_embedder import SentenceTransformerEmbedder


class DocumentDB:
    def __init__(self, id: str, content: str, metadata: Optional[Dict] = None):
        self.id = id
        self.content = content
        self.metadata = metadata or {}


class FaissDatabase:
    def __init__(self,
                 embedding_model: Optional[SentenceTransformerEmbedder] = None,
                 dimension: int = 384,  # 默认维度
                 normalize: bool = False):  # 默认归一化
        """
        初始化向量数据库
        :param embedding_model: 用于生成向量的嵌入模型，默认为None
        :param dimension: 向量的维度，默认为384
        :param normalize: 是否对向量进行L2归一化，默认为True
        """
        if embedding_model is None:
            self.embedding_model = SentenceTransformerEmbedder()  # 默认嵌入模型
        else:
            self.embedding_model = embedding_model

        self.dimension = dimension
        self.index = faiss.IndexFlat(dimension, faiss.METRIC_L2)
        self.documents = []
        self.index_to_docstore_id = []
        self._normalize_L2 = normalize

    @classmethod
    def from_documents(cls, documents: List[DocumentDB], emedder) -> 'FaissDatabase':
        """
        根据文档构建向量数据库
        :param documents: 文档列表

        Args:
            emedder: 
        """
        db = cls(embedding_model=emedder)
        for doc in documents:
            embedding = emedder.encode(doc.content).astype('float32')
            embedding = np.reshape(embedding, (1, -1))  # 确保形状为 (1, dimension)
            db.index.add(embedding)
            db.documents.append(doc)
            db.index_to_docstore_id.append(doc.id)

        return db

    def add_documents(self, docs: List[DocumentDB]):
        """添加单个文档到数据库"""
        for doc in docs:
            embedding = self.embedding_model.encode(doc.content).astype('float32')
            embedding = np.reshape(embedding, (1, -1))  # 确保形状为 (1, dimension)
            self.index.add(embedding)
            self.documents.append(doc)
            self.index_to_docstore_id.append(doc.id)

    def similarity_search_with_score(self, query: str, k: int = 4,
                                     filter: Optional[Union[Callable, Dict[str, Any]]] = None,
                                     fetch_k: int = 20, **kwargs: Any) -> List[Tuple[DocumentDB, float]]:
        """
        根据查询返回相似文档和相应评分
        :param query: 查询文本
        :param k: 返回的文档数量
        :param filter: 过滤条件
        :param fetch_k: 先获取的文档数量
        :return: 相似文档和相应评分的列表
        """
        query_vector = self.embedding_model.encode(query).astype('float32')
        query_vector = np.reshape(query_vector, (1, -1))  # 确保形状为 (1, dimension)
        if self._normalize_L2:
            faiss.normalize_L2(query_vector)

        scores, indices = self.index.search(query_vector, k if filter is None else fetch_k)
        docs = []

        if filter is not None:
            filter_func = self._create_filter_func(filter)

        for j, i in enumerate(indices[0]):
            if i == -1:
                continue

            doc_id = self.index_to_docstore_id[i]
            doc = next((d for d in self.documents if d.id == doc_id), None)
            if doc is None:
                raise ValueError(f"Could not find document for id {doc_id}")

            if filter is not None:
                if filter_func(doc.metadata):
                    docs.append((doc, scores[0][j]))
            else:
                docs.append((doc, scores[0][j]))

        score_threshold = kwargs.get("score_threshold")
        if score_threshold is not None:
            cmp = operator.le  # L2 distance越小越相似
            docs = [
                (doc, similarity)
                for doc, similarity in docs
                if cmp(similarity, score_threshold)
            ]

        return docs[:k]

    def save_local(self, file_path: str):
        """保存当前向量数据库到本地文件"""
        with open(file_path, 'wb') as f:
            pickle.dump({
                'index': self.index,
                'documents': self.documents,
                'index_to_docstore_id': self.index_to_docstore_id,
                'embedding_model': self.embedding_model,
                'dimension': self.dimension,
                '_normalize_L2': self._normalize_L2
            }, f)

        return file_path

    @classmethod
    def load(cls, file_path: str):
        """从本地文件加载向量数据库"""
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"No such file: '{file_path}'")

        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                index = data['index']
                _documents = data['documents']
                dimension = data['dimension']
                normalize = data['_normalize_L2']
                embedding_model = data['embedding_model']
                index_to_docstore_id = data['index_to_docstore_id']
                database = cls(embedding_model=embedding_model, dimension=dimension, normalize=normalize)
                database.index = index
                database.index_to_docstore_id = index_to_docstore_id
                database.documents = _documents
                return database
        except EOFError:
            raise ValueError(f"File '{file_path}' is corrupted or empty.")
        except Exception as e:
            raise RuntimeError(f"An error occurred while loading the database: {e}")

    def _create_filter_func(self, filter: Union[Callable, Dict[str, Any]]) -> Callable:
        """
        创建过滤函数
        :param filter: 过滤条件
        :return: 过滤函数
        """
        if callable(filter):
            return filter
        else:
            def filter_func(metadata):
                return all(metadata.get(k) == v for k, v in filter.items())

            return filter_func

