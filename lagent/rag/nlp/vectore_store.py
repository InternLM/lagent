import operator
import os
import pickle

import numpy as np
import faiss
from typing import List, Tuple, Optional, Union, Callable, Dict, Any

from lagent.rag.nlp.sentence_transformer_embedder import SentenceTransformerEmbedder


class DocumentDB:
    """
        Represents a document with an ID, content, and optional metadata.It is used for building database.

        Args:
            id (str): The unique identifier for the document.
            content (str): The textual content of the document.
            metadata (Optional[Dict]): Additional metadata for the document.
                Defaults to an empty dictionary if not provided.
        """
    def __init__(self, id: str, content: str, metadata: Optional[Dict] = None):
        self.id = id
        self.content = content
        self.metadata = metadata or {}


class FaissDatabase:
    def __init__(self,
                 embedding_model: Optional[SentenceTransformerEmbedder] = None,
                 dimension: int = 384,
                 normalize: bool = False):
        """
        Initializes the Faiss vector database.

        Args:
            embedding_model (Optional[SentenceTransformerEmbedder]):
                The embedder model used to generate vectors. Defaults to None,
                which initializes a default SentenceTransformerEmbedder.
            dimension (int): The dimensionality of the vectors. Defaults to 384.
            normalize (bool): Whether to apply L2 normalization to vectors.
                Defaults to False.
        """
        if embedding_model is None:
            self.embedding_model = SentenceTransformerEmbedder()
        else:
            self.embedding_model = embedding_model

        self.dimension = dimension
        self.index = faiss.IndexFlat(dimension, faiss.METRIC_L2)
        self.documents = []
        self.index_to_docstore_id = []
        self._normalize_L2 = normalize

    @classmethod
    def from_documents(cls, documents: List[DocumentDB], embedder) -> 'FaissDatabase':
        """
        Constructs a FaissDatabase from a list of documents.

        Args:
            documents (List[DocumentDB]): A list of DocumentDB instances to add to the database.
            embedder (SentenceTransformerEmbedder): The embedder used to generate document vectors.

        Returns:
            FaissDatabase: An instance of FaissDatabase populated with the provided documents.
        """
        db = cls(embedding_model=embedder)
        for doc in documents:
            embedding = embedder.encode(doc.content).astype('float32')
            embedding = np.reshape(embedding, (1, -1))  # 确保形状为 (1, dimension)
            db.index.add(embedding)
            db.documents.append(doc)
            db.index_to_docstore_id.append(doc.id)

        return db

    def add_documents(self, docs: List[DocumentDB]):
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
        Retrieves similar documents to the query along with their similarity scores.

        Args:
            query (str): The query text.
            k (int): The number of top documents to return. Defaults to 4.
            filter (Optional[Union[Callable[[Dict[str, Any]], bool], Dict[str, Any]]]):
                A filter to apply to the documents. Can be a callable or a dictionary of conditions.
                Defaults to None.
            fetch_k (int): The number of documents to fetch initially before applying the filter.
                Defaults to 20.
            **kwargs (Any): Additional keyword arguments. Supports 'score_threshold' to filter results.

        Returns:
            List[Tuple[DocumentDB, float]]: A list of tuples containing the DocumentDB instances and their
                corresponding similarity scores.
        """
        query_vector = self.embedding_model.encode(query).astype('float32')
        query_vector = np.reshape(query_vector, (1, -1))
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
            cmp = operator.le
            docs = [
                (doc, similarity)
                for doc, similarity in docs
                if cmp(similarity, score_threshold)
            ]

        return docs[:k]

    def save_local(self, file_path: str):
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
        Creates a filter function based on the provided filter criteria.

        Args:
            filter (Union[Callable[[Dict[str, Any]], bool], Dict[str, Any]]):
                The filter criteria, either as a callable or a dictionary of key-value pairs.

        Returns:
            Callable[[Dict[str, Any]], bool]: A function that takes metadata and returns a boolean
                indicating whether the metadata satisfies the filter criteria.
        """
        if callable(filter):
            return filter
        else:
            def filter_func(metadata):
                return all(metadata.get(k) == v for k, v in filter.items())

            return filter_func

