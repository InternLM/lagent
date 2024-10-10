from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
import os


class SentenceTransformerEmbedder:
    def __init__(
            self,
            model_name: str = 'all-MiniLM-L6-v2',
            device: Optional[str] = "cpu",  # 设备类型，设置为 "cpu"
            prefix: str = "",
            suffix: str = "",
            batch_size: int = 32,
            normalize_embeddings: bool = True,
            model_path='sentence_transformer_min.pkl',
    ):
        """
        初始化自定义文本嵌入器。

        :param model_name: 模型名称或路径
        :param device: 设备类型（'cpu' 或 'cuda'）
        :param prefix: 添加到每个文本开头的字符串
        :param suffix: 添加到每个文本结尾的字符串
        :param batch_size: 批量大小
        :param normalize_embeddings: 是否对嵌入进行归一化
        """

        self.model_name = model_name
        self.device = device
        self.prefix = prefix
        self.suffix = suffix
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.model_path = model_path
        self.model = self.load_model()

    def __call__(self, text: str):
        return self.model.encode(text)

    def encode(self, text: str):
        return self.model.encode(text)

    def load_model(self):
        # 如果模型文件已经存在，直接加载
        if os.path.exists(self.model_path):
            try:
                print(f"Loading model from {self.model_path}...")
                with open(self.model_path, 'rb') as f:
                    model = pickle.load(f)
            except Exception as e:
                raise EOFError(f"Error loading model: {e}")
        else:
            # 首次加载模型并保存
            print(f"Downloading and initializing model {self.model_name}...")
            model = SentenceTransformer(self.model_name, device=self.device)
            print(f"Saving model to {self.model_path}...")
            with open(self.model_path, 'wb') as f:
                pickle.dump(model, f)

        return model

    def embed_documents(self, texts: List[str]) -> Optional[np.ndarray]:
        """
        对文本列表进行嵌入。

        :param texts: 要嵌入的文本列表
        :return: 嵌入后的文本矩阵
        """
        if not isinstance(texts, list):
            raise TypeError("The 'texts' parameter should be a list of strings.")
        if not all(isinstance(text, str) for text in texts):
            raise ValueError("All elements in the 'texts' list should be strings.")

        # 添加前缀和后缀
        texts = [self.prefix + text + self.suffix for text in texts]

        try:
            # 使用 SentenceTransformer 进行嵌入
            embeddings = self.model.encode(texts, batch_size=self.batch_size, show_progress_bar=True)
        except Exception as e:
            raise RuntimeError(f"Failed to embed texts: {e}")

        # 归一化嵌入（如果需要）
        if self.normalize_embeddings:
            try:
                embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            except Exception as e:
                raise RuntimeError(f"Failed to normalize embeddings: {e}")

        return embeddings
