from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
import os


class SentenceTransformerEmbedder:
    def __init__(
            self,
            model_name: str = 'all-MiniLM-L6-v2',
            device: Optional[str] = "cpu",
            prefix: str = "",
            suffix: str = "",
            batch_size: int = 32,
            normalize_embeddings: bool = True,
            model_path='sentence_transformer_min.pkl',
    ):
        """
        Initializes a custom text embedder using SentenceTransformer.

        Args:
            model_name (str): The name or path of the SentenceTransformer model.
                Defaults to 'all-MiniLM-L6-v2'.
            device (Optional[str]): The device to run the model on ('cpu' or 'cuda').
                Defaults to "cpu".
            prefix (str): A string to prepend to each input text.
                Defaults to an empty string.
            suffix (str): A string to append to each input text.
                Defaults to an empty string.
            batch_size (int): The number of texts to process in a single batch.
                Defaults to 32.
            normalize_embeddings (bool): Whether to normalize the embeddings.
                Defaults to True.
            model_path (str): The file path to save/load the model.
                Defaults to 'sentence_transformer_min.pkl'.
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
        if os.path.exists(self.model_path):
            try:
                print(f"Loading model from {self.model_path}...")
                with open(self.model_path, 'rb') as f:
                    model = pickle.load(f)
            except Exception as e:
                raise EOFError(f"Error loading model: {e}")
        else:
            print(f"Downloading and initializing model {self.model_name}...")
            model = SentenceTransformer(self.model_name, device=self.device)
            print(f"Saving model to {self.model_path}...")
            with open(self.model_path, 'wb') as f:
                pickle.dump(model, f)

        return model

    def embed_documents(self, texts: List[str]) -> Optional[np.ndarray]:
        """
        Embeds a list of texts into their corresponding embeddings.

        Args:
            texts (List[str]): A list of texts to embed.

        Returns:
            Optional[np.ndarray]: A matrix of embeddings for the input texts.

        Raises:
            TypeError: If 'texts' is not a list of strings.
            ValueError: If any element in 'texts' is not a string.
            RuntimeError: If embedding or normalization fails.
        """
        if not isinstance(texts, list):
            raise TypeError("The 'texts' parameter should be a list of strings.")
        if not all(isinstance(text, str) for text in texts):
            raise ValueError("All elements in the 'texts' list should be strings.")

        texts = [self.prefix + text + self.suffix for text in texts]

        try:
            embeddings = self.model.encode(texts, batch_size=self.batch_size, show_progress_bar=True)
        except Exception as e:
            raise RuntimeError(f"Failed to embed texts: {e}")

        if self.normalize_embeddings:
            try:
                embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            except Exception as e:
                raise RuntimeError(f"Failed to normalize embeddings: {e}")

        return embeddings
