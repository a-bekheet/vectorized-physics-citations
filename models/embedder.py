"""Embedding generation using Sentence-BERT."""
from typing import Optional, Union
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class PaperEmbedder:
    """Generates embeddings for academic papers using Sentence-BERT."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        batch_size: int = 64,
        max_seq_length: int = 512,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.model = SentenceTransformer(model_name, device=device)
        self.model.max_seq_length = max_seq_length
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

    def embed(
        self,
        texts: list[str],
        normalize: bool = True,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        )
        return embeddings

    def embed_single(self, text: str, normalize: bool = True) -> np.ndarray:
        """Generate embedding for a single text."""
        embedding = self.model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        )
        return embedding[0]

    def save_embeddings(
        self,
        embeddings: np.ndarray,
        path: Union[str, Path],
    ) -> None:
        """Save embeddings to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, embeddings)
        print(f"Saved embeddings {embeddings.shape} to {path}")

    @staticmethod
    def load_embeddings(path: Union[str, Path]) -> np.ndarray:
        """Load embeddings from disk."""
        embeddings = np.load(path)
        print(f"Loaded embeddings {embeddings.shape} from {path}")
        return embeddings

    def compute_similarity(
        self,
        query_embedding: np.ndarray,
        corpus_embeddings: np.ndarray,
    ) -> np.ndarray:
        """Compute cosine similarity between query and corpus."""
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        similarities = np.dot(corpus_embeddings, query_embedding.T).flatten()
        return similarities

    def find_similar(
        self,
        query_embedding: np.ndarray,
        corpus_embeddings: np.ndarray,
        top_k: int = 10,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Find top-k most similar embeddings."""
        similarities = self.compute_similarity(query_embedding, corpus_embeddings)
        top_indices = np.argsort(similarities)[::-1][:top_k]
        top_scores = similarities[top_indices]
        return top_indices, top_scores

    def __repr__(self) -> str:
        return (
            f"PaperEmbedder(model={self.model_name}, "
            f"dim={self.embedding_dim}, device={self.device})"
        )
