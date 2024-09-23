"""FAISS-based similarity search index."""
from typing import Optional, Union
from pathlib import Path

import numpy as np
import faiss


class SearchIndex:
    """FAISS index for fast similarity search."""

    def __init__(
        self,
        dimension: int = 384,
        index_type: str = "flat",
        nlist: int = 100,
        nprobe: int = 10,
    ):
        self.dimension = dimension
        self.index_type = index_type
        self.nlist = nlist
        self.nprobe = nprobe
        self.index: Optional[faiss.Index] = None
        self.is_trained = False

    def build(self, embeddings: np.ndarray) -> None:
        """Build the search index from embeddings."""
        embeddings = embeddings.astype(np.float32)
        n_vectors = embeddings.shape[0]

        if self.index_type == "flat":
            self.index = faiss.IndexFlatIP(self.dimension)
        elif self.index_type == "ivf":
            quantizer = faiss.IndexFlatIP(self.dimension)
            actual_nlist = min(self.nlist, n_vectors // 10)
            self.index = faiss.IndexIVFFlat(
                quantizer, self.dimension, actual_nlist, faiss.METRIC_INNER_PRODUCT
            )
            self.index.train(embeddings)
            self.index.nprobe = self.nprobe
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")

        self.index.add(embeddings)
        self.is_trained = True
        print(f"Built {self.index_type} index with {n_vectors} vectors")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Search for top-k nearest neighbors."""
        if self.index is None:
            raise RuntimeError("Index not built. Call build() first.")

        query = query_embedding.astype(np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)

        scores, indices = self.index.search(query, top_k)
        return indices[0], scores[0]

    def batch_search(
        self,
        query_embeddings: np.ndarray,
        top_k: int = 10,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Search for multiple queries at once."""
        if self.index is None:
            raise RuntimeError("Index not built. Call build() first.")

        queries = query_embeddings.astype(np.float32)
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)

        scores, indices = self.index.search(queries, top_k)
        return indices, scores

    def save(self, path: Union[str, Path]) -> None:
        """Save index to disk."""
        if self.index is None:
            raise RuntimeError("Index not built. Call build() first.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))
        print(f"Saved index to {path}")

    def load(self, path: Union[str, Path]) -> None:
        """Load index from disk."""
        self.index = faiss.read_index(str(path))
        self.is_trained = True
        print(f"Loaded index from {path} ({self.index.ntotal} vectors)")

    @property
    def size(self) -> int:
        """Return number of vectors in index."""
        if self.index is None:
            return 0
        return self.index.ntotal

    def __repr__(self) -> str:
        return (
            f"SearchIndex(type={self.index_type}, "
            f"dim={self.dimension}, size={self.size})"
        )
