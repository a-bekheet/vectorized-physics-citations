"""
cpp_embedder - Pure C++ sentence embeddings for Python

A high-performance sentence embedding library implemented in pure C++,
providing Python bindings via ctypes. Drop-in replacement for the
Python PaperEmbedder class.

Example usage:
    from cpp_embedder import Embedder
    import numpy as np

    embedder = Embedder("model.weights")
    vec = embedder.embed("hello world")  # returns (384,) numpy array
    vecs = embedder.embed_batch(["hello", "world"])  # returns (2, 384) numpy array
"""

__version__ = "1.0.0"

# Model constants (matching all-MiniLM-L6-v2)
EMBEDDING_DIM = 384
VOCAB_SIZE = 30522
MAX_POSITION_EMBEDDINGS = 512
NUM_LAYERS = 6
NUM_ATTENTION_HEADS = 12

# Import main classes
from .embedder import Embedder, Tokenizer
from .embedder import (
    EmbedderError,
    FileNotFoundError as EmbedderFileNotFoundError,
    FormatError,
    TokenizerError,
    InferenceError,
)
from .embedder import (
    cosine_similarity,
    find_similar,
    normalize,
)

__all__ = [
    # Version
    "__version__",
    # Constants
    "EMBEDDING_DIM",
    "VOCAB_SIZE",
    "MAX_POSITION_EMBEDDINGS",
    "NUM_LAYERS",
    "NUM_ATTENTION_HEADS",
    # Classes
    "Embedder",
    "Tokenizer",
    # Exceptions
    "EmbedderError",
    "EmbedderFileNotFoundError",
    "FormatError",
    "TokenizerError",
    "InferenceError",
    # Utility functions
    "cosine_similarity",
    "find_similar",
    "normalize",
]
