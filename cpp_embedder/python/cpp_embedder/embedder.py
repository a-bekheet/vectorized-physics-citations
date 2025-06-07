"""
Python bindings for cpp_embedder using ctypes.

This module provides a Pythonic interface to the C++ embedder library,
handling library loading, type conversions, and memory management.
"""

import ctypes
import os
import platform
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import numpy.typing as npt

# =============================================================================
# Exception Classes
# =============================================================================


class EmbedderError(Exception):
    """Base exception for all embedder errors."""

    def __init__(self, message: str, code: int = 0):
        super().__init__(message)
        self._code = code

    @property
    def code(self) -> int:
        """Return the underlying C++ error code."""
        return self._code


class FileNotFoundError(EmbedderError):
    """Weight file or vocabulary file not found."""
    pass


class FormatError(EmbedderError):
    """Invalid weight file format."""
    pass


class TokenizerError(EmbedderError):
    """Tokenization error."""
    pass


class InferenceError(EmbedderError):
    """Model inference error."""
    pass


# Error code to exception mapping
_ERROR_MAP = {
    1: FileNotFoundError,   # CPP_EMBEDDER_ERROR_FILE_NOT_FOUND
    2: FormatError,         # CPP_EMBEDDER_ERROR_FILE_READ
    3: FormatError,         # CPP_EMBEDDER_ERROR_FILE_FORMAT
    10: FormatError,        # CPP_EMBEDDER_ERROR_INVALID_MAGIC
    11: FormatError,        # CPP_EMBEDDER_ERROR_UNSUPPORTED_VERSION
    12: FormatError,        # CPP_EMBEDDER_ERROR_WEIGHT_MISMATCH
    13: FormatError,        # CPP_EMBEDDER_ERROR_CHECKSUM_MISMATCH
    20: TokenizerError,     # CPP_EMBEDDER_ERROR_VOCAB_NOT_LOADED
    21: TokenizerError,     # CPP_EMBEDDER_ERROR_TOKEN_NOT_FOUND
    22: TokenizerError,     # CPP_EMBEDDER_ERROR_SEQUENCE_TOO_LONG
    30: InferenceError,     # CPP_EMBEDDER_ERROR_MODEL_NOT_LOADED
    31: InferenceError,     # CPP_EMBEDDER_ERROR_INVALID_INPUT
    32: InferenceError,     # CPP_EMBEDDER_ERROR_COMPUTATION
    40: MemoryError,        # CPP_EMBEDDER_ERROR_ALLOCATION_FAILED
    50: ValueError,         # CPP_EMBEDDER_ERROR_INVALID_CONFIG
    60: ValueError,         # CPP_EMBEDDER_ERROR_NULL_POINTER
    61: ValueError,         # CPP_EMBEDDER_ERROR_BUFFER_TOO_SMALL
}

# =============================================================================
# Library Loading
# =============================================================================


def _get_lib_name() -> str:
    """Get the platform-specific library name."""
    system = platform.system()
    if system == "Darwin":
        return "libcpp_embedder.dylib"
    elif system == "Windows":
        return "cpp_embedder.dll"
    else:  # Linux and others
        return "libcpp_embedder.so"


def _find_library() -> Optional[Path]:
    """Find the shared library in common locations."""
    lib_name = _get_lib_name()

    # Search paths in order of priority
    search_paths = [
        # 1. Environment variable
        os.environ.get("CPP_EMBEDDER_LIB_PATH"),
        # 2. Alongside the Python package
        Path(__file__).parent / lib_name,
        Path(__file__).parent / "lib" / lib_name,
        # 3. In the build directory (for development)
        Path(__file__).parent.parent.parent / "build" / lib_name,
        Path(__file__).parent.parent.parent / "build" / "lib" / lib_name,
        # 4. System library paths
        Path("/usr/local/lib") / lib_name,
        Path("/usr/lib") / lib_name,
    ]

    # On macOS, also check Homebrew locations
    if platform.system() == "Darwin":
        search_paths.extend([
            Path("/opt/homebrew/lib") / lib_name,
            Path("/usr/local/opt/cpp_embedder/lib") / lib_name,
        ])

    for path in search_paths:
        if path is None:
            continue
        path = Path(path)
        if path.exists():
            return path

    return None


def _load_library() -> ctypes.CDLL:
    """Load the cpp_embedder shared library."""
    lib_path = _find_library()

    if lib_path is None:
        lib_name = _get_lib_name()
        raise OSError(
            f"Could not find {lib_name}. "
            f"Set CPP_EMBEDDER_LIB_PATH environment variable or "
            f"install the library to a standard location."
        )

    try:
        return ctypes.CDLL(str(lib_path))
    except OSError as e:
        raise OSError(f"Failed to load library from {lib_path}: {e}")


# Global library instance (lazy loaded)
_lib: Optional[ctypes.CDLL] = None


def _get_lib() -> ctypes.CDLL:
    """Get the loaded library, loading it if necessary."""
    global _lib
    if _lib is None:
        _lib = _load_library()
        _setup_function_signatures(_lib)
    return _lib


def _setup_function_signatures(lib: ctypes.CDLL) -> None:
    """Set up ctypes function signatures for type safety."""

    # Version functions
    lib.embedder_version.argtypes = []
    lib.embedder_version.restype = ctypes.c_char_p

    lib.embedder_build_info.argtypes = []
    lib.embedder_build_info.restype = ctypes.c_char_p

    # Error functions
    lib.embedder_get_error.argtypes = []
    lib.embedder_get_error.restype = ctypes.c_char_p

    lib.embedder_get_error_code.argtypes = []
    lib.embedder_get_error_code.restype = ctypes.c_int

    lib.embedder_clear_error.argtypes = []
    lib.embedder_clear_error.restype = None

    # Lifecycle functions
    lib.embedder_create.argtypes = [ctypes.c_char_p]
    lib.embedder_create.restype = ctypes.c_void_p

    lib.embedder_create_with_config.argtypes = [
        ctypes.c_char_p,  # weights_path
        ctypes.c_uint32,  # max_seq_length
        ctypes.c_int,     # normalize
        ctypes.c_uint32,  # num_threads
    ]
    lib.embedder_create_with_config.restype = ctypes.c_void_p

    lib.embedder_load.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.embedder_load.restype = ctypes.c_int

    lib.embedder_destroy.argtypes = [ctypes.c_void_p]
    lib.embedder_destroy.restype = None

    lib.embedder_is_loaded.argtypes = [ctypes.c_void_p]
    lib.embedder_is_loaded.restype = ctypes.c_int

    # Property functions
    lib.embedder_get_dim.argtypes = [ctypes.c_void_p]
    lib.embedder_get_dim.restype = ctypes.c_uint32

    lib.embedder_get_max_seq_length.argtypes = [ctypes.c_void_p]
    lib.embedder_get_max_seq_length.restype = ctypes.c_uint32

    lib.embedder_get_vocab_size.argtypes = [ctypes.c_void_p]
    lib.embedder_get_vocab_size.restype = ctypes.c_uint32

    # Embedding functions
    lib.embedder_embed.argtypes = [
        ctypes.c_void_p,                              # handle
        ctypes.c_char_p,                              # text
        ctypes.POINTER(ctypes.c_float),               # output
        ctypes.c_uint32,                              # output_size
    ]
    lib.embedder_embed.restype = ctypes.c_int

    lib.embedder_embed_ex.argtypes = [
        ctypes.c_void_p,                              # handle
        ctypes.c_char_p,                              # text
        ctypes.POINTER(ctypes.c_float),               # output
        ctypes.c_uint32,                              # output_size
        ctypes.c_int,                                 # normalize
    ]
    lib.embedder_embed_ex.restype = ctypes.c_int

    lib.embedder_embed_batch.argtypes = [
        ctypes.c_void_p,                              # handle
        ctypes.POINTER(ctypes.c_char_p),              # texts
        ctypes.c_uint32,                              # num_texts
        ctypes.POINTER(ctypes.c_float),               # output
        ctypes.c_uint32,                              # output_size
    ]
    lib.embedder_embed_batch.restype = ctypes.c_int

    lib.embedder_embed_batch_ex.argtypes = [
        ctypes.c_void_p,                              # handle
        ctypes.POINTER(ctypes.c_char_p),              # texts
        ctypes.c_uint32,                              # num_texts
        ctypes.POINTER(ctypes.c_float),               # output
        ctypes.c_uint32,                              # output_size
        ctypes.c_int,                                 # normalize
    ]
    lib.embedder_embed_batch_ex.restype = ctypes.c_int

    # Tokenizer functions
    lib.embedder_get_tokenizer.argtypes = [ctypes.c_void_p]
    lib.embedder_get_tokenizer.restype = ctypes.c_void_p

    lib.tokenizer_encode.argtypes = [
        ctypes.c_void_p,                              # tokenizer
        ctypes.c_char_p,                              # text
        ctypes.POINTER(ctypes.c_uint32),              # output
        ctypes.c_uint32,                              # output_size
        ctypes.POINTER(ctypes.c_uint32),              # actual_length
        ctypes.c_uint32,                              # max_length
    ]
    lib.tokenizer_encode.restype = ctypes.c_int

    lib.tokenizer_decode.argtypes = [
        ctypes.c_void_p,                              # tokenizer
        ctypes.POINTER(ctypes.c_uint32),              # token_ids
        ctypes.c_uint32,                              # num_tokens
        ctypes.c_char_p,                              # output
        ctypes.c_uint32,                              # output_size
        ctypes.c_int,                                 # skip_special
    ]
    lib.tokenizer_decode.restype = ctypes.c_int

    lib.tokenizer_cls_token_id.argtypes = [ctypes.c_void_p]
    lib.tokenizer_cls_token_id.restype = ctypes.c_uint32

    lib.tokenizer_sep_token_id.argtypes = [ctypes.c_void_p]
    lib.tokenizer_sep_token_id.restype = ctypes.c_uint32

    lib.tokenizer_pad_token_id.argtypes = [ctypes.c_void_p]
    lib.tokenizer_pad_token_id.restype = ctypes.c_uint32

    lib.tokenizer_unk_token_id.argtypes = [ctypes.c_void_p]
    lib.tokenizer_unk_token_id.restype = ctypes.c_uint32

    # Utility functions
    lib.embedder_cosine_similarity.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_uint32,
    ]
    lib.embedder_cosine_similarity.restype = ctypes.c_float

    lib.embedder_normalize.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_uint32,
    ]
    lib.embedder_normalize.restype = None

    lib.embedder_norm.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_uint32,
    ]
    lib.embedder_norm.restype = ctypes.c_float

    lib.embedder_dot_product.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_uint32,
    ]
    lib.embedder_dot_product.restype = ctypes.c_float


def _check_error(code: int) -> None:
    """Check error code and raise appropriate exception."""
    if code == 0:
        return

    lib = _get_lib()
    message = lib.embedder_get_error()
    if message:
        message = message.decode("utf-8")
    else:
        message = f"Unknown error (code {code})"

    exc_class = _ERROR_MAP.get(code, EmbedderError)
    raise exc_class(message, code)


# =============================================================================
# Tokenizer Class
# =============================================================================


class Tokenizer:
    """
    WordPiece tokenizer wrapper.

    This class provides access to the C++ tokenizer through the embedder.
    The tokenizer is owned by the embedder and should not be used after
    the embedder is destroyed.
    """

    def __init__(self, handle: ctypes.c_void_p):
        """Initialize with a tokenizer handle (internal use only)."""
        self._handle = handle
        self._lib = _get_lib()

    def encode(
        self,
        text: str,
        max_length: int = 256,
    ) -> List[int]:
        """
        Encode text to token IDs.

        Args:
            text: Input text to encode.
            max_length: Maximum sequence length (including special tokens).

        Returns:
            List of token IDs with padding.

        Raises:
            TokenizerError: If encoding fails.
        """
        text_bytes = text.encode("utf-8")
        output = (ctypes.c_uint32 * max_length)()
        actual_length = ctypes.c_uint32()

        code = self._lib.tokenizer_encode(
            self._handle,
            text_bytes,
            output,
            max_length,
            ctypes.byref(actual_length),
            max_length,
        )
        _check_error(code)

        return list(output[:actual_length.value])

    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """
        Decode token IDs to text.

        Args:
            token_ids: List of token IDs to decode.
            skip_special_tokens: Whether to skip special tokens in output.

        Returns:
            Decoded text string.

        Raises:
            TokenizerError: If decoding fails.
        """
        num_tokens = len(token_ids)
        ids_array = (ctypes.c_uint32 * num_tokens)(*token_ids)
        output_size = 4096  # Reasonable buffer size
        output = ctypes.create_string_buffer(output_size)

        code = self._lib.tokenizer_decode(
            self._handle,
            ids_array,
            num_tokens,
            output,
            output_size,
            1 if skip_special_tokens else 0,
        )
        _check_error(code)

        return output.value.decode("utf-8")

    @property
    def cls_token_id(self) -> int:
        """Return [CLS] token ID."""
        return self._lib.tokenizer_cls_token_id(self._handle)

    @property
    def sep_token_id(self) -> int:
        """Return [SEP] token ID."""
        return self._lib.tokenizer_sep_token_id(self._handle)

    @property
    def pad_token_id(self) -> int:
        """Return [PAD] token ID."""
        return self._lib.tokenizer_pad_token_id(self._handle)

    @property
    def unk_token_id(self) -> int:
        """Return [UNK] token ID."""
        return self._lib.tokenizer_unk_token_id(self._handle)


# =============================================================================
# Embedder Class
# =============================================================================


class Embedder:
    """
    Sentence embedder using all-MiniLM-L6-v2 architecture.

    Drop-in replacement for the Python PaperEmbedder class with
    accelerated C++ inference.

    Example:
        embedder = Embedder("model.weights")
        vec = embedder.embed("hello world")  # shape (384,)
        vecs = embedder.embed_batch(["hello", "world"])  # shape (2, 384)
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        *,
        max_seq_length: int = 256,
        normalize_embeddings: bool = True,
        num_threads: int = 0,
    ) -> None:
        """
        Initialize the embedder.

        Args:
            weights_path: Path to binary weight file. If None, model must
                         be loaded separately via load().
            max_seq_length: Maximum input sequence length (1-512).
            normalize_embeddings: Whether to L2-normalize output embeddings.
            num_threads: Number of threads for computation (0=auto-detect).

        Raises:
            FileNotFoundError: If weights_path is provided but file not found.
            FormatError: If weight file has invalid format.
        """
        self._lib = _get_lib()
        self._handle: Optional[ctypes.c_void_p] = None
        self._tokenizer: Optional[Tokenizer] = None

        weights_bytes = weights_path.encode("utf-8") if weights_path else None

        self._handle = self._lib.embedder_create_with_config(
            weights_bytes,
            max_seq_length,
            1 if normalize_embeddings else 0,
            num_threads,
        )

        if self._handle is None:
            code = self._lib.embedder_get_error_code()
            _check_error(code if code != 0 else 99)

    def __del__(self) -> None:
        """Clean up resources."""
        if self._handle is not None:
            self._lib.embedder_destroy(self._handle)
            self._handle = None

    def __enter__(self) -> "Embedder":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        if self._handle is not None:
            self._lib.embedder_destroy(self._handle)
            self._handle = None

    def load(self, weights_path: str) -> None:
        """
        Load model weights from file.

        Args:
            weights_path: Path to binary weight file.

        Raises:
            FileNotFoundError: If file not found.
            FormatError: If file has invalid format.
        """
        if self._handle is None:
            raise RuntimeError("Embedder has been destroyed")

        weights_bytes = weights_path.encode("utf-8")
        code = self._lib.embedder_load(self._handle, weights_bytes)
        _check_error(code)

    @property
    def is_loaded(self) -> bool:
        """Return True if model is loaded and ready."""
        if self._handle is None:
            return False
        return self._lib.embedder_is_loaded(self._handle) != 0

    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension (384)."""
        if self._handle is None:
            return 0
        return self._lib.embedder_get_dim(self._handle)

    @property
    def max_seq_length(self) -> int:
        """Return maximum sequence length."""
        if self._handle is None:
            return 0
        return self._lib.embedder_get_max_seq_length(self._handle)

    @property
    def tokenizer(self) -> Tokenizer:
        """Access the underlying tokenizer."""
        if self._handle is None:
            raise RuntimeError("Embedder has been destroyed")

        if self._tokenizer is None:
            tok_handle = self._lib.embedder_get_tokenizer(self._handle)
            self._tokenizer = Tokenizer(tok_handle)

        return self._tokenizer

    def embed(
        self,
        texts: Union[str, List[str]],
        *,
        normalize: bool = True,
        show_progress: bool = False,
        batch_size: int = 32,
    ) -> npt.NDArray[np.float32]:
        """
        Generate embeddings for text(s).

        This method matches the signature of PaperEmbedder.embed().

        Args:
            texts: Single text string or list of texts.
            normalize: Whether to L2-normalize embeddings.
            show_progress: Whether to show progress bar (currently ignored).
            batch_size: Batch size for processing (currently ignored).

        Returns:
            numpy.ndarray of shape:
                - [embedding_dim] if single text
                - [num_texts, embedding_dim] if list of texts

        Raises:
            InferenceError: If inference fails.
        """
        if isinstance(texts, str):
            return self.embed_single(texts, normalize=normalize)
        else:
            return self.embed_batch(texts, normalize=normalize)

    def embed_single(
        self,
        text: str,
        normalize: bool = True,
    ) -> npt.NDArray[np.float32]:
        """
        Generate embedding for a single text.

        Args:
            text: Input text string.
            normalize: Whether to L2-normalize the embedding.

        Returns:
            numpy.ndarray of shape [embedding_dim].

        Raises:
            InferenceError: If inference fails.
        """
        if self._handle is None:
            raise RuntimeError("Embedder has been destroyed")

        dim = self.embedding_dim
        output = np.empty(dim, dtype=np.float32)
        output_ptr = output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        text_bytes = text.encode("utf-8")

        code = self._lib.embedder_embed_ex(
            self._handle,
            text_bytes,
            output_ptr,
            dim,
            1 if normalize else 0,
        )
        _check_error(code)

        return output

    def embed_batch(
        self,
        texts: List[str],
        *,
        normalize: bool = True,
    ) -> npt.NDArray[np.float32]:
        """
        Generate embeddings for a batch of texts.

        Args:
            texts: List of input texts.
            normalize: Whether to L2-normalize embeddings.

        Returns:
            numpy.ndarray of shape [num_texts, embedding_dim].

        Raises:
            InferenceError: If inference fails.
        """
        if self._handle is None:
            raise RuntimeError("Embedder has been destroyed")

        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        num_texts = len(texts)
        dim = self.embedding_dim
        output_size = num_texts * dim

        output = np.empty((num_texts, dim), dtype=np.float32)
        output_ptr = output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        # Convert Python strings to C strings
        text_bytes = [t.encode("utf-8") for t in texts]
        texts_array = (ctypes.c_char_p * num_texts)(*text_bytes)

        code = self._lib.embedder_embed_batch_ex(
            self._handle,
            texts_array,
            num_texts,
            output_ptr,
            output_size,
            1 if normalize else 0,
        )
        _check_error(code)

        return output

    def embed_into(
        self,
        texts: List[str],
        out: npt.NDArray[np.float32],
        *,
        normalize: bool = True,
    ) -> None:
        """
        Generate embeddings directly into a pre-allocated array.

        Args:
            texts: List of input texts.
            out: Pre-allocated output array of shape [len(texts), embedding_dim].
                Must be C-contiguous float32.
            normalize: Whether to L2-normalize embeddings.

        Raises:
            ValueError: If out has wrong shape or dtype.
            InferenceError: If inference fails.
        """
        if self._handle is None:
            raise RuntimeError("Embedder has been destroyed")

        if not texts:
            return

        num_texts = len(texts)
        dim = self.embedding_dim

        # Validate output array
        if out.dtype != np.float32:
            raise ValueError(f"Output array must be float32, got {out.dtype}")
        if not out.flags["C_CONTIGUOUS"]:
            raise ValueError("Output array must be C-contiguous")
        if out.shape != (num_texts, dim):
            raise ValueError(
                f"Output array has wrong shape: expected ({num_texts}, {dim}), "
                f"got {out.shape}"
            )

        output_ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        output_size = num_texts * dim

        text_bytes = [t.encode("utf-8") for t in texts]
        texts_array = (ctypes.c_char_p * num_texts)(*text_bytes)

        code = self._lib.embedder_embed_batch_ex(
            self._handle,
            texts_array,
            num_texts,
            output_ptr,
            output_size,
            1 if normalize else 0,
        )
        _check_error(code)

    @property
    def metadata(self) -> Dict[str, Any]:
        """
        Return model metadata.

        Returns:
            Dictionary with model information.
        """
        return {
            "model_name": "all-MiniLM-L6-v2",
            "model_version": "1.0.0",
            "embedding_dim": self.embedding_dim,
            "vocab_size": 30522,
            "num_layers": 6,
            "num_attention_heads": 12,
            "max_seq_length": self.max_seq_length,
        }

    def __repr__(self) -> str:
        """Return string representation."""
        status = "loaded" if self.is_loaded else "not loaded"
        return (
            f"Embedder(embedding_dim={self.embedding_dim}, "
            f"max_seq_length={self.max_seq_length}, "
            f"status={status})"
        )


# =============================================================================
# Utility Functions
# =============================================================================


def cosine_similarity(
    a: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
) -> Union[float, npt.NDArray[np.float32]]:
    """
    Compute cosine similarity between embeddings.

    Args:
        a: First embedding(s), shape [dim] or [n, dim].
        b: Second embedding(s), shape [dim] or [m, dim].

    Returns:
        - float if both are 1D
        - np.ndarray of shape [n] if a is 2D, b is 1D
        - np.ndarray of shape [m] if a is 1D, b is 2D
        - np.ndarray of shape [n, m] if both are 2D
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    # Handle 1D vs 2D cases
    a_is_1d = a.ndim == 1
    b_is_1d = b.ndim == 1

    if a_is_1d:
        a = a.reshape(1, -1)
    if b_is_1d:
        b = b.reshape(1, -1)

    # Normalize
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)

    # Compute similarities
    similarities = np.dot(a_norm, b_norm.T)

    # Reshape output based on input shapes
    if a_is_1d and b_is_1d:
        return float(similarities[0, 0])
    elif a_is_1d:
        return similarities[0]
    elif b_is_1d:
        return similarities[:, 0]
    else:
        return similarities


def find_similar(
    query: npt.NDArray[np.float32],
    corpus: npt.NDArray[np.float32],
    top_k: int = 10,
) -> Tuple[npt.NDArray[np.intp], npt.NDArray[np.float32]]:
    """
    Find top-k most similar embeddings in corpus.

    Args:
        query: Query embedding, shape [dim].
        corpus: Corpus embeddings, shape [n, dim].
        top_k: Number of results to return.

    Returns:
        Tuple of (indices, scores) where:
        - indices: np.ndarray of shape [top_k] with corpus indices
        - scores: np.ndarray of shape [top_k] with similarity scores

        Results are sorted by descending similarity.
    """
    query = np.asarray(query, dtype=np.float32)
    corpus = np.asarray(corpus, dtype=np.float32)

    if query.ndim != 1:
        raise ValueError(f"Query must be 1D, got shape {query.shape}")
    if corpus.ndim != 2:
        raise ValueError(f"Corpus must be 2D, got shape {corpus.shape}")

    # Compute all similarities
    similarities = cosine_similarity(query, corpus)

    # Get top-k indices
    top_k = min(top_k, len(corpus))
    indices = np.argpartition(similarities, -top_k)[-top_k:]
    indices = indices[np.argsort(similarities[indices])[::-1]]

    return indices, similarities[indices]


def normalize(embeddings: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """
    L2-normalize embeddings.

    Args:
        embeddings: Embeddings of shape [dim] or [n, dim].

    Returns:
        Normalized embeddings with same shape.
    """
    embeddings = np.asarray(embeddings, dtype=np.float32)

    if embeddings.ndim == 1:
        norm = np.linalg.norm(embeddings)
        if norm > 0:
            return embeddings / norm
        return embeddings.copy()
    else:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        return embeddings / norms
