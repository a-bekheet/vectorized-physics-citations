# Python Bindings Interface Contract

## Overview

This document defines the Python interface for the `cpp_embedder` library, providing a drop-in replacement for the existing Python `PaperEmbedder` class with identical semantics but accelerated C++ inference.

## Module Name

```python
import cpp_embedder
```

---

## Module-Level Constants

```python
# Library version
cpp_embedder.__version__: str  # e.g., "1.0.0"

# Model constants
cpp_embedder.EMBEDDING_DIM: int = 384
cpp_embedder.VOCAB_SIZE: int = 30522
cpp_embedder.MAX_POSITION_EMBEDDINGS: int = 512
cpp_embedder.NUM_LAYERS: int = 6
cpp_embedder.NUM_ATTENTION_HEADS: int = 12
```

---

## Exception Classes

```python
class EmbedderError(Exception):
    """Base exception for all embedder errors."""

    @property
    def code(self) -> int:
        """Return the underlying C++ error code."""
        ...


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
```

---

## Tokenizer Class

```python
class Tokenizer:
    """WordPiece tokenizer compatible with BERT uncased vocabulary."""

    def __init__(
        self,
        vocab_path: Optional[str] = None,
        *,
        do_lower_case: bool = True,
        max_chars_per_word: int = 200,
        unk_token: str = "[UNK]",
        sep_token: str = "[SEP]",
        cls_token: str = "[CLS]",
        pad_token: str = "[PAD]",
    ) -> None:
        """
        Initialize the tokenizer.

        Args:
            vocab_path: Path to vocabulary file. If None, vocabulary must
                        be loaded separately via load_vocab().
            do_lower_case: Whether to lowercase input before tokenization.
            max_chars_per_word: Maximum characters per word before character split.
            unk_token: Unknown token string.
            sep_token: Separator token string.
            cls_token: Classification token string.
            pad_token: Padding token string.

        Raises:
            FileNotFoundError: If vocab_path is provided but file not found.
            FormatError: If vocabulary file has invalid format.
        """
        ...

    def load_vocab(self, path: str) -> None:
        """
        Load vocabulary from a text file (one token per line).

        Args:
            path: Path to vocabulary file.

        Raises:
            FileNotFoundError: If file not found.
            FormatError: If file has invalid format.
        """
        ...

    @property
    def is_loaded(self) -> bool:
        """Return True if vocabulary is loaded."""
        ...

    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        ...

    def tokenize(
        self,
        text: str,
        max_length: int = 256,
        *,
        add_special_tokens: bool = True,
        padding: bool = True,
        return_tensors: Optional[str] = None,
    ) -> TokenizedInput:
        """
        Tokenize a single text string.

        Args:
            text: Input text to tokenize.
            max_length: Maximum sequence length (including special tokens).
            add_special_tokens: Whether to add [CLS] and [SEP] tokens.
            padding: Whether to pad to max_length.
            return_tensors: If "np", return numpy arrays; otherwise lists.

        Returns:
            TokenizedInput object with input_ids, attention_mask, token_type_ids.

        Raises:
            TokenizerError: If tokenization fails.
        """
        ...

    def tokenize_batch(
        self,
        texts: List[str],
        max_length: int = 256,
        *,
        add_special_tokens: bool = True,
        padding: bool = True,
        return_tensors: Optional[str] = None,
    ) -> BatchTokenizedInput:
        """
        Tokenize a batch of text strings.

        Args:
            texts: List of input texts to tokenize.
            max_length: Maximum sequence length (including special tokens).
            add_special_tokens: Whether to add [CLS] and [SEP] tokens.
            padding: Whether to pad to max_length.
            return_tensors: If "np", return numpy arrays; otherwise lists.

        Returns:
            BatchTokenizedInput with batched arrays/lists.

        Raises:
            TokenizerError: If tokenization fails.
        """
        ...

    def encode(self, text: str, max_length: int = 256) -> List[int]:
        """
        Tokenize and convert to token IDs.

        Args:
            text: Input text.
            max_length: Maximum sequence length.

        Returns:
            List of token IDs.
        """
        ...

    def decode(
        self,
        token_ids: List[int],
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        """
        Convert token IDs back to text.

        Args:
            token_ids: List of token IDs.
            skip_special_tokens: Whether to skip special tokens in output.

        Returns:
            Decoded text string.
        """
        ...

    def token_to_id(self, token: str) -> int:
        """Convert token string to ID. Returns unk_token_id if not found."""
        ...

    def id_to_token(self, id: int) -> str:
        """Convert ID to token string. Raises IndexError if out of range."""
        ...

    @property
    def cls_token_id(self) -> int:
        """Return [CLS] token ID."""
        ...

    @property
    def sep_token_id(self) -> int:
        """Return [SEP] token ID."""
        ...

    @property
    def pad_token_id(self) -> int:
        """Return [PAD] token ID."""
        ...

    @property
    def unk_token_id(self) -> int:
        """Return [UNK] token ID."""
        ...
```

---

## TokenizedInput Classes

```python
class TokenizedInput:
    """Container for tokenized single input."""

    @property
    def input_ids(self) -> Union[List[int], np.ndarray]:
        """Token IDs."""
        ...

    @property
    def attention_mask(self) -> Union[List[int], np.ndarray]:
        """Attention mask (1 for real tokens, 0 for padding)."""
        ...

    @property
    def token_type_ids(self) -> Union[List[int], np.ndarray]:
        """Token type IDs (segment IDs)."""
        ...

    @property
    def tokens(self) -> List[str]:
        """Original token strings (for debugging)."""
        ...

    def __len__(self) -> int:
        """Return sequence length (including padding)."""
        ...


class BatchTokenizedInput:
    """Container for tokenized batch input."""

    @property
    def input_ids(self) -> Union[List[List[int]], np.ndarray]:
        """Token IDs with shape [batch_size, seq_length]."""
        ...

    @property
    def attention_mask(self) -> Union[List[List[int]], np.ndarray]:
        """Attention masks with shape [batch_size, seq_length]."""
        ...

    @property
    def token_type_ids(self) -> Union[List[List[int]], np.ndarray]:
        """Token type IDs with shape [batch_size, seq_length]."""
        ...

    def __len__(self) -> int:
        """Return batch size."""
        ...

    def __getitem__(self, index: int) -> TokenizedInput:
        """Get individual TokenizedInput by index."""
        ...
```

---

## Embedder Class

```python
class Embedder:
    """
    Sentence embedder using all-MiniLM-L6-v2 architecture.

    Drop-in replacement for the Python PaperEmbedder class.
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
            max_seq_length: Maximum input sequence length (truncates longer).
            normalize_embeddings: Whether to L2-normalize output embeddings.
            num_threads: Number of threads for computation (0 = auto-detect).

        Raises:
            FileNotFoundError: If weights_path is provided but file not found.
            FormatError: If weight file has invalid format.
        """
        ...

    def load(self, weights_path: str) -> None:
        """
        Load model weights from file.

        Args:
            weights_path: Path to binary weight file.

        Raises:
            FileNotFoundError: If file not found.
            FormatError: If file has invalid format.
        """
        ...

    @property
    def is_loaded(self) -> bool:
        """Return True if model is loaded and ready."""
        ...

    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension (384)."""
        ...

    @property
    def max_seq_length(self) -> int:
        """Return maximum sequence length."""
        ...

    @property
    def tokenizer(self) -> Tokenizer:
        """Access the underlying tokenizer."""
        ...

    def embed(
        self,
        texts: Union[str, List[str]],
        *,
        normalize: bool = True,
        show_progress: bool = False,
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Generate embeddings for text(s).

        This method matches the signature of PaperEmbedder.embed().

        Args:
            texts: Single text string or list of texts.
            normalize: Whether to L2-normalize embeddings (overrides config).
            show_progress: Whether to show progress bar for batch processing.
            batch_size: Batch size for processing (affects memory usage).

        Returns:
            numpy.ndarray of shape:
                - [embedding_dim] if single text
                - [num_texts, embedding_dim] if list of texts

        Raises:
            InferenceError: If inference fails.
        """
        ...

    def embed_single(
        self,
        text: str,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Generate embedding for a single text.

        This method matches the signature of PaperEmbedder.embed_single().

        Args:
            text: Input text string.
            normalize: Whether to L2-normalize the embedding.

        Returns:
            numpy.ndarray of shape [embedding_dim].

        Raises:
            InferenceError: If inference fails.
        """
        ...

    def embed_tokenized(
        self,
        tokenized: Union[TokenizedInput, BatchTokenizedInput],
        *,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Generate embeddings from pre-tokenized input.

        Args:
            tokenized: Pre-tokenized input from Tokenizer.
            normalize: Whether to L2-normalize embeddings.

        Returns:
            numpy.ndarray of embeddings.

        Raises:
            InferenceError: If inference fails.
        """
        ...

    @property
    def metadata(self) -> Dict[str, Any]:
        """
        Return model metadata from weight file.

        Returns:
            Dictionary with keys: model_name, model_version, embedding_dim,
            vocab_size, num_layers, num_attention_heads, etc.
        """
        ...

    def __repr__(self) -> str:
        """Return string representation."""
        ...
```

---

## Utility Functions

```python
def cosine_similarity(
    a: np.ndarray,
    b: np.ndarray,
) -> Union[float, np.ndarray]:
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
    ...


def find_similar(
    query: np.ndarray,
    corpus: np.ndarray,
    top_k: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
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
    ...


def normalize(embeddings: np.ndarray) -> np.ndarray:
    """
    L2-normalize embeddings.

    Args:
        embeddings: Embeddings of shape [dim] or [n, dim].

    Returns:
        Normalized embeddings with same shape.
    """
    ...
```

---

## NumPy Array Interop

### Memory Layout

- All returned `np.ndarray` objects use C-contiguous (row-major) layout
- Data type is `np.float32` for embeddings
- Data type is `np.int32` for token IDs and masks
- Arrays are writable and own their data (no shared memory with C++)

### Zero-Copy Options

For performance-critical applications, the following zero-copy methods are available:

```python
class Embedder:
    def embed_into(
        self,
        texts: List[str],
        out: np.ndarray,
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
        ...
```

---

## Compatibility with PaperEmbedder

The `Embedder` class is designed as a drop-in replacement for `PaperEmbedder`:

| PaperEmbedder Method | Embedder Method | Notes |
|---------------------|-----------------|-------|
| `__init__(model_name, device, batch_size, max_seq_length)` | `__init__(weights_path, max_seq_length, normalize_embeddings, num_threads)` | Different params |
| `embed(texts, normalize, show_progress)` | `embed(texts, normalize, show_progress, batch_size)` | Compatible |
| `embed_single(text, normalize)` | `embed_single(text, normalize)` | Identical |
| `embedding_dim` | `embedding_dim` | Identical |
| `compute_similarity(query, corpus)` | `cosine_similarity(query, corpus)` | Module function |
| `find_similar(query, corpus, top_k)` | `find_similar(query, corpus, top_k)` | Module function |

### Migration Example

```python
# Before (Python + PyTorch)
from models import PaperEmbedder

embedder = PaperEmbedder(
    model_name="all-MiniLM-L6-v2",
    device="cpu",
    batch_size=64,
    max_seq_length=512,
)
embedding = embedder.embed_single("Hello world")


# After (Pure C++)
import cpp_embedder

embedder = cpp_embedder.Embedder(
    weights_path="model.weights",
    max_seq_length=512,
)
embedding = embedder.embed_single("Hello world")
```

---

## Thread Safety

1. **`Tokenizer`**: Thread-safe after construction. Multiple threads may call methods concurrently.

2. **`Embedder`**: Thread-safe after loading. Multiple threads may call `embed()` concurrently.

3. **GIL Release**: Long-running operations (embedding computation) release the GIL to allow other Python threads to run.

---

## Build Requirements

The Python bindings require:
- Python >= 3.8
- NumPy >= 1.19
- pybind11 >= 2.10 (build only)

---

## Type Stubs

Type stubs (`cpp_embedder.pyi`) are provided for IDE support:

```python
# cpp_embedder.pyi
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import numpy.typing as npt

__version__: str
EMBEDDING_DIM: int
VOCAB_SIZE: int
MAX_POSITION_EMBEDDINGS: int
NUM_LAYERS: int
NUM_ATTENTION_HEADS: int

class EmbedderError(Exception):
    @property
    def code(self) -> int: ...

class FileNotFoundError(EmbedderError): ...
class FormatError(EmbedderError): ...
class TokenizerError(EmbedderError): ...
class InferenceError(EmbedderError): ...

class TokenizedInput:
    @property
    def input_ids(self) -> Union[List[int], npt.NDArray[np.int32]]: ...
    @property
    def attention_mask(self) -> Union[List[int], npt.NDArray[np.int32]]: ...
    @property
    def token_type_ids(self) -> Union[List[int], npt.NDArray[np.int32]]: ...
    @property
    def tokens(self) -> List[str]: ...
    def __len__(self) -> int: ...

class BatchTokenizedInput:
    @property
    def input_ids(self) -> Union[List[List[int]], npt.NDArray[np.int32]]: ...
    @property
    def attention_mask(self) -> Union[List[List[int]], npt.NDArray[np.int32]]: ...
    @property
    def token_type_ids(self) -> Union[List[List[int]], npt.NDArray[np.int32]]: ...
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> TokenizedInput: ...

class Tokenizer:
    def __init__(
        self,
        vocab_path: Optional[str] = None,
        *,
        do_lower_case: bool = True,
        max_chars_per_word: int = 200,
        unk_token: str = "[UNK]",
        sep_token: str = "[SEP]",
        cls_token: str = "[CLS]",
        pad_token: str = "[PAD]",
    ) -> None: ...
    def load_vocab(self, path: str) -> None: ...
    @property
    def is_loaded(self) -> bool: ...
    @property
    def vocab_size(self) -> int: ...
    def tokenize(
        self,
        text: str,
        max_length: int = 256,
        *,
        add_special_tokens: bool = True,
        padding: bool = True,
        return_tensors: Optional[str] = None,
    ) -> TokenizedInput: ...
    def tokenize_batch(
        self,
        texts: List[str],
        max_length: int = 256,
        *,
        add_special_tokens: bool = True,
        padding: bool = True,
        return_tensors: Optional[str] = None,
    ) -> BatchTokenizedInput: ...
    def encode(self, text: str, max_length: int = 256) -> List[int]: ...
    def decode(self, token_ids: List[int], *, skip_special_tokens: bool = True) -> str: ...
    def token_to_id(self, token: str) -> int: ...
    def id_to_token(self, id: int) -> str: ...
    @property
    def cls_token_id(self) -> int: ...
    @property
    def sep_token_id(self) -> int: ...
    @property
    def pad_token_id(self) -> int: ...
    @property
    def unk_token_id(self) -> int: ...

class Embedder:
    def __init__(
        self,
        weights_path: Optional[str] = None,
        *,
        max_seq_length: int = 256,
        normalize_embeddings: bool = True,
        num_threads: int = 0,
    ) -> None: ...
    def load(self, weights_path: str) -> None: ...
    @property
    def is_loaded(self) -> bool: ...
    @property
    def embedding_dim(self) -> int: ...
    @property
    def max_seq_length(self) -> int: ...
    @property
    def tokenizer(self) -> Tokenizer: ...
    def embed(
        self,
        texts: Union[str, List[str]],
        *,
        normalize: bool = True,
        show_progress: bool = False,
        batch_size: int = 32,
    ) -> npt.NDArray[np.float32]: ...
    def embed_single(self, text: str, normalize: bool = True) -> npt.NDArray[np.float32]: ...
    def embed_tokenized(
        self,
        tokenized: Union[TokenizedInput, BatchTokenizedInput],
        *,
        normalize: bool = True,
    ) -> npt.NDArray[np.float32]: ...
    def embed_into(
        self,
        texts: List[str],
        out: npt.NDArray[np.float32],
        *,
        normalize: bool = True,
    ) -> None: ...
    @property
    def metadata(self) -> Dict[str, Any]: ...
    def __repr__(self) -> str: ...

def cosine_similarity(
    a: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
) -> Union[float, npt.NDArray[np.float32]]: ...

def find_similar(
    query: npt.NDArray[np.float32],
    corpus: npt.NDArray[np.float32],
    top_k: int = 10,
) -> Tuple[npt.NDArray[np.intp], npt.NDArray[np.float32]]: ...

def normalize(embeddings: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]: ...
```

---

## Assumptions

1. **Python GIL**: The GIL is released during computation-heavy operations
2. **NumPy arrays**: All arrays use C-contiguous layout and standard dtypes
3. **String encoding**: All Python strings are assumed to be valid Unicode
4. **Memory ownership**: Returned arrays own their data (no shared memory)
