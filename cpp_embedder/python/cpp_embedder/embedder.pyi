"""Type stubs for cpp_embedder.embedder module."""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt

class EmbedderError(Exception):
    def __init__(self, message: str, code: int = ...) -> None: ...
    @property
    def code(self) -> int: ...

class FileNotFoundError(EmbedderError): ...
class FormatError(EmbedderError): ...
class TokenizerError(EmbedderError): ...
class InferenceError(EmbedderError): ...

class Tokenizer:
    def encode(
        self,
        text: str,
        max_length: int = ...,
    ) -> List[int]: ...
    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = ...,
    ) -> str: ...
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
        weights_path: Optional[str] = ...,
        *,
        max_seq_length: int = ...,
        normalize_embeddings: bool = ...,
        num_threads: int = ...,
    ) -> None: ...
    def __enter__(self) -> Embedder: ...
    def __exit__(
        self,
        exc_type: Any,
        exc_val: Any,
        exc_tb: Any,
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
        normalize: bool = ...,
        show_progress: bool = ...,
        batch_size: int = ...,
    ) -> npt.NDArray[np.float32]: ...
    def embed_single(
        self,
        text: str,
        normalize: bool = ...,
    ) -> npt.NDArray[np.float32]: ...
    def embed_batch(
        self,
        texts: List[str],
        *,
        normalize: bool = ...,
    ) -> npt.NDArray[np.float32]: ...
    def embed_into(
        self,
        texts: List[str],
        out: npt.NDArray[np.float32],
        *,
        normalize: bool = ...,
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
    top_k: int = ...,
) -> Tuple[npt.NDArray[np.intp], npt.NDArray[np.float32]]: ...

def normalize(
    embeddings: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]: ...
